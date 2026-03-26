from __future__ import annotations

import asyncio
import math
import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from pydantic import Field
from pydrantic import RunConfig

from cartridges.clients.base import ClientConfig
from cartridges.benchmark.datasets import BenchmarkItem, load_dataset_items
from cartridges.benchmark.scorers import SCORER_REGISTRY, ScorerFn
from cartridges.utils import get_logger

logger = get_logger(__name__)


class BenchmarkConfig(RunConfig):
    # ---- dataset ----
    dataset_name: str                          # "mmlu", "qasper", "hellaswag", "truthfulqa", "local"
    dataset_path: Optional[str] = None         # HF id override or local file path
    subset: Optional[str] = None               # e.g. "abstract_algebra" for MMLU
    split: str = "test"
    num_few_shot: int = 0
    prompt_template: Optional[str] = None      # override the default per-dataset prompt
    max_samples: Optional[int] = None          # limit for quick runs

    # ---- inference ----
    client: ClientConfig
    temperature: float = 0.0
    max_completion_tokens: int = 256
    batch_size: int = 16
    max_concurrent_batches: int = 4

    # ---- scoring ----
    scorer: str = "exact_match"                # key into SCORER_REGISTRY

    # ---- output ----
    name: Optional[str] = "benchmark"
    output_dir: str = Field(default=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."))
    seed: int = 42

    def run(self):
        asyncio.run(_run_benchmark(self))


async def _run_benchmark(config: BenchmarkConfig):
    t_start = time.time()

    # --- load data ---
    items = load_dataset_items(
        config.dataset_name,
        dataset_path=config.dataset_path,
        subset=config.subset,
        split=config.split,
        num_few_shot=config.num_few_shot,
        prompt_template=config.prompt_template,
        max_samples=config.max_samples,
        seed=config.seed,
    )

    if not items:
        logger.error("No benchmark items loaded — aborting.")
        return

    # --- resolve scorer ---
    scorer_fn: ScorerFn = SCORER_REGISTRY.get(config.scorer)
    if scorer_fn is None:
        raise ValueError(
            f"Unknown scorer '{config.scorer}'. Available: {list(SCORER_REGISTRY.keys())}"
        )

    # --- instantiate client ---
    client = config.client.instantiate()
    logger.info(
        f"Running benchmark '{config.dataset_name}' "
        f"({len(items)} items, batch_size={config.batch_size}, "
        f"max_concurrent={config.max_concurrent_batches})"
    )

    # --- run batched inference ---
    total_batches = math.ceil(len(items) / config.batch_size)
    results: list[dict] = [None] * len(items)  # type: ignore[list-item]
    semaphore = asyncio.Semaphore(config.max_concurrent_batches)

    async def _process_batch(batch_idx: int):
        start = batch_idx * config.batch_size
        end = min(start + config.batch_size, len(items))
        batch_items = items[start:end]

        chats = [
            [{"role": "user", "content": item.prompt}]
            for item in batch_items
        ]

        async with semaphore:
            t0 = time.time()
            response = await client.chat(
                chats=chats,
                temperature=config.temperature,
                max_completion_tokens=config.max_completion_tokens,
            )
            elapsed = time.time() - t0

        logger.info(
            f"Batch {batch_idx + 1}/{total_batches} completed in {elapsed:.1f}s"
        )

        for local_idx, (item, sample) in enumerate(
            zip(batch_items, response.samples)
        ):
            prediction = sample.text or ""
            score = scorer_fn(prediction, item.ground_truth, metadata=item.metadata)
            results[start + local_idx] = {
                "prompt": item.prompt,
                "ground_truth": item.ground_truth,
                "prediction": prediction,
                "score": score,
                **item.metadata,
            }

    tasks = [_process_batch(i) for i in range(total_batches)]
    await asyncio.gather(*tasks)

    # --- build output dataframe ---
    df = pd.DataFrame(results)

    # --- save locally ---
    run_dir = Path(config.output_dir) / (config.name or "benchmark")
    run_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = run_dir / "results.parquet"
    csv_path = run_dir / "results.csv"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)

    # --- summary ---
    elapsed_total = time.time() - t_start
    avg_score = df["score"].mean()
    num_correct = (df["score"] == 1.0).sum()
    total = len(df)

    summary = (
        f"\n{'=' * 60}\n"
        f"  Benchmark: {config.dataset_name}"
        f"{(' / ' + config.subset) if config.subset else ''}\n"
        f"  Scorer:    {config.scorer}\n"
        f"  Samples:   {total}\n"
        f"  Score:     {avg_score:.4f}  ({num_correct}/{total} correct)\n"
        f"  Time:      {elapsed_total:.1f}s\n"
        f"  Output:    {run_dir.absolute()}\n"
        f"{'=' * 60}"
    )
    logger.info(summary)
    print(summary)

    # --- per-subset breakdown (e.g. MMLU subjects) ---
    if "subject" in df.columns:
        breakdown = (
            df.groupby("subject")["score"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "accuracy", "count": "n"})
            .sort_values("accuracy")
        )
        breakdown.to_csv(run_dir / "breakdown_by_subject.csv")
        logger.info(f"Per-subject breakdown saved to {run_dir / 'breakdown_by_subject.csv'}")
        print(f"\nPer-subject breakdown (bottom 10):\n{breakdown.head(10).to_string()}")

    return df
