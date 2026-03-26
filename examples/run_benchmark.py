"""
Example: run benchmarks with a Tokasaurus or OpenAI-compatible server.

Usage (with a running server):
    python run_benchmark.py

Or override via CLI (pydrantic):
    python run_benchmark.py dataset_name=hellaswag max_samples=200

To auto-launch a Tokasaurus server, see run_benchmark_with_server.py
or wrap with EvaluateTokaConfig the same way as qasper_synthesize_with_server.py.
"""

import os
from pathlib import Path

import pydrantic

from cartridges.benchmark import BenchmarkConfig
from cartridges.clients.tokasaurus import TokasaurusClient
from cartridges.clients.base import CartridgeConfig
from cartridges.clients.openai import OpenAIClient

# ---------- client configs ----------

# Model only (no cartridge)
toka_client = TokasaurusClient.Config(
    url="http://localhost:10210",
    model_name="Qwen/Qwen3-4B",
)

# # Model + cartridge — cartridges are sent with every chat() request
# # and automatically route to the /custom/cartridge/ endpoint on the Toka server
# toka_client = TokasaurusClient.Config(
#     url="http://localhost:10210",
#     model_name="Qwen/Qwen3-4B",
#     cartridges=[
#         CartridgeConfig(
#             id="qtris123/longhealth-p1-10_8192_1024_no-cartridge",
#             source="huggingface",
#             force_redownload=False,
#         )
#     ],
# )

# openai_client = OpenAIClient.Config(
#     base_url="http://localhost:8000/v1",
#     model_name="Qwen/Qwen3-4B",
# )

# ---------- benchmark configs ----------

configs = [
    # MMLU  — 5-shot, multiple-choice scoring
    BenchmarkConfig(
        name="mmlu_abstract_algebra",
        dataset_name="mmlu",
        subset="abstract_algebra",
        split="test",
        num_few_shot=5,
        client=toka_client,
        scorer="multiple_choice",
        temperature=0.0,
        max_completion_tokens=16,
        batch_size=32,
        max_concurrent_batches=4,
        max_samples=None,
        output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "./outputs"),
    ),

    # HellaSwag — 0-shot, multiple-choice
    BenchmarkConfig(
        name="hellaswag_quick",
        dataset_name="hellaswag",
        split="validation",
        num_few_shot=0,
        client=toka_client,
        scorer="multiple_choice",
        temperature=0.0,
        max_completion_tokens=16,
        batch_size=32,
        max_concurrent_batches=4,
        max_samples=200,
        output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "./outputs"),
    ),

    # QASPER — uses rewritten dataset (same as cartridges.data.qasper.evals)
    BenchmarkConfig(
        name="qasper_f1",
        dataset_name="qasper",
        # subset selects the HF split: "question" by default (matches QasperEvalDataset)
        # dataset_path can override the HF repo (default: qtris123/qtris123qasper-rewrite-gpt-4.1-MT-task)
        num_few_shot=0,
        client=toka_client,
        scorer="f1",
        temperature=0.0,
        max_completion_tokens=256,
        batch_size=16,
        max_concurrent_batches=4,
        max_samples=100,
        output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "./outputs"),
    ),

    # LongHealth — model + cartridge, fuzzy option matching (same as longhealth/evals.py)
    BenchmarkConfig(
        name="longhealth_mc",
        dataset_name="longhealth",
        # subset = comma-separated patient IDs, or None for all patients
        subset="patient_01,patient_02,patient_03,patient_04,patient_05,"
               "patient_06,patient_07,patient_08,patient_09,patient_10",
        client=toka_client,
        scorer="longhealth_mc",
        temperature=0.0,
        max_completion_tokens=512,
        batch_size=16,
        max_concurrent_batches=4,
        output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "./outputs"),
    ),

    # # Local dataset example
    # BenchmarkConfig(
    #     name="my_custom_eval",
    #     dataset_name="local",
    #     dataset_path="/path/to/my_data.csv",  # expects 'question' and 'answer' columns
    #     client=toka_client,
    #     scorer="exact_match",
    #     temperature=0.0,
    #     max_completion_tokens=128,
    #     batch_size=16,
    #     output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "./outputs"),
    # ),
]

if __name__ == "__main__":
    pydrantic.main(configs)
