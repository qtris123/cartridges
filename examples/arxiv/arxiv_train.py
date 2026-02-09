import os
from pathlib import Path
import pydrantic

from cartridges.initialization import KVFromText
from cartridges.train import TrainConfig, LossEvalConfig, GenerationEvalConfig
from cartridges.models import HFModelConfig, FlexQwen3ForCausalLM, FlexLlamaForCausalLM
from cartridges.datasets import DataSource, GenerateEvalDataset, TrainDataset, LossEvalDataset



config = TrainConfig(
    model=HFModelConfig(
        pretrained_model_name_or_path="meta-llama/llama-3.2-1B-Instruct",
        model_cls=FlexLlamaForCausalLM,
    ),
    kv_cache_initializer=KVFromText.Config(
        text_source=os.path.join(os.environ["CARTRIDGES_DIR"], "examples/arxiv/cartridges.tex"),
        max_tokens=None
    ),
    
    lr=2e-2,
    epochs=1,
    global_batch_size=32,

    dataset=TrainDataset.Config(
        data_sources=[
            # TODO: replace below with your own dataset you just synthesized and 
            # remove our huggingface dataset below
            # DataSource(path="path/to/your/dataset.parquet", type="local"),    
            DataSource(path="/scratch/scholar/vo43/train.parquet", type="local"),
        ],
        top_k_logits=20,
        packed_seq_length=2048,
        packing_mode="truncate",
        max_train_samples=10000
    ),

    loss_eval_every_n_steps=16,
    loss_evals=[
        LossEvalConfig(
            dataset=LossEvalDataset.Config(
                data_source=DataSource(
                    path="/scratch/scholar/vo43/train_eval.parquet",
                    type="local",
                ),
                packed_seq_length=2048,
            ),
            name_for_wandb="arxiv_synthesize",
            max_eval_samples=1000
        )
    ],

    generate_eval_every_n_steps=128,
    generate_evals=[
        GenerationEvalConfig(
            dataset=GenerateEvalDataset.Config(
                data_source=DataSource(
                    path="/scratch/scholar/vo43/train_eval.parquet",
                    type="local",
                ),
            ),
            name_for_wandb="arxiv-train",
            batch_size=16,
            max_eval_samples=1000
        )
    ],
    distributed_backend="gloo",

    save_every_n_steps=512,
    name="cartridges-tutorial-train",
)


if __name__ == "__main__":
    pydrantic.main(config)