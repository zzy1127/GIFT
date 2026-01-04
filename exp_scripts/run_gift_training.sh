#!/bin/bash

# GIFT: Gibbs Initialization with Finite Temperature - Training Script

export PYTHONPATH=$PYTHONPATH:$(pwd)/verl
export WANDB_MODE=offline

# Configuration
project_name=gift_training
experiment_name=gift_experiment

# Data and model paths (REQUIRED: set these to your own paths)
train_file="/path/to/your/train.parquet"
val_file="/path/to/your/val.parquet"
model_path="/path/to/your/base_model"

# Data keys (adjust according to your dataset)
prompt_key=sft_prompt
response_key=solution

# Launch training
torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --master_port=29501 \
    -m verl.trainer.fsdp_gift_trainer \
    data.train_files=${train_file} \
    data.val_files=${val_file} \
    data.prompt_key=${prompt_key} \
    data.response_key=${response_key} \
    data.train_batch_size=128 \
    data.micro_batch_size_per_gpu=2 \
    data.max_length=8192 \
    data.truncation=right \
    model.partial_pretrain=${model_path} \
    model.fsdp_config.cpu_offload=False \
    model.enable_gradient_checkpointing=True \
    optim.lr=1e-5 \
    optim.lr_scheduler=constant \
    optim.warmup_steps_ratio=0 \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.total_epochs=10 \
    trainer.save_freq=78 \
    trainer.test_freq=78 \
    trainer.beta=20.0 \
    trainer.smooth_lambda=0.0 \
    trainer.default_local_dir=./train_results/${project_name}/${experiment_name} \
    trainer.logger='["console","wandb"]' \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8

