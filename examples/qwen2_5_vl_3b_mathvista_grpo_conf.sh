#!/bin/bash

set -x

MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct  # replace it with your local file path

python3 -m verl.trainer.main \
    data.max_prompt_length=4096 \
    config=examples/config_conf.yaml \
    data.rollout_batch_size=16 \
    data.val_batch_size=16 \
    worker.actor.global_batch_size=8 \
    data.train_files=Byanka/mathvista-freeform@train \
    data.val_files=Byanka/mathvista-freeform@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen2_5_vl_3b_mathvista_grpo_conf \
    trainer.n_gpus_per_node=2
