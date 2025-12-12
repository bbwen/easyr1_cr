#!/bin/bash

set -x

MODEL_PATH=Qwen/Qwen3-4B 

python3 -m verl.trainer.main \
    config=examples/config_ori.yaml \
    data.train_files=mehuldamani/hotpot_qa@train  \
    data.val_files=mehuldamani/hotpot_qa@test \
    data.max_response_length=4096 \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen3_4b_hotpot_grpo_ori \
    trainer.n_gpus_per_node=2

