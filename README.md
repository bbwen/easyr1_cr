EasyR1_cr

EasyR1_cr is a research extension of the EasyR1 framework for confidence-aware reinforcement learning in large language models (LLMs) and vision-language models (VLMs).

This repository implements Reflective Confidence (ReCon) and VeriConf-GRPO, a reinforcement learning method that trains models to produce accurate, calibrated self-reported confidence, enabling safer automation, abstention, and evaluation in high-stakes reasoning tasks.

This repo builds on the original EasyR1 framework
https://github.com/hiyouga/EasyR1

✨ Key Contributions

Reflective Confidence (ReCon): explicit reasoning, reflection, and confidence output format

VeriConf-GRPO: GRPO training with rewards for correctness and confidence calibration

Unified evaluation framework for:

HotpotQA (multi-hop text reasoning)

MathVista (vision-language mathematical reasoning)

Support for multiple confidence baselines

Calibration metrics:

Accuracy

AUROC

Brier Score

Expected Calibration Error (ECE)

🧠 Core Idea: Reflective Confidence (ReCon)

Standard LLMs often exhibit systematic overconfidence.
ReCon addresses this by forcing models to explicitly reflect on uncertainty.

Required Output Format (VeriConf-GRPO)
<think>
  reasoning process
</think>

<answer>
  final answer
</answer>

<analysis>
  reflection on uncertainty and limitations
</analysis>

<confidence>
  confidence score in [0,1]
</confidence>


This structured output enables reliable extraction and calibration of confidence.

📦 Installation
Recommended: Docker
docker run --gpus all -it --ipc=host \
  -v $(pwd):/workspace/easyr1_cr \
  hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0 bash


Inside container:

cd /workspace/easyr1_cr

Local Environment (Advanced)
conda create -n easyr1_cr python=3.10
conda activate easyr1_cr

pip install torch datasets transformers ray vllm accelerate sentencepiece
pip install git+https://github.com/huggingface/transformers.git

📂 Repository Structure
├── evaluation.py
├── dataset_processing.py
├── system_prompts.py
├── eval/
│   ├── eval_utils.py
│   ├── eval_args.py
│   └── check_functions.py
├── examples/
│   ├── config_conf.yaml         # VeriConf-GRPO
│   ├── config_ori.yaml          # Vanilla GRPO
│   ├── format_prompt/
│   └── reward_function/
├── scripts/
│   ├── model_merger.py
│   └── evaluate_checkpoint.py
├── eval_configs/
└── README.md

🚀 Training with GRPO + Reflective Confidence

EasyR1_cr supports two GRPO variants.
The training config determines the resulting model.

GRPO Variants
Training Config	Resulting Model
config_grpo_ori.yaml	Vanilla GRPO
config_grpo_conf.yaml	VeriConf-GRPO
HotpotQA (Text-Only)
VeriConf-GRPO
python3 -m verl.trainer.main \
  config=examples/config_conf.yaml \
  data.train_files=hotpotqa@train \
  data.val_files=hotpotqa@validation \
  worker.actor.model.model_path=Qwen/Qwen2.5-3B-Instruct \
  trainer.experiment_name=qwen2_5_3b_hotpot_grpo_conf \
  trainer.n_gpus_per_node=2

Vanilla GRPO
python3 -m verl.trainer.main \
  config=examples/config_ori.yaml \
  data.train_files=hotpotqa@train \
  data.val_files=hotpotqa@validation \
  worker.actor.model.model_path=Qwen/Qwen2.5-3B-Instruct \
  trainer.experiment_name=qwen2_5_3b_hotpot_grpo_ori \
  trainer.n_gpus_per_node=2

MathVista-mini (Vision-Language)
python3 -m verl.trainer.main \
  config=examples/config_conf.yaml \
  data.train_files=data/mathvista_train.jsonl \
  data.val_files=data/mathvista_test.jsonl \
  worker.actor.model.model_path=Qwen/Qwen2.5-VL-3B-Instruct \
  trainer.experiment_name=qwen2_5_vl_3b_mathvista_grpo_conf \
  trainer.n_gpus_per_node=2


Important notes for VLMs

Prompts must include <image> tokens

Recommended:

data:
  max_prompt_length: 8192
  image_size: 336

🔧 Reward Design
Vanilla GRPO

Reward: answer correctness only

Confidence not optimized

VeriConf-GRPO

Answer correctness

Output format adherence

Confidence calibration reward

🔗 Checkpoint Merging (Required)

Training produces sharded actor checkpoints.
Before evaluation, they must be merged.

python scripts/model_merger.py \
  --local_dir checkpoints/easy_r1/<experiment>/global_step_x/actor


This creates:

actor/huggingface/
├── config.json
├── model.safetensors
└── generation_config.json


Only the huggingface/ directory should be used for evaluation.

🧪 Evaluation

Evaluation is driven by a JSON config:

one global dataset block

multiple model blocks

Example: HotpotQA Evaluation Config
[
  {
    "dataset_name": "hotpotqa",
    "dataset_config": "fullwiki",
    "split": "test",
    "hash_key": "question",
    "store_name": "eval_outputs/hotpotqa",
    "gpu_memory_utilization": 0.8,
    "log_path": "results/hotpotqa"
  },
  {
    "name": "Vanilla-GRPO",
    "model": "checkpoints/easy_r1/qwen2_5_3b_hotpot_grpo_ori/global_step_x/actor/huggingface",
    "check_fn": "confidence_verifier",
    "sys_prompt_name": "tac",
    "vllm_task": ["confidence_at_end", "ans_at_end"]
  },
  {
    "name": "VeriConf-GRPO",
    "model": "checkpoints/easy_r1/qwen2_5_3b_hotpot_grpo_conf/global_step_x/actor/huggingface",
    "check_fn": "confidence_verifier",
    "sys_prompt_name": "tac",
    "vllm_task": ["confidence_at_end", "ans_at_end"]
  },
  {
    "name": "Answer-Prob",
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "check_fn": "confidence_verifier",
    "sys_prompt_name": "gen",
    "vllm_task": ["confidence_prob", "ans_at_end"]
  }
]

Running Evaluation
python evaluation.py --config eval_configs/hotpotqa_eval.json


Outputs:

predictions and confidence

Accuracy, AUROC, Brier, ECE

saved under eval_outputs/ and results/

🧠 Key Takeaways

Training config defines model behavior

Checkpoint merging is mandatory

Evaluation config ensures fair comparison

VeriConf-GRPO improves calibration without hurting accuracy

Works for both LLMs and VLMs