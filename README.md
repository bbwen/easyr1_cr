# Reflective Confidence 

Reflective Confidence (Recon) is a framework for confidence-aware reinforcement learning in large language models (LLMs) and vision-language models (VLMs).

This repository implements Reflective Confidence (ReCon) and VeriConf-GRPO, a reinforcement learning method that trains models to produce accurate and calibrated self-reported confidence, enabling safer automation, abstention, and evaluation in high-stakes reasoning tasks.

This repo builds on the original EasyR1 framework for RL training:
https://github.com/hiyouga/EasyR1

---

## Key Contributions

- Reflective Confidence (ReCon): explicit reasoning, reflection, and confidence output format
- VeriConf-GRPO: GRPO training with rewards for correctness and confidence calibration
- Unified evaluation framework for:
  - HotpotQA (multi-hop text reasoning)
  - MathVista (vision-language mathematical reasoning)
- Multiple confidence baselines:
  - Verbalized confidence
  - Token-probability confidence
- Calibration metrics:
  - Accuracy
  - AUROC
  - Brier Score
  - Expected Calibration Error (ECE)

---

## Core Idea: Reflective Confidence (ReCon)

Standard LLMs often exhibit systematic overconfidence.
ReCon addresses this by forcing models to explicitly reflect on uncertainty.

### Required Output Format (VeriConf-GRPO)

```
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
```

---

## Installation

### Docker (Recommended)

```
docker run --gpus all -it --ipc=host \
  -v $(pwd):/workspace/easyr1_cr \
  hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0 bash
```

---

### Local Environment (Advanced)

```
conda create -n easyr1_cr python=3.10
conda activate easyr1_cr

pip install torch datasets transformers ray vllm accelerate sentencepiece
pip install git+https://github.com/huggingface/transformers.git
```

---


### Repository Structure

```
├── examples/
│   ├── config_conf.yaml
│   ├── config_ori.yaml
|   |── qwen2_5_3b_hotpotqa_grpo_conf.sh
|   |── qwen2_5_3b_hotpotqa_grpo_ori.sh
|   |── qwen2_5_vl_3b_mathvista_grpo_conf.sh
|   |── qwen2_5_vl_3b_mathvista_grpo_ori.sh
│   ├── config_ori.yaml
 
│   ├── format_prompt/
│   └── reward_function/

|   
├── evaluation.py
|── evaluation_vl.py
├── dataset_processing.py
├── system_prompts.py
├── eval/
│   ├── eval_utils.py
│   ├── eval_args.py
│   └── check_functions.py
├── scripts/
│   ├── model_merger.py
├── eval_configs/
└── README.md
```


## Training with GRPO + Reflective Confidence

Two GRPO variants are supported.

| Training Config | Resulting Model |
|----------------|---------------|
| config_grpo_ori.yaml | Vanilla GRPO |
| config_grpo_conf.yaml | VeriConf-GRPO |

---

## HotpotQA (Text-Only)

VeriConf-GRPO

```
bash qwen2_5_3b_hotpotqa_grpo_conf.sh

```

Vanilla GRPO

```
bash qwen2_5_3b_hotpotqa_grpo_ori.sh

```


Similar settings for MathVista-mini (Vision-Language)


## Checkpoint Merging (Required)

```
python scripts/model_merger.py \
  --local_dir checkpoints/easy_r1/<experiment>/global_step_x/actor
```

Use the resulting `actor/huggingface/` directory for evaluation.

---

## Evaluation

Evaluation is driven by a JSON config:

one global dataset block

multiple model blocks

## Running Evaluation for text-only tasks

```
python evaluation.py --config eval_configs/hotpotqa_eval.json
```


## Running Evaluation for vl tasks

```
python evaluation_vl.py --config eval_configs/mathvista_eval.json
```

Example: HotpotQA Evaluation Config: eval_configs/hotpotqa_eval.json
```
[
  {
    "dataset_name": "hotpotqa",
    "hash_key": "problem",
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

```

