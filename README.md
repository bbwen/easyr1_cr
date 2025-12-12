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

## Training with GRPO + Reflective Confidence

Two GRPO variants are supported.

| Training Config | Resulting Model |
|----------------|---------------|
| config_grpo_ori.yaml | Vanilla GRPO |
| config_grpo_conf.yaml | VeriConf-GRPO |

---

## Checkpoint Merging (Required)

```
python scripts/model_merger.py \
  --local_dir checkpoints/easy_r1/<experiment>/global_step_x/actor
```

Use the resulting `actor/huggingface/` directory for evaluation.

---

## Evaluation

```
python evaluation.py --config eval_configs/hotpotqa_eval.json
```

Metrics include Accuracy, AUROC, Brier Score, and ECE.

---

## Citation

```
@software{easyr1_cr_2025,
  title = {EasyR1_cr: Reinforcement Learning with Reflective Confidence},
  author = {Bingbing Wen et al.},
  year = {2025},
  url = {https://github.com/bbwen/easyr1_cr}
}
```

---

## License

Apache-2.0 License
