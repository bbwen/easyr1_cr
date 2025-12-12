# EasyR1: An Efficient, Scalable, Multi-Modality RL Training Framework

[![GitHub Repo stars](https://img.shields.io/github/stars/hiyouga/EasyR1)](https://github.com/hiyouga/EasyR1/stargazers)
[![Twitter](https://img.shields.io/twitter/follow/llamafactory_ai)](https://twitter.com/llamafactory_ai)
[![Docker Pulls](https://img.shields.io/docker/pulls/hiyouga/verl)](https://hub.docker.com/r/hiyouga/verl/tags)

### Used by [Amazon Web Services](https://aws.amazon.com/cn/blogs/china/building-llm-model-hub-based-on-llamafactory-and-easyr1/)

This project is a clean fork of the original [veRL](https://github.com/volcengine/verl) project to support vision language models, we thank all the authors for providing such a high-performance RL training framework.

EasyR1 is efficient and scalable due to the design of **[HybirdEngine](https://arxiv.org/abs/2409.19256)** and the latest release of **[vLLM](https://github.com/vllm-project/vllm)**'s SPMD mode.

## Features

- Supported models
  - Llama3/Qwen2/Qwen2.5/Qwen3 language models
  - Qwen2-VL/Qwen2.5-VL/Qwen3-VL vision language models
  - DeepSeek-R1 distill models

- Supported algorithms
  - GRPO
  - DAPO ![new](https://img.shields.io/badge/new-orange)
  - Reinforce++
  - ReMax
  - RLOO
  - GSPO ![new](https://img.shields.io/badge/new-orange)
  - CISPO ![new](https://img.shields.io/badge/new-orange)

- Supported datasets
  - Any text, vision-text dataset in a [specific format](#custom-dataset)

- Supported tricks
  - Padding-free training
  - Resuming from the latest/best checkpoint
  - Wandb & SwanLab & Mlflow & Tensorboard tracking

## Requirements

### Software Requirements

- Python 3.9+
- transformers>=4.54.0
- flash-attn>=2.4.3
- vllm>=0.8.3

We provide a [Dockerfile](./Dockerfile) to easily build environments.

We recommend using the [pre-built docker image](https://hub.docker.com/r/hiyouga/verl) in EasyR1.

```bash
docker pull hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0
docker run -it --ipc=host --gpus=all hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0
```

If your environment does not support Docker, you can consider using **Apptainer**:

```bash
apptainer pull easyr1.sif docker://hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0
apptainer shell --nv --cleanenv --bind /mnt/your_dir:/mnt/your_dir easyr1.sif
```

Use `USE_MODELSCOPE_HUB=1` to download models from the ModelScope hub.

### Hardware Requirements

\* *estimated*

| Method                   | Bits |  1.5B  |   3B   |   7B   |   32B   |   72B   |
| ------------------------ | ---- | ------ | ------ | ------ | ------- | ------- |
| GRPO Full Fine-Tuning    |  AMP | 2*24GB | 4*40GB | 8*40GB | 16*80GB | 32*80GB |
| GRPO Full Fine-Tuning    | BF16 | 1*24GB | 1*40GB | 4*40GB |  8*80GB | 16*80GB |

> [!NOTE]
> Use `worker.actor.fsdp.torch_dtype=bf16` and `worker.actor.optim.strategy=adamw_bf16` to enable bf16 training.
>
> We are working hard to reduce the VRAM in RL training, LoRA support will be integrated in next updates.

## Tutorial: Run Qwen2.5-VL GRPO on [Geometry3K](https://huggingface.co/datasets/hiyouga/geometry3k) Dataset in Just 3 Steps

![image](assets/qwen2_5_vl_7b_geo.png)

### Installation

```bash
git clone https://github.com/hiyouga/EasyR1.git
cd EasyR1
pip install -e .
```

### GRPO Training Examples

#### Geometry3K (Vision-Language)

```bash
bash examples/qwen2_5_vl_7b_geo3k_grpo.sh
```

#### HotpotQA (Text-Only)

**With Confidence (Brier Score Reward)**:
```bash
# Uses math_conf.py reward function with Brier score
bash examples/qwen3_4b_hotpot_grpo_conf.sh
```

**Original (Accuracy Only)**:
```bash
# Uses math_ori.py reward function (accuracy only)
bash examples/qwen3_4b_hotpot_grpo_ori.sh
```

#### MathVista (Vision-Language)

**With Confidence (Brier Score Reward)**:
```bash
# Uses math_conf.py reward function with Brier score
bash examples/qwen2_5_vl_3b_mathvista_grpo_conf.sh
```

**Original (Accuracy Only)**:
```bash
# Uses math_ori.py reward function (accuracy only)
bash examples/qwen2_5_vl_3b_mathvista_grpo_ori.sh
```

> [!NOTE]
> **Reward Function Differences**:
> - **`_conf` variants**: Use `math_conf.py` reward function which includes **Brier score** for confidence calibration. This rewards models that provide well-calibrated confidence estimates (confidence matches correctness).
> - **`_ori` variants**: Use `math_ori.py` reward function which only considers format compliance and answer accuracy.
> 
> See [Reward Functions Documentation](examples/reward_function/README.md) for detailed information about the Brier score implementation.

### Merge Checkpoint in Hugging Face Format

```bash
python3 scripts/model_merger.py --local_dir checkpoints/easy_r1/exp_name/global_step_1/actor
```

> [!TIP]
> If you encounter issues with connecting to Hugging Face, consider using `export HF_ENDPOINT=https://hf-mirror.com`.
>
> If you want to use SwanLab logger, consider using `bash examples/qwen2_5_vl_7b_geo3k_swanlab.sh`.

## Evaluation

EasyR1 provides evaluation scripts for assessing model performance on various datasets. Evaluation supports both text-only and vision-language models.

### Evaluation Scripts

- **`evaluation.py`**: For text-only models (e.g., HotpotQA)
- **`evaluation_vl.py`**: For vision-language models (e.g., MathVista, Geometry3K)
- **`scripts/evaluate_checkpoint.py`**: For evaluating EasyR1-trained checkpoints

### Running Evaluation

Evaluation uses JSON configuration files located in `eval_configs/`. Each config file specifies:
- Dataset name and split
- Model paths to evaluate
- System prompts
- Evaluation tasks (confidence extraction, answer extraction)
- Output storage locations

#### Text-Only Evaluation (HotpotQA)

```bash
# Evaluate on HotpotQA
CUDA_VISIBLE_DEVICES=0 python evaluation.py --config eval_configs/Hotpot-models/hotpot-vanilla-eval-em.json
```

#### Vision-Language Evaluation (MathVista)

```bash
# Evaluate on MathVista
CUDA_VISIBLE_DEVICES=0 python evaluation_vl.py --config eval_configs/Hotpot-models/mathvista_eval-em.json
```

#### Evaluating EasyR1 Checkpoints

```bash
# Evaluate a trained checkpoint
python scripts/evaluate_checkpoint.py \
    --model_path checkpoints/easy_r1/exp_name/global_step_1/actor \
    --dataset_name mehuldamani/hotpot_qa \
    --split test \
    --output_dir eval_results
```

### Evaluation Config Format

Evaluation configs are JSON files with the following structure:

```json
[
  {
    "dataset_name": "mehuldamani/hotpot_qa",
    "hash_key": "problem",
    "store_name": "eval_outputs/Hotpot-models-fresh/hotpot-vanilla-eval-em",
    "gpu_memory_utilization": 0.9,
    "log_path": "results/Hotpot-models-fresh/hotpot-vanilla-eval-em"
  },
  {
    "name": "Base",
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "check_fn": "confidence_verifier",
    "sys_prompt_name": "tac",
    "vllm_task": ["confidence_at_end", "ans_at_end"]
  }
]
```

**Global Config (first entry)**:
- `dataset_name`: HuggingFace dataset name or local path
- `hash_key`: Key to use for dataset hashing/deduplication
- `store_name`: Where to save evaluation results
- `gpu_memory_utilization`: vLLM GPU memory utilization (0.0-1.0)
- `log_path`: Path to save metrics JSON file

**Model Configs (subsequent entries)**:
- `name`: Model identifier for results
- `model`: Model path (HuggingFace or local)
- `check_fn`: Function to extract confidence/answers (`confidence_verifier` or `llm_confidence_verifier`)
- `sys_prompt_name`: System prompt to use (see `system_prompts.py`)
- `vllm_task`: Tasks to extract from responses:
  - `confidence_at_end`: Extract confidence from `<confidence>` tags
  - `ans_at_end`: Extract answer from `<answer>` tags
  - `confidence_prob`: Extract confidence as probability

### Evaluation Metrics

The evaluation scripts compute:
- **Accuracy**: Exact match between predicted and ground truth answers
- **Brier Score**: Calibration metric (1 - (correctness - confidence)²)
- **ECE (Expected Calibration Error)**: Measures calibration quality
- **AUROC**: Area under ROC curve for confidence-based classification
- **Pass@k**: Accuracy at top-k confidence thresholds

Results are saved to:
- `{log_path}/metrics.json`: Aggregated metrics
- `{store_name}/`: Full evaluation dataset with predictions

### Example Evaluation Run

```bash
# Run evaluation on HotpotQA
bash eval_runs.sh

# Or run specific evaluation
CUDA_VISIBLE_DEVICES=0 python evaluation.py --config eval_configs/Hotpot-models/hotpot-vanilla-eval-em.json
```

> [!NOTE]
> Evaluation results are cached. If you re-run evaluation with the same config, it will load existing results and only evaluate new models. Use `--fresh` flag (if supported) to force re-evaluation.

## Custom Dataset

Please refer to the example datasets to prepare your own dataset.

- Text dataset: https://huggingface.co/datasets/hiyouga/math12k
- Image-text dataset: https://huggingface.co/datasets/hiyouga/geometry3k
- Multi-image-text dataset: https://huggingface.co/datasets/hiyouga/journeybench-multi-image-vqa
- Text-image mixed dataset: https://huggingface.co/datasets/hiyouga/rl-mixed-dataset

## Data Processing Scripts

This repository includes data processing scripts for preparing datasets in EasyR1 format. The scripts are located in the `data/` directory.

### Dataset Format

All datasets should follow this format:
```json
{
  "images": ["path/to/image.png"],  // List of image paths (empty for text-only datasets)
  "problem": "<image>Your question here",  // Problem text (prepend <image> for vision tasks)
  "answer": "Answer text"
}
```

### MathVista Processing

**Script**: `data/data_transform.py`

Processes MathVista dataset for vision-language tasks:
- Filters for `free_form` questions
- Extracts and saves images to `data/mathvista_images_geo3k/`
- Prepends `<image>` tag to problem text
- Splits into train/test (80/20)
- Generates JSONL files: `mathvista_train.jsonl` and `mathvista_test.jsonl`

**Usage**:
```bash
python data/data_transform.py
```

**Output Format**:
- Images saved to `data/mathvista_images_geo3k/XXXXX.png`
- Problem format: `"<image>Question text"`
- Images field: `["data/mathvista_images_geo3k/XXXXX.png"]`

### MathVista HuggingFace Upload

**Script**: `data/mathvista_upload_hf.py`

Downloads MathVista, processes it, and uploads to HuggingFace Hub:
- Downloads from `AI4Math/MathVista`
- Filters for `free_form` questions
- Renames columns: `question` → `problem`, `decoded_image` → `image`
- Prepends `<image>` tag to problem text
- Splits into train/test (80/20 by default)
- Uploads to HuggingFace Hub

**Usage**:
```bash
# Upload to HuggingFace (requires authentication)
python data/mathvista_upload_hf.py --repo_id username/mathvista-freeform

# Custom test size
python data/mathvista_upload_hf.py --repo_id username/mathvista-freeform --test-size 0.3

# Use different source split
python data/mathvista_upload_hf.py --repo_id username/mathvista-freeform --split test

# Process without uploading (for testing)
python data/mathvista_upload_hf.py --repo_id username/mathvista-freeform --no-upload
```

**Authentication**:
Before uploading, authenticate with HuggingFace:
```bash
hf auth login
# Or set environment variable:
export HF_TOKEN="your_token_here"
```

### HotpotQA Processing

**Script**: `data/hotpotqa_processing.py`

Processes HotpotQA dataset for text-only tasks:
- Loads from `mehuldamani/hotpot_qa`
- Filters out entries with empty answers
- Splits into train/test (80/20)
- Generates JSONL files: `hotpotqa_train.jsonl` and `hotpotqa_test.jsonl`

**Usage**:
```bash
python data/hotpotqa_processing.py
```

**Output Format**:
- Images field: `[]` (empty, as HotpotQA is text-only)
- Problem format: Raw question text (no `<image>` tag)
- Answer format: Answer text

**Note**: HotpotQA is a text-only dataset, so the `images` field is always an empty list.

### Data Processing Features

All processing scripts include:
- **Train/Test Split**: 80/20 split with configurable seed (default: 42)
- **Empty Answer Filtering**: Automatically filters out entries with empty answers
- **Format Consistency**: Ensures all datasets follow the same EasyR1 format
- **Relative Paths**: Image paths are relative to EasyR1 root directory

### Uploading to HuggingFace

For uploading processed datasets to HuggingFace Hub, see:
- `data/upload_to_huggingface.py` - Generic upload script for JSONL files
- `data/mathvista_upload_hf.py` - MathVista-specific upload with processing

Both scripts require HuggingFace authentication. See the scripts for detailed usage instructions.

## How to Understand GRPO in EasyR1

![image](assets/easyr1_grpo.png)

- To learn about the GRPO algorithm, you can refer to [Hugging Face's blog](https://huggingface.co/docs/trl/v0.16.1/en/grpo_trainer).

## How to Run 70B+ Model in Multi-node Environment

1. Start the Ray head node.

```bash
ray start --head --port=6379 --dashboard-host=0.0.0.0
```

2. Start the Ray worker node and connect to the head node.

```bash
ray start --address=<head_node_ip>:6379
```

3. Check the Ray resource pool.

```bash
ray status
```

4. Run training script on the Ray head node only.

```bash
bash examples/qwen2_5_vl_7b_geo3k_grpo.sh
```

See the **[veRL's official doc](https://verl.readthedocs.io/en/latest/start/multinode.html)** for more details about multi-node training and Ray debugger.

## Other Baselines

We also reproduced the following two baselines of the [R1-V](https://github.com/deep-agent/R1-V) project.
- [CLEVR-70k-Counting](examples/baselines/qwen2_5_vl_3b_clevr.sh): Train the Qwen2.5-VL-3B-Instruct model on counting problem.
- [GeoQA-8k](examples/baselines/qwen2_5_vl_3b_geoqa8k.sh): Train the Qwen2.5-VL-3B-Instruct model on GeoQA problem.

## Performance Baselines

See [baselines.md](assets/baselines.md).

## Awesome Work using EasyR1

- **MMR1**: Enhancing Multimodal Reasoning with Variance-Aware Sampling and Open Resources. [![[code]](https://img.shields.io/github/stars/LengSicong/MMR1)](https://github.com/LengSicong/MMR1) [![[arxiv]](https://img.shields.io/badge/arxiv-2509.21268-blue)](https://arxiv.org/abs/2509.21268)
- **Vision-R1**: Incentivizing Reasoning Capability in Multimodal Large Language Models. [![[code]](https://img.shields.io/github/stars/Osilly/Vision-R1)](https://github.com/Osilly/Vision-R1) [![[arxiv]](https://img.shields.io/badge/arxiv-2503.06749-blue)](https://arxiv.org/abs/2503.06749)
- **Seg-Zero**: Reasoning-Chain Guided Segmentation via Cognitive Reinforcement. [![[code]](https://img.shields.io/github/stars/dvlab-research/Seg-Zero)](https://github.com/dvlab-research/Seg-Zero) [![[arxiv]](https://img.shields.io/badge/arxiv-2503.06520-blue)](https://arxiv.org/abs/2503.06520)
- **MetaSpatial**: Reinforcing 3D Spatial Reasoning in VLMs for the Metaverse. [![[code]](https://img.shields.io/github/stars/PzySeere/MetaSpatial)](https://github.com/PzySeere/MetaSpatial) [![[arxiv]](https://img.shields.io/badge/arxiv-2503.18470-blue)](https://arxiv.org/abs/2503.18470)
- **Temporal-R1**: Envolving Temporal Reasoning Capability into LMMs via Temporal Consistent Reward. [![[code]](https://img.shields.io/github/stars/appletea233/Temporal-R1)](https://github.com/appletea233/Temporal-R1) [![[arxiv]](https://img.shields.io/badge/arxiv-2506.01908-blue)](https://arxiv.org/abs/2506.01908)
- **NoisyRollout**: Reinforcing Visual Reasoning with Data Augmentation. [![[code]](https://img.shields.io/github/stars/John-AI-Lab/NoisyRollout)](https://github.com/John-AI-Lab/NoisyRollout) [![[arxiv]](https://img.shields.io/badge/arxiv-2504.13055-blue)](https://arxiv.org/pdf/2504.13055)
- **GUI-R1**: A Generalist R1-Style Vision-Language Action Model For GUI Agents. [![[code]](https://img.shields.io/github/stars/ritzz-ai/GUI-R1)](https://github.com/ritzz-ai/GUI-R1) [![[arxiv]](https://img.shields.io/badge/arxiv-2504.10458-blue)](https://arxiv.org/abs/2504.10458)
- **R1-Track**: Direct Application of MLLMs to Visual Object Tracking via Reinforcement Learning. [![[code]](https://img.shields.io/github/stars/Wangbiao2/R1-Track)](https://github.com/Wangbiao2/R1-Track)
- **VisionReasoner**: Unified Visual Perception and Reasoning via Reinforcement Learning. [![[code]](https://img.shields.io/github/stars/dvlab-research/VisionReasoner)](https://github.com/dvlab-research/VisionReasoner) [![[arxiv]](https://img.shields.io/badge/arxiv-2505.12081-blue)](https://arxiv.org/abs/2505.12081)
- **MM-UPT**: Unsupervised Post-Training for Multi-Modal LLM Reasoning via GRPO. [![[code]](https://img.shields.io/github/stars/waltonfuture/MM-UPT)](https://github.com/waltonfuture/MM-UPT) [![[arxiv]](https://img.shields.io/badge/arxiv-2505.22453-blue)](https://arxiv.org/pdf/2505.22453)
- **RL-with-Cold-Start**: Advancing Multimodal Reasoning via Reinforcement Learning with Cold Start. [![[code]](https://img.shields.io/github/stars/waltonfuture/RL-with-Cold-Start)](https://github.com/waltonfuture/RL-with-Cold-Start) [![[arxiv]](https://img.shields.io/badge/arxiv-2505.22334-blue)](https://arxiv.org/pdf/2505.22334)
- **ViGoRL**: Grounded Reinforcement Learning for Visual Reasoning. [![[code]](https://img.shields.io/github/stars/Gabesarch/grounded-rl)](https://github.com/Gabesarch/grounded-rl) [![[arxiv]](https://img.shields.io/badge/arxiv-2505.22334-blue)](https://arxiv.org/abs/2505.23678)
- **Revisual-R1**: Advancing Multimodal Reasoning: From Optimized Cold Start to Staged Reinforcement Learning. [![[code]](https://img.shields.io/github/stars/CSfufu/Revisual-R1)](https://github.com/CSfufu/Revisual-R1) [![[arxiv]](https://img.shields.io/badge/arxiv-2506.04207-blue)](https://arxiv.org/abs/2506.04207)
- **SophiaVL-R1**: Reinforcing MLLMs Reasoning with Thinking Reward. [![[code]](https://img.shields.io/github/stars/kxfan2002/SophiaVL-R1)](https://github.com/kxfan2002/SophiaVL-R1) [![[arxiv]](https://img.shields.io/badge/arxiv-2505.17018-blue)](https://arxiv.org/abs/2505.17018)
- **Vision-Matters**: Simple Visual Perturbations Can Boost Multimodal Math Reasoning. [![[code]](https://img.shields.io/github/stars/YutingLi0606/Vision-Matters)](https://github.com/YutingLi0606/Vision-Matters) [![[arxiv]](https://img.shields.io/badge/arxiv-2506.09736-blue)](https://arxiv.org/abs/2506.09736)
- **VTool-R1**: VLMs Learn to Think with Images via Reinforcement Learning on Multimodal Tool Use. [![[code]](https://img.shields.io/github/stars/VTOOL-R1/vtool-r1)](https://github.com/VTOOL-R1/vtool-r1) [![[arxiv]](https://img.shields.io/badge/arxiv-2505.19255-blue)](https://arxiv.org/abs/2505.19255)
- **Long-RL**: Scaling RL to Long Sequences. [![[code]](https://img.shields.io/github/stars/NVlabs/Long-RL)](https://github.com/NVlabs/Long-RL) [![[arxiv]](https://img.shields.io/badge/arxiv-2507.07966-blue)](https://arxiv.org/abs/2507.07966)
- **EditGRPO**: Reinforcement Learning with Post-Rollout Edits for Clinically Accurate Chest X-Ray Report Generation. [![[code]](https://img.shields.io/github/stars/taokz/EditGRPO)](https://github.com/taokz/EditGRPO)
- **ARES**: Multimodal Adaptive Reasoning via Difficulty-Aware Token-Level Entropy Shaping. [![[code]](https://img.shields.io/github/stars/shawn0728/ARES)](https://github.com/shawn0728/ARES) [![[arxiv]](https://img.shields.io/badge/arxiv-2510.08457-blue)](https://arxiv.org/abs/2510.08457)
- **VPPO**: Spotlight on Token Perception for Multimodal Reinforcement Learning. [![[code]](https://img.shields.io/github/stars/huaixuheqing/VPPO-RL)](https://github.com/huaixuheqing/VPPO-RL) [![[arxiv]](https://img.shields.io/badge/arxiv-2510.09285-blue)](https://arxiv.org/abs/2510.09285)
- **IE-Critic-R1**: Advancing the Explanatory Measurement of Text-Driven Image Editing for Human Perception Alignment. [![[code]](https://img.shields.io/github/stars/Coobiw/IE-Critic-R1)](https://github.com/Coobiw/IE-Critic-R1) [![[arxiv]](https://img.shields.io/badge/arxiv-2511.18055-blue)](https://arxiv.org/abs/2511.18055)
- **OneThinker**: All-in-one Reasoning Model for Image and Video. [![[code]](https://img.shields.io/github/stars/tulerfeng/OneThinker)](https://github.com/tulerfeng/OneThinker) [![[arxiv]](https://img.shields.io/badge/arxiv-2512.03043-blue)](https://arxiv.org/abs/2512.03043)


## TODO

- Support LoRA (high priority).
- Support ulysses parallelism for VLMs (middle priority).
- Support more VLM architectures.

> [!NOTE]
> We will not provide scripts for supervised fine-tuning and inference in this project. If you have such requirements, we recommend using [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

### Known bugs

These features are temporarily disabled for now, we plan to fix them one-by-one in the future updates.

- Vision language models are not compatible with ulysses parallelism yet.

## Discussion Group

👋 Join our [WeChat group](https://github.com/hiyouga/llamafactory-community/blob/main/wechat/easyr1.jpg).

## FAQs

> ValueError: Image features and image tokens do not match: tokens: 8192, features 9800

Increase the `data.max_prompt_length` or reduce the `data.max_pixels`.

> RuntimeError: CUDA Error: out of memory at /workspace/csrc/cumem_allocator.cpp:62

Reduce the `worker.rollout.gpu_memory_utilization` and enable `worker.actor.offload.offload_params`.

> RuntimeError: 0 active drivers ([]). There should only be one.

Uninstall `deepspeed` from the current python environment.

## Citation

Core contributors: [Yaowei Zheng](https://github.com/hiyouga), [Junting Lu](https://github.com/AL-377), [Shenzhi Wang](https://github.com/Shenzhi-Wang), [Zhangchi Feng](https://github.com/BUAADreamer), [Dongdong Kuang](https://github.com/Kuangdd01) and Yuwen Xiong

We also thank Guangming Sheng and Chi Zhang for helpful discussions.

```bibtex
@misc{zheng2025easyr1,
  title        = {EasyR1: An Efficient, Scalable, Multi-Modality RL Training Framework},
  author       = {Yaowei Zheng, Junting Lu, Shenzhi Wang, Zhangchi Feng, Dongdong Kuang, Yuwen Xiong},
  howpublished = {\url{https://github.com/hiyouga/EasyR1}},
  year         = {2025}
}
```

We recommend to also cite the original work.

```bibtex
@article{sheng2024hybridflow,
  title   = {HybridFlow: A Flexible and Efficient RLHF Framework},
  author  = {Guangming Sheng and Chi Zhang and Zilingfeng Ye and Xibin Wu and Wang Zhang and Ru Zhang and Yanghua Peng and Haibin Lin and Chuan Wu},
  year    = {2024},
  journal = {arXiv preprint arXiv: 2409.19256}
}
```
