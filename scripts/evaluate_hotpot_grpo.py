#!/usr/bin/env python3
"""
Evaluation script for EasyR1 checkpoints on HotpotQA - matching recom format exactly.

This script:
- Uses TABC_LONG_PROMPT system prompt (same as recom)
- Uses PROBLEM: format for prompts (same as recom)
- Uses exact_match_score for HotpotQA (same as recom)
- Extracts answers and confidence from <answer> and <confidence> tags (same as recom)
- Computes metrics: accuracy, Brier score, ECE, AUROC, pass@k (same as recom)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Add EasyR1 root to path
EASYR1_ROOT = Path(__file__).resolve().parent.parent
if str(EASYR1_ROOT) not in sys.path:
    sys.path.append(str(EASYR1_ROOT))

# TABC_LONG_PROMPT from recom (exact copy)
TABC_LONG_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind, provides the user with the final answer, then analyzes its confidence about the solution and then provides the user with its confidence level. "
    "The confidence level is a number between 0 and 1 (inclusive) enclosed within <confidence> </confidence> tags. The final answer is enclosed between <answer> </answer> tags."
    "The analysis about confidence and uncertainty is enclosed within <analysis> </analysis> tags. The assistant should reason about its confidence in the solution and its uncertainty in the solution within these tags."
    "Here are some guidelines for the analysis: "
    "1. Your task is to point out things where the model could be wrong in its thinking, or things where there might be ambiguity in the solution steps, or in the reasoning process itself.\n" 
    "2. You should not suggest ways of fixing the response, your job is only to reason about uncertainties.\n"
    "3. For some questions, the response might be correct. In these cases, It is also okay to have only a small number of uncertainties and then explictly say that I am unable to spot more uncertainties.\n"
    "4. Uncertainties might be different from errors. For example, uncertainties may arise from ambiguities in the question, or from the application of a particular lemma/proof. \n"
    "5. If there are alternate potential approaches that may lead to different answers, you should mention them.\n"
    "6. List out plausible uncertainties, do not make generic statements, be as specific about uncertainties as possible.\n"
    "7. Enclose this uncertainty analysis within <analysis> </analysis> tags.\n"
    "The final format that must be followed is : <think> reasoning process here </think><answer> final answer here </answer> <analysis> analysis about confidence and uncertainty here</analysis> <confidence> confidence level here (number between 0 and 1) </confidence>"
)

# Evaluation utilities (from recom - exact copy)
def normalize_answer(s):
    """Normalize answer for exact match comparison."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

import string

def exact_match_score(prediction: str, ground_truth: str) -> bool:
    """Check if prediction matches ground truth using exact match (same as recom)."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def compute_pass_n(evals, k):
    """Compute pass@k metric (same as recom)."""
    n = len(evals[0])
    corrects, totals = [], []
    for i in range(len(evals)):
        eval_list = evals[i]
        count = sum(1 for j in range(n) if eval_list[j] == 1)
        corrects.append(count)
        totals.append(n)
    return estimate_pass_at_k(totals, corrects, k).mean()

def estimate_pass_at_k(num_samples, num_correct, k):
    """Estimates pass@k of each problem and returns them in an array (same as recom)."""
    def estimator(n: int, c: int, k: int) -> float:
        """Calculates 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
    
    import itertools
    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])

def get_brier(correctness, confidence):
    """Compute Brier score (same as recom)."""
    return np.mean((confidence - correctness) ** 2)

def get_ece(correctness, confidence):
    """Compute Expected Calibration Error (same as recom)."""
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    bin_indices = np.digitize(confidence, bin_edges) - 1
    
    ece = 0
    for i in range(n_bins):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            bin_conf = np.mean(confidence[mask])
            bin_acc = np.mean(correctness[mask])
            bin_weight = np.sum(mask) / len(confidence)
            ece += bin_weight * np.abs(bin_conf - bin_acc)
    return ece

def get_auroc(correctness, confidence):
    """Compute Area Under ROC Curve (same as recom)."""
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(correctness, confidence)
    return auc(fpr, tpr)

# Regex patterns (same as recom)
ANSWER_PATTERN = r"<answer>(.*?)</answer>"
CONFIDENCE_PATTERN = r"<confidence>(.*?)</confidence>"


def confidence_extractor(response: str) -> tuple[int, float]:
    """Extract confidence from response (same as recom).
    
    Returns:
        (format_valid, confidence_value): (1, conf) if valid confidence found, (0, 0.0) otherwise
    """
    conf_matches = re.findall(CONFIDENCE_PATTERN, response, re.DOTALL | re.MULTILINE)
    last_confidence = conf_matches[-1] if conf_matches else ""
    if last_confidence == "":
        return 0, 0.0
    try:
        confidence = float(last_confidence)
        if confidence > 1 and confidence <= 100:
            return 1, confidence / 100
        elif confidence >= 0 and confidence <= 1:
            return 1, confidence
        else:
            return 0, 0.0
    except Exception:
        # Extract first number in string
        first_number = re.search(r"-?\d+(?:\.\d+)?", last_confidence)
        if first_number:
            first_number = float(first_number.group())
            if first_number >= 0 and first_number <= 1:
                return 1, first_number
            elif first_number > 1 and first_number <= 100:
                return 1, first_number / 100
            else:
                return 0, 0.0
        else:
            return 0, 0.0


def extract_answer(response: str) -> str:
    """Extract answer from response (same as recom)."""
    ans_matches = re.findall(ANSWER_PATTERN, response, re.DOTALL | re.MULTILINE)
    return ans_matches[-1].strip() if ans_matches else ""


def check_correctness(pred_answer: str, gold_answer: str) -> int:
    """Check if predicted answer matches gold answer using exact match (same as recom)."""
    return 1 if exact_match_score(pred_answer, gold_answer) else 0


def merge_checkpoint(checkpoint_path: str) -> str:
    """Merge sharded checkpoint into a single HuggingFace model."""
    actor_path = checkpoint_path
    if not actor_path.endswith("actor"):
        actor_path = os.path.join(checkpoint_path, "actor")
    
    huggingface_path = os.path.join(actor_path, "huggingface")
    
    # Check if already merged
    if os.path.exists(huggingface_path) and os.path.exists(os.path.join(huggingface_path, "config.json")):
        print(f"Checkpoint already merged at {huggingface_path}")
        return huggingface_path
    
    # Use model_merger.py to merge
    print(f"Merging checkpoint from {actor_path}...")
    merger_script = os.path.join(EASYR1_ROOT, "scripts", "model_merger.py")
    
    result = subprocess.run(
        [sys.executable, merger_script, "--local_dir", actor_path],
        capture_output=True,
        text=True,
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"Failed to merge checkpoint: {result.stderr}")
    
    print(f"Checkpoint merged to {huggingface_path}")
    return huggingface_path


def evaluate_hotpot_grpo(
    checkpoint_path: str,
    dataset_name: str = "mehuldamani/hotpot_qa",
    split: str = "test",
    num_samples: int | None = None,
    n_generations: int = 1,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    gpu_memory_utilization: float = 0.9,
    output_dir: str | None = None,
    merge_checkpoint_first: bool = True,
):
    """Evaluate HotpotQA GRPO model and compute metrics (same as recom).
    
    This function matches recom's evaluate_hotpot_grpo exactly.
    """
    # Merge checkpoint if needed
    if merge_checkpoint_first:
        model_path = merge_checkpoint(checkpoint_path)
    else:
        model_path = checkpoint_path
    
    print(f"Loading dataset: {dataset_name} (split: {split})")
    dataset = load_dataset(dataset_name, split=split)
    if num_samples is not None:
        dataset = dataset.select(range(num_samples))
    
    print(f"Dataset size: {len(dataset)}")
    
    # Format prompts (same as recom)
    print("Formatting prompts...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    def make_prompt(example):
        # Handle both 'question' and 'problem' keys (HotpotQA uses 'problem')
        question = example.get("question", example.get("problem", ""))
        user_format = f"\n\nPROBLEM: {question}\n\n"
        prompt = [
            {"role": "system", "content": TABC_LONG_PROMPT},
            {"role": "user", "content": user_format},
        ]
        return {
            "prompt": tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
            ),
            "answer": example.get("answer", ""),
            "question": question,
        }
    
    dataset = dataset.map(make_prompt)
    
    # Prepare texts for generation
    texts = [example["prompt"] for example in dataset]
    print(f"Sample prompt:\n{texts[0][:500]}...")
    
    # Generate outputs
    print(f"Loading model: {model_path}")
    llm = LLM(model=model_path, gpu_memory_utilization=gpu_memory_utilization, trust_remote_code=True)
    
    sampling_params = SamplingParams(
        n=n_generations,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=42,
    )
    
    print(f"Generating {n_generations} outputs per example...")
    outputs = llm.generate(texts, sampling_params=sampling_params)
    
    # Extract answers and confidence (same as recom)
    print("Extracting answers and confidence...")
    evals = []
    c_lengths = []
    confidence_levels = []
    conf_format_levels = []
    extracted_answers_list = []
    
    for i, output in enumerate(tqdm(outputs, desc="Processing outputs")):
        eval_list = []
        c_len_list = []
        conf_list = []
        conf_format_list = []
        answers_list = []
        
        gold_answer = dataset[i]["answer"]
        
        for j in range(n_generations):
            pred_response = output.outputs[j].text
            pred_answer = extract_answer(pred_response)
            answers_list.append(pred_answer)
            
            correctness = check_correctness(pred_answer, gold_answer)
            conf_format, conf_level = confidence_extractor(pred_response)
            
            eval_list.append(correctness)
            c_len_list.append(len(pred_response))
            conf_list.append(conf_level)
            conf_format_list.append(conf_format)
        
        evals.append(eval_list)
        c_lengths.append(c_len_list)
        confidence_levels.append(conf_list)
        conf_format_levels.append(conf_format_list)
        extracted_answers_list.append(answers_list)
    
    # Compute metrics (same as recom)
    print("\n" + "=" * 60)
    print("COMPUTING EVALUATION METRICS")
    print("=" * 60)
    
    metrics = {}
    
    # Pass@k
    pass_k_vals = [1, n_generations] if n_generations > 1 else [1]
    for k in pass_k_vals:
        if k <= n_generations:
            pass_k = compute_pass_n(evals, k)
            metrics[f"pass@{k}"] = float(pass_k)
    
    # Calibration metrics
    correctness_array = np.array(evals).flatten()
    confidence_array = np.array(confidence_levels).flatten()
    
    # Core metrics (same as recom)
    accuracy = np.mean(correctness_array)
    brier_score = get_brier(correctness_array, confidence_array)
    ece = get_ece(correctness_array, confidence_array)
    auroc = get_auroc(correctness_array, confidence_array)
    
    metrics["accuracy"] = float(accuracy)
    metrics["brier_score"] = float(brier_score)
    metrics["ece"] = float(ece)
    metrics["auroc"] = float(auroc)
    
    # Additional metrics (same as recom)
    metrics["completion_length"] = float(np.mean(np.array(c_lengths)))
    metrics["confidence_level"] = float(np.mean(confidence_array))
    metrics["confidence_format_adherence"] = float(np.mean(np.array(conf_format_levels)))
    metrics["num_samples"] = len(evals)
    
    # Print metrics (same format as recom)
    print("\nEvaluation Metrics:")
    print(f"  Accuracy:           {metrics['accuracy']:.4f}")
    print(f"  AUROC:              {metrics['auroc']:.4f}")
    print(f"  Brier Score:        {metrics['brier_score']:.4f}")
    print(f"  ECE:                {metrics['ece']:.4f}")
    print(f"\nAdditional Metrics:")
    print(f"  Mean Confidence:    {metrics['confidence_level']:.4f}")
    print(f"  Completion Length:  {metrics['completion_length']:.2f}")
    print(f"  Format Adherence:   {metrics['confidence_format_adherence']:.4f}")
    if n_generations > 1:
        print(f"  Pass@{n_generations}:      {metrics[f'pass@{n_generations}']:.4f}")
    print(f"  Num Samples:        {metrics['num_samples']}")
    print("=" * 60)
    
    # Save results
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics
        metrics_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"\nMetrics saved to: {metrics_path}")
        
        # Save detailed results
        results = []
        for i in range(len(dataset)):
            result = {
                "question": dataset[i]["question"],
                "gold_answer": dataset[i]["answer"],
                "predictions": [],
            }
            for j in range(n_generations):
                result["predictions"].append({
                    "answer": extracted_answers_list[i][j],
                    "correct": evals[i][j],
                    "confidence": confidence_levels[i][j],
                    "response": outputs[i].outputs[j].text,
                })
            results.append(result)
        
        results_path = os.path.join(output_dir, "detailed_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Detailed results saved to: {results_path}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate EasyR1 checkpoint on HotpotQA using recom format (exact match)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint (global_step_X/actor or merged model path)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mehuldamani/hotpot_qa",
        help="Dataset name (default: mehuldamani/hotpot_qa)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split (default: test)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all)",
    )
    parser.add_argument(
        "--n_generations",
        type=int,
        default=1,
        help="Number of generations per example (default: 1)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature (default: 0.0)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=4096,
        help="Maximum tokens to generate (default: 4096)",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization for vLLM (default: 0.9)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results (default: None)",
    )
    parser.add_argument(
        "--no_merge",
        action="store_true",
        help="Skip checkpoint merging (use if already merged)",
    )
    
    args = parser.parse_args()
    
    evaluate_hotpot_grpo(
        checkpoint_path=args.checkpoint,
        dataset_name=args.dataset,
        split=args.split,
        num_samples=args.num_samples,
        n_generations=args.n_generations,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        output_dir=args.output_dir,
        merge_checkpoint_first=not args.no_merge,
    )


if __name__ == "__main__":
    import sys
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

