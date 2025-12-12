# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Any

from mathruler.grader import extract_boxed_content, grade_answer


# Metadata
REWARD_NAME = "math"
REWARD_TYPE = "batch"


def format_reward(response: str) -> float:
    pattern = re.compile(r".*?</think>.*?<answer>.*?</answer>.*", re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    if not format_match:
        return 0.0
    
    # Also check that confidence is a valid number between 0 and 1
    confidence_pattern = r"<confidence>(.*?)</confidence>"
    confidence_matches = re.findall(confidence_pattern, response, re.DOTALL | re.MULTILINE)
    if confidence_matches:
        try:
            confidence = float(confidence_matches[-1].strip())
            if 0.0 <= confidence <= 1.0:
                return 1.0
        except Exception:
            pass
    
    return 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def brier_reward(response: str, ground_truth: str) -> float:
    """Brier score reward function: 1 - (correctness - confidence)²"""
    confidence_pattern = r"<confidence>(.*?)</confidence>"
    
    # Check format first
    format_score = format_reward(response)
    if format_score == 0.0:
        return 0.0
    
    # Get correctness (binary: 0 or 1)
    correctness = accuracy_reward(response, ground_truth)
    
    # Extract confidence
    confidence_matches = re.findall(confidence_pattern, response, re.DOTALL | re.MULTILINE)
    last_confidence = confidence_matches[-1] if confidence_matches else ""
    
    if last_confidence == "":
        return 0.0
    
    try:
        conf = float(last_confidence.strip())
        # Clamp confidence to [0, 1]
        conf = max(0.0, min(1.0, conf))
        # Brier score: 1 - (correctness - confidence)²
        reward = 1.0 - (correctness - conf) ** 2
        return reward
    except Exception:
        return 0.0


def compute_score(reward_inputs: list[dict[str, Any]], format_weight: float = 0.1) -> list[dict[str, float]]:
    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        brier_score = brier_reward(response, reward_input["ground_truth"])
        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
                "format": format_score,
                "accuracy": accuracy_score,
            }
        )

    return scores
