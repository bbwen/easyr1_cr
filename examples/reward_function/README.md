# Reward Functions

This directory contains reward functions for different tasks in EasyR1. Each reward function implements a `compute_score` function that takes reward inputs and returns scores.

## Math Reward Functions

### `math_ori.py` - Original Math Reward Function

Standard math reward function that computes:
- **Format Score**: Checks if response follows the required format (`<think>...<answer>...</answer>...`)
- **Accuracy Score**: Binary correctness (1.0 if correct, 0.0 if incorrect)
- **Overall Score**: Weighted combination of format and accuracy

**Features**:
- Validates response format
- Extracts and grades answers using `mathruler.grader`
- Returns format, accuracy, and overall scores

**Usage**:
```yaml
worker:
  reward:
    reward_function: ./examples/reward_function/math_ori.py:compute_score
```

### `math_conf.py` - Math Reward Function with Brier Score

Enhanced math reward function that includes **Brier score** for confidence calibration:
- **Format Score**: Checks if response follows the required format with confidence tag
- **Accuracy Score**: Binary correctness (1.0 if correct, 0.0 if incorrect)
- **Brier Score**: Calibration metric that rewards well-calibrated confidence predictions
- **Overall Score**: Weighted combination of format and accuracy

**Brier Score Implementation**:
The Brier score measures the calibration of confidence predictions. It's computed as:
```
Brier Score = 1 - (correctness - confidence)²
```

Where:
- `correctness` is binary (0 or 1) based on answer accuracy
- `confidence` is extracted from `<confidence>...</confidence>` tags in the response
- Higher Brier scores indicate better calibration (confidence matches correctness)

**Key Differences from `math_ori.py`**:
1. Requires `<analysis>...</analysis>` and `<confidence>...</confidence>` tags in format validation
2. Computes and returns Brier score in the score dictionary
3. Brier score rewards models that provide well-calibrated confidence estimates

**Usage**:
```yaml
worker:
  reward:
    reward_function: ./examples/reward_function/math_conf.py:compute_score
```

**Output Format**:
```python
{
    "overall": float,    # Weighted combination: (1 - format_weight) * accuracy + format_weight * format
    "format": float,      # Format compliance score (0.0 or 1.0)
    "accuracy": float,   # Answer correctness (0.0 or 1.0)
    "brier": float       # Brier score (0.0 to 1.0, higher is better)
}
```

## Other Reward Functions

- **`r1v.py`**: Vision-language reward function with Brier score support
- **`dapo.py`**: DAPO-specific reward function
- **`android_gui.py`**: Android GUI task reward function

## Reward Function Interface

All reward functions must implement:

```python
def compute_score(reward_inputs: list[dict[str, Any]], **kwargs) -> list[dict[str, float]]:
    """
    Compute rewards for a batch of responses.
    
    Args:
        reward_inputs: List of dictionaries containing:
            - "response": str - Model response
            - "ground_truth": str - Ground truth answer
            - Other task-specific fields
    
    Returns:
        List of score dictionaries with keys like "overall", "format", "accuracy", etc.
    """
    pass
```

## Brier Score Details

The Brier score is a proper scoring rule that measures the accuracy of probabilistic predictions. In the context of confidence calibration:

- **Perfect Calibration**: When confidence = correctness, Brier score = 1.0
- **Overconfident**: When confidence > correctness, Brier score < 1.0
- **Underconfident**: When confidence < correctness, Brier score < 1.0

**Example**:
- Correct answer with confidence 0.9: Brier = 1 - (1 - 0.9)² = 0.99
- Correct answer with confidence 0.5: Brier = 1 - (1 - 0.5)² = 0.75
- Incorrect answer with confidence 0.1: Brier = 1 - (0 - 0.1)² = 0.99
- Incorrect answer with confidence 0.9: Brier = 1 - (0 - 0.9)² = 0.19

This encourages models to:
1. Be confident when they're correct
2. Be uncertain when they're incorrect
3. Provide well-calibrated confidence estimates

