"""
Specialized metrics for function calling with diffusion.

These metrics help understand model behavior specific to our use case:
- NULL token prediction patterns
- Field-level accuracy  
- Diffusion convergence behavior
- Parse and validation rates
"""
import torch
import json
from typing import Dict, List, Any, Optional
from collections import defaultdict
import re


def extract_tool_call_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract and parse tool call JSON from text."""
    pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        return None
    try:
        return json.loads(matches[0])
    except json.JSONDecodeError:
        return None


def calculate_null_token_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    mask_positions: torch.Tensor,
    null_token_id: Optional[int]
) -> Dict[str, float]:
    """
    Calculate metrics about NULL token predictions.
    
    Returns:
        - null_prediction_rate: % of predictions that are NULL
        - null_accuracy: Accuracy on NULL positions
        - real_token_accuracy: Accuracy on non-NULL positions
        - null_precision: Of predicted NULLs, how many are correct
        - null_recall: Of actual NULLs, how many are predicted
    """
    if null_token_id is None or mask_positions.sum() == 0:
        return {}
    
    # Get masked positions
    pred_masked = predictions[mask_positions]
    label_masked = labels[mask_positions]
    
    # Separate NULL vs real tokens
    label_is_null = label_masked == null_token_id
    pred_is_null = pred_masked == null_token_id
    
    # NULL prediction rate
    null_pred_rate = pred_is_null.float().mean().item()
    
    # Accuracy on NULL positions
    null_correct = (pred_masked[label_is_null] == label_masked[label_is_null]).float()
    null_acc = null_correct.mean().item() if label_is_null.any() else 0.0
    
    # Accuracy on real token positions
    real_correct = (pred_masked[~label_is_null] == label_masked[~label_is_null]).float()
    real_acc = real_correct.mean().item() if (~label_is_null).any() else 0.0
    
    # Precision: of predicted NULLs, how many are correct
    null_precision = (pred_is_null & label_is_null).float().sum() / pred_is_null.float().sum() if pred_is_null.any() else 0.0
    
    # Recall: of actual NULLs, how many are predicted
    null_recall = (pred_is_null & label_is_null).float().sum() / label_is_null.float().sum() if label_is_null.any() else 0.0
    
    return {
        "null_prediction_rate": null_pred_rate,
        "null_accuracy": null_acc,
        "real_token_accuracy": real_acc,
        "null_precision": null_precision.item() if isinstance(null_precision, torch.Tensor) else null_precision,
        "null_recall": null_recall.item() if isinstance(null_recall, torch.Tensor) else null_recall,
    }


def calculate_field_level_metrics(
    predicted_tool_calls: List[Optional[Dict[str, Any]]],
    ground_truth_tool_calls: List[Optional[Dict[str, Any]]]
) -> Dict[str, float]:
    """
    Calculate field-level accuracy metrics.
    
    Returns:
        - tool_name_accuracy: % correct tool names
        - field_exact_match_rate: Average % of fields exactly correct per example
        - required_field_accuracy: Accuracy on fields that must be present
        - per_field_accuracy: Accuracy for specific field names
    """
    if not predicted_tool_calls or not ground_truth_tool_calls:
        return {}
    
    tool_name_correct = 0
    field_matches = []
    per_field_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    
    for pred, gt in zip(predicted_tool_calls, ground_truth_tool_calls):
        if pred is None or gt is None:
            continue
        
        # Tool name accuracy
        if pred.get("name") == gt.get("name"):
            tool_name_correct += 1
        
        # Field-level accuracy
        pred_args = pred.get("arguments", {})
        gt_args = gt.get("arguments", {})
        
        if not gt_args:
            continue
        
        field_correct = 0
        for field_name, gt_value in gt_args.items():
            pred_value = pred_args.get(field_name)
            is_correct = pred_value == gt_value
            
            if is_correct:
                field_correct += 1
            
            # Track per-field stats
            per_field_stats[field_name]["total"] += 1
            if is_correct:
                per_field_stats[field_name]["correct"] += 1
        
        field_match_rate = field_correct / len(gt_args) if gt_args else 0.0
        field_matches.append(field_match_rate)
    
    # Calculate per-field accuracy for common fields
    per_field_acc = {}
    for field_name, stats in per_field_stats.items():
        if stats["total"] >= 3:  # Only report if seen at least 3 times
            per_field_acc[f"field_{field_name}_accuracy"] = stats["correct"] / stats["total"]
    
    return {
        "tool_name_accuracy": tool_name_correct / len(predicted_tool_calls),
        "field_exact_match_rate": sum(field_matches) / len(field_matches) if field_matches else 0.0,
        **per_field_acc
    }


def calculate_parse_metrics(
    generated_texts: List[str],
    has_tool_calls: List[bool]
) -> Dict[str, float]:
    """
    Calculate parsing and validation metrics.
    
    Returns:
        - json_parse_success_rate: % of outputs that parse as valid JSON
        - tool_call_format_success_rate: % with correct <tool_call> format
        - false_positive_rate: % of non-tool examples that generated tool calls
        - false_negative_rate: % of tool examples that didn't generate tool calls
    """
    parse_success = 0
    format_success = 0
    false_positives = 0
    false_negatives = 0
    
    for text, should_have_tool_call in zip(generated_texts, has_tool_calls):
        has_format = "<tool_call>" in text and "</tool_call>" in text
        
        if has_format:
            format_success += 1
            
            # Try to parse
            parsed = extract_tool_call_json(text)
            if parsed is not None:
                parse_success += 1
        
        # False positive/negative
        if has_format and not should_have_tool_call:
            false_positives += 1
        elif not has_format and should_have_tool_call:
            false_negatives += 1
    
    n_tool = sum(has_tool_calls)
    n_no_tool = len(has_tool_calls) - n_tool
    
    return {
        "json_parse_success_rate": parse_success / len(generated_texts) if generated_texts else 0.0,
        "tool_call_format_rate": format_success / len(generated_texts) if generated_texts else 0.0,
        "false_positive_rate": false_positives / n_no_tool if n_no_tool > 0 else 0.0,
        "false_negative_rate": false_negatives / n_tool if n_tool > 0 else 0.0,
    }


def calculate_scaffold_metrics(
    scaffold_sizes: List[int],
    mask_counts: List[int],
    null_counts: List[int]
) -> Dict[str, float]:
    """
    Calculate scaffold and masking statistics.
    
    Returns:
        - avg_scaffold_size: Average scaffold size in tokens
        - avg_mask_count: Average number of masked positions
        - avg_null_ratio: Average ratio of NULL to mask tokens
        - scaffold_size_std: Standard deviation of scaffold sizes
    """
    if not scaffold_sizes:
        return {}
    
    import numpy as np
    
    null_ratios = [n / m if m > 0 else 0.0 for n, m in zip(null_counts, mask_counts)]
    
    return {
        "avg_scaffold_size": np.mean(scaffold_sizes),
        "avg_mask_count": np.mean(mask_counts),
        "avg_null_ratio": np.mean(null_ratios),
        "scaffold_size_std": np.std(scaffold_sizes),
        "max_scaffold_size": np.max(scaffold_sizes),
        "min_scaffold_size": np.min(scaffold_sizes),
    }


def calculate_confidence_metrics(
    confidence_scores: List[float],
    is_correct: List[bool]
) -> Dict[str, float]:
    """
    Calculate confidence-related metrics.
    
    Returns:
        - avg_confidence: Average confidence of predictions
        - confidence_correct: Average confidence when correct
        - confidence_incorrect: Average confidence when incorrect
        - overconfidence_rate: % of high-confidence wrong predictions
    """
    if not confidence_scores:
        return {}
    
    import numpy as np
    
    confidence_correct = [c for c, correct in zip(confidence_scores, is_correct) if correct]
    confidence_incorrect = [c for c, correct in zip(confidence_scores, is_correct) if not correct]
    
    # High confidence wrong predictions (confidence > 0.8 but wrong)
    overconfident_wrong = sum(1 for c, correct in zip(confidence_scores, is_correct) 
                              if c > 0.8 and not correct)
    
    return {
        "avg_confidence": np.mean(confidence_scores),
        "confidence_when_correct": np.mean(confidence_correct) if confidence_correct else 0.0,
        "confidence_when_incorrect": np.mean(confidence_incorrect) if confidence_incorrect else 0.0,
        "overconfidence_rate": overconfident_wrong / len(confidence_scores),
    }
