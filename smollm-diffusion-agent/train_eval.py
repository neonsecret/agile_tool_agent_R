"""
Evaluation functions and metric computations for training.

Handles validation, functional evaluation, and NULL token metrics.
"""

import torch
import random
from typing import Dict, Optional
from tqdm.auto import tqdm
from accelerate import Accelerator

from data.device_utils import empty_cache


def _sum_across_processes(tensor: torch.Tensor, accelerator: Accelerator) -> torch.Tensor:
    if accelerator.num_processes > 1:
        return accelerator.reduce(tensor, reduction="sum")
    return tensor


def _compute_null_counts(predictions, labels, mask_positions, null_token_id):
    if null_token_id is None:
        return None
    pred_masked = predictions[mask_positions]
    label_masked = labels[mask_positions]
    label_is_null = label_masked == null_token_id
    pred_is_null = pred_masked == null_token_id
    null_pred = pred_is_null.sum()
    null_label = label_is_null.sum()
    null_correct = (pred_is_null & label_is_null).sum()
    real_correct = (pred_masked[~label_is_null] == label_masked[~label_is_null]).sum()
    real_total = (~label_is_null).sum()
    total_masked = mask_positions.sum()
    return {
        "null_pred": null_pred,
        "null_label": null_label,
        "null_correct": null_correct,
        "real_correct": real_correct,
        "real_total": real_total,
        "total_masked": total_masked,
    }


def _null_metrics_from_counts(counts):
    if counts is None or counts["total_masked"].item() == 0:
        return {}
    total_masked = counts["total_masked"].item()
    null_pred = counts["null_pred"].item()
    null_label = counts["null_label"].item()
    null_correct = counts["null_correct"].item()
    real_correct = counts["real_correct"].item()
    real_total = counts["real_total"].item()
    null_prediction_rate = null_pred / total_masked if total_masked > 0 else 0.0
    null_accuracy = null_correct / null_label if null_label > 0 else 0.0
    real_token_accuracy = real_correct / real_total if real_total > 0 else 0.0
    null_precision = null_correct / null_pred if null_pred > 0 else 0.0
    null_recall = null_correct / null_label if null_label > 0 else 0.0
    metrics = {
        "null_prediction_rate": null_prediction_rate,
        "null_accuracy": null_accuracy,
        "real_token_accuracy": real_token_accuracy,
        "null_precision": null_precision,
        "null_recall": null_recall,
    }
    if "unique_tokens" in counts and "pred_tokens" in counts:
        unique = counts["unique_tokens"].item()
        total = counts["pred_tokens"].item()
        metrics["token_diversity"] = unique / total if total > 0 else 0.0
    if "repetitions" in counts:
        reps = counts["repetitions"].item()
        total = counts["pred_tokens"].item() if "pred_tokens" in counts else total_masked
        metrics["token_repetition_rate"] = reps / max(1, total - 1)
    return metrics


def _compute_diversity_counts(predictions, mask_positions):
    pred_masked = predictions[mask_positions]
    if pred_masked.numel() == 0:
        return {"unique_tokens": torch.tensor(0), "pred_tokens": torch.tensor(0), "repetitions": torch.tensor(0)}
    unique = torch.unique(pred_masked).numel()
    total = pred_masked.numel()
    consecutive_same = (pred_masked[1:] == pred_masked[:-1]).sum()
    return {
        "unique_tokens": torch.tensor(unique, device=predictions.device),
        "pred_tokens": torch.tensor(total, device=predictions.device),
        "repetitions": consecutive_same,
    }


def _init_scaffold_stats(device):
    return {
        "count": torch.tensor(0.0, device=device),
        "sum": torch.tensor(0.0, device=device),
        "sum_sq": torch.tensor(0.0, device=device),
        "mask_sum": torch.tensor(0.0, device=device),
        "null_sum": torch.tensor(0.0, device=device),
        "null_ratio_sum": torch.tensor(0.0, device=device),
        "min": torch.tensor(float("inf"), device=device),
        "max": torch.tensor(float("-inf"), device=device),
    }


def _update_scaffold_stats(stats, scaffold_size, null_count):
    stats["count"] += 1.0
    stats["sum"] += scaffold_size
    stats["sum_sq"] += scaffold_size * scaffold_size
    stats["mask_sum"] += scaffold_size
    stats["null_sum"] += null_count
    null_ratio = null_count / scaffold_size if scaffold_size > 0 else 0.0
    stats["null_ratio_sum"] += null_ratio
    stats["min"] = torch.minimum(stats["min"], scaffold_size)
    stats["max"] = torch.maximum(stats["max"], scaffold_size)


def _scaffold_metrics_from_stats(stats, accelerator: Accelerator):
    if stats["count"].item() == 0:
        return {}
    stats["count"] = _sum_across_processes(stats["count"], accelerator)
    stats["sum"] = _sum_across_processes(stats["sum"], accelerator)
    stats["sum_sq"] = _sum_across_processes(stats["sum_sq"], accelerator)
    stats["mask_sum"] = _sum_across_processes(stats["mask_sum"], accelerator)
    stats["null_sum"] = _sum_across_processes(stats["null_sum"], accelerator)
    stats["null_ratio_sum"] = _sum_across_processes(stats["null_ratio_sum"], accelerator)

    count = stats["count"].item()
    mean = stats["sum"].item() / count if count > 0 else 0.0
    variance = stats["sum_sq"].item() / count - mean * mean if count > 0 else 0.0
    std = variance ** 0.5 if variance > 0 else 0.0

    min_vals = accelerator.gather(stats["min"].unsqueeze(0))
    max_vals = accelerator.gather(stats["max"].unsqueeze(0))

    return {
        "avg_scaffold_size": mean,
        "avg_null_ratio": stats["null_ratio_sum"].item() / count if count > 0 else 0.0,
        "scaffold_size_std": std,
        "max_scaffold_size": max_vals.max().item(),
        "min_scaffold_size": min_vals.min().item(),
    }


def evaluate(model, eval_dataloader, accelerator, null_token_id=None, return_logits=True):
    """Evaluate the model on validation set with enhanced metrics."""
    model.eval()
    total_loss = torch.tensor(0.0, device=accelerator.device)
    total_diffusion_loss = torch.tensor(0.0, device=accelerator.device)
    num_batches = torch.tensor(0.0, device=accelerator.device)
    null_counts = None
    scaffold_stats = _init_scaffold_stats(accelerator.device)

    eval_bar = tqdm(eval_dataloader, desc="Evaluating", disable=not accelerator.is_local_main_process)

    with torch.no_grad():
        for batch in eval_bar:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                scaffold_mask=batch["scaffold_mask"],
                return_logits=return_logits,
            )

            if outputs["loss"] is not None:
                total_loss += outputs["loss"].detach()
                num_batches += 1

                losses_detail = outputs.get("losses", {})
                if "diffusion" in losses_detail:
                    total_diffusion_loss += losses_detail["diffusion"].detach()

            predictions = outputs.get("predictions")
            if predictions is None and "logits" in outputs:
                logits = outputs["logits"]
                predictions = torch.argmax(logits, dim=-1)
            if predictions is not None:
                mask_positions = outputs.get("mask_positions", batch["scaffold_mask"])
                batch_null_counts = _compute_null_counts(
                    predictions,
                    batch["labels"],
                    mask_positions,
                    null_token_id,
                )
                if batch_null_counts is not None:
                    if null_counts is None:
                        null_counts = {k: v.clone() for k, v in batch_null_counts.items()}
                    else:
                        for key, value in batch_null_counts.items():
                            null_counts[key] += value

                for i in range(batch["scaffold_mask"].size(0)):
                    scaffold_mask = batch["scaffold_mask"][i]
                    labels = batch["labels"][i]
                    scaffold_size = scaffold_mask.sum().float()
                    if null_token_id is not None:
                        null_count = ((labels == null_token_id) & scaffold_mask).sum().float()
                    else:
                        null_count = torch.tensor(0.0, device=accelerator.device)
                    _update_scaffold_stats(scaffold_stats, scaffold_size, null_count)

    model.train()

    total_loss = _sum_across_processes(total_loss, accelerator)
    total_diffusion_loss = _sum_across_processes(total_diffusion_loss, accelerator)
    num_batches = _sum_across_processes(num_batches, accelerator)
    metrics = {
        "eval/total_loss": total_loss.item() / max(num_batches.item(), 1.0),
        "eval/diffusion_loss": total_diffusion_loss.item() / max(num_batches.item(), 1.0),
    }

    if null_counts is not None and null_token_id is not None:
        for key, value in null_counts.items():
            null_counts[key] = _sum_across_processes(value, accelerator)
        null_metrics = _null_metrics_from_counts(null_counts)
        for key, value in null_metrics.items():
            metrics[f"eval/{key}"] = value

    scaffold_metrics = _scaffold_metrics_from_stats(scaffold_stats, accelerator)
    for key, value in scaffold_metrics.items():
        metrics[f"eval/{key}"] = value

    return metrics
