"""
MLX training and evaluation functions.

Handles forward passes, loss computation, and evaluation loops for MLX.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import torch
import numpy as np
from typing import Dict, Optional, Tuple

from model.mlx_hybrid_model import log_softmax
from train_mlx_utils import clip_grad_norm, all_reduce_grads, check_nan


def collate_fn(batch: list) -> Dict[str, mx.array]:
    """Collate function for batching (converts torch tensors from dataset to MLX)."""

    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.numpy()
        return np.array(x)

    input_ids = [to_numpy(item['input_ids']) for item in batch]
    attention_mask = [to_numpy(item['attention_mask']) for item in batch]
    scaffold_mask = [to_numpy(item['scaffold_mask']) for item in batch]
    labels = [to_numpy(item['labels']) for item in batch]

    max_len = max(len(ids) for ids in input_ids)

    def pad_sequence(seqs, pad_value):
        return mx.array(np.stack([
            np.pad(seq, (0, max_len - len(seq)), constant_values=pad_value)
            for seq in seqs
        ]))

    batch_dict = {
        "input_ids": pad_sequence(input_ids, 0),
        "attention_mask": pad_sequence(attention_mask, 0),
        "scaffold_mask": pad_sequence(scaffold_mask, False),
        "labels": pad_sequence(labels, -100),
    }

    if 'router_label' in batch[0]:
        batch_dict["router_labels"] = mx.array([item['router_label'] for item in batch])

    return batch_dict


def evaluate(
        model,
        eval_loader: list,
        train_router: bool,
) -> Dict[str, float]:
    """Evaluate the model on validation set."""
    total_loss = 0.0
    total_diffusion_loss = 0.0
    total_router_loss = 0.0
    total_router_correct = 0
    total_router_samples = 0
    num_eval_batches = 0

    for batch in eval_loader:
        current_router_labels = batch.get("router_labels") if train_router else None

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            scaffold_mask=batch["scaffold_mask"],
            router_labels=current_router_labels,
            training=False,
        )
        mx.eval(outputs["loss"])

        if outputs["loss"] is not None:
            loss_val = float(outputs["loss"])
            total_loss += loss_val
            num_eval_batches += 1

            losses_detail = outputs.get("losses", {})
            if "diffusion" in losses_detail:
                mx.eval(losses_detail["diffusion"])
                total_diffusion_loss += float(losses_detail["diffusion"])
            if "router" in losses_detail:
                mx.eval(losses_detail["router"])
                total_router_loss += float(losses_detail["router"])

            if train_router and "router_logits" in outputs and current_router_labels is not None:
                router_preds = mx.argmax(outputs["router_logits"], axis=-1)
                correct = mx.sum(router_preds == current_router_labels)
                mx.eval(correct)
                total_router_correct += int(correct)
                total_router_samples += current_router_labels.shape[0]

    metrics = {
        "eval/total_loss": total_loss / max(num_eval_batches, 1),
        "eval/diffusion_loss": total_diffusion_loss / max(num_eval_batches, 1),
    }

    if train_router and total_router_samples > 0:
        metrics["eval/router_loss"] = total_router_loss / max(num_eval_batches, 1)
        metrics["eval/router_accuracy"] = total_router_correct / total_router_samples

    return metrics


def train_step(
        model,
        batch: Dict,
        optimizer: optim.Optimizer,
        train_router: bool,
        world=None,
        max_grad_norm: float = 1.0,
        debug: bool = False,
) -> Tuple[mx.array, Optional[Dict]]:
    """Single training step with gradient clipping. Returns (loss, debug_info)."""
    hidden_states = model._get_hidden_states(batch["input_ids"], batch.get("attention_mask"))
    mx.eval(hidden_states)

    if debug or True:
        hs_check = check_nan(hidden_states, "hidden_states")
        if hs_check:
            return mx.array(float('nan')), {"reason": "hidden_states_nan", "details": hs_check}

    local_debug = {}

    def loss_fn(diff_params, router_params):
        model.diffusion_head.update(diff_params)
        model.router_head.update(router_params)

        total_loss = mx.array(0.0, dtype=mx.float32)
        labels = batch["labels"]
        scaffold_mask = batch["scaffold_mask"]

        if labels is not None and scaffold_mask is not None:
            diff_loss = model.diffusion_head.training_step(
                tokens=labels,
                hidden_states=hidden_states,
                scaffold_mask=scaffold_mask,
            )
            total_loss = total_loss + diff_loss
            if float(diff_loss) == 0.0:
                local_debug["diffusion_zero_loss"] = {
                    "scaffold_sum": float(mx.sum(scaffold_mask.astype(mx.float32))),
                    "reason": "no_mask_or_no_valid",
                }

        if train_router:
            router_labels = batch.get("router_labels")
            if router_labels is not None:
                router_logits = model.router_head(hidden_states)
                log_probs = log_softmax(router_logits, axis=-1)
                nll = -mx.take_along_axis(
                    log_probs,
                    mx.expand_dims(router_labels.astype(mx.int32), -1),
                    axis=-1
                ).squeeze(-1)
                total_loss = total_loss + mx.mean(nll)

        return total_loss

    diff_params = dict(model.diffusion_head.parameters())
    router_params = dict(model.router_head.parameters())

    loss_and_grad_fn = mx.value_and_grad(loss_fn, argnums=(0, 1))
    loss, (diff_grads, router_grads) = loss_and_grad_fn(diff_params, router_params)
    mx.eval(loss)

    loss_val = float(loss)
    if np.isnan(loss_val):
        debug_msg = f"NaN in loss computation. hidden_states: min={float(mx.min(hidden_states)):.4f}, max={float(mx.max(hidden_states)):.4f}"
        return loss, debug_msg

    if world is not None and world.size() > 1:
        diff_grads = all_reduce_grads(diff_grads, world)
        if train_router:
            router_grads = all_reduce_grads(router_grads, world)
        loss = mx.distributed.all_sum(loss) / world.size()
        mx.eval(loss)

    diff_grads = clip_grad_norm(diff_grads, max_grad_norm)
    if train_router:
        router_grads = clip_grad_norm(router_grads, max_grad_norm)

    new_diff_params = optimizer.apply_gradients(diff_grads, diff_params)
    model.diffusion_head.update(new_diff_params)

    if train_router:
        new_router_params = optimizer.apply_gradients(router_grads, router_params)
        model.router_head.update(new_router_params)

    return loss, local_debug
