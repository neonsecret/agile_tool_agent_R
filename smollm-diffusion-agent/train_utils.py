"""
Training utilities and helper functions.

Extracted from train.py for better code organization.
"""

import torch
import torch.distributed as dist
import yaml
from typing import Dict, Optional, Sequence, Tuple
from accelerate import Accelerator
from torch.optim.lr_scheduler import LambdaLR, OneCycleLR


class ClampedCosineScheduler(LambdaLR):
    """Cosine scheduler with warmup that decays to min_lr_ratio after total_training_steps.
    
    The default HuggingFace get_cosine_schedule_with_warmup continues the cosine
    curve past total_training_steps, which causes LR to cycle back up. This wrapper
    decays to a minimum LR and stays there.
    """
    def __init__(self, optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.01, last_epoch=-1):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.min_lr_ratio = min_lr_ratio  # e.g., 0.01 = decay to 1% of peak LR
        super().__init__(optimizer, self._lr_lambda, last_epoch=last_epoch)

    def _lr_lambda(self, current_step):
        if current_step >= self.num_training_steps:
            return self.min_lr_ratio
        if current_step < self.num_warmup_steps:
            return float(current_step) / float(max(1, self.num_warmup_steps))
        import math
        progress = float(current_step - self.num_warmup_steps) / float(
            max(1, self.num_training_steps - self.num_warmup_steps)
        )
        # Cosine decay from 1.0 to min_lr_ratio
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine_decay


def _split_muon_params(module: torch.nn.Module) -> Tuple[list, list]:
    muon_params = []
    adamw_params = []
    for _, param in module.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2:
            muon_params.append(param)
        else:
            adamw_params.append(param)
    return muon_params, adamw_params


def _build_adamw_optimizer(
    params: Sequence[torch.nn.Parameter],
    training_cfg: Dict,
) -> Tuple[torch.optim.Optimizer, Dict[str, float]]:
    optim_cfg = training_cfg.get("optimizer", {})
    adamw_cfg = optim_cfg.get("adamw", {})
    lr = float(adamw_cfg.get("lr", training_cfg["learning_rate"]))
    betas = tuple(adamw_cfg.get("betas", (0.9, 0.95)))
    eps = float(adamw_cfg.get("eps", 1e-8))
    weight_decay = float(adamw_cfg.get("weight_decay", 0.01))
    optimizer = torch.optim.AdamW(
        params,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )
    info = {
        "name": "adamw",
        "lr": lr,
        "weight_decay": weight_decay,
        "betas": betas,
        "eps": eps,
    }
    return optimizer, info


def _build_muon_optimizer(
    module: torch.nn.Module,
    training_cfg: Dict,
) -> Tuple[torch.optim.Optimizer, Dict[str, float]]:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError(
            "Muon requires torch.distributed to be initialized. Use accelerate launch "
            "or set training.optimizer.name to 'adamw'."
        )
    from muon import MuonWithAuxAdam

    muon_params, adamw_params = _split_muon_params(module)
    if not muon_params and not adamw_params:
        raise ValueError("No trainable parameters found for optimizer.")

    optim_cfg = training_cfg.get("optimizer", {})
    muon_cfg = optim_cfg.get("muon", {})
    adamw_cfg = optim_cfg.get("adamw", {})

    muon_lr = float(muon_cfg.get("lr", training_cfg["learning_rate"]))
    muon_momentum = float(muon_cfg.get("momentum", 0.95))
    muon_weight_decay = float(muon_cfg.get("weight_decay", 0.0))

    adamw_lr = float(adamw_cfg.get("lr", training_cfg["learning_rate"]))
    adamw_betas = tuple(adamw_cfg.get("betas", (0.9, 0.95)))
    adamw_eps = float(adamw_cfg.get("eps", 1e-10))
    adamw_weight_decay = float(adamw_cfg.get("weight_decay", 0.0))

    param_groups = []
    if muon_params:
        param_groups.append(
            {
                "params": muon_params,
                "use_muon": True,
                "lr": muon_lr,
                "momentum": muon_momentum,
                "weight_decay": muon_weight_decay,
            }
        )
    if adamw_params:
        param_groups.append(
            {
                "params": adamw_params,
                "use_muon": False,
                "lr": adamw_lr,
                "betas": adamw_betas,
                "eps": adamw_eps,
                "weight_decay": adamw_weight_decay,
            }
        )
    optimizer = MuonWithAuxAdam(param_groups)
    info = {
        "name": "muon",
        "muon_lr": muon_lr,
        "muon_momentum": muon_momentum,
        "muon_weight_decay": muon_weight_decay,
        "adamw_lr": adamw_lr,
        "adamw_weight_decay": adamw_weight_decay,
    }
    return optimizer, info


def _resolve_muon_availability() -> Tuple[bool, str]:
    if not dist.is_available():
        return False, "torch.distributed not available"
    if not dist.is_initialized():
        return False, "torch.distributed not initialized"
    if dist.get_world_size() <= 1:
        return False, "single_process"
    return True, "ok"


def build_optimizer(
    module: torch.nn.Module,
    training_cfg: Dict,
) -> Tuple[torch.optim.Optimizer, Dict[str, float]]:
    optim_cfg = training_cfg.get("optimizer", {})
    name = optim_cfg.get("name", "adamw")
    params = [p for p in module.parameters() if p.requires_grad]
    if name == "muon":
        can_use_muon, reason = _resolve_muon_availability()
        if not can_use_muon:
            optimizer, info = _build_adamw_optimizer(params, training_cfg)
            info["fallback_from"] = "muon"
            info["fallback_reason"] = reason
            return optimizer, info
        return _build_muon_optimizer(module, training_cfg)
    return _build_adamw_optimizer(params, training_cfg)


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_collate_fn(bucket_sizes, max_seq_len, pad_token_id: int):
    """Create a collate_fn that pads to the next bucket size >= batch max length.

    This keeps shapes more stable for torch.compile and CUDA graphs.
    """
    bucket_sizes = sorted(bucket_sizes)

    def collate_fn(batch):
        max_len = max(item["input_ids"].size(0) for item in batch)
        pad_to = None
        for b in bucket_sizes:
            if b >= max_len:
                pad_to = b
                break
        if pad_to is None:
            pad_to = bucket_sizes[-1]
        pad_to = min(pad_to, max_seq_len)

        def pad_tensor(t: torch.Tensor, pad_value: int):
            if t.size(0) > pad_to:
                return t[:pad_to]
            if t.size(0) == pad_to:
                return t
            pad_width = pad_to - t.size(0)
            return torch.nn.functional.pad(t, (0, pad_width), value=pad_value)

        input_ids = [pad_tensor(item["input_ids"], pad_value=pad_token_id) for item in batch]
        attention_mask = [pad_tensor(item["attention_mask"], pad_value=0) for item in batch]
        scaffold_mask = [pad_tensor(item["scaffold_mask"], pad_value=0) for item in batch]
        labels = [pad_tensor(item["labels"], pad_value=-100) for item in batch]

        batch_dict = {
            "input_ids": torch.stack(input_ids, dim=0),
            "attention_mask": torch.stack(attention_mask, dim=0),
            "scaffold_mask": torch.stack(scaffold_mask, dim=0),
            "labels": torch.stack(labels, dim=0),
        }

        return batch_dict

    return collate_fn


def _build_clamped_cosine_scheduler(optimizer, training_cfg, total_training_steps):
    warmup_steps = training_cfg.get("warmup_steps", 2500)
    warmup_steps = min(warmup_steps, int(0.1 * total_training_steps))
    min_lr_ratio = training_cfg.get("min_lr_ratio", 0.01)
    scheduler = ClampedCosineScheduler(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
        min_lr_ratio=min_lr_ratio,
    )
    info = {
        "name": "clamped_cosine",
        "warmup_steps": warmup_steps,
        "min_lr_ratio": min_lr_ratio,
    }
    return scheduler, info


def _resolve_one_cycle_max_lr(optimizer, training_cfg):
    scheduler_cfg = training_cfg.get("scheduler", {})
    max_lr = scheduler_cfg.get("max_lr", training_cfg["learning_rate"])
    if isinstance(max_lr, (list, tuple)):
        max_lr_values = [float(v) for v in max_lr]
    else:
        max_lr_values = [float(max_lr) for _ in optimizer.param_groups]
    if len(max_lr_values) != len(optimizer.param_groups):
        raise ValueError(
            f"OneCycleLR max_lr length {len(max_lr_values)} does not match "
            f"param_groups length {len(optimizer.param_groups)}."
        )
    return max_lr_values


def _build_one_cycle_scheduler(optimizer, training_cfg, total_training_steps):
    scheduler_cfg = training_cfg.get("scheduler", {})
    max_lr = _resolve_one_cycle_max_lr(optimizer, training_cfg)
    pct_start = float(scheduler_cfg.get("pct_start", 0.1))
    warmup_steps = training_cfg.get("warmup_steps")
    use_warmup_steps = scheduler_cfg.get("use_warmup_steps", True)
    if use_warmup_steps and warmup_steps is not None and total_training_steps > 0:
        pct_start = max(0.01, min(0.5, float(warmup_steps) / float(total_training_steps)))
    anneal_strategy = scheduler_cfg.get("anneal_strategy", "cos")
    div_factor = float(scheduler_cfg.get("div_factor", 25.0))
    final_div_factor = float(scheduler_cfg.get("final_div_factor", 1e4))
    cycle_momentum = bool(scheduler_cfg.get("cycle_momentum", False))
    three_phase = bool(scheduler_cfg.get("three_phase", False))
    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_training_steps,
        pct_start=pct_start,
        anneal_strategy=anneal_strategy,
        div_factor=div_factor,
        final_div_factor=final_div_factor,
        cycle_momentum=cycle_momentum,
        three_phase=three_phase,
    )
    info = {
        "name": "one_cycle",
        "max_lr": max_lr,
        "pct_start": pct_start,
        "use_warmup_steps": use_warmup_steps,
        "anneal_strategy": anneal_strategy,
        "div_factor": div_factor,
        "final_div_factor": final_div_factor,
    }
    return scheduler, info


def build_scheduler(optimizer, training_cfg, total_training_steps):
    scheduler_cfg = training_cfg.get("scheduler", {})
    name = scheduler_cfg.get("name", "clamped_cosine")
    if name == "one_cycle":
        return _build_one_cycle_scheduler(optimizer, training_cfg, total_training_steps)
    return _build_clamped_cosine_scheduler(optimizer, training_cfg, total_training_steps)


def safe_unwrap_model(model):
    """Avoid accelerate.unwrap_model to prevent optional deepspeed import side effects."""
    if hasattr(model, "module"):
        return model.module
    return model


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, accelerator):
    import os
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}\n"
            f"Please ensure the checkpoint file exists or set resume_from_checkpoint: false"
        )

    accelerator.print(f"[CHECKPOINT] Loading checkpoint from: {checkpoint_path}")
    accelerator.print(f"[CHECKPOINT] Checkpoint file exists: {os.path.exists(checkpoint_path)}")
    checkpoint = torch.load(checkpoint_path, map_location=accelerator.device)
    accelerator.print(f"[CHECKPOINT] Checkpoint loaded successfully")

    unwrapped_model = safe_unwrap_model(model)
    checkpoint_state = checkpoint['model_state_dict']
    
    if hasattr(unwrapped_model, 'load_trainable_state_dict'):
        missing_keys, unexpected_keys = unwrapped_model.load_trainable_state_dict(
            checkpoint_state, strict=False
        )
        accelerator.print(f"[CHECKPOINT] Loaded trainable parameters (diffusion_head)")
        if missing_keys:
            accelerator.print(f"[CHECKPOINT] Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            accelerator.print(f"[CHECKPOINT] Unexpected keys: {len(unexpected_keys)}")
    else:
        filtered_state = {k: v for k, v in checkpoint_state.items() 
                         if k.startswith('diffusion_head.')}
        if filtered_state:
            unwrapped_model.load_state_dict(filtered_state, strict=False)
            accelerator.print(f"[CHECKPOINT] Loaded {len(filtered_state)} trainable parameters")
        else:
            accelerator.print("[CHECKPOINT] WARNING: No trainable parameters found in checkpoint")
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    accelerator.print(f"[CHECKPOINT] Loaded optimizer state")

    if 'scheduler_state_dict' in checkpoint and scheduler is not None:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            accelerator.print(f"[CHECKPOINT] Loaded scheduler state")
        except (KeyError, TypeError) as e:
            accelerator.print(f"[CHECKPOINT] WARNING: Scheduler state incompatible (type mismatch), using fresh scheduler")
            accelerator.print(f"[CHECKPOINT] (Error: {type(e).__name__}: {e})")
    else:
        accelerator.print(f"[CHECKPOINT] No scheduler state in checkpoint (or scheduler is None)")

    start_epoch = checkpoint['epoch'] + 1
    best_eval_loss = checkpoint['eval_loss']

    accelerator.print(f"[CHECKPOINT] Checkpoint info:")
    accelerator.print(f"  - Previous epoch: {checkpoint['epoch']}")
    accelerator.print(f"  - Resuming from epoch: {start_epoch}")
    accelerator.print(f"  - Best eval loss: {best_eval_loss:.4f}")

    return start_epoch, best_eval_loss
