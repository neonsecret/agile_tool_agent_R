#!/usr/bin/env python
"""
Evaluate warm-start diffusion approach on BFCL benchmark.

Compares:
1. Base AR model (baseline - ~94% on BFCL simple)
2. Standard diffusion (current - ~21% on BFCL simple)
3. Warm-start diffusion (AR draft + diffusion refinement)
"""

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from bfcl_eval.eval_checker.ast_eval.ast_checker import simple_function_checker
from bfcl_eval.constants.enums import Language


@dataclass
class EvalResult:
    model_name: str
    accuracy: float
    total: int
    correct: int
    mean_latency_ms: float
    peak_memory_mb: float
    errors: Dict[str, int]
    extra_info: Dict[str, Any] = None


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_memory_mb(device):
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    elif device.type == "mps":
        return torch.mps.driver_allocated_memory() / (1024 ** 2)
    return 0


def reset_memory(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def parse_json(field):
    if isinstance(field, str):
        return json.loads(field)
    return field


def get_function_schema(functions, func_name):
    for f in functions:
        if f.get("name") == func_name:
            return f
    return None


def format_model_output(func_name: str, args: dict) -> dict:
    return {func_name: args}


def format_ground_truth(raw_gt) -> dict:
    if isinstance(raw_gt, str):
        raw_gt = json.loads(raw_gt)
    if isinstance(raw_gt, list) and raw_gt:
        raw_gt = raw_gt[0]
    if isinstance(raw_gt, dict):
        if "name" in raw_gt:
            return {raw_gt["name"]: raw_gt.get("arguments", {})}
        func_name = list(raw_gt.keys())[0]
        return raw_gt
    return {}


def load_bfcl_data(test_category="simple", limit=None):
    ds = load_dataset("llamastack/bfcl_v3")
    data = [ex for ex in ds["train"] if ex.get("test_category") == test_category]
    if limit:
        data = data[:limit]
    return data


def build_registry(functions):
    registry = {}
    for f in functions:
        name = f.get("name")
        if name:
            registry[name] = f
    return registry


def evaluate_warmstart_model(data, device, show_steps=False):
    """Evaluate warm-start diffusion model."""
    import yaml
    from transformers import AutoTokenizer
    from data.config_utils import validate_and_adjust_config
    from model.hybrid_model import HybridSmolLM
    from inference_warmstart import WarmStartGenerator, WarmStartConfig

    print("Loading SmolLM3-Diffusion for warm-start evaluation...")

    config_path = ROOT_DIR / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config = validate_and_adjust_config(config, device)

    base_model_id = config["model"]["base_model_id"]
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = HybridSmolLM(
        base_model_id=base_model_id,
        load_in_4bit=config["quantization"]["enabled"],
        diffusion_config=config["diffusion"],
        max_seq_length=config["training"]["max_seq_len"],
        use_flash_attention=config["model"].get("use_flash_attention", True),
        use_gradient_checkpointing=False,
        device=device,
    )
    model.eval()

    checkpoint_path = ROOT_DIR / config["training"]["checkpoint_path"]
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            stripped = {}
            for k, v in state_dict.items():
                if k.startswith("diffusion_head."):
                    stripped[k[len("diffusion_head."):]] = v
                else:
                    stripped[k] = v
            model.diffusion_head.load_state_dict(stripped)
        else:
            model.diffusion_head.load_state_dict(checkpoint)
        print(f"Loaded diffusion checkpoint from {checkpoint_path}")

    from data.utils import resolve_mask_token, resolve_null_token
    data_cfg = config.get("data", {})
    _, mask_token_id = resolve_mask_token(tokenizer, data_cfg.get("mask_token"))
    _, null_token_id = resolve_null_token(tokenizer, data_cfg.get("null_token"))
    model.diffusion_head.set_mask_token_id(mask_token_id)
    if null_token_id is not None:
        model.diffusion_head.set_null_token_id(null_token_id)

    infer_cfg = config.get("inference", {})
    generator = WarmStartGenerator(model, tokenizer, device)
    warmstart_config = WarmStartConfig(
        diffusion_steps=infer_cfg.get("steps", 4),
        temperature=infer_cfg.get("temperature", 0.0),
        refinement_ratio=0.3,
        min_confidence_to_keep=0.8,
        use_cuda_graph=infer_cfg.get("use_cuda_graph", True),
        show_steps=show_steps,
        reencode_hidden_states_every=1,
    )

    reset_memory(device)
    correct = 0
    total = 0
    latencies = []
    error_counts = {}
    positions_refined_total = 0
    total_scaffold_total = 0

    for example in tqdm(data, desc="Evaluating Warm-Start Diffusion"):
        raw_functions = example.get("functions")
        functions = parse_json(raw_functions) if raw_functions else []
        registry = build_registry(functions)
        if not registry:
            continue

        query = example.get("question", "")
        if not query:
            turns = parse_json(example.get("turns", []))
            for turn in turns[0] if turns else []:
                if turn.get("role") == "user":
                    query = turn.get("content", "")
                    break

        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        result = generator.generate(
            prompt=query,
            tool_registry=registry,
            config=warmstart_config,
        )

        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)

        positions_refined_total += result.get("positions_refined", 0)
        total_scaffold_total += result.get("total_scaffold", 0)

        prediction = result.get("tool_call_parsed")
        tool_name = result.get("tool_name", "")

        raw_gt = parse_json(example.get("ground_truth", {}))
        gt = format_ground_truth(raw_gt)
        gt_func_name = list(gt.keys())[0] if gt else ""

        pred_name = prediction.get("name", "") if prediction else tool_name
        pred_args = prediction.get("arguments", {}) if prediction else {}
        model_output = format_model_output(pred_name, pred_args)

        func_schema = get_function_schema(functions, gt_func_name)
        if not func_schema:
            continue

        check_result = simple_function_checker(
            func_description=func_schema,
            model_output=model_output,
            possible_answer=gt,
            language=Language.PYTHON,
            model_name="gorilla-openfunctions-v2",
        )

        total += 1
        if check_result["valid"]:
            correct += 1
        else:
            err_type = check_result.get("error_type", "unknown")
            error_counts[err_type] = error_counts.get(err_type, 0) + 1

    del model, generator
    gc.collect()
    peak_mem = get_memory_mb(device)
    reset_memory(device)

    avg_refined = positions_refined_total / total if total > 0 else 0
    avg_scaffold = total_scaffold_total / total if total > 0 else 0

    return EvalResult(
        model_name="SmolLM3-WarmStart",
        accuracy=correct / total if total > 0 else 0,
        total=total,
        correct=correct,
        mean_latency_ms=sum(latencies) / len(latencies) if latencies else 0,
        peak_memory_mb=peak_mem,
        errors=error_counts,
        extra_info={
            "avg_positions_refined": avg_refined,
            "avg_scaffold_size": avg_scaffold,
            "refinement_rate": avg_refined / avg_scaffold if avg_scaffold > 0 else 0,
        },
    )


def print_results(results: List[EvalResult]):
    print("\n" + "=" * 80)
    print("BFCL EVALUATION RESULTS (AST Checker)")
    print("=" * 80)
    print("-" * 80)
    print(f"{'Model':<25} {'Accuracy':<12} {'Correct':<12} {'Latency (ms)':<15} {'Memory (MB)'}")
    print("-" * 80)
    for r in results:
        print(f"{r.model_name:<25} {r.accuracy*100:.1f}% {'':<5} "
              f"{r.correct}/{r.total} {'':<5} {r.mean_latency_ms:.0f} {'':<10} {r.peak_memory_mb:.0f}")
    print("-" * 80)
    print("\nError breakdown:")
    for r in results:
        if r.errors:
            print(f"  {r.model_name}:")
            for err, cnt in sorted(r.errors.items(), key=lambda x: -x[1]):
                print(f"    {err}: {cnt}")
        if r.extra_info:
            print(f"  Extra info:")
            for k, v in r.extra_info.items():
                print(f"    {k}: {v:.2f}" if isinstance(v, float) else f"    {k}: {v}")
    print("=" * 80)


def save_results(results: List[EvalResult], output_path: str):
    data = []
    for r in results:
        data.append({
            "model_name": r.model_name,
            "accuracy": r.accuracy,
            "total": r.total,
            "correct": r.correct,
            "mean_latency_ms": r.mean_latency_ms,
            "peak_memory_mb": r.peak_memory_mb,
            "errors": r.errors,
            "extra_info": r.extra_info,
        })
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Evaluate warm-start diffusion on BFCL")
    parser.add_argument(
        "--test-category",
        type=str,
        default="simple",
        help="BFCL test category",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of samples",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark/results/warmstart_eval.json",
    )
    parser.add_argument(
        "--show-steps",
        action="store_true",
        help="Show generation steps",
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    data = load_bfcl_data(test_category=args.test_category, limit=args.limit)
    print(f"Loaded {len(data)} BFCL examples ({args.test_category})")

    results = []
    result = evaluate_warmstart_model(data, device, show_steps=args.show_steps)
    results.append(result)

    print_results(results)
    save_results(results, args.output)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
