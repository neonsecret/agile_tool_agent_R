"""
Berkeley Function Call Leaderboard (BFCL) Evaluation Script.
Uses the smollm-diffusion-agent inference stack.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
SMOLLM_DIR = ROOT_DIR / "smollm-diffusion-agent"
sys.path.insert(0, str(SMOLLM_DIR))

from data.budget_utils import DEFAULT_MAX_BUDGET, MIN_FIELD_BUDGET
from data.config_utils import (
    get_inference_kwargs,
    get_model_kwargs,
    validate_and_adjust_config,
)
from data.device_utils import get_device
from data.utils import resolve_mask_token, resolve_null_token
from inference import FunctionCallGenerator, GenerationConfig
from model.hybrid_model import HybridSmolLM


def _load_config() -> Dict[str, Any]:
    config_path = SMOLLM_DIR / "config.yaml"
    with config_path.open("r") as handle:
        return yaml.safe_load(handle)


def _resolve_checkpoint_path(config: Dict[str, Any]) -> Path:
    training_cfg = config.get("training", {})
    checkpoint_path = training_cfg.get(
        "checkpoint_path",
        "checkpoints/best_model/model.pt",
    )
    path = Path(checkpoint_path)
    if not path.is_absolute():
        path = SMOLLM_DIR / checkpoint_path
    return path


def _build_budget_config(config: Dict[str, Any]) -> Dict[str, int]:
    data_cfg = config.get("data", {})
    dynamic_budget_cfg = data_cfg.get("dynamic_budget", {})
    max_tokens = dynamic_budget_cfg.get(
        "max_tokens",
        data_cfg.get("mask_budget", DEFAULT_MAX_BUDGET),
    )
    min_tokens = dynamic_budget_cfg.get("min_tokens", MIN_FIELD_BUDGET)
    extra_tokens = dynamic_budget_cfg.get("extra_tokens", 0)
    return {
        "min_tokens": min_tokens,
        "max_tokens": max_tokens,
        "extra_tokens": extra_tokens,
    }


def _build_expansion_config(config: Dict[str, Any]) -> Dict[str, Any]:
    infer_cfg = config.get("inference", {})
    expansion_cfg = infer_cfg.get("expansion", {})
    return {
        "enabled": expansion_cfg.get("enabled", False),
        "max_rounds": expansion_cfg.get("max_rounds", 0),
        "expand_tokens": expansion_cfg.get("expand_tokens", 4),
        "tail_window": expansion_cfg.get("tail_window", 4),
        "tail_null_threshold": expansion_cfg.get("tail_null_threshold", 0.5),
    }


def _build_generation_config(config: Dict[str, Any]) -> GenerationConfig:
    infer_cfg = config.get("inference", {})
    remask_cfg = infer_cfg.get("remasking", {})
    return GenerationConfig(
        steps=infer_cfg.get("steps", 4),
        temperature=infer_cfg.get("temperature", 0.0),
        cfg_scale=infer_cfg.get("cfg_scale", 0.0),
        use_cuda_graph=infer_cfg.get("use_cuda_graph", False),
        enable_remasking=remask_cfg.get("enabled", True),
        remask_ratio=remask_cfg.get("remask_ratio", 0.2),
        min_lock_confidence=remask_cfg.get("min_lock_confidence", 0.7),
        reencode_hidden_states_every=infer_cfg.get("reencode_hidden_states_every", 0),
    )


def _build_tool_registry(functions: List[Dict[str, Any]]) -> Dict[str, Any]:
    tool_registry = {}
    for func in functions:
        name = func.get("name")
        if not name:
            continue
        tool_registry[name] = func
    return tool_registry


def _parse_json_field(field: Any) -> Any:
    if isinstance(field, str):
        return json.loads(field)
    return field


def _parse_ground_truth(raw_ground_truth: Any) -> Dict[str, Any]:
    if isinstance(raw_ground_truth, dict) and "name" in raw_ground_truth:
        return raw_ground_truth

    if not raw_ground_truth:
        return {"name": "", "arguments": {}}

    if isinstance(raw_ground_truth, list):
        first_call = raw_ground_truth[0] if raw_ground_truth else {}
    else:
        first_call = raw_ground_truth

    if isinstance(first_call, dict) and "name" in first_call:
        return first_call

    if isinstance(first_call, dict):
        func_name = list(first_call.keys())[0]
        params = first_call[func_name]
        arguments = {}
        if isinstance(params, dict):
            for param_name, values in params.items():
                if isinstance(values, list) and values:
                    arguments[param_name] = values[0]
                else:
                    arguments[param_name] = values
        return {"name": func_name, "arguments": arguments}

    return {"name": "", "arguments": {}}


def _values_equal(expected: Any, predicted: Any) -> bool:
    if expected is predicted:
        return True
    if expected is None or predicted is None:
        return expected is None and predicted is None
    if isinstance(expected, bool) or isinstance(predicted, bool):
        return type(expected) is type(predicted) and expected == predicted
    if isinstance(expected, (int, float)) and isinstance(predicted, (int, float)):
        return float(expected) == float(predicted)
    if isinstance(expected, str) and isinstance(predicted, str):
        return expected == predicted
    if isinstance(expected, list) and isinstance(predicted, list):
        if len(expected) != len(predicted):
            return False
        return all(_values_equal(e, p) for e, p in zip(expected, predicted))
    if isinstance(expected, dict) and isinstance(predicted, dict):
        if set(expected.keys()) != set(predicted.keys()):
            return False
        return all(_values_equal(expected[k], predicted[k]) for k in expected.keys())
    return False


def _arguments_equal(expected: Dict[str, Any], predicted: Dict[str, Any]) -> bool:
    if set(expected.keys()) != set(predicted.keys()):
        return False
    for key, value in expected.items():
        if key not in predicted:
            return False
        if not _values_equal(value, predicted[key]):
            return False
    return True


def _build_model(
    config: Dict[str, Any],
    tokenizer,
    device: torch.device,
) -> HybridSmolLM:
    model_kwargs = get_model_kwargs(config, device)
    model_kwargs["vocab_size"] = len(tokenizer)
    model = HybridSmolLM(**model_kwargs)

    checkpoint_path = _resolve_checkpoint_path(config)
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        checkpoint_state = checkpoint.get("model_state_dict", {})
        checkpoint_state = {k: v.to(device) for k, v in checkpoint_state.items()}
        if hasattr(model, "load_trainable_state_dict"):
            model.load_trainable_state_dict(checkpoint_state, strict=False)
        else:
            model_state = model.state_dict()
            filtered_state = {
                k: v
                for k, v in checkpoint_state.items()
                if k in model_state and k.startswith("diffusion_head.")
            }
            if filtered_state:
                model.load_state_dict(filtered_state, strict=False)
    else:
        print(f"No checkpoint found at {checkpoint_path}, using untrained model")

    mask_token_str, mask_token_id = resolve_mask_token(
        tokenizer, config.get("data", {}).get("mask_token", None)
    )
    model.diffusion_head.set_mask_token_id(mask_token_id)

    null_token_str, null_token_id = resolve_null_token(
        tokenizer, config.get("data", {}).get("null_token", None)
    )
    if null_token_id is not None:
        model.diffusion_head.set_null_token_id(null_token_id)
    del mask_token_str, null_token_str

    model.eval()
    return model


def _build_generator(
    config: Dict[str, Any],
    tokenizer,
    device: torch.device,
    model: HybridSmolLM,
) -> Tuple[FunctionCallGenerator, GenerationConfig]:
    infer_kwargs = get_inference_kwargs(config, device)
    budget_config = _build_budget_config(config)
    expansion_config = _build_expansion_config(config)
    generator = FunctionCallGenerator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        use_torch_compile=infer_kwargs.get("use_torch_compile", False),
        use_cuda_graph=infer_kwargs.get("use_cuda_graph", False),
        max_seq_len=infer_kwargs.get("max_seq_len", 2048),
        budget_config=budget_config,
        expansion_config=expansion_config,
    )
    generation_config = _build_generation_config(config)
    return generator, generation_config


class BFCLEvaluator:
    """Evaluator for Berkeley Function Call Leaderboard."""

    def __init__(self, generator: FunctionCallGenerator, gen_config: GenerationConfig):
        self.generator = generator
        self.gen_config = gen_config

    def evaluate_single(
        self,
        query: str,
        functions: List[Dict[str, Any]],
        ground_truth: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], bool]:
        tool_registry = _build_tool_registry(functions)
        if not tool_registry:
            return {}, False

        result = self.generator.generate_unified(
            prompt=query,
            tool_registry=tool_registry,
            config=self.gen_config,
        )
        prediction = result.get("tool_call_parsed") or {}

        pred_name = prediction.get("name", "").strip()
        pred_args = prediction.get("arguments", {})
        if not isinstance(pred_args, dict):
            return prediction, False

        gt_name = ground_truth.get("name", "").strip()
        gt_args = ground_truth.get("arguments", {})
        if not isinstance(gt_args, dict):
            return prediction, False

        if pred_name != gt_name:
            return prediction, False

        is_correct = _arguments_equal(gt_args, pred_args)
        return prediction, is_correct

    def evaluate_dataset(
        self,
        test_category: str = "simple",
        output_path: Optional[str] = None,
        limit: Optional[int] = None,
        data_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        if data_path:
            with open(data_path, "r") as handle:
                dataset = json.load(handle)
        else:
            ds = load_dataset("llamastack/bfcl_v3")
            dataset = [
                ex for ex in ds["train"] if ex.get("test_category") == test_category
            ]

        if limit:
            dataset = dataset[:limit]

        results = []
        correct = 0
        total = 0
        bar = tqdm(dataset)
        for idx, example in enumerate(bar):
            try:
                raw_functions = example.get("functions")
                functions = _parse_json_field(raw_functions) if raw_functions else []
                raw_ground_truth = example.get("ground_truth")
                ground_truth = _parse_ground_truth(_parse_json_field(raw_ground_truth))

                query = example.get("question", "")
                if not query:
                    turns = _parse_json_field(example.get("turns", []))
                    for turn in turns[0] if turns else []:
                        if turn.get("role") == "user":
                            query = turn.get("content", "")
                            break

                prediction, is_correct = self.evaluate_single(
                    query, functions, ground_truth
                )

                results.append(
                    {
                        "id": example.get("id", idx),
                        "query": query,
                        "prediction": prediction,
                        "ground_truth": ground_truth,
                        "correct": is_correct,
                    }
                )
                total += 1
                if is_correct:
                    correct += 1
                bar.set_description_str(f"correct: {correct}/{total}")
            except Exception as exc:
                results.append(
                    {
                        "id": example.get("id", idx),
                        "error": str(exc),
                    }
                )

        accuracy = correct / total if total > 0 else 0.0
        metrics = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "error_rate": 1 - accuracy if total > 0 else 1.0,
        }

        if output_path:
            output_data = {"metrics": metrics, "results": results}
            with open(output_path, "w") as handle:
                json.dump(output_data, handle, indent=2)

        return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SmolLM Diffusion Agent on BFCL."
    )
    parser.add_argument(
        "--test-category",
        type=str,
        default="simple",
        help="BFCL test category to evaluate (default: simple)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Optional path to local BFCL-style JSON dataset",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="bfcl_results.json",
        help="Path to save evaluation results",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples to evaluate (for testing)",
    )

    args = parser.parse_args()

    config = _load_config()
    device = get_device()
    config = validate_and_adjust_config(config, device)

    tokenizer = AutoTokenizer.from_pretrained(
        config.get("model", {}).get("base_model_id", "HuggingFaceTB/SmolLM3-3B")
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = _build_model(config, tokenizer, device)
    generator, gen_config = _build_generator(config, tokenizer, device, model)

    evaluator = BFCLEvaluator(generator=generator, gen_config=gen_config)
    metrics = evaluator.evaluate_dataset(
        test_category=args.test_category,
        output_path=args.output_path,
        limit=args.limit,
        data_path=args.data_path,
    )

    print("\n=== Evaluation Results ===")
    print(f"Total examples: {metrics['total']}")
    print(f"Correct: {metrics['correct']}")
    print(f"Accuracy: {metrics['accuracy']:.2%}")


if __name__ == "__main__":
    main()
