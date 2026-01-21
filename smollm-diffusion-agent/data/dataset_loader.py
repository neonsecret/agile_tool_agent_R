import json
import hashlib
import pickle
import torch
import torch.distributed as dist
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from datasets import load_dataset
import yaml
from torch.utils.data import Dataset

from .schema import build_schema_template
from .smollm3_prompting import apply_smollm3_chat_template, encode_tool_call_wrapper
from .utils import resolve_mask_token, resolve_null_token
from .multi_dataset_loader import load_multi_dataset_from_config
from .dataset_processing import (
    parse_tools_schema,
    map_role,
    build_messages,
    extract_tool_call_from_message,
)

# Budget configuration - matches guide.md recommendations
# Minimum budget ensures consistency across fields (prevents too-small templates)
# Maximum budget prevents excessive memory usage
MIN_FIELD_BUDGET = 32
DEFAULT_MAX_BUDGET = 48


class SmartScaffoldDataset(Dataset):
    def __init__(
            self,
            tokenizer,
            split: str = "train",
            max_seq_len: int = 1024,
            max_new_tokens: int = 256,
            limit: Optional[int] = None,
            mask_token: Optional[str] = None,
            null_token: Optional[str] = None,
            chat_sampling_rate: float = 0.1,
            mask_budget: int = 48,
            system_message: str = "/no_think",
            max_history_messages: int = 12,
            data_config: Optional[Dict[str, Any]] = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_new_tokens = max_new_tokens
        self.limit = limit
        self.chat_sampling_rate = chat_sampling_rate
        self.mask_budget = mask_budget
        self.system_message = system_message
        self.max_history_messages = max_history_messages
        self.padding_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.eos_token_id = tokenizer.eos_token_id

        if mask_token is None:
            raise ValueError(
                "mask_token must be provided. Use resolve_mask_token() from utils to get "
                "the mask token string from config, or pass it explicitly."
            )

        self.mask_token, self.mask_token_id = resolve_mask_token(tokenizer, mask_token)

        # NULL token for self-adaptive masking (variable-length fields)
        self.null_token = None
        self.null_token_id = None
        if null_token is not None:
            self.null_token, self.null_token_id = resolve_null_token(tokenizer, null_token)

        self.config = data_config or self._load_default_config()
        self.dynamic_budget = self._get_dynamic_budget_config(self.config)
        self.ds = self._load_dataset(split, self.config)
        self.processed_examples = self._load_or_build_processed_examples()

    def _load_dataset(self, split: str, data_config: Dict[str, Any]):
        return load_multi_dataset_from_config(data_config)

    def _load_default_config(self) -> Dict[str, Any]:
        config_path = Path(__file__).resolve().parents[1] / "config.yaml"
        with config_path.open("r") as handle:
            return yaml.safe_load(handle)

    def _get_dynamic_budget_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        data_cfg = config.get("data", {})
        dynamic_cfg = data_cfg.get("dynamic_budget", {})
        enabled = dynamic_cfg.get("enabled", False)
        max_tokens = dynamic_cfg.get("max_tokens")
        if max_tokens is None:
            max_tokens = data_cfg.get("mask_budget")
        return {
            "enabled": enabled,
            "min_tokens": dynamic_cfg.get("min_tokens", 0),
            "extra_tokens": dynamic_cfg.get("extra_tokens", 0),
            "max_tokens": max_tokens,
        }

    def _process_dataset(self):
        processed = []
        print("Processing dataset examples...")
        failed = 0
        for i, example in enumerate(self.ds):
            if self.limit and i >= self.limit:
                break

            conversations = example.get("conversations", [])
            tools_schema = parse_tools_schema(example.get("tools", "[]"))
            if tools_schema is None:
                failed += 1
                continue

            for msg_idx, msg in enumerate(conversations):
                if msg['from'] == 'gpt' and '<tool_call>' in msg['value']:
                    self._process_tool_call(msg, msg_idx, conversations, tools_schema, processed)

        print(f"Processed {len(processed)} examples, {failed} failed.")
        return processed

    def _cache_config(self) -> Dict[str, Any]:
        data_cfg = self.config.get("data", {})
        cache_cfg = data_cfg.get("cache", {})
        return {
            "enabled": cache_cfg.get("enabled", False),
            "dir": cache_cfg.get("dir", "data_cache"),
        }

    def _cache_fingerprint(self) -> str:
        data_cfg = self.config.get("data", {})
        training_cfg = self.config.get("training", {})
        payload = {
            "datasets": data_cfg.get("datasets"),
            "dataset_name": data_cfg.get("dataset_name"),
            "limit": data_cfg.get("limit"),
            "mask_budget": self.mask_budget,
            "dynamic_budget": self.dynamic_budget,
            "mask_token": self.mask_token,
            "null_token": self.null_token,
            "system_message": self.system_message,
            "max_history_messages": self.max_history_messages,
            "max_new_tokens": self.max_new_tokens,
            "max_seq_len": training_cfg.get("max_seq_len"),
            "tokenizer": getattr(self.tokenizer, "name_or_path", None),
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]

    def _cache_path(self) -> Optional[Path]:
        cache_cfg = self._cache_config()
        if not cache_cfg["enabled"]:
            return None
        root = Path(__file__).resolve().parents[1]
        cache_dir = root / cache_cfg["dir"]
        cache_dir.mkdir(parents=True, exist_ok=True)
        fingerprint = self._cache_fingerprint()
        return cache_dir / f"processed_{fingerprint}.pkl"

    def _load_cache(self, path: Path) -> List[Dict[str, Any]]:
        print(f"Loading processed dataset cache: {path}")
        with path.open("rb") as handle:
            return pickle.load(handle)

    def _save_cache(self, path: Path, processed: List[Dict[str, Any]]) -> None:
        tmp_path = path.with_suffix(".tmp")
        with tmp_path.open("wb") as handle:
            pickle.dump(processed, handle, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(path)
        print(f"Saved processed dataset cache: {path}")

    @staticmethod
    def _is_distributed() -> bool:
        return dist.is_available() and dist.is_initialized()

    def _load_or_build_processed_examples(self) -> List[Dict[str, Any]]:
        cache_path = self._cache_path()
        if cache_path is None:
            return self._process_dataset()
        if cache_path.exists():
            return self._load_cache(cache_path)

        if self._is_distributed() and dist.get_rank() != 0:
            dist.barrier()
            return self._load_cache(cache_path)

        processed = self._process_dataset()
        self._save_cache(cache_path, processed)

        if self._is_distributed():
            dist.barrier()

        return processed

    def _process_tool_call(self, msg, msg_idx, conversations, tools_schema, processed):
        full_text = msg['value']
        tool_call_json = extract_tool_call_from_message(full_text)

        if tool_call_json is None:
            return

        tool_name = tool_call_json.get("name")
        tool_args = tool_call_json.get("arguments", {})

        tool_schema = next((t for t in tools_schema if isinstance(t, dict) and t.get('name') == tool_name), None)
        if not tool_schema:
            return

        messages = build_messages(
            conversations, msg_idx, self.system_message, self.max_history_messages
        )

        # Use Robust Schema Builder
        fields = []
        target_tokens_map = {}

        params = tool_schema.get("parameters", {})
        if "properties" in params:
            props = params["properties"]
        else:
            props = params

        for key in props.keys():
            val = tool_args.get(key, "")
            if not isinstance(val, str):
                val = json.dumps(val)

            val_ids = self.tokenizer.encode(val, add_special_tokens=False)
            # EOS tokens are structural - Python/template handles them, not the model
            # This aligns with schema scaffolding: Python does syntax, LLM does semantics

            budget = self._compute_budget(len(val_ids))
            fields.append((key, budget))
            target_tokens_map[key] = val_ids

        # Skip if no fields (empty schema)
        if not fields:
            return

        # Build Template with optional NULL token support
        template = build_schema_template(
            self.tokenizer,
            fields,
            self.mask_token,
            null_token=self.null_token,
            include_codeblock=False
        )

        # Validate template was created successfully
        if template is None or len(template.field_segments) == 0:
            return

        processed.append(
            {
                "messages": messages,
                "tools_schema": tools_schema,
                "tool_name": tool_name,
                "template": template,
                "target_tokens_map": target_tokens_map
            }
        )

    def __len__(self):
        return len(self.processed_examples)

    def _compute_budget(self, value_length: int) -> int:
        if not self.dynamic_budget["enabled"]:
            return min(max(value_length, MIN_FIELD_BUDGET), self.mask_budget)

        budget = value_length + self.dynamic_budget["extra_tokens"]
        min_tokens = self.dynamic_budget["min_tokens"]
        if min_tokens is not None:
            budget = max(budget, min_tokens)
        max_tokens = self.dynamic_budget["max_tokens"]
        if max_tokens is not None:
            budget = min(budget, max_tokens)
        return max(budget, 0)

    def __getitem__(self, idx):
        ex = self.processed_examples[idx]
        tools_schema = ex.get("tools_schema")
        messages = ex.get("messages")
        if messages is None:
            raise ValueError("Dataset example missing `messages` field.")

        prompt_ids = apply_smollm3_chat_template(
            self.tokenizer,
            messages=messages,
            tools=tools_schema,
            add_generation_prompt=True,
        )

        # If no template (Chat example), return minimal dict (no diffusion training)
        if ex["template"] is None:
            input_ids = torch.tensor(prompt_ids, dtype=torch.long)

            # Truncate if too long
            if len(input_ids) > self.max_seq_len:
                input_ids = input_ids[-self.max_seq_len:]

            attention_mask = torch.ones_like(input_ids)
            scaffold_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            labels = torch.full_like(input_ids, -100)

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "scaffold_mask": scaffold_mask,
                "labels": labels,
                "is_tool": False,
            }

        tool_name = ex.get("tool_name")
        if tool_name is None:
            raise ValueError("Tool example missing `tool_name`.")

        template = ex["template"]
        target_map = ex["target_tokens_map"]

        tool_call_parts = encode_tool_call_wrapper(self.tokenizer, tool_name)
        prefix_ids = tool_call_parts.prefix_ids
        suffix_ids = tool_call_parts.suffix_ids

        tail_len = len(prefix_ids) + len(template.tokens) + len(suffix_ids)
        if len(prompt_ids) + tail_len > self.max_seq_len:
            keep = self.max_seq_len - tail_len
            if keep <= 0:
                raise ValueError(
                    "max_seq_len is too small to fit tool_call wrapper + scaffold. "
                    f"Need at least {tail_len} tokens, got max_seq_len={self.max_seq_len}."
                )
            prompt_ids = prompt_ids[-keep:]

        full_input_ids = list(prompt_ids) + list(prefix_ids) + list(template.tokens) + list(suffix_ids)

        scaffold_mask = [0] * len(full_input_ids)
        labels = [-100] * len(full_input_ids)

        prompt_len = len(prompt_ids)
        prefix_len = len(prefix_ids)

        for segment in template.field_segments:
            tgt = target_map.get(segment.name, [])
            for i, pos in enumerate(segment.value_positions):
                global_pos = prompt_len + prefix_len + pos
                scaffold_mask[global_pos] = 1
                if i < len(tgt):
                    labels[global_pos] = tgt[i]
                else:
                    # Use NULL token for unused slots (enables automatic budget handling)
                    # Model learns to predict NULL for positions beyond actual content
                    if self.null_token_id is not None:
                        labels[global_pos] = self.null_token_id
                    else:
                        labels[global_pos] = -100  # Fallback to ignore
        input_ids = torch.tensor(full_input_ids, dtype=torch.long)
        scaffold_mask_tensor = torch.tensor(scaffold_mask, dtype=torch.bool)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "scaffold_mask": scaffold_mask_tensor,
            "labels": labels_tensor,
            "is_tool": True,
        }


if __name__ == '__main__':
    raise SystemExit("Run dataset tests via smollm-diffusion-agent/test_dataset.py")
