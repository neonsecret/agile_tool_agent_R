import json
import random
import re
from typing import Any, Dict, List, Optional, Sequence

import torch
from datasets import load_dataset
from torch.utils.data import Dataset

from .schema import build_schema_template
from .smollm3_prompting import apply_smollm3_chat_template, encode_tool_call_wrapper
from .utils import resolve_mask_token, resolve_null_token


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

        # Load dataset (Hermes reasoning + tool use)
        self.ds = load_dataset("interstellarninja/hermes_reasoning_tool_use", split=split)
        self.processed_examples = self._process_dataset()

    def _map_role(self, role: str) -> str:
        if role == "gpt":
            return "assistant"
        if role == "human":
            return "user"
        return role

    def _build_messages(
        self,
        conversations: Sequence[Dict[str, Any]],
        msg_idx: int,
    ) -> List[Dict[str, str]]:
        """
        Build a SmolLM3-compatible `messages` list for tokenizer.apply_chat_template.

        The returned messages ALWAYS start with a system message, and end with the
        last message before `msg_idx` (typically a user message). We truncate older
        history so the scaffold fits under `max_seq_len`.
        """
        history: List[Dict[str, str]] = [{"role": "system", "content": self.system_message}]

        upto = conversations[:msg_idx]
        mapped = [
            {"role": self._map_role(m.get("from", "")), "content": m.get("value", "")}
            for m in upto
            if m.get("from") is not None and m.get("value") is not None
        ]
        if self.max_history_messages > 0:
            mapped = mapped[-self.max_history_messages:]
        history.extend(mapped)
        return history

    def _process_dataset(self):
        processed = []
        print("Processing dataset examples...")
        failed = 0
        for i, example in enumerate(self.ds):
            if self.limit and i >= self.limit:
                break

            conversations = example.get("conversations", [])
            tools_schema_str = example.get("tools", "[]")

            try:
                tools_schema = json.loads(tools_schema_str)
            except json.JSONDecodeError:
                failed += 1
                continue

            # 1. Positive Examples (Tool Calls)
            for msg_idx, msg in enumerate(conversations):
                if msg['from'] == 'gpt' and '<tool_call>' in msg['value']:
                    self._process_tool_call(msg, msg_idx, conversations, tools_schema, processed)

            # 2. Synthetic Negative Examples (Direct Answer)
            # Randomly select non-tool-call assistant turns
            # MediaTek augmentation strategy: Remove relevant tools -> force direct answer
            # Here we simplify: Just pick regular chat turns and label them as "chat" (router_label=0)
            # Since TRAIN_ROUTER is disabled, these will just serve as "no diffusion" examples if we want.
            # But diffusion head needs mask. So "no diffusion" examples are skipped for diffusion loss.
            # But useful for router training later.

            for msg_idx, msg in enumerate(conversations):
                if (
                    msg.get("from") == "gpt"
                    and "<tool_call>" not in msg.get("value", "")
                    and random.random() < self.chat_sampling_rate
                ):
                    # Router-only example: conversation up to this assistant turn.
                    messages = self._build_messages(conversations, msg_idx)
                    processed.append(
                        {
                            "messages": messages,
                            "tools_schema": tools_schema,
                            "template": None,
                            "target_tokens_map": None,
                            "tool_name": None,
                            "router_label": 0,
                        }
                    )

        print(f"Processed {len(processed)} examples, {failed} failed.")
        return processed

    def _process_tool_call(self, msg, msg_idx, conversations, tools_schema, processed):
        full_text = msg['value']
        tool_call_matches = list(re.finditer(r'<tool_call>(.*?)</tool_call>', full_text, re.DOTALL))

        if not tool_call_matches:
            return

        match = tool_call_matches[0]
        tool_call_json_str = match.group(1).strip()

        try:
            tool_call_json = json.loads(tool_call_json_str)
        except json.JSONDecodeError:
            return

        tool_name = tool_call_json.get("name")
        tool_args = tool_call_json.get("arguments", {})

        tool_schema = next((t for t in tools_schema if isinstance(t, dict) and t.get('name') == tool_name), None)
        if not tool_schema:
            return

        messages = self._build_messages(conversations, msg_idx)

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

            # Automatic budget: min for consistency, max to prevent excessive memory
            budget = min(max(len(val_ids), MIN_FIELD_BUDGET), self.mask_budget)
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
                "target_tokens_map": target_tokens_map,
                "router_label": 1,
            }
        )

    def __len__(self):
        return len(self.processed_examples)

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

        # If no template (Chat example), return minimal dict for Router Training only
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
                "router_label": ex["router_label"]
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
            "router_label": ex["router_label"]
        }


if __name__ == '__main__':
    raise SystemExit("Run dataset tests via smollm-diffusion-agent/test_dataset.py")
