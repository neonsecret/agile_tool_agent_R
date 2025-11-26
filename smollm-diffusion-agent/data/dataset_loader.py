import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import json
import re
import random

from .schema import build_schema_template
from .utils import resolve_mask_token


class SmartScaffoldDataset(Dataset):
    def __init__(self, tokenizer, split="train", max_seq_len=1024, max_new_tokens=256, limit=None,
                 mask_token=None):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_new_tokens = max_new_tokens
        self.limit = limit
        self.padding_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.eos_token_id = tokenizer.eos_token_id

        if mask_token is None:
            raise ValueError(
                "mask_token must be provided. Use resolve_mask_token() from utils to get "
                "the mask token string from config, or pass it explicitly."
            )
        
        self.mask_token, self.mask_token_id = resolve_mask_token(tokenizer, mask_token)

        # Load dataset
        self.ds = load_dataset("interstellarninja/hermes_reasoning_tool_use", split=split)
        self.processed_examples = self._process_dataset()

    def _process_dataset(self):
        processed = []
        print("Processing dataset examples...")

        for i, example in enumerate(self.ds):
            if self.limit and i >= self.limit: break
            if i > 5000: break

            conversations = example.get("conversations", [])
            tools_schema_str = example.get("tools", "[]")

            try:
                tools_schema = json.loads(tools_schema_str)
            except:
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
                if msg['from'] == 'gpt' and '<tool_call>' not in msg['value'] and random.random() < 0.1:
                    # 10% sampling of chat turns
                    prompt, _ = self._build_prompt_context(msg_idx, conversations, msg['value'])
                    # Add router label 0 (Chat)
                    processed.append({
                        "prompt": prompt,
                        "template": None,  # No diffusion
                        "target_tokens_map": None,
                        "router_label": 0
                    })

        print(f"Processed {len(processed)} examples.")
        return processed

    def _build_prompt_context(self, msg_idx, conversations, current_text):
        # Context up to this message
        prompt = ""
        for prev_msg in conversations[:msg_idx]:
            role = prev_msg['from']
            if role == 'gpt': role = 'assistant'
            if role == 'human': role = 'user'
            prompt += f"<|im_start|>{role}\n{prev_msg['value']}<|im_end|>\n"

        # Current message starts
        # For tool call, we cut off before <tool_call>
        # For chat, we just prompt
        return prompt, current_text

    def _process_tool_call(self, msg, msg_idx, conversations, tools_schema, processed):
        full_text = msg['value']
        tool_call_matches = list(re.finditer(r'<tool_call>(.*?)</tool_call>', full_text, re.DOTALL))

        if not tool_call_matches:
            return

        match = tool_call_matches[0]
        tool_call_json_str = match.group(1).strip()

        try:
            tool_call_json = json.loads(tool_call_json_str)
        except:
            return

        tool_name = tool_call_json.get("name")
        tool_args = tool_call_json.get("arguments", {})

        tool_schema = next((t for t in tools_schema if isinstance(t, dict) and t.get('name') == tool_name), None)
        if not tool_schema:
            return

        # Context
        context_text = full_text[:match.start()]
        prompt_prefix, _ = self._build_prompt_context(msg_idx, conversations, full_text)

        prompt = prompt_prefix + f"<|im_start|>assistant\n{context_text}"
        prompt += "<|decision:use_tool|>\n" + f"<|tool_name:{tool_name}|>\n"

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
            if self.eos_token_id is not None:
                val_ids.append(self.eos_token_id)

            budget = max(len(val_ids), 32)
            fields.append((key, budget))
            target_tokens_map[key] = val_ids

        # Build Template
        template = build_schema_template(
            self.tokenizer,
            fields,
            self.mask_token,
            include_codeblock=False
        )

        processed.append({
            "prompt": prompt,
            "template": template,
            "target_tokens_map": target_tokens_map,
            "router_label": 1  # Tool
        })

    def __len__(self):
        return len(self.processed_examples)

    def __getitem__(self, idx):
        ex = self.processed_examples[idx]

        prompt_ids = self.tokenizer.encode(ex["prompt"], add_special_tokens=False)

        # If no template (Chat example), return minimal dict for Router Training only
        if ex["template"] is None:
            input_ids = torch.tensor(prompt_ids, dtype=torch.long)
            # Pad/Mask everything else
            return {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
                "scaffold_mask": torch.zeros_like(input_ids, dtype=torch.bool),  # No diffusion
                "labels": torch.full_like(input_ids, -100),
                "diffusion_steps": torch.tensor(0),  # Dummy
                "router_label": ex["router_label"]
            }

        template = ex["template"]
        target_map = ex["target_tokens_map"]

        full_input_ids = list(prompt_ids)
        full_input_ids.extend(template.tokens)

        scaffold_mask = [0] * len(prompt_ids)

        template_mask = [0] * len(template.tokens)
        labels = [-100] * len(full_input_ids)

        for segment in template.field_segments:
            tgt = target_map.get(segment.name, [])
            for i, pos in enumerate(segment.value_positions):
                template_mask[pos] = 1
                global_pos = len(prompt_ids) + pos
                if i < len(tgt):
                    labels[global_pos] = tgt[i]
                else:
                    labels[global_pos] = -100

        scaffold_mask.extend(template_mask)

        input_ids = torch.tensor(full_input_ids, dtype=torch.long)
        scaffold_mask_tensor = torch.tensor(scaffold_mask, dtype=torch.bool)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
            scaffold_mask_tensor = scaffold_mask_tensor[:self.max_seq_len]
            labels_tensor = labels_tensor[:self.max_seq_len]
            attention_mask = attention_mask[:self.max_seq_len]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "scaffold_mask": scaffold_mask_tensor,
            "labels": labels_tensor,
            "diffusion_steps": torch.randint(0, 4, (1,)).item(),
            "router_label": ex["router_label"]
        }
