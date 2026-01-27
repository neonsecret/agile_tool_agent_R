"""
Warm-start diffusion inference pipeline.

Uses base AR model's output as initialization for diffusion refinement,
addressing the "blind diffusion" problem where the diffusion head operates
on MASK tokens with no semantic context.

Key insight: Instead of starting from all MASKs, we start from the base
AR model's draft (94% accurate) and let diffusion refine it.
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from data.schema import SchemaTemplate, build_schema_template
from data.smollm3_prompting import (
    apply_smollm3_chat_template,
    parse_first_tool_call,
    encode_tool_call_wrapper,
)
from data.budget_utils import build_fields_from_schema, MIN_FIELD_BUDGET
from data.dataset_processing import get_tool_schema_properties

from inference import FunctionCallGenerator, GenerationConfig, GenerationOutput


@dataclass
class WarmStartConfig:
    diffusion_steps: int = 4
    temperature: float = 0.0
    refinement_ratio: float = 0.3
    min_confidence_to_keep: float = 0.8
    use_cuda_graph: bool = True
    show_steps: bool = False
    reencode_hidden_states_every: int = 1


def generate_ar_draft(
    model,
    tokenizer: AutoTokenizer,
    prompt: str,
    tool_registry: Dict[str, Any],
    device: torch.device,
    system_message: str = "/no_think",
) -> Optional[Dict[str, Any]]:
    """Generate a complete tool call using base AR model."""
    tools_list = list(tool_registry.values())
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]
    
    prompt_ids = apply_smollm3_chat_template(
        tokenizer, messages, tools=tools_list, add_generation_prompt=True
    )
    prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    attention_mask = torch.ones_like(prompt_tensor)
    
    with torch.no_grad():
        output_ids = model.base_llm.generate(
            prompt_tensor,
            attention_mask=attention_mask,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    gen_ids = output_ids[0, len(prompt_ids):]
    generated = tokenizer.decode(gen_ids, skip_special_tokens=False)
    
    del attention_mask
    return parse_first_tool_call(generated)


def build_warmstart_sequence(
    tokenizer: AutoTokenizer,
    prompt: str,
    tool_name: str,
    ar_args: Dict[str, Any],
    tool_schema: Dict[str, Any],
    template: SchemaTemplate,
    tools: List[Dict[str, Any]],
    device: torch.device,
    max_seq_len: int = 2048,
) -> Tuple[torch.Tensor, int, int, torch.Tensor]:
    """Build sequence with AR draft values instead of MASKs."""
    tool_call_parts = encode_tool_call_wrapper(tokenizer, tool_name)
    prefix_ids = tool_call_parts.prefix_ids
    suffix_ids = tool_call_parts.suffix_ids
    
    messages = [
        {"role": "system", "content": "/no_think"},
        {"role": "user", "content": prompt},
    ]
    prompt_ids_list = apply_smollm3_chat_template(
        tokenizer, messages, tools=tools, add_generation_prompt=True
    )
    
    props = get_tool_schema_properties(tool_schema)
    field_values = {}
    for segment in template.field_segments:
        val = ar_args.get(segment.name, None)
        val_json = json.dumps(val)
        val_ids = tokenizer.encode(val_json, add_special_tokens=False)
        field_values[segment.name] = val_ids
    
    filled_tokens = list(template.tokens)
    for segment in template.field_segments:
        val_ids = field_values.get(segment.name, [])
        positions = segment.value_positions
        for i, pos in enumerate(positions):
            if i < len(val_ids):
                filled_tokens[pos] = val_ids[i]
            else:
                filled_tokens[pos] = template.null_token_id or template.mask_token_id
    
    tail_len = len(prefix_ids) + len(filled_tokens) + len(suffix_ids)
    if len(prompt_ids_list) + tail_len > max_seq_len:
        keep = max_seq_len - tail_len
        if keep <= 0:
            raise ValueError("Prompt too large")
        prompt_ids_list = prompt_ids_list[-keep:]
    
    prompt_ids = torch.tensor(prompt_ids_list, dtype=torch.long, device=device).unsqueeze(0)
    prefix = torch.tensor(prefix_ids, dtype=torch.long, device=device).unsqueeze(0)
    suffix = torch.tensor(suffix_ids, dtype=torch.long, device=device).unsqueeze(0)
    filled_tensor = torch.tensor(filled_tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    sequence = torch.cat([prompt_ids, prefix, filled_tensor, suffix], dim=1)
    prompt_length = prompt_ids.shape[1]
    prefix_length = prefix.shape[1]
    
    scaffold_mask = torch.zeros_like(sequence, dtype=torch.bool, device=device)
    for segment in template.field_segments:
        for pos in segment.value_positions:
            scaffold_mask[:, prompt_length + prefix_length + pos] = True
    
    return sequence, prompt_length, prefix_length, scaffold_mask


def compute_token_confidence(
    model,
    sequence: torch.Tensor,
    scaffold_mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Compute confidence scores for each scaffold position."""
    attention_mask = torch.ones_like(sequence)
    
    with torch.no_grad():
        outputs = model.get_hidden_states(
            input_ids=sequence,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = outputs.hidden_states[-1]
    
    t = torch.zeros(sequence.shape[0], device=device)
    logits = model.diffusion_head.predict(
        hidden_states, sequence, t, scaffold_mask=scaffold_mask
    )
    
    log_probs = F.log_softmax(logits, dim=-1)
    current_tokens = sequence.clone()
    token_log_probs = log_probs.gather(-1, current_tokens.unsqueeze(-1)).squeeze(-1)
    
    confidence = torch.zeros_like(sequence, dtype=torch.float)
    confidence[scaffold_mask] = torch.exp(token_log_probs[scaffold_mask])
    
    del attention_mask, outputs, hidden_states, logits, log_probs
    return confidence


def select_positions_to_refine(
    confidence: torch.Tensor,
    scaffold_mask: torch.Tensor,
    refinement_ratio: float,
    min_confidence: float,
) -> torch.Tensor:
    """Select low-confidence positions for refinement."""
    scaffold_indices = torch.nonzero(scaffold_mask[0], as_tuple=False).squeeze(-1)
    if scaffold_indices.numel() == 0:
        return torch.tensor([], dtype=torch.long, device=confidence.device)
    
    scaffold_conf = confidence[0, scaffold_indices]
    low_conf_mask = scaffold_conf < min_confidence
    
    if not low_conf_mask.any():
        return torch.tensor([], dtype=torch.long, device=confidence.device)
    
    num_to_refine = max(1, int(scaffold_indices.numel() * refinement_ratio))
    _, worst_indices = torch.topk(scaffold_conf, num_to_refine, largest=False)
    
    return scaffold_indices[worst_indices]


def refine_with_diffusion(
    model,
    tokenizer: AutoTokenizer,
    sequence: torch.Tensor,
    scaffold_mask: torch.Tensor,
    positions_to_refine: torch.Tensor,
    config: WarmStartConfig,
    template: SchemaTemplate,
    device: torch.device,
) -> torch.Tensor:
    """Refine selected positions using diffusion denoising."""
    if positions_to_refine.numel() == 0:
        return sequence
    
    refined_sequence = sequence.clone()
    refined_sequence[0, positions_to_refine] = template.mask_token_id
    
    refined_mask = torch.zeros_like(sequence, dtype=torch.bool, device=device)
    refined_mask[0, positions_to_refine] = True
    
    attention_mask = torch.ones_like(sequence)
    initial_mask_count = positions_to_refine.numel()
    
    for step in range(config.diffusion_steps):
        mask_positions = (refined_sequence == template.mask_token_id) & refined_mask
        mask_indices = torch.nonzero(mask_positions[0], as_tuple=False).squeeze(-1)
        
        if mask_indices.numel() == 0:
            break
        
        if config.reencode_hidden_states_every > 0 and step % config.reencode_hidden_states_every == 0:
            with torch.no_grad():
                outputs = model.get_hidden_states(
                    input_ids=refined_sequence,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
                hidden_states = outputs.hidden_states[-1]
        
        remaining = mask_indices.numel()
        t_val = float(remaining) / float(initial_mask_count)
        t = torch.full((1,), t_val, device=device, dtype=torch.float)
        
        with torch.no_grad():
            logits = model.diffusion_head.predict(
                hidden_states, refined_sequence, t, scaffold_mask=scaffold_mask
            )
        
        logits[:, :, template.mask_token_id] = -float('inf')
        
        if config.temperature > 0:
            logits = logits / config.temperature
        
        predictions = torch.argmax(logits, dim=-1)
        
        log_probs = F.log_softmax(logits[0, mask_indices], dim=-1)
        mask_conf = log_probs.gather(-1, predictions[0, mask_indices].unsqueeze(-1)).squeeze(-1)
        
        remaining_steps = config.diffusion_steps - step - 1
        if remaining_steps <= 0:
            k = remaining
        else:
            k = max(1, remaining // (remaining_steps + 1))
        
        topk = torch.topk(mask_conf, min(k, remaining))
        selected = mask_indices[topk.indices]
        refined_sequence[0, selected] = predictions[0, selected]
        
        if config.show_steps:
            revealed_tokens = [tokenizer.decode([predictions[0, s].item()]) for s in selected[:5]]
            print(f"  Refinement step {step + 1}: revealed {len(selected)} tokens: {revealed_tokens}")
        
        del logits, log_probs, mask_conf
    
    del attention_mask
    return refined_sequence


class WarmStartGenerator:
    """Generator that uses AR draft as warm-start for diffusion."""
    
    def __init__(
        self,
        model,
        tokenizer: AutoTokenizer,
        device: torch.device,
        max_seq_len: int = 2048,
        budget_config: Optional[Dict[str, int]] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_seq_len = max_seq_len
        self.budget_config = budget_config or {
            "min_tokens": 0,
            "max_tokens": 32,
            "extra_tokens": 2,
        }
        
        self.model.eval()
    
    def generate(
        self,
        prompt: str,
        tool_registry: Dict[str, Any],
        config: Optional[WarmStartConfig] = None,
    ) -> Dict[str, Any]:
        """Generate with warm-start from AR draft."""
        cfg = config or WarmStartConfig()
        
        if not tool_registry:
            return {"mode": "chat", "error": "No tools provided"}
        
        ar_draft = generate_ar_draft(
            self.model,
            self.tokenizer,
            prompt,
            tool_registry,
            self.device,
        )
        
        if ar_draft is None:
            return {"mode": "chat", "error": "AR draft failed"}
        
        tool_name = ar_draft.get("name")
        ar_args = ar_draft.get("arguments", {})
        
        if not tool_name or tool_name not in tool_registry:
            return {"mode": "chat", "error": f"Unknown tool: {tool_name}"}
        
        tool_schema = tool_registry[tool_name]
        
        fields = self._build_fields_for_tool(tool_schema, ar_args)
        
        mask_token_str = self.tokenizer.convert_ids_to_tokens(
            [self.model.diffusion_head.mask_token_id]
        )[0]
        null_token_id = self.model.diffusion_head.null_token_id
        null_token_str = (
            self.tokenizer.convert_ids_to_tokens([null_token_id])[0]
            if null_token_id is not None
            else None
        )
        
        template = build_schema_template(
            tokenizer=self.tokenizer,
            fields=fields,
            mask_token=mask_token_str,
            null_token=null_token_str,
            include_codeblock=False,
        )
        
        sequence, prompt_length, prefix_length, scaffold_mask = build_warmstart_sequence(
            self.tokenizer,
            prompt,
            tool_name,
            ar_args,
            tool_schema,
            template,
            list(tool_registry.values()),
            self.device,
            self.max_seq_len,
        )
        
        confidence = compute_token_confidence(
            self.model, sequence, scaffold_mask, self.device
        )
        
        positions_to_refine = select_positions_to_refine(
            confidence,
            scaffold_mask,
            cfg.refinement_ratio,
            cfg.min_confidence_to_keep,
        )
        
        if cfg.show_steps:
            total_scaffold = scaffold_mask.sum().item()
            print(f"Warm-start: {positions_to_refine.numel()}/{total_scaffold} positions to refine")
        
        if positions_to_refine.numel() > 0:
            refined_sequence = refine_with_diffusion(
                self.model,
                self.tokenizer,
                sequence,
                scaffold_mask,
                positions_to_refine,
                cfg,
                template,
                self.device,
            )
        else:
            refined_sequence = sequence
        
        response_tokens = refined_sequence[0, prompt_length:].cpu()
        if null_token_id is not None:
            response_tokens = response_tokens[response_tokens != null_token_id]
        
        text = self.tokenizer.decode(
            response_tokens,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        
        parsed = parse_first_tool_call(text)
        
        return {
            "mode": "tool",
            "tool_name": tool_name,
            "tool_call": text,
            "tool_call_parsed": parsed,
            "ar_draft": ar_draft,
            "positions_refined": int(positions_to_refine.numel()),
            "total_scaffold": int(scaffold_mask.sum().item()),
        }
    
    def _build_fields_for_tool(
        self,
        tool_schema: Dict[str, Any],
        tool_args: Dict[str, Any],
    ) -> List[Tuple[str, int]]:
        props = get_tool_schema_properties(tool_schema)
        fields = []
        for key in props.keys():
            val = tool_args.get(key, None)
            val_json = json.dumps(val)
            val_ids = self.tokenizer.encode(val_json, add_special_tokens=False)
            budget = min(
                max(
                    len(val_ids) + self.budget_config.get("extra_tokens", 0),
                    self.budget_config.get("min_tokens", MIN_FIELD_BUDGET),
                ),
                self.budget_config.get("max_tokens", 48),
            )
            fields.append((key, budget))
        return fields
