"""
Diffusion-specific operations for inference.

Handles sequence building, masking, hidden state caching, and template operations.
"""

from typing import List, Optional, Tuple, Dict, Any
import torch

from data.schema import SchemaTemplate, build_schema_template
from data.smollm3_prompting import encode_tool_call_wrapper
from data.budget_utils import DEFAULT_MAX_BUDGET


class DiffusionOperations:
    """Handles all diffusion-related operations for inference."""

    def __init__(
            self,
            model,
            tokenizer,
            device: torch.device,
            max_seq_len: int = 2048,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_seq_len = max_seq_len

    def _strip_null_tokens(self, tokens: torch.Tensor, null_token_id: Optional[int]) -> torch.Tensor:
        """Strip NULL tokens from output for variable-length field handling.
        
        During generation, the model may predict NULL tokens for unused slots.
        These should be removed before decoding to get clean output.
        
        Args:
            tokens: 1D tensor of token IDs (already on CPU)
            null_token_id: NULL token ID to filter out, or None to skip filtering
        
        Returns:
            Filtered tensor with NULL tokens removed
        """
        if null_token_id is None:
            return tokens
        mask = tokens != null_token_id
        return tokens[mask]

    def _build_prompt_parts(
            self,
            prompt: str,
            tool_name: str,
            tools: Optional[List[Dict[str, Any]]],
            system_message: str,
            messages_builder,
    ) -> Tuple[List[int], List[int], List[int]]:
        tool_call_parts = encode_tool_call_wrapper(self.tokenizer, tool_name)
        prefix_ids = tool_call_parts.prefix_ids
        suffix_ids = tool_call_parts.suffix_ids

        messages = messages_builder(prompt, system_message=system_message)
        from data.smollm3_prompting import apply_smollm3_chat_template
        prompt_ids_list = apply_smollm3_chat_template(
            self.tokenizer,
            messages=messages,
            tools=tools,
            add_generation_prompt=True,
        )
        return prompt_ids_list, prefix_ids, suffix_ids

    def _truncate_prompt_ids(self, prompt_ids_list: List[int], tail_len: int) -> List[int]:
        if len(prompt_ids_list) + tail_len <= self.max_seq_len:
            return prompt_ids_list
        keep = self.max_seq_len - tail_len
        if keep <= 0:
            raise ValueError("Prompt too large to fit tool scaffold under max_seq_len.")
        return prompt_ids_list[-keep:]

    def _build_sequence(
            self,
            prompt_ids_list: List[int],
            prefix_ids: List[int],
            suffix_ids: List[int],
            template: SchemaTemplate,
    ) -> Tuple[torch.Tensor, int, int]:
        tail_len = len(prefix_ids) + len(template.tokens) + len(suffix_ids)
        prompt_ids_list = self._truncate_prompt_ids(prompt_ids_list, tail_len)

        prompt_ids = torch.tensor(prompt_ids_list, dtype=torch.long, device=self.device).unsqueeze(0)
        prefix = torch.tensor(prefix_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        suffix = torch.tensor(suffix_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        template_tensor = template.to_tensor(self.device).unsqueeze(0)

        sequence = torch.cat([prompt_ids, prefix, template_tensor, suffix], dim=1)
        prompt_length = prompt_ids.shape[1]
        prefix_length = prefix.shape[1]
        return sequence, prompt_length, prefix_length

    def _build_scaffold_mask(
            self,
            sequence: torch.Tensor,
            prompt_length: int,
            prefix_length: int,
            template: SchemaTemplate,
    ) -> torch.Tensor:
        scaffold_mask = torch.zeros_like(sequence, dtype=torch.bool, device=self.device)
        for segment in template.field_segments:
            for pos in segment.value_positions:
                scaffold_mask[:, prompt_length + prefix_length + pos] = True
        return scaffold_mask

    def _cache_hidden_states(
            self,
            sequence: torch.Tensor,
            scaffold_mask: torch.Tensor,
            mask_token_id: int,
    ) -> torch.Tensor:
        attention_mask = torch.ones_like(sequence)
        masked_sequence = sequence.clone()
        masked_sequence[scaffold_mask] = mask_token_id
        outputs = self.model.get_hidden_states(
            input_ids=masked_sequence,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states_cached = outputs.hidden_states[-1].detach()
        del outputs, attention_mask, masked_sequence
        return hidden_states_cached

    def _compute_hidden_states(self, sequence: torch.Tensor) -> torch.Tensor:
        attention_mask = torch.ones_like(sequence)
        outputs = self.model.get_hidden_states(
            input_ids=sequence,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = outputs.hidden_states[-1].detach()
        del outputs, attention_mask
        return hidden_states

    def _predict_from_cached_hidden_states(
            self,
            hidden_states: torch.Tensor,
            current_tokens: torch.Tensor,
            t: torch.Tensor,
            use_cuda_graph: bool,
            cuda_graph_runner,
            scaffold_mask: Optional[torch.Tensor] = None,
            prompt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict logits using cached base hidden states.

        This matches training, where the base model is run once on the masked scaffold
        and the diffusion head uses token embeddings to reflect the current state.
        """
        diffusion_head_dtype = next(self.model.diffusion_head.parameters()).dtype
        hidden_states = hidden_states.to(dtype=diffusion_head_dtype)

        if use_cuda_graph and cuda_graph_runner.is_enabled():
            logits = cuda_graph_runner.run(
                hidden_states,
                current_tokens,
                t,
                scaffold_mask=scaffold_mask,
                prompt_mask=prompt_mask,
            )
        else:
            logits = self.model.diffusion_head.predict(
                hidden_states,
                current_tokens,
                t,
                scaffold_mask=scaffold_mask,
                prompt_mask=prompt_mask,
            )

        return logits


class TemplateExpansionOperations:
    """Handles template expansion for dynamic budgeting."""

    def __init__(
            self,
            tokenizer,
            generator,
    ):
        self.tokenizer = tokenizer
        self.generator = generator

    @property
    def expansion_config(self):
        """Access live expansion_config from parent generator."""
        return self.generator.expansion_config

    @property
    def budget_config(self):
        """Access live budget_config from parent generator."""
        return self.generator.budget_config

    def _detect_overflow_fields(
            self,
            sequence: torch.Tensor,
            prompt_length: int,
            prefix_length: int,
            template: SchemaTemplate,
            null_token_id: Optional[int],
    ) -> List[str]:
        if null_token_id is None:
            return []
        tail_window = self.expansion_config.get("tail_window", 0)
        if tail_window <= 0:
            return []
        threshold = self.expansion_config.get("tail_null_threshold", 0.5)
        fields = []
        for segment in template.field_segments:
            positions = segment.value_positions
            if not positions:
                continue
            window = positions[-min(tail_window, len(positions)):]
            abs_positions = [
                prompt_length + prefix_length + pos
                for pos in window
            ]
            tokens = sequence[0, abs_positions]
            null_ratio = (tokens == null_token_id).float().mean().item()
            if null_ratio < threshold:
                fields.append(segment.name)
        return fields

    def _expand_template(
            self,
            template: SchemaTemplate,
            fields_to_expand: List[str],
    ) -> Optional[SchemaTemplate]:
        expand_tokens = self.expansion_config.get("expand_tokens", 0)
        if not fields_to_expand or expand_tokens <= 0:
            return None
        max_tokens = self.budget_config.get("max_tokens", DEFAULT_MAX_BUDGET)
        include_codeblock = self._template_has_codeblock(template)
        fields = []
        changed = False
        for segment in template.field_segments:
            length = segment.length
            if segment.name in fields_to_expand:
                new_length = min(length + expand_tokens, max_tokens)
                if new_length != length:
                    changed = True
            else:
                new_length = length
            fields.append((segment.name, new_length))
        if not changed:
            return None
        return build_schema_template(
            tokenizer=self.tokenizer,
            fields=fields,
            mask_token=template.mask_token,
            null_token=template.null_token,
            include_codeblock=include_codeblock,
        )

    def _template_has_codeblock(self, template: SchemaTemplate) -> bool:
        return template.text.lstrip().startswith("```json")

    def _apply_warm_start(
            self,
            old_sequence: torch.Tensor,
            old_template: SchemaTemplate,
            old_prompt_length: int,
            old_prefix_length: int,
            new_sequence: torch.Tensor,
            new_template: SchemaTemplate,
            new_prompt_length: int,
            new_prefix_length: int,
    ) -> None:
        old_segments = {seg.name: seg for seg in old_template.field_segments}
        for new_seg in new_template.field_segments:
            old_seg = old_segments.get(new_seg.name)
            if old_seg is None:
                continue
            copy_len = min(old_seg.length, new_seg.length)
            for idx in range(copy_len):
                old_pos = (
                        old_prompt_length
                        + old_prefix_length
                        + old_seg.value_positions[idx]
                )
                new_pos = new_prompt_length + new_prefix_length + new_seg.value_positions[idx]
                new_sequence[0, new_pos] = old_sequence[0, old_pos]
