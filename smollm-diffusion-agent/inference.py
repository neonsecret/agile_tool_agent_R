"""
Self-adaptive generation loop for diffusion LLMs with schema scaffolding.

Refactored to use modular components for better code organization.
Based on: dLLM-CtrlGen/decoding/generator.py
Implements the S3 (Schema Scaffolding) denoising loop with top-K remasking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from tqdm import trange

from model.hybrid_model import HybridSmolLM
from data.schema import SchemaTemplate, build_schema_template
from data.utils import validate_mask_token_consistency
from data.budget_utils import build_fields_from_schema, MIN_FIELD_BUDGET
from data.device_utils import empty_cache
from data.smollm3_prompting import parse_first_tool_call

from inference_utils import (
    _default_budget_config,
    _default_expansion_config,
    _apply_gumbel_noise,
    _can_use_torch_compile_mps,
)
from inference_generation import GenerationOperations
from inference_diffusion import DiffusionOperations, TemplateExpansionOperations
from inference_cuda_graph import CUDAGraphRunner


@dataclass
class GenerationConfig:
    """Hyperparameters controlling the denoising schedule."""

    steps: int = 4
    temperature: float = 0.0
    cfg_scale: float = 0.0
    topk_remask: Optional[int] = None
    use_fp16: bool = False
    clear_cache: bool = True
    show_steps: bool = False
    use_cuda_graph: bool = True
    # Running Confidence Remasking (RCR) - allows token revision
    enable_remasking: bool = True
    remask_ratio: float = 0.2  # Fraction of low-confidence tokens to remask
    min_lock_confidence: float = 0.7  # Min confidence to keep token locked


@dataclass
class TraceStep:
    """Trace information for visualising the denoising process."""

    step: int
    revealed_indices: Sequence[int]
    revealed_tokens: Optional[Sequence[str]] = None


@dataclass
class GenerationOutput:
    """Structured output returned by the generator."""

    text: str
    token_ids: List[int]
    steps_executed: int
    trace: List[TraceStep] = field(default_factory=list)


@dataclass
class GenerationState:
    sequence: torch.Tensor
    prompt_length: int
    prefix_length: int
    template: SchemaTemplate


class FunctionCallGenerator:
    """Runs the S3 denoising loop with top-K remasking.
    
    Pipeline:
    - Function name generation (AR from base model)
    - Diffusion parameter filling (S3 strategy)
    
    Optimizations:
    - KV cache reuse: Caches base LLM KV states for prompt, reuses across diffusion steps
    - torch.compile: JIT compiles diffusion head for faster execution
    - CUDA graphs: Captures diffusion head forward pass to reduce kernel launch overhead
    """

    def __init__(
            self,
            model: HybridSmolLM,
            tokenizer: AutoTokenizer,
            device: torch.device,
            use_torch_compile: bool = False,
            use_cuda_graph: bool = True,
            max_seq_len: int = 2048,
            budget_config: Optional[Dict[str, int]] = None,
            expansion_config: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        self.max_seq_len = max_seq_len
        self.budget_config = budget_config or _default_budget_config()
        self.expansion_config = expansion_config or _default_expansion_config()

        self._kv_cache: Optional[Tuple] = None
        self._cached_prompt_length: int = 0
        self._cached_prompt_hidden_states: Optional[torch.Tensor] = None

        is_cuda = device.type == "cuda"
        is_mps = device.type == "mps"
        
        if use_torch_compile and not is_cuda and not is_mps:
            print(f"Warning: torch.compile on {device.type} may not be optimized, disabling")
            use_torch_compile = False
        
        if use_cuda_graph and not is_cuda:
            use_cuda_graph = False

        self._compiled = False
        if use_torch_compile and hasattr(torch, 'compile'):
            if is_mps:
                if _can_use_torch_compile_mps(self.device):
                    self._compile_diffusion_head()
                else:
                    print("torch.compile on MPS requires PyTorch 2.1+, using eager mode")
            else:
                self._compile_diffusion_head()

        self.MODE_CHAT = 0
        self.MODE_TOOL = 1
        self.MODE_THINK = 2

        self._system_message_tool = "/no_think"
        self._system_message_chat = "/think"
        
        self.generation_ops = GenerationOperations(
            model, tokenizer, device,
            self._system_message_tool, self._system_message_chat
        )
        self.diffusion_ops = DiffusionOperations(
            model, tokenizer, device, max_seq_len
        )
        # Pass self as owner so expansion_ops can access live config references
        self.expansion_ops = TemplateExpansionOperations(
            tokenizer, self
        )
        self.cuda_graph_runner = CUDAGraphRunner(
            model, device, use_cuda_graph
        )

    def _compile_diffusion_head(self):
        """Apply torch.compile to diffusion head for JIT optimization."""
        try:
            if self.device.type == "mps" and torch.backends.mps.is_available():
                compile_mode = "default"
                device_info = "MPS"
            else:
                compile_mode = "reduce-overhead"
                device_info = self.device.type.upper()

            self.model.diffusion_head = torch.compile(
                self.model.diffusion_head,
                mode=compile_mode,
                fullgraph=False
            )
            self._compiled = True
            print(f"Diffusion head compiled with torch.compile() ({device_info} mode: {compile_mode})")
        except Exception as e:
            print(f"torch.compile failed, using eager mode: {e}")
            self._compiled = False

    def clear_cache(self):
        """Clear KV cache and CUDA graph state."""
        self._kv_cache = None
        self._cached_prompt_length = 0
        self._cached_prompt_hidden_states = None
        self.cuda_graph_runner.clear()

    def generate_chat(self, prompt: str, max_new_tokens: int = 256) -> str:
        return self.generation_ops.generate_chat(prompt, max_new_tokens)
    
    def generate_think(self, prompt: str, max_new_tokens: int = 512) -> str:
        return self.generation_ops.generate_think(prompt, max_new_tokens)
    
    def select_tool_call(
        self, prompt: str, tool_registry: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        return self.generation_ops.select_tool_call(prompt, tool_registry)

    def _resolve_budget(self, config: GenerationConfig, total_masks: int) -> int:
        """Calculate number of tokens to reveal per step."""
        if config.topk_remask is not None:
            return config.topk_remask
        return max(1, math.ceil(total_masks / config.steps))

    def _expansion_enabled(self, template: SchemaTemplate) -> bool:
        if not self.expansion_config.get("enabled", False):
            return False
        if template.null_token_id is None:
            return False
        return self.expansion_config.get("max_rounds", 0) > 0

    # Backward compatibility wrappers for tests that call private methods
    def _detect_overflow_fields(self, state: GenerationState) -> List[str]:
        """Wrapper for backward compatibility with tests."""
        return self.expansion_ops._detect_overflow_fields(
            state.sequence, state.prompt_length, state.prefix_length,
            state.template, state.template.null_token_id
        )
    
    def _expand_template(self, template: SchemaTemplate, fields_to_expand: List[str]) -> Optional[SchemaTemplate]:
        """Wrapper for backward compatibility with tests."""
        return self.expansion_ops._expand_template(template, fields_to_expand)
    
    def _apply_warm_start(
        self,
        warm_state: GenerationState,
        new_sequence: torch.Tensor,
        new_template: SchemaTemplate,
        new_prompt_length: int,
        new_prefix_length: int,
    ) -> None:
        """Wrapper for backward compatibility with tests."""
        self.expansion_ops._apply_warm_start(
            warm_state.sequence, warm_state.template,
            warm_state.prompt_length, warm_state.prefix_length,
            new_sequence, new_template, new_prompt_length, new_prefix_length,
        )

    def generate_unified(
            self,
            prompt: str,
            tool_registry: Optional[dict] = None,
            config: Optional[GenerationConfig] = None,
    ) -> Dict[str, any]:
        """
        Complete unified inference pipeline.
        
        Flow:
        1. Base model decides whether to use tool (via native tool calling)
        2. If Tool mode:
           a. Base model generates tool name
           b. Run diffusion for parameters
        3. If no tools: Use AR generation
        """
        cfg = config or GenerationConfig()
        
        if tool_registry is None or len(tool_registry) == 0:
            response = self.generate_chat(prompt)
            return {"mode": "chat", "response": response, "note": "No tools provided"}

        tool_call = self.select_tool_call(prompt, tool_registry)
        if tool_call is None:
            response = self.generate_chat(prompt)
            return {"mode": "chat", "response": response}

        tool_name = tool_call.get("name")
        if not tool_name or tool_name not in tool_registry:
            response = self.generate_chat(prompt)
            return {
                "mode": "chat",
                "response": response,
                "note": f"Tool selection failed or unknown tool: {tool_name}",
            }
        
        tool_schema = tool_registry[tool_name]
        
        min_budget = self.budget_config.get("min_tokens", MIN_FIELD_BUDGET)
        max_budget = self.budget_config.get("max_tokens", 48)
        extra_budget = self.budget_config.get("extra_tokens", 0)
        fields = build_fields_from_schema(
            tool_schema,
            self.tokenizer,
            min_budget=min_budget,
            max_budget=max_budget,
            extra_budget=extra_budget,
        )
        
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
        
        output = self.generate(
            prompt=prompt,
            template=template,
            config=cfg,
            tool_name=tool_name,
            tools=list(tool_registry.values()),
        )
        
        parsed = parse_first_tool_call(output.text)
        return {
            "mode": "tool",
            "tool_name": tool_name,
            "tool_call": output.text,
            "tool_call_parsed": parsed,
            "steps_executed": output.steps_executed,
        }

    def generate(
            self,
            prompt: str,
            template: SchemaTemplate,
            config: Optional[GenerationConfig] = None,
            trace: bool = False,
            callback: Optional[Callable[[int, torch.Tensor, int], None]] = None,
            tool_name: str = "generic_tool",
            tools: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> GenerationOutput:
        cfg = config or GenerationConfig()
        if not self._expansion_enabled(template):
            return self._generate_single_pass(
                prompt=prompt,
                template=template,
                config=cfg,
                trace=trace,
                callback=callback,
                tool_name=tool_name,
                tools=tools,
            )

        max_rounds = self.expansion_config.get("max_rounds", 0)
        current_template = template
        warm_state = None
        output = None

        for _ in range(max_rounds + 1):
            output, state = self._generate_single_pass(
                prompt=prompt,
                template=current_template,
                config=cfg,
                trace=trace,
                callback=callback,
                tool_name=tool_name,
                tools=tools,
                return_state=True,
                warm_start_state=warm_state,
            )
            # Use wrapper method so tests can mock it
            fields_to_expand = self._detect_overflow_fields(state)
            if not fields_to_expand:
                return output
            new_template = self._expand_template(current_template, fields_to_expand)
            if new_template is None:
                return output
            warm_state = state
            current_template = new_template

        return output

    def _generate_single_pass(
            self,
            prompt: str,
            template: SchemaTemplate,
            config: Optional[GenerationConfig] = None,
            trace: bool = False,
            callback: Optional[Callable[[int, torch.Tensor, int], None]] = None,
            tool_name: str = "generic_tool",
            tools: Optional[Sequence[Dict[str, Any]]] = None,
            return_state: bool = False,
            warm_start_state: Optional[GenerationState] = None,
    ) -> GenerationOutput:
        """Main inference loop combining AR + Scaffolding + Diffusion."""
        cfg = config or GenerationConfig()

        validate_mask_token_consistency(
            self.model.diffusion_head.mask_token_id,
            template.mask_token_id,
            context=" in FunctionCallGenerator._generate_single_pass()"
        )

        self._kv_cache = None
        self._cached_prompt_length = 0
        self._cached_prompt_hidden_states = None

        prompt_ids_list, prefix_ids, suffix_ids = self.diffusion_ops._build_prompt_parts(
            prompt=prompt,
            tool_name=tool_name,
            tools=tools,
            system_message=self._system_message_tool,
            messages_builder=self.generation_ops._build_messages,
        )

        with torch.no_grad():
            sequence, prompt_length, prefix_length = self.diffusion_ops._build_sequence(
                prompt_ids_list=prompt_ids_list,
                prefix_ids=prefix_ids,
                suffix_ids=suffix_ids,
                template=template,
            )

            if warm_start_state is not None:
                self.expansion_ops._apply_warm_start(
                    warm_start_state.sequence, warm_start_state.template,
                    warm_start_state.prompt_length, warm_start_state.prefix_length,
                    sequence, template, prompt_length, prefix_length,
                )

            scaffold_mask = self.diffusion_ops._build_scaffold_mask(
                sequence, prompt_length, prefix_length, template,
            )
            initial_variable_count = int(scaffold_mask.sum().item())

            budget = self._resolve_budget(cfg, initial_variable_count)

            generation_trace: List[TraceStep] = []
            mask_positions = torch.zeros_like(sequence, dtype=torch.bool)

            # Running Confidence Remasking (RCR) state
            seq_len = sequence.shape[1]
            running_max_conf = torch.full((seq_len,), float('-inf'), device=self.device)
            revealed_positions = torch.zeros(seq_len, dtype=torch.bool, device=self.device)

            hidden_states_cached = self.diffusion_ops._cache_hidden_states(
                sequence, scaffold_mask, template.mask_token_id,
            )

            for step in trange(cfg.steps, desc="Diffusion steps", disable=not cfg.show_steps):
                mask_positions.fill_(False)
                mask_positions[sequence == template.mask_token_id] = True
                mask_positions = mask_positions & scaffold_mask
                mask_indices = torch.nonzero(mask_positions[0], as_tuple=False).squeeze(-1)

                if mask_indices.numel() == 0:
                    executed_steps = step
                    break

                remaining = int(mask_indices.numel())
                t_val = float(remaining) / float(initial_variable_count)
                t = torch.full(
                    (sequence.shape[0],),
                    t_val,
                    device=self.device,
                    dtype=torch.float,
                )

                logits = self.diffusion_ops._predict_from_cached_hidden_states(
                    hidden_states_cached, sequence, t,
                    cfg.use_cuda_graph, self.cuda_graph_runner
                )

                if cfg.show_steps and step == 0:
                    mask_logits = logits[0, mask_indices]
                    valid_mask = torch.ones_like(mask_logits, dtype=torch.bool)
                    valid_mask[:, template.mask_token_id] = False
                    valid_logits = mask_logits[valid_mask]
                    print(f"  Logits stats: mean={valid_logits.mean():.4f}, std={valid_logits.std():.4f}")
                    print(f"  Logits range: [{valid_logits.min():.4f}, {valid_logits.max():.4f}]")

                logits[:, :, template.mask_token_id] = -float('inf')

                if cfg.show_steps and step <= 1:
                    with torch.no_grad():
                        for i in range(min(3, len(mask_indices))):
                            pos_logits = logits[0, mask_indices[i]]
                            top5 = torch.topk(pos_logits, 5)
                            top5_ids = top5.indices.cpu().tolist()
                            top5_tokens = [self.tokenizer.decode([tid], skip_special_tokens=False) for tid in top5_ids]
                            print(f"  Position {i} top-5: {list(zip(top5_tokens, top5_ids))}")

                logits = _apply_gumbel_noise(logits, cfg.temperature)
                predictions = torch.argmax(logits, dim=-1)

                if cfg.show_steps:
                    mask_predictions = predictions[0, mask_indices]
                    unique_preds = torch.unique(mask_predictions)
                    print(f"  Unique predicted token IDs at masked positions: {unique_preds.tolist()[:10]}")
                    if template.mask_token_id in unique_preds:
                        print(f"  WARNING: Predicting mask token {template.mask_token_id}!")

                log_probs = F.log_softmax(logits[0, mask_indices], dim=-1)
                del logits

                mask_conf = log_probs.gather(
                    -1, predictions[0, mask_indices].unsqueeze(-1)
                ).squeeze(-1)
                del log_probs

                # Update running max confidence for masked positions
                running_max_conf[mask_indices] = torch.maximum(
                    running_max_conf[mask_indices],
                    mask_conf
                )

                remaining = mask_indices.numel()
                remaining_steps = cfg.steps - step
                if remaining_steps <= 1:
                    k = remaining
                else:
                    k = min(budget, remaining)

                topk = torch.topk(mask_conf, k)
                selected = mask_indices[topk.indices]
                selected_conf = mask_conf[topk.indices]
                del mask_conf

                if cfg.show_steps:
                    revealed_token_ids = predictions[0, selected].cpu().tolist()
                    revealed_tokens = [self.tokenizer.decode([tid], skip_special_tokens=False) for tid in revealed_token_ids[:5]]
                    print(f"  Revealing {len(selected)} tokens: {revealed_token_ids[:5]}")
                    print(f"  Decoded: {revealed_tokens}")

                del mask_indices, topk

                sequence[0, selected] = predictions[0, selected]
                revealed_positions[selected] = True
                del predictions

                # Running Confidence Remasking (RCR) - allow token revision
                if cfg.enable_remasking and step < cfg.steps - 1:
                    # Find revealed tokens with low running confidence
                    revealed_indices = torch.nonzero(
                        revealed_positions & scaffold_mask[0], as_tuple=False
                    ).squeeze(-1)

                    if revealed_indices.numel() > 0:
                        revealed_conf = running_max_conf[revealed_indices]
                        # Convert log probs to probs for threshold comparison
                        revealed_probs = torch.exp(revealed_conf)

                        # Remask tokens below confidence threshold
                        low_conf_mask = revealed_probs < cfg.min_lock_confidence

                        if low_conf_mask.any():
                            # Remask only a fraction to avoid oscillation
                            num_to_remask = max(1, int(low_conf_mask.sum() * cfg.remask_ratio))
                            low_conf_indices = revealed_indices[low_conf_mask]

                            if low_conf_indices.numel() > 0:
                                # Select lowest confidence tokens to remask
                                low_conf_values = revealed_conf[low_conf_mask]
                                _, worst_indices = torch.topk(
                                    low_conf_values, min(num_to_remask, low_conf_indices.numel()),
                                    largest=False
                                )
                                remask_positions = low_conf_indices[worst_indices]

                                # Remask these tokens
                                sequence[0, remask_positions] = template.mask_token_id
                                revealed_positions[remask_positions] = False

                                if cfg.show_steps:
                                    print(f"  RCR: Remasking {len(remask_positions)} low-confidence tokens")

                if cfg.show_steps:
                    response_tokens_step = sequence[0, prompt_length:].cpu()

                    unique_in_seq = torch.unique(response_tokens_step).tolist()
                    print(f"  Unique tokens in sequence now: {unique_in_seq[:15]}")

                    text_step = self.tokenizer.decode(
                        response_tokens_step,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    print(f"Step {step + 1}/{cfg.steps}: {text_step}")

                if trace:
                    generation_trace.append(
                        TraceStep(
                            step=step,
                            revealed_indices=selected.cpu().tolist(),
                            revealed_tokens=None,
                        )
                    )

                if callback is not None:
                    callback(step, sequence.clone(), prompt_length)
            else:
                executed_steps = cfg.steps

            response_tokens = sequence[0, prompt_length:].cpu()

            null_token_id = template.null_token_id
            response_tokens_clean = self.diffusion_ops._strip_null_tokens(response_tokens, null_token_id)

            if trace:
                for trace_step in generation_trace:
                    trace_step.revealed_tokens = [
                        self.tokenizer.decode(
                            [int(sequence[0, idx].item())],
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False,
                        )
                        for idx in trace_step.revealed_indices
                    ]

            text = self.tokenizer.decode(
                response_tokens_clean,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )

            state = None
            if return_state:
                state = GenerationState(
                    sequence=sequence.cpu(),
                    prompt_length=prompt_length,
                    prefix_length=prefix_length,
                    template=template,
                )

            del sequence, mask_positions

            self._kv_cache = None
            self._cached_prompt_length = 0
            self._cached_prompt_hidden_states = None

            if cfg.clear_cache:
                empty_cache(self.device)

        output = GenerationOutput(
            text=text,
            token_ids=response_tokens_clean.tolist(),
            steps_executed=executed_steps,
            trace=generation_trace,
        )

        if return_state:
            return output, state
        return output
