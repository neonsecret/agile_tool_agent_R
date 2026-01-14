"""
Self-adaptive generation loop for diffusion LLMs with schema scaffolding.

Based on: dLLM-CtrlGen/decoding/generator.py
Implements the S3 (Schema Scaffolding) denoising loop with top-K remasking.

Optimizations:
- KV cache reuse for prompt portion (avoid recomputing base LLM for same prompt)
- torch.compile for diffusion head (JIT compilation speedup)
- CUDA graphs for diffusion head forward pass (reduce kernel launch overhead)
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from tqdm import trange

import yaml

from model.hybrid_model import HybridSmolLM
from data.schema import SchemaTemplate, build_schema_template
from data.utils import resolve_mask_token, resolve_null_token, validate_mask_token_consistency
from data.smollm3_prompting import (
    apply_smollm3_chat_template,
    encode_tool_call_wrapper,
    parse_first_tool_call,
)
from data.budget_utils import build_fields_from_schema, print_budget_info, MIN_FIELD_BUDGET, DEFAULT_MAX_BUDGET
from data.device_utils import empty_cache, synchronize, get_device
from data.config_utils import validate_and_adjust_config, get_model_kwargs, get_inference_kwargs, print_device_capabilities


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _is_cuda_graph_supported(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.is_available()


def _can_use_torch_compile_mps(device: torch.device) -> bool:
    """Check if torch.compile works on MPS (requires PyTorch 2.1+)."""
    if device.type != "mps" or not torch.backends.mps.is_available():
        return False

    # torch.compile on MPS requires PyTorch 2.1+
    try:
        version = torch.__version__.split('.')
        major, minor = int(version[0]), int(version[1])
        return major >= 2 and minor >= 1
    except (ValueError, IndexError):
        return False


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


def _apply_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Apply Gumbel noise for temperature-based sampling."""
    if temperature <= 0.0:
        return logits
    noise = torch.rand_like(logits)
    noise = noise.clamp_min(1e-10)
    gumbel = -torch.log(-torch.log(noise))
    return logits / temperature + gumbel


class FunctionCallGenerator:
    """Runs the S3 denoising loop with top-K remasking.
    
    Now includes full pipeline:
    - Router mode selection (Chat/Think/Tool)
    - Decision token generation (use_tool/answer)
    - Function name generation (AR)
    - Diffusion parameter filling (existing S3 strategy)
    
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
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        self.max_seq_len = max_seq_len

        # KV cache storage (kept for future use, currently caching hidden states once)
        self._kv_cache: Optional[Tuple] = None
        self._cached_prompt_length: int = 0
        self._cached_prompt_hidden_states: Optional[torch.Tensor] = None

        # Validate device-specific optimizations
        is_cuda = device.type == "cuda"
        is_mps = device.type == "mps"
        
        if use_torch_compile and not is_cuda and not is_mps:
            print(f"Warning: torch.compile on {device.type} may not be optimized, disabling")
            use_torch_compile = False
        
        if use_cuda_graph and not is_cuda:
            use_cuda_graph = False

        # torch.compile optimization
        self._compiled = False
        if use_torch_compile and hasattr(torch, 'compile'):
            if is_mps:
                if _can_use_torch_compile_mps(self.device):
                    self._compile_diffusion_head()
                else:
                    print("torch.compile on MPS requires PyTorch 2.1+, using eager mode")
            else:
                self._compile_diffusion_head()

        # CUDA graph optimization (CUDA only)
        self._cuda_graph_enabled = use_cuda_graph and _is_cuda_graph_supported(device)
        self._cuda_graph: Optional[torch.cuda.CUDAGraph] = None
        self._graph_inputs: Dict[str, torch.Tensor] = {}
        self._graph_outputs: Optional[torch.Tensor] = None
        
        # Mode labels
        self.MODE_CHAT = 0
        self.MODE_TOOL = 1
        self.MODE_THINK = 2

        # SmolLM3 chat template configuration
        self._system_message_tool = "/no_think"
        self._system_message_chat = "/think"

    def _compile_diffusion_head(self):
        """Apply torch.compile to diffusion head for JIT optimization.
        
        Uses MPS-specific compilation mode when running on Apple Silicon.
        MPS works better with "default" mode (PyTorch 2.1+).
        """
        try:
            # Determine compilation mode based on device
            if self.device.type == "mps" and torch.backends.mps.is_available():
                # MPS-specific: use "default" mode for better compatibility
                # PyTorch 2.1+ has improved MPS support
                compile_mode = "default"
                device_info = "MPS"
            else:
                # CUDA/CPU: use "reduce-overhead" for maximum performance
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

    def _setup_cuda_graph(self, hidden_states: torch.Tensor,
                          current_tokens: torch.Tensor, t: torch.Tensor):
        """Capture CUDA graph for diffusion head forward pass."""
        if not self._cuda_graph_enabled:
            return

        # Warmup runs required before graph capture
        for _ in range(3):
            _ = self.model.diffusion_head.predict(hidden_states, current_tokens, t)

        synchronize(self.device)

        # Create static input tensors for graph
        self._graph_inputs = {
            'hidden_states': hidden_states.clone(),
            'current_tokens': current_tokens.clone(),
            't': t.clone()
        }

        # Capture graph
        self._cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._cuda_graph):
            self._graph_outputs = self.model.diffusion_head.predict(
                self._graph_inputs['hidden_states'],
                self._graph_inputs['current_tokens'],
                self._graph_inputs['t']
            )

        print("CUDA graph captured for diffusion head")

    def _run_cuda_graph(self, hidden_states: torch.Tensor,
                        current_tokens: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Execute captured CUDA graph with new inputs."""
        # Copy new data to static graph inputs
        self._graph_inputs['hidden_states'].copy_(hidden_states)
        self._graph_inputs['current_tokens'].copy_(current_tokens)
        self._graph_inputs['t'].copy_(t)

        # Replay graph
        self._cuda_graph.replay()

        return self._graph_outputs.clone()

    def clear_cache(self):
        """Clear KV cache and CUDA graph state."""
        self._kv_cache = None
        self._cached_prompt_length = 0
        self._cached_prompt_hidden_states = None
        self._cuda_graph = None
        self._graph_inputs = {}
        self._graph_outputs = None

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
        # Filter out NULL tokens - works on CPU tensors
        mask = tokens != null_token_id
        return tokens[mask]
    
    def route_mode(self, prompt: str) -> int:
        """
        Use router head to classify mode: Chat (0), Tool (1), or Think (2).
        
        Args:
            prompt: User query
        
        Returns:
            Mode index: 0=Chat, 1=Tool, 2=Think
        """
        with torch.no_grad():
            # Encode prompt
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
            prompt_ids = prompt_ids.to(self.device)
            
            # Get hidden states from base model
            outputs = self.model.get_hidden_states(
                input_ids=prompt_ids,
                attention_mask=torch.ones_like(prompt_ids),
                output_hidden_states=True
            )
            hidden_states = outputs.hidden_states[-1]
            
            # Router prediction
            router_logits = self.model.router_head(
                hidden_states, attention_mask=torch.ones_like(prompt_ids)
            )
            mode = torch.argmax(router_logits, dim=-1).item()
            
            return mode
    
    def _build_messages(self, prompt: str, system_message: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]

    def _generate_base(
        self,
        prompt: str,
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        system_message: Optional[str] = None,
        max_new_tokens: int = 256,
        do_sample: bool = False,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> Tuple[str, str]:
        """
        Generate using the frozen base model with SmolLM3's official chat template.

        Returns:
            (generated_text_raw, generated_text_clean)
        """
        if system_message is None:
            system_message = self._system_message_chat

        messages = self._build_messages(prompt, system_message=system_message)
        prompt_ids_list = apply_smollm3_chat_template(
            self.tokenizer,
            messages=messages,
            tools=tools,
            add_generation_prompt=True,
        )
        prompt_ids = torch.tensor(prompt_ids_list, dtype=torch.long, device=self.device).unsqueeze(0)
        prompt_len = prompt_ids.shape[1]

        if self.model.use_unsloth:
            raise NotImplementedError("Unsloth inference is not implemented for base generation.")

        output_ids = self.model.base_llm.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        gen_ids = output_ids[0, prompt_len:]
        generated_raw = self.tokenizer.decode(
            gen_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        generated_clean = self.tokenizer.decode(
            gen_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return generated_raw, generated_clean

    def generate_chat(self, prompt: str, max_new_tokens: int = 256) -> str:
        """
        Standard autoregressive generation for chat mode.
        
        Args:
            prompt: User query
            max_new_tokens: Maximum tokens to generate
        
        Returns:
            Generated response
        """
        raw, clean = self._generate_base(
            prompt,
            tools=None,
            system_message=self._system_message_chat,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        del raw
        return clean.strip()
    
    def generate_think(self, prompt: str, max_new_tokens: int = 512) -> str:
        raw, clean = self._generate_base(
            prompt,
            tools=None,
            system_message=self._system_message_chat,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        del raw
        return clean.strip()
    
    def select_tool_call(
        self, prompt: str, tool_registry: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Use SmolLM3's official tool calling template to decide whether to call a tool.

        Returns parsed tool call dict (with "name" and "arguments") or None.
        """
        tools_list = list(tool_registry.values())
        generated_raw, _ = self._generate_base(
            prompt,
            tools=tools_list,
            system_message=self._system_message_tool,
            max_new_tokens=256,
            do_sample=False,
        )
        return parse_first_tool_call(generated_raw)

    def _initialise_sequence(
            self, prompt_ids: torch.Tensor, template: SchemaTemplate
    ) -> torch.Tensor:
        """Concatenate prompt with scaffold template."""
        template_tensor = template.to_tensor(self.device).unsqueeze(0)
        return torch.cat([prompt_ids, template_tensor], dim=1)

    def _resolve_budget(self, config: GenerationConfig, total_masks: int) -> int:
        """Calculate number of tokens to reveal per step."""
        if config.topk_remask is not None:
            return config.topk_remask
        return max(1, math.ceil(total_masks / config.steps))

    def _forward(
            self,
            sequence: torch.Tensor,
            prompt_mask: torch.Tensor,
            template: SchemaTemplate,
            timestep: torch.Tensor,
            cfg_scale: float = 0.0,
            prompt_length: int = 0,
            use_cuda_graph: bool = True,
    ) -> torch.Tensor:
        """Forward pass through diffusion head (base hidden states are cached externally)."""
        raise NotImplementedError("Use _predict_from_cached_hidden_states() instead.")

    def _predict_from_cached_hidden_states(
        self,
        hidden_states: torch.Tensor,
        current_tokens: torch.Tensor,
        t: torch.Tensor,
        use_cuda_graph: bool,
    ) -> torch.Tensor:
        """
        Predict logits using cached base hidden states.

        This matches training, where the base model is run once on the masked scaffold
        and the diffusion head uses token embeddings to reflect the current state.
        """
        diffusion_head_dtype = next(self.model.diffusion_head.parameters()).dtype
        hidden_states = hidden_states.to(dtype=diffusion_head_dtype)

        # Use CUDA graph if enabled and available
        if use_cuda_graph and self._cuda_graph_enabled:
            if self._cuda_graph is None:
                # First call - setup CUDA graph
                self._setup_cuda_graph(hidden_states, current_tokens, t)

            if self._cuda_graph is not None:
                # Check if shapes match (CUDA graphs require fixed shapes)
                if (hidden_states.shape == self._graph_inputs['hidden_states'].shape and
                        current_tokens.shape == self._graph_inputs['current_tokens'].shape):
                    logits = self._run_cuda_graph(hidden_states, current_tokens, t)
                else:
                    # Shape mismatch - fall back to regular forward
                    logits = self.model.diffusion_head.predict(hidden_states, current_tokens, t)
            else:
                logits = self.model.diffusion_head.predict(hidden_states, current_tokens, t)
        else:
            logits = self.model.diffusion_head.predict(hidden_states, current_tokens, t)

        return logits

    def generate_unified(
            self,
            prompt: str,
            tool_registry: Optional[dict] = None,
            config: Optional[GenerationConfig] = None,
            use_router: bool = True,
    ) -> Dict[str, any]:
        """
        Complete unified inference pipeline with routing.
        
        Flow:
        1. Router classifies mode (Chat/Think/Tool)
        2. If Tool mode:
           a. Generate decision token (use_tool/answer)
           b. Generate function name
           c. Run diffusion for parameters
        3. If Chat/Think: Use AR generation
        
        Args:
            prompt: User query
            tool_registry: Dict of available tools {name: schema}
            config: Generation configuration
            use_router: Whether to use router (False = skip to tool mode for testing)
        
        Returns:
            Dict with 'mode', 'response', and optional 'tool_call' info
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
            
        fields = build_fields_from_schema(
            tool_schema,
            self.tokenizer,
            min_budget=MIN_FIELD_BUDGET,
            max_budget=DEFAULT_MAX_BUDGET,
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
        """
        Main inference loop combining AR + Scaffolding + Diffusion.

        Based on dLLM-CtrlGen generator.py lines 151-278.
        
        Note: Inference processes one example at a time (batch_size=1), so padding/bucketing
        is not required. CUDA graphs will capture fixed shapes from the first forward pass.
        
        Optimizations enabled via config:
        - use_kv_cache: Cache base LLM KV states for prompt (default: True)
        - use_cuda_graph: Use CUDA graphs for diffusion head (default: False)

        Args:
            prompt: User query
            template: SchemaTemplate with mask positions
            config: Generation configuration
            trace: Whether to trace denoising steps
            callback: Optional callback function per step

        Returns:
            GenerationOutput with text and metadata
        """
        cfg = config or GenerationConfig()

        validate_mask_token_consistency(
            self.model.diffusion_head.mask_token_id,
            template.mask_token_id,
            context=" in FunctionCallGenerator.generate()"
        )

        # Clear KV cache for new prompt
        self._kv_cache = None
        self._cached_prompt_length = 0
        self._cached_prompt_hidden_states = None

        tool_call_parts = encode_tool_call_wrapper(self.tokenizer, tool_name)
        prefix_ids = tool_call_parts.prefix_ids
        suffix_ids = tool_call_parts.suffix_ids

        messages = self._build_messages(prompt, system_message=self._system_message_tool)
        prompt_ids_list = apply_smollm3_chat_template(
            self.tokenizer,
            messages=messages,
            tools=tools,
            add_generation_prompt=True,
        )

        tail_len = len(prefix_ids) + len(template.tokens) + len(suffix_ids)
        if len(prompt_ids_list) + tail_len > self.max_seq_len:
            # Inference should preserve tool_call + scaffold; truncate prompt from the left.
            keep = self.max_seq_len - tail_len
            if keep <= 0:
                raise ValueError("Prompt too large to fit tool scaffold under max_seq_len.")
            prompt_ids_list = prompt_ids_list[-keep:]

        with torch.no_grad():
            prompt_ids = torch.tensor(
                prompt_ids_list, dtype=torch.long, device=self.device
            ).unsqueeze(0)
            prefix = torch.tensor(prefix_ids, dtype=torch.long, device=self.device).unsqueeze(0)
            suffix = torch.tensor(suffix_ids, dtype=torch.long, device=self.device).unsqueeze(0)
            template_tensor = template.to_tensor(self.device).unsqueeze(0)
            sequence = torch.cat([prompt_ids, prefix, template_tensor, suffix], dim=1)
            prompt_length = prompt_ids.shape[1]
            prefix_length = prefix.shape[1]
            del prompt_ids, prefix, suffix, template_tensor

            prompt_mask = torch.zeros_like(sequence, dtype=torch.bool, device=self.device)
            prompt_mask[:, :prompt_length] = True

            scaffold_mask = torch.zeros_like(sequence, dtype=torch.bool, device=self.device)
            for segment in template.field_segments:
                for pos in segment.value_positions:
                    scaffold_mask[:, prompt_length + prefix_length + pos] = True
            initial_variable_count = int(scaffold_mask.sum().item())

            budget = self._resolve_budget(cfg, initial_variable_count)

            generation_trace: List[TraceStep] = []
            mask_positions = torch.zeros_like(sequence, dtype=torch.bool)

            # Cache base hidden states ONCE on the initial fully-masked scaffold (matches training).
            attention_mask = torch.ones_like(sequence)
            outputs = self.model.get_hidden_states(
                input_ids=sequence,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            hidden_states_cached = outputs.hidden_states[-1].detach()
            del outputs, attention_mask

            for step in trange(cfg.steps, desc="Diffusion steps", disable=not cfg.show_steps):
                mask_positions.fill_(False)
                mask_positions[sequence == template.mask_token_id] = True
                mask_positions = mask_positions & scaffold_mask
                mask_indices = torch.nonzero(mask_positions[0], as_tuple=False).squeeze(-1)

                if mask_indices.numel() == 0:
                    executed_steps = step
                    break

                # Set t based on current masked fraction (consistent with training noise rate).
                remaining = int(mask_indices.numel())
                t_val = float(remaining) / float(initial_variable_count)
                t = torch.full(
                    (sequence.shape[0],),
                    t_val,
                    device=self.device,
                    dtype=torch.float,
                )

                logits = self._predict_from_cached_hidden_states(
                    hidden_states_cached,
                    sequence,
                    t,
                    use_cuda_graph=cfg.use_cuda_graph,
                )
                
                # CRITICAL FIX: Prevent NULL token from being predicted at masked positions
                # NULL tokens should only exist in pre-generated padding, not be predicted
                if template.null_token_id is not None:
                    logits[:, :, template.null_token_id] = -float('inf')
                
                # Also prevent mask token from being predicted (it's an input marker, not output)
                logits[:, :, template.mask_token_id] = -float('inf')
                
                # Debug: Check logits statistics
                if cfg.show_steps and step == 0:
                    mask_logits = logits[0, mask_indices]
                    print(f"  Logits stats: mean={mask_logits.mean():.4f}, std={mask_logits.std():.4f}")
                    print(f"  Logits range: [{mask_logits.min():.4f}, {mask_logits.max():.4f}]")
                    print(f"  Mask token logit (ID {template.mask_token_id}): {mask_logits[0, template.mask_token_id]:.4f}")
                
                # Debug: Check top predictions for each masked position
                if cfg.show_steps and step <= 1:
                    with torch.no_grad():
                        for i in range(min(3, len(mask_indices))):  # Show first 3 masked positions
                            pos_logits = logits[0, mask_indices[i]]
                            top5 = torch.topk(pos_logits, 5)
                            top5_ids = top5.indices.cpu().tolist()
                            top5_tokens = [self.tokenizer.decode([tid], skip_special_tokens=False) for tid in top5_ids]
                            print(f"  Position {i} top-5: {list(zip(top5_tokens, top5_ids))}")
                
                logits = _apply_gumbel_noise(logits, cfg.temperature)
                predictions = torch.argmax(logits, dim=-1)

                # Debug: Check what's being predicted
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
                
                # Penalize special tokens in confidence selection
                # We want to prioritize revealing real content tokens over NULL/special tokens
                # This is necessary because the model learned to confidently predict NULL tokens
                # for padding positions in variable-length fields during training
                special_token_mask = (predictions[0, mask_indices] >= 128000)  # Reserved special tokens start at 128000
                if special_token_mask.any():
                    # Apply VERY large negative penalty to special tokens
                    # We essentially force the model to reveal non-special tokens first
                    mask_conf = mask_conf.clone()
                    mask_conf[special_token_mask] = -1e10  # Effectively remove from consideration

                remaining = mask_indices.numel()
                remaining_steps = cfg.steps - step
                if remaining_steps <= 1:
                    k = remaining
                else:
                    k = min(budget, remaining)

                topk = torch.topk(mask_conf, k)
                del mask_conf

                selected = mask_indices[topk.indices]
                
                # Debug: Show what tokens are being revealed
                if cfg.show_steps:
                    revealed_token_ids = predictions[0, selected].cpu().tolist()
                    revealed_tokens = [self.tokenizer.decode([tid], skip_special_tokens=False) for tid in revealed_token_ids[:5]]
                    print(f"  Revealing {len(selected)} tokens: {revealed_token_ids[:5]}")
                    print(f"  Decoded: {revealed_tokens}")
                
                del mask_indices, topk

                sequence[0, selected] = predictions[0, selected]
                del predictions

                if cfg.show_steps:
                    response_tokens_step = sequence[0, prompt_length:].cpu()
                    
                    # Debug: Show sequence state after update
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

            # Strip NULL tokens for clean output (self-adaptive masking)
            null_token_id = template.null_token_id
            response_tokens_clean = self._strip_null_tokens(response_tokens, null_token_id)

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

            del sequence, prompt_mask, mask_positions

            # Clear internal caches
            self._kv_cache = None
            self._cached_prompt_length = 0
            self._cached_prompt_hidden_states = None

            if cfg.clear_cache:
                empty_cache(self.device)

        return GenerationOutput(
            text=text,
            token_ids=response_tokens_clean.tolist(),  # Use cleaned tokens (NULL tokens removed)
            steps_executed=executed_steps,
            trace=generation_trace,
        )


def demo_inference():
    """Demo function showing how to use the generator with automatic budgeting."""
    # Print device capabilities and get device
    print_device_capabilities()
    device = get_device()
    print(f"Using device: {device}")

    # Load and validate config
    config = load_config()
    config = validate_and_adjust_config(config, device)
    
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B")

    # Get inference kwargs from config
    inference_cfg = config.get("inference", {})
    inference_kwargs = get_inference_kwargs(config, device)
    
    steps = inference_cfg.get("steps", 4)
    temperature = inference_cfg.get("temperature", 0.0)
    cfg_scale = inference_cfg.get("cfg_scale", 0.0)
    max_seq_length = inference_kwargs.get("max_seq_len", 2048)
    use_torch_compile = inference_kwargs["use_torch_compile"]
    use_cuda_graph = inference_kwargs["use_cuda_graph"]

    # Resolve mask token
    data_cfg = config.get("data", {})
    mask_token_config = data_cfg.get("mask_token", None)
    mask_token_str, mask_token_id = resolve_mask_token(tokenizer, mask_token_config)

    # Resolve NULL token for self-adaptive masking
    null_token_config = data_cfg.get("null_token", None)
    null_token_str, null_token_id = resolve_null_token(tokenizer, null_token_config)
    
    # Get budget settings from config (matches training)
    min_field_budget = MIN_FIELD_BUDGET
    max_field_budget = data_cfg.get("mask_budget", DEFAULT_MAX_BUDGET)

    print(f"Using mask token: {mask_token_str} (ID: {mask_token_id})")
    print(f"Budget range: {min_field_budget}-{max_field_budget} tokens per field")
    print(f"Tokenizer vocab size: {len(tokenizer)}")

    # Check for checkpoint
    training_cfg = config.get("training", {})
    checkpoint_path = training_cfg.get("checkpoint_path", "checkpoints/best_model/model.pt")
    
    # Get model kwargs using config utils
    model_kwargs = get_model_kwargs(config, device)
    model_kwargs['vocab_size'] = len(tokenizer)
    
    model = HybridSmolLM(**model_kwargs)

    # Load checkpoint if it exists
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        checkpoint_state = checkpoint['model_state_dict']
        model_state = model.state_dict()

        filtered_state = {}
        skipped_keys = []
        loaded_keys = []

        for key, value in checkpoint_state.items():
            if key.startswith('diffusion_head.') or key.startswith('router_head.'):
                if key in model_state:
                    if model_state[key].shape == value.shape:
                        filtered_state[key] = value.cpu()
                        loaded_keys.append(key)
                    else:
                        skipped_keys.append(f"{key} (shape mismatch: {model_state[key].shape} vs {value.shape})")
                else:
                    skipped_keys.append(f"{key} (not in model)")
            else:
                skipped_keys.append(f"{key} (base_llm, skipped)")

        if filtered_state:
            model.load_state_dict(filtered_state, strict=False)
            print(f"Loaded {len(loaded_keys)} trainable head weights from checkpoint")
            if skipped_keys:
                print(f"Skipped {len(skipped_keys)} keys (base_llm or incompatible)")
        else:
            print("Warning: No compatible weights found in checkpoint, using untrained heads")

        if 'epoch' in checkpoint:
            print(f"Checkpoint info: epoch {checkpoint['epoch']}, eval loss: {checkpoint.get('eval_loss', 'N/A')}")
    else:
        print(f"No checkpoint found at {checkpoint_path}, using untrained model")

    # Set model to eval mode and configure
    model.to(device)
    model.eval()
    
    model.diffusion_head.set_mask_token_id(mask_token_id)

    # Set NULL token ID if available
    if null_token_id is not None:
        model.diffusion_head.set_null_token_id(null_token_id)

    # Initialize generator with optimizations (from config + device defaults)
    generator = FunctionCallGenerator(
        model,
        tokenizer,
        device,
        use_torch_compile=use_torch_compile,
        use_cuda_graph=use_cuda_graph,
        max_seq_len=max_seq_length,
    )

    print(f"Optimizations: torch.compile={use_torch_compile}, cuda_graph={use_cuda_graph}")

    prompt = "What's the weather in London?"
    
    # Define tool registry (in production, this would come from API/tool registry)
    tool_registry = {
        "get_weather": {
            "name": "get_weather",
            "parameters": {
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name or location"
                    },
                    "units": {
                        "type": "string",
                        "enum": ["C", "F"],
                        "description": "Temperature units"
                    }
                }
            }
        },
        "search_web": {
            "name": "search_web",
            "parameters": {
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    }
                }
            }
        }
    }

    config = GenerationConfig(
        steps=steps,
        temperature=temperature,
        cfg_scale=cfg_scale,
        show_steps=True,
        use_cuda_graph=use_cuda_graph
    )
    
    print("\n" + "="*80)
    print("UNIFIED PIPELINE DEMO")
    print("="*80)
    
    # Test unified pipeline
    result = generator.generate_unified(
        prompt=prompt,
        tool_registry=tool_registry,
        config=config,
        use_router=True  # Enable full routing
    )
    
    print(f"\n{'='*80}")
    print("RESULT")
    print("="*80)
    print(f"Mode: {result['mode']}")
    if 'decision' in result:
        print(f"Decision: {result['decision']}")
    if 'tool_name' in result:
        print(f"Tool: {result['tool_name']}")
    if 'tool_call' in result:
        print(f"Tool Call: {result['tool_call']}")
    if 'response' in result:
        print(f"Response: {result['response']}")
    if 'steps_executed' in result:
        print(f"Steps: {result['steps_executed']}")
    
    # Also demo the old direct method for comparison
    print(f"\n{'='*80}")
    print("DIRECT TOOL MODE (no routing)")
    print("="*80)
    
    tool_schema = tool_registry["get_weather"]
    fields = build_fields_from_schema(
        tool_schema,
        tokenizer,
        min_budget=min_field_budget,
        max_budget=max_field_budget
    )
    
    print("\nAutomatic budget calculation:")
    print_budget_info(fields)

    template = build_schema_template(
        tokenizer=tokenizer,
        fields=fields,
        mask_token=mask_token_str,
        null_token=null_token_str,
        include_codeblock=False
    )

    validate_mask_token_consistency(
        model.diffusion_head.mask_token_id,
        template.mask_token_id,
        context=" in demo_inference()"
    )

    print(f"Scaffold template: {template.text}")

    output = generator.generate(
        prompt=prompt,
        template=template,
        config=config,
        trace=True,
        tool_name="get_weather"
    )

    print(f"\nGenerated text: {output.text}")
    print(f"Steps executed: {output.steps_executed}")


if __name__ == "__main__":
    demo_inference()
