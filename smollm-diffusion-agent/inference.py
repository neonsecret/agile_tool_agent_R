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
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from tqdm import trange

import yaml

from model.hybrid_model import HybridSmolLM
from data.schema import SchemaTemplate, build_schema_template
from data.utils import resolve_mask_token, resolve_null_token, validate_mask_token_consistency
from data.budget_utils import build_fields_from_schema, print_budget_info, MIN_FIELD_BUDGET, DEFAULT_MAX_BUDGET
from data.device_utils import empty_cache, synchronize


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
    use_kv_cache: bool = True
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
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

        # KV cache storage
        self._kv_cache: Optional[Tuple] = None
        self._cached_prompt_length: int = 0
        self._cached_prompt_hidden_states: Optional[torch.Tensor] = None

        # torch.compile optimization
        # For MPS: check if PyTorch version supports it (2.1+)
        # For CUDA/CPU: use if available
        self._compiled = False
        if use_torch_compile and hasattr(torch, 'compile'):
            if self.device.type == "mps":
                if _can_use_torch_compile_mps(self.device):
                    self._compile_diffusion_head()
                else:
                    print("torch.compile on MPS requires PyTorch 2.1+, using eager mode")
            else:
                self._compile_diffusion_head()

        # CUDA graph optimization
        self._cuda_graph_enabled = use_cuda_graph and _is_cuda_graph_supported(device)
        self._cuda_graph: Optional[torch.cuda.CUDAGraph] = None
        self._graph_inputs: Dict[str, torch.Tensor] = {}
        self._graph_outputs: Optional[torch.Tensor] = None

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

    def _build_messages(self, prompt: str) -> List[Dict[str, str]]:
        """Build chat template messages matching training format.
        
        Training format includes:
        - System message with full conversation
        - Special tokens: <|decision:use_tool|>, <|tool_name:...|>
        """
        return [{"role": "user", "content": prompt}]

    def _encode_prompt(self, prompt: str, tool_name: str = "get_weather") -> torch.Tensor:
        """Encode prompt using format that matches training data.
        
        During training, prompts have this structure:
        <|im_start|>system
        ...system prompt...
        <|im_end|>
        <|im_start|>user  
        ...user query...
        <|im_end|>
        <|im_start|>assistant
        <|decision:use_tool|>
        <|tool_name:TOOL_NAME|>
        
        This matches the format from dataset_loader.py lines 131-135
        """
        # Build structured prompt matching training format
        system_prompt = "You are a helpful assistant with access to tools."
        
        structured_prompt = (
            f"<|im_start|>system\n"
            f"{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
            f"<|decision:use_tool|>\n"
            f"<|tool_name:{tool_name}|>\n"
        )
        
        chat_ids = self.tokenizer.encode(structured_prompt, add_special_tokens=False)
        return torch.tensor(chat_ids, dtype=torch.long, device=self.device).unsqueeze(0)

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

    def _forward_with_kv_cache(
            self,
            sequence: torch.Tensor,
            prompt_length: int,
            use_cache: bool = True,
    ) -> torch.Tensor:
        """Get hidden states using KV cache optimization.
        
        On first call: Compute full sequence, cache KV states for prompt portion.
        On subsequent calls: Reuse cached KV states for prompt, only compute scaffold portion.
        
        Note: KV cache only works with PyTorch backend. MLX backend runs full forward each time.
        """
        attention_mask = torch.ones_like(sequence)

        with torch.no_grad():
            if use_cache and self._kv_cache is not None and self._cached_prompt_length == prompt_length and not self.model.use_mlx:
                # Reuse cached KV states for prompt + compute scaffold tokens
                scaffold_tokens = sequence[:, prompt_length:]
                
                # Create attention mask for full sequence (prompt + scaffold)
                full_attention_mask = torch.ones_like(sequence)
                
                # Create position IDs for scaffold tokens (continuing from prompt)
                scaffold_position_ids = torch.arange(
                    prompt_length,
                    sequence.shape[1],
                    device=self.device
                ).unsqueeze(0)
                
                # Forward with cached KV (only processes scaffold tokens)
                outputs = self.model.get_hidden_states(
                    input_ids=scaffold_tokens,
                    attention_mask=full_attention_mask,
                    past_key_values=self._kv_cache,
                    position_ids=scaffold_position_ids,
                    output_hidden_states=True,
                    use_cache=True
                )
                
                # Concatenate cached prompt hidden states with new scaffold hidden states
                if hasattr(self, '_cached_prompt_hidden_states') and self._cached_prompt_hidden_states is not None:
                    scaffold_hidden = outputs.hidden_states[-1]
                    hidden_states = torch.cat([self._cached_prompt_hidden_states, scaffold_hidden], dim=1)
                else:
                    # Fallback: run full forward (cache not fully initialized)
                    outputs = self.model.get_hidden_states(
                        input_ids=sequence,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        use_cache=False
                    )
                    hidden_states = outputs.hidden_states[-1]
            else:
                # First call or cache miss - compute full sequence and cache
                outputs = self.model.get_hidden_states(
                    input_ids=sequence,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=use_cache
                )
                hidden_states = outputs.hidden_states[-1]

                if use_cache and hasattr(outputs, 'past_key_values') and outputs.past_key_values is not None and not self.model.use_mlx:
                    # Cache KV states and prompt hidden states
                    self._kv_cache = outputs.past_key_values
                    self._cached_prompt_length = prompt_length
                    self._cached_prompt_hidden_states = hidden_states[:, :prompt_length, :].detach()

        return hidden_states

    def _forward(
            self,
            sequence: torch.Tensor,
            prompt_mask: torch.Tensor,
            template: SchemaTemplate,
            timestep: float,
            cfg_scale: float = 0.0,
            prompt_length: int = 0,
            use_kv_cache: bool = True,
            use_cuda_graph: bool = True,
    ) -> torch.Tensor:
        """Forward pass through the model with optimizations."""
        with torch.no_grad():
            # Get hidden states (with optional KV cache)
            hidden_states = self._forward_with_kv_cache(
                sequence, prompt_length, use_cache=use_kv_cache
            )

            # Convert hidden_states to match diffusion head dtype (bfloat16)
            diffusion_head_dtype = next(self.model.diffusion_head.parameters()).dtype
            hidden_states = hidden_states.to(dtype=diffusion_head_dtype)

        # Convert timestep to tensor if needed
        if isinstance(timestep, (int, float)):
            t = torch.full((sequence.shape[0],), timestep, device=self.device, dtype=torch.float)
        else:
            t = timestep

        # Use CUDA graph if enabled and available
        if use_cuda_graph and self._cuda_graph_enabled:
            if self._cuda_graph is None:
                # First call - setup CUDA graph
                self._setup_cuda_graph(hidden_states, sequence, t)

            if self._cuda_graph is not None:
                # Check if shapes match (CUDA graphs require fixed shapes)
                if (hidden_states.shape == self._graph_inputs['hidden_states'].shape and
                        sequence.shape == self._graph_inputs['current_tokens'].shape):
                    logits = self._run_cuda_graph(hidden_states, sequence, t)
                else:
                    # Shape mismatch - fall back to regular forward
                    logits = self.model.diffusion_head.predict(hidden_states, sequence, t)
            else:
                logits = self.model.diffusion_head.predict(hidden_states, sequence, t)
        else:
            logits = self.model.diffusion_head.predict(hidden_states, sequence, t)

        return logits

    def generate(
            self,
            prompt: str,
            template: SchemaTemplate,
            config: Optional[GenerationConfig] = None,
            trace: bool = False,
            callback: Optional[Callable[[int, torch.Tensor, int], None]] = None,
            tool_name: str = "generic_tool",
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

        with torch.no_grad():
            prompt_ids = self._encode_prompt(prompt, tool_name)
            sequence = self._initialise_sequence(prompt_ids, template)
            prompt_length = prompt_ids.shape[1]
            del prompt_ids

            prompt_mask = torch.zeros_like(sequence, dtype=torch.bool, device=self.device)
            prompt_mask[:, :prompt_length] = True

            variable_positions = [
                prompt_length + position
                for segment in template.field_segments
                for position in segment.value_positions
            ]
            initial_variable_count = len(variable_positions)
            del variable_positions

            budget = self._resolve_budget(cfg, initial_variable_count)

            generation_trace: List[TraceStep] = []
            mask_positions = torch.zeros_like(sequence, dtype=torch.bool)

            # S3 strategy: Use fixed t=0 (fully denoised state) for all steps
            # The model was trained with random t ∈ [0, 1], but inference uses t=0
            # to get the best prediction, then uses confidence-based top-K masking
            t = torch.zeros(sequence.shape[0], device=self.device)

            for step in trange(cfg.steps, desc="Diffusion steps", disable=not cfg.show_steps):
                mask_positions.fill_(False)
                mask_positions[sequence == template.mask_token_id] = True
                mask_positions[:, :prompt_length] = False
                mask_indices = torch.nonzero(mask_positions[0], as_tuple=False).squeeze(-1)

                if mask_indices.numel() == 0:
                    executed_steps = step
                    break

                # Always use t=0 (S3 strategy: predict clean tokens, use confidence for masking)
                logits = self._forward(
                    sequence,
                    prompt_mask,
                    template,
                    t,
                    cfg.cfg_scale,
                    prompt_length=prompt_length,
                    use_kv_cache=cfg.use_kv_cache,
                    use_cuda_graph=cfg.use_cuda_graph
                )
                
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
                skip_special_tokens=True,
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
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Load config and tokenizer first
    config = load_config()
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B")

    # Inference config
    inference_cfg = config.get("inference", {})
    use_kv_cache = inference_cfg.get("use_kv_cache", True)
    use_torch_compile = inference_cfg.get("use_torch_compile", False)
    use_cuda_graph = inference_cfg.get("use_cuda_graph", True)
    steps = inference_cfg.get("steps", 4)
    temperature = inference_cfg.get("temperature", 0.0)
    cfg_scale = inference_cfg.get("cfg_scale", 0.0)

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
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    diff_cfg = config.get("diffusion", {})
    quant_cfg = config.get("quantization", {})
    checkpoint_path = training_cfg.get("checkpoint_path", "checkpoints/best_model/model.pt")

    quantize_enabled = quant_cfg.get("enabled", False)
    quantize_bits = quant_cfg.get("bits", 4)
    load_in_4bit = quantize_enabled and quantize_bits == 4

    base_model_id = model_cfg.get("base_model_id", "HuggingFaceTB/SmolLM3-3B")
    mlx_base_model_id = model_cfg.get("mlx_base_model_id", "mlx-community/SmolLM3-3B-4bit")
    backend = model_cfg.get("backend", None)

    model = HybridSmolLM(
        base_model_id=base_model_id,
        mlx_base_model_id=mlx_base_model_id,
        load_in_4bit=load_in_4bit,
        diffusion_config=diff_cfg,
        vocab_size=len(tokenizer),
        backend=backend
    )

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
    if not model.use_mlx:
        model.to(device)
    else:
        # For MLX: only move PyTorch heads to device, not the MLX base model
        model.diffusion_head.to(device)
        model.router_head.to(device)
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
        use_cuda_graph=use_cuda_graph
    )

    print(f"Optimizations: torch.compile={use_torch_compile}, cuda_graph={use_cuda_graph}, kv_cache={use_kv_cache}")

    prompt = "What's the weather in London?"
    tool_name = "get_weather"

    # Define tool schema (in production, this would come from API/tool registry)
    tool_schema = {
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
    }

    # Automatically build fields from schema with proper budgeting
    # No hardcoded numbers - fully automatic!
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
        include_codeblock=False  # Match training format
    )

    validate_mask_token_consistency(
        model.diffusion_head.mask_token_id,
        template.mask_token_id,
        context=" in demo_inference()"
    )

    print(f"Scaffold template: {template.text}")

    config = GenerationConfig(
        steps=steps,
        temperature=temperature,
        cfg_scale=cfg_scale,
        show_steps=True,
        use_kv_cache=use_kv_cache,
        use_cuda_graph=use_cuda_graph
    )

    output = generator.generate(
        prompt=prompt,
        template=template,
        config=config,
        trace=True,
        tool_name=tool_name  # Pass tool name for proper prompt formatting
    )

    print(f"\nGenerated text: {output.text}")
    print(f"Steps executed: {output.steps_executed}")


if __name__ == "__main__":
    demo_inference()
