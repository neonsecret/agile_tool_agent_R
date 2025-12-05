"""
MLX-compatible hybrid model for training on Apple Silicon.

This module provides MLX implementations of:
- HybridSmolLM model with quantized base LLM
- Diffusion head with bidirectional attention
- Router head for classification
"""

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from typing import Optional, Dict, Tuple, List
import numpy as np
import math


def log_softmax(x: mx.array, axis: int = -1) -> mx.array:
    """
    Numerically stable log_softmax implementation.
    
    log_softmax(x) = x - logsumexp(x)
    """
    # Try to use mx.logsumexp if available, otherwise compute manually
    try:
        if hasattr(mx, 'logsumexp'):
            return x - mx.logsumexp(x, axis=axis, keepdims=True)
    except AttributeError:
        pass

    # Fallback: manually compute logsumexp for numerical stability
    x_max = mx.max(x, axis=axis, keepdims=True)
    x_shifted = x - x_max
    log_sum_exp = x_max + mx.log(mx.sum(mx.exp(x_shifted), axis=axis, keepdims=True))
    return x - log_sum_exp


class LogLinearNoise:
    """Log Linear noise schedule for discrete diffusion (MLX version).
    
    Built such that 1 - 1/e^(n(t)) interpolates between 0 and
    ~1 when t varies from 0 to 1. Total noise is
    -log(1 - (1 - eps) * t), so the sigma will be
    (1 - eps) * t.
    """

    def __init__(self, eps: float = 1e-3):
        self.eps = eps
        self.sigma_max = self.total_noise(mx.array(1.0))
        self.sigma_min = eps + self.total_noise(mx.array(0.0))

    def total_noise(self, t: mx.array) -> mx.array:
        """Total noise ie integral_0^t g(t) dt + g(0)."""
        return -mx.log1p(-(1 - self.eps) * t)

    def rate_noise(self, t: mx.array) -> mx.array:
        """Rate of change of noise ie g(t)."""
        return (1 - self.eps) / (1 - (1 - self.eps) * t)

    def __call__(self, t: mx.array) -> Tuple[mx.array, mx.array]:
        """Get total noise and rate of noise."""
        return self.total_noise(t), self.rate_noise(t)


class Dropout(nn.Module):
    """Dropout layer for MLX (applies during training)."""

    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p

    def __call__(self, x: mx.array, training: bool = True) -> mx.array:
        if not training or self.p == 0:
            return x
        mask = mx.random.bernoulli(1 - self.p, x.shape)
        return x * mask / (1 - self.p)


class BidirectionalAttentionBlock(nn.Module):
    """Transformer encoder block with bidirectional (non-causal) self-attention.
    
    Unlike autoregressive attention, this allows each position to attend to all
    other positions, enabling global constraint verification for structured output.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        self.dropout = Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def __call__(
            self, x: mx.array, attention_mask: Optional[mx.array] = None, training: bool = True
    ) -> mx.array:
        """
        Args:
            x: [batch, seq_len, hidden_dim]
            attention_mask: Optional [batch, seq_len] boolean mask (True = attend)
            training: Whether in training mode (applies dropout)
        
        Returns:
            [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape

        # Pre-norm
        normed = self.norm1(x)

        # Compute Q, K, V
        q = self.q_proj(normed)
        q = mx.reshape(q, (batch_size, seq_len, self.num_heads, self.head_dim))
        q = mx.transpose(q, (0, 2, 1, 3))  # [batch, heads, seq, head_dim]

        k = self.k_proj(normed)
        k = mx.reshape(k, (batch_size, seq_len, self.num_heads, self.head_dim))
        k = mx.transpose(k, (0, 2, 1, 3))

        v = self.v_proj(normed)
        v = mx.reshape(v, (batch_size, seq_len, self.num_heads, self.head_dim))
        v = mx.transpose(v, (0, 2, 1, 3))

        # Bidirectional attention (no causal mask)
        attn_weights = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) / self.scale

        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask for broadcasting: [batch, 1, 1, seq_len]
            mask = mx.expand_dims(mx.expand_dims(attention_mask, 1), 2)
            attn_weights = mx.where(mask, attn_weights, mx.array(-1e9))

        attn_weights = mx.softmax(attn_weights, axis=-1)
        attn_weights = self.dropout(attn_weights, training=training)

        attn_output = mx.matmul(attn_weights, v)
        attn_output = mx.transpose(attn_output, (0, 2, 1, 3))
        attn_output = mx.reshape(attn_output, (batch_size, seq_len, self.hidden_dim))
        attn_output = self.out_proj(attn_output)

        # Residual connection with dropout
        x = x + self.dropout(attn_output, training=training)

        # MLP with pre-norm and residual
        x = x + self.dropout(self.mlp(self.norm2(x)), training=training)

        return x


class SchemaDiffusionHead(nn.Module):
    """Schema-constrained diffusion head for function calling (MLX version).
    
    Combines:
    - mdlm diffusion mechanics (forward/reverse diffusion with LogLinearNoise)
    - Bidirectional attention for global constraint verification
    - Schema scaffolding support (masks only scaffold positions)
    - NULL token support for self-adaptive masking (variable-length fields)
    """

    def __init__(
            self,
            input_dim: int,
            vocab_size: int,
            hidden_dim: int = 1024,
            num_layers: int = 2,
            num_steps: int = 4,
            label_smoothing: float = 0.1,
            use_bidirectional: bool = True,
            num_heads: int = 8,
    ):
        """
        Args:
            input_dim: Dimension of hidden states from base model
            vocab_size: Size of vocabulary
            hidden_dim: Hidden dimension for diffusion head
            num_layers: Number of attention/residual blocks
            num_steps: Number of diffusion steps for training
            label_smoothing: Label smoothing factor (0.0 = no smoothing, 0.1 = 10% smoothing)
            use_bidirectional: If True, use bidirectional attention; else use residual MLPs
            num_heads: Number of attention heads (only used if use_bidirectional=True)
        """
        super().__init__()
        self.num_steps = num_steps
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.label_smoothing = label_smoothing
        self.use_bidirectional = use_bidirectional

        self.noise = LogLinearNoise()

        self.time_embed = nn.Embedding(num_steps + 1, hidden_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)

        if use_bidirectional:
            # Bidirectional attention blocks for global constraint verification
            self.denoise_blocks = [
                BidirectionalAttentionBlock(hidden_dim, num_heads=num_heads)
                for _ in range(num_layers)
            ]
        else:
            # Original residual MLP blocks (faster but no global attention)
            self.denoise_blocks = [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(num_layers)
            ]

        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        self.mask_token_id: Optional[int] = None
        self.null_token_id: Optional[int] = None

    def set_mask_token_id(self, mask_token_id: int):
        """Set the mask token ID (should be called during initialization)."""
        self.mask_token_id = mask_token_id

    def set_null_token_id(self, null_token_id: int):
        """Set the NULL token ID for self-adaptive masking."""
        self.null_token_id = null_token_id

    def forward_diffusion(
            self, tokens: mx.array, scaffold_mask: mx.array, t: mx.array
    ) -> Tuple[mx.array, mx.array]:
        """
        Add noise (mask tokens) based on timestep t.
        
        From mdlm diffusion.py q_xt method.
        
        Args:
            tokens: [batch, seq_len] clean tokens (may contain -100 for ignore positions)
            scaffold_mask: [batch, seq_len] boolean mask (True = scaffold position)
            t: [batch] timestep in [0, 1]
        
        Returns:
            noisy_tokens: [batch, seq_len] tokens with masks
            mask_positions: [batch, seq_len] boolean mask of which positions were masked
        """
        if self.mask_token_id is None:
            raise ValueError("mask_token_id not set. Call set_mask_token_id() first.")

        valid_mask = tokens >= 0
        scaffold_mask = scaffold_mask & valid_mask

        sigma_t, _ = self.noise(t)
        move_chance = 1 - mx.exp(-sigma_t)
        move_chance = mx.expand_dims(move_chance, -1)

        # Generate mask based on noise schedule
        mask_probs = mx.random.uniform(shape=tokens.shape) < move_chance
        mask_positions = scaffold_mask & mask_probs

        # Apply masks to tokens
        noisy_tokens = mx.where(mask_positions, self.mask_token_id, tokens)

        return noisy_tokens, mask_positions

    def predict(
            self,
            hidden_states: mx.array,
            current_tokens: mx.array,
            t: mx.array,
            attention_mask: Optional[mx.array] = None,
            training: bool = True,
    ) -> mx.array:
        """
        Predict original tokens from noisy version.
        
        Uses bidirectional attention (if enabled) for global constraint verification:
        - Each position can attend to all other positions
        - Enables seeing "location" field while predicting "unit" field
        - Improves consistency in structured JSON output
        
        Args:
            hidden_states: [batch, seq_len, input_dim] from base model
            current_tokens: [batch, seq_len] current noisy state
            t: [batch] or scalar timestep
            attention_mask: Optional [batch, seq_len] boolean mask
            training: Whether in training mode
        
        Returns:
            logits: [batch, seq_len, vocab_size] predictions
        """
        # Ensure float32 for computation
        hidden_states = hidden_states.astype(mx.float32)

        context = self.input_proj(hidden_states)
        safe_tokens = mx.where(current_tokens < 0, 0, current_tokens)
        token_emb = self.token_emb(safe_tokens.astype(mx.int32))

        # Handle timestep conversion
        if isinstance(t, (int, float)):
            t = mx.full((hidden_states.shape[0],), int(t * self.num_steps), dtype=mx.int32)
        elif t.dtype in (mx.float32, mx.float16, mx.bfloat16):
            t = (t * self.num_steps).astype(mx.int32)
            t = mx.clip(t, 0, self.num_steps)

        t_emb = mx.expand_dims(self.time_embed(t.astype(mx.int32)), 1)

        x = context + token_emb + t_emb

        if self.use_bidirectional:
            for block in self.denoise_blocks:
                x = block(x, attention_mask=attention_mask, training=training)
        else:
            for block in self.denoise_blocks:
                x = x + block(x)

        logits = self.output_proj(x)
        return logits

    def training_step(
            self, tokens: mx.array, hidden_states: mx.array, scaffold_mask: mx.array,
            debug: bool = False
    ) -> mx.array:
        """Full training forward pass with diffusion loss."""
        batch_size = tokens.shape[0]

        # Cast hidden_states to float32 for stable computation
        hidden_states = hidden_states.astype(mx.float32)

        scaffold_mask = scaffold_mask.astype(mx.bool_)
        valid_labels = tokens >= 0
        valid_scaffold_mask = scaffold_mask & valid_labels

        # Early return if no valid positions
        num_valid = mx.sum(valid_scaffold_mask.astype(mx.float32))
        mx.eval(num_valid)
        if float(num_valid) == 0:
            return mx.array(0.0)

        t = mx.random.uniform(shape=(batch_size,))

        noisy_tokens, mask_positions = self.forward_diffusion(
            tokens, valid_scaffold_mask, t
        )

        logits = self.predict(hidden_states, noisy_tokens, t, training=True)
        logits = logits.astype(mx.float32)  # Ensure float32 for loss computation

        valid_mask_positions = mask_positions & valid_labels

        flat_logits = mx.reshape(logits, (-1, self.vocab_size))
        flat_labels = mx.reshape(tokens, (-1,))
        flat_mask = mx.reshape(valid_mask_positions, (-1,)).astype(mx.float32)

        num_masked = mx.sum(flat_mask)
        mx.eval(num_masked)
        if float(num_masked) == 0:
            return mx.array(0.0)

        # Numerically stable log softmax
        logits_max = mx.max(flat_logits, axis=-1, keepdims=True)
        logits_shifted = flat_logits - logits_max
        log_sum_exp = mx.log(mx.sum(mx.exp(logits_shifted), axis=-1, keepdims=True) + 1e-10)
        log_probs = logits_shifted - log_sum_exp

        flat_labels_safe = mx.clip(flat_labels, 0, self.vocab_size - 1).astype(mx.int32)
        nll = -mx.take_along_axis(
            log_probs,
            mx.expand_dims(flat_labels_safe, -1),
            axis=-1
        ).squeeze(-1)

        smooth_loss = -mx.mean(log_probs, axis=-1)

        per_position_loss = (1 - self.label_smoothing) * nll + self.label_smoothing * smooth_loss

        masked_loss = per_position_loss * flat_mask
        loss = mx.sum(masked_loss) / mx.maximum(num_masked, mx.array(1.0))

        return loss

    def __call__(
            self,
            hidden_states: mx.array,
            current_tokens: mx.array,
            step_ids: mx.array,
            scaffold_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Forward pass for compatibility with existing code.
        
        Args:
            hidden_states: [batch, seq_len, input_dim]
            current_tokens: [batch, seq_len]
            step_ids: [batch] timestep indices
            scaffold_mask: [batch, seq_len] (unused in forward, kept for compatibility)
        
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        return self.predict(hidden_states, current_tokens, step_ids, training=False)


class RouterHead(nn.Module):
    """Router head for classification (MLX version)."""

    def __init__(self, hidden_size: int, num_classes: int = 2):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, num_classes)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """
        Args:
            hidden_states: [batch, seq_len, hidden]
        
        Returns:
            logits: [batch, num_classes]
        """
        # Pool the last token
        pooled = hidden_states[:, -1, :]
        return self.classifier(pooled)


class HybridSmolLMMLX(nn.Module):
    """Hybrid model with quantized base LLM and trainable heads (MLX version).
    
    Key differences from PyTorch version:
    - Uses mlx_lm for model loading (native quantization support)
    - Base model is frozen (parameters not tracked for gradients)
    - Uses MLX's unified memory (no explicit device placement)
    """

    def __init__(
            self,
            base_model_id: str = "HuggingFaceTB/SmolLM3-3B",
            quantize_bits: Optional[int] = 4,
            diffusion_config: Optional[Dict] = None,
            vocab_size: Optional[int] = None,
    ):
        """
        Args:
            base_model_id: HuggingFace model ID or local path
            quantize_bits: Quantization bits (4 or 8), None for no quantization
            diffusion_config: Configuration dict for diffusion head
            vocab_size: Override vocabulary size (default: from model config)
        """
        super().__init__()

        # Load base model using mlx_lm
        # mlx_lm.load returns (model, tokenizer) tuple
        print(f"Loading base model: {base_model_id}")
        if quantize_bits:
            print(f"Using {quantize_bits}-bit quantization")
            self.base_model, self.tokenizer = load(base_model_id)
            # Note: mlx_lm quantizes during load if model is already quantized
            # For on-the-fly quantization, use mlx_lm.convert separately
        else:
            self.base_model, self.tokenizer = load(base_model_id)

        self._log_base_model_dtype()

        # Get model configuration
        # mlx_lm models store config in model.args or model.config
        model_args = self._get_model_args()
        hidden_size = model_args.get("hidden_size", 2048)

        if vocab_size is None:
            vocab_size = model_args.get("vocab_size", len(self.tokenizer))

        print(f"Hidden size: {hidden_size}, Vocab size: {vocab_size}")

        # Initialize trainable heads
        if diffusion_config is None:
            diffusion_config = {}

        hidden_dim = diffusion_config.get("hidden_dim", 1024)
        num_layers = diffusion_config.get("num_layers", 2)
        num_steps = diffusion_config.get("num_steps", 4)
        label_smoothing = diffusion_config.get("label_smoothing", 0.1)
        use_bidirectional = diffusion_config.get("use_bidirectional", True)
        num_heads = diffusion_config.get("num_heads", 8)

        self.diffusion_head = SchemaDiffusionHead(
            input_dim=hidden_size,
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_steps=num_steps,
            label_smoothing=label_smoothing,
            use_bidirectional=use_bidirectional,
            num_heads=num_heads,
        )

        self.router_head = RouterHead(hidden_size, num_classes=2)
        self._hidden_size = hidden_size

        # Note: Keep heads in float32 for now - bfloat16 can cause NaN issues in MLX
        # self._cast_heads_to_dtype(mx.bfloat16)

    def _log_base_model_dtype(self):
        """Log a sample parameter dtype/shape to confirm quantized vs full precision load."""
        try:
            params = self.base_model.parameters()
            if isinstance(params, dict) and params:
                name, param = next(iter(params.items()))
                print(f"Base model param sample: {name}, dtype={getattr(param, 'dtype', None)}, shape={getattr(param, 'shape', None)}")
        except Exception as e:
            print(f"Could not inspect base model parameters: {e}")

    def _cast_heads_to_dtype(self, dtype=mx.bfloat16):
        """Cast trainable heads to specified dtype (default bfloat16 to match PyTorch)."""
        def cast_params(module):
            params = module.parameters()
            new_params = {}
            for k, v in params.items():
                if isinstance(v, mx.array) and v.dtype == mx.float32:
                    new_params[k] = v.astype(dtype)
                elif isinstance(v, dict):
                    new_params[k] = {kk: vv.astype(dtype) if isinstance(vv, mx.array) and vv.dtype == mx.float32 else vv for kk, vv in v.items()}
                else:
                    new_params[k] = v
            module.update(new_params)

        cast_params(self.diffusion_head)
        cast_params(self.router_head)
        print(f"Heads cast to {dtype}")

    def _get_model_args(self) -> Dict:
        """Extract model configuration from mlx_lm model."""
        # mlx_lm models have different structures depending on version
        if hasattr(self.base_model, "args"):
            # Newer mlx_lm format
            args = self.base_model.args
            if hasattr(args, "__dict__"):
                return vars(args)
            return {"hidden_size": getattr(args, "hidden_size", 2048)}
        elif hasattr(self.base_model, "config"):
            config = self.base_model.config
            if isinstance(config, dict):
                return config
            elif hasattr(config, "__dict__"):
                return vars(config)

        # Fallback: try to infer from model structure
        return {"hidden_size": 2048, "vocab_size": 32000}

    def _get_hidden_states(
            self, input_ids: mx.array, attention_mask: Optional[mx.array] = None
    ) -> mx.array:
        """
        Extract hidden states from base model.
        
        mlx_lm models use a different forward API than transformers.
        This method handles the model-specific extraction.
        """
        # mlx_lm LLM models typically have this structure:
        # model.model.embed_tokens -> embeddings
        # model.model.layers -> transformer layers
        # model.model.norm -> final layer norm

        model = self.base_model

        # Try different model structures
        if hasattr(model, "model"):
            # Wrapper model (e.g., LlamaForCausalLM)
            inner_model = model.model
        else:
            inner_model = model

        # Get embeddings
        if hasattr(inner_model, "embed_tokens"):
            x = inner_model.embed_tokens(input_ids)
        elif hasattr(inner_model, "embeddings"):
            x = inner_model.embeddings(input_ids)
        elif hasattr(inner_model, "wte"):
            x = inner_model.wte(input_ids)
        else:
            raise ValueError("Could not find embedding layer in model")

        # Create attention mask if needed (causal mask is handled by layers)
        # mlx_lm typically uses None for full attention
        mask = None
        if attention_mask is not None:
            # Some models need explicit mask
            pass  # Most mlx_lm models handle masking internally

        # Run through transformer layers
        if hasattr(inner_model, "layers"):
            for i, layer in enumerate(inner_model.layers):
                # Different layers have different signatures
                if hasattr(layer, "__call__"):
                    # Try common signatures
                    try:
                        x = layer(x, mask=mask)
                    except TypeError:
                        try:
                            x = layer(x)
                        except TypeError:
                            try:
                                x = layer(x, attention_mask=mask)
                            except Exception as e:
                                raise ValueError(f"Layer {i} failed with error: {e}")

        # Apply final layer norm if present
        if hasattr(inner_model, "norm"):
            x = inner_model.norm(x)
        elif hasattr(inner_model, "ln_f"):
            x = inner_model.ln_f(x)

        return x

    def freeze_base_model(self):
        """Freeze base model parameters (no gradients tracked)."""
        # In MLX, we control which parameters are trained by what we pass to the optimizer
        # This method is provided for API compatibility
        pass

    def trainable_parameters(self) -> Dict:
        """Return only the trainable parameters (diffusion_head + router_head)."""
        params = {}
        # Get diffusion head parameters
        for name, param in self.diffusion_head.parameters().items():
            params[f"diffusion_head.{name}"] = param
        # Get router head parameters
        for name, param in self.router_head.parameters().items():
            params[f"router_head.{name}"] = param
        return params

    def __call__(
            self,
            input_ids: mx.array,
            attention_mask: Optional[mx.array] = None,
            labels: Optional[mx.array] = None,
            scaffold_mask: Optional[mx.array] = None,
            router_labels: Optional[mx.array] = None,
            training: bool = True,
    ) -> Dict[str, mx.array]:
        """
        Forward pass for training.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len] (optional)
            labels: Ground truth tokens for diffusion loss [batch, seq_len]
            scaffold_mask: Boolean mask for scaffold positions [batch, seq_len]
            router_labels: Ground truth labels for router [batch]
            training: Whether in training mode
        
        Returns:
            dict with 'loss', 'losses', 'router_logits', 'hidden_states'
        """
        # 1. Run Base LLM to get Context Embeddings (frozen, no gradient tracking)
        # In MLX, we simply don't include base_model params in optimizer
        hidden_states = self._get_hidden_states(input_ids, attention_mask)

        # 2. Router Forward
        router_logits = self.router_head(hidden_states)

        total_loss = mx.array(0.0)
        losses = {}

        # 3. Diffusion Loss (if applicable)
        if labels is not None and scaffold_mask is not None:
            num_scaffold = mx.sum(scaffold_mask)
            if num_scaffold > 0:
                diff_loss = self.diffusion_head.training_step(
                    tokens=labels,
                    hidden_states=hidden_states,
                    scaffold_mask=scaffold_mask,
                )
                total_loss = total_loss + diff_loss
                losses["diffusion"] = diff_loss

        # 4. Router Loss (if training router)
        if router_labels is not None:
            # Cross-entropy loss
            log_probs = log_softmax(router_logits, axis=-1)
            nll = -mx.take_along_axis(
                log_probs,
                mx.expand_dims(router_labels.astype(mx.int32), -1),
                axis=-1
            ).squeeze(-1)
            router_loss = mx.mean(nll)
            total_loss = total_loss + router_loss
            losses["router"] = router_loss

        has_loss = len(losses) > 0
        return {
            "loss": total_loss if has_loss else None,
            "losses": losses,
            "router_logits": router_logits,
            "hidden_states": hidden_states,  # Expose for debugging/inference
        }
