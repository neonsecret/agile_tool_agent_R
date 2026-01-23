"""
Schema-constrained diffusion head for function calling.

Combines:
- mdlm diffusion mechanics (forward/reverse diffusion with LogLinearNoise)
- Bidirectional attention for global constraint verification
- Schema scaffolding support (masks only scaffold positions)
- NULL token support for self-adaptive masking (variable-length fields)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from .noise_schedule import LogLinearNoise
from .attention_blocks import BidirectionalAttentionBlock
from .attention_blocks_optimized import BidirectionalAttentionBlockOptimized

logger = logging.getLogger(__name__)


def _create_sinusoidal_positions(max_len, dim):
    """Create sinusoidal position embeddings."""
    position = torch.arange(max_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    pe = torch.zeros(max_len, dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class PromptCrossAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, prompt_states, prompt_mask=None):
        key_padding_mask = None
        if prompt_mask is not None:
            key_padding_mask = ~prompt_mask
        attn_out, _ = self.cross_attn(
            x, prompt_states, prompt_states,
            key_padding_mask=key_padding_mask,
        )
        return self.norm(x + attn_out)


class SchemaDiffusionHead(nn.Module):
    def __init__(self, input_dim, vocab_size, hidden_dim=1024, num_layers=2, num_steps=4,
                 label_smoothing=0.1, use_bidirectional=True, num_heads=8,
                 null_loss_weight=0.3, null_prediction_penalty=0.0, entropy_weight=0.05,
                 use_optimized_attention=True, training_temperature=1.0,
                 repetition_penalty=0.0, max_seq_len=2048, use_attention_mask=False,
                 t_sampling="uniform", t_high_prob=0.0, t_high_range=None,
                 use_prompt_cross_attention=False, prompt_cross_attention_heads=None,
                 use_field_position=False, field_position_max_len=64):
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
            entropy_weight: Weight for entropy regularization (prevents token collapse)
            use_optimized_attention: If True, use SDPA-optimized attention (2-3x faster)
            training_temperature: Temperature for logits during training (>1.0 smooths, <1.0 sharpens)
            repetition_penalty: Penalty for consecutive identical token predictions
            max_seq_len: Maximum sequence length for position embeddings
        """
        super().__init__()
        self.num_steps = num_steps
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.label_smoothing = label_smoothing
        self.use_bidirectional = use_bidirectional
        self.null_loss_weight = null_loss_weight
        self.null_prediction_penalty = null_prediction_penalty
        self.entropy_weight = entropy_weight
        self.use_optimized_attention = use_optimized_attention
        self.training_temperature = training_temperature
        self.repetition_penalty = repetition_penalty
        self.use_attention_mask = use_attention_mask
        self.t_sampling = t_sampling
        self.t_high_prob = t_high_prob
        self.use_prompt_cross_attention = use_prompt_cross_attention
        self.use_field_position = use_field_position
        self.field_position_max_len = field_position_max_len

        if t_high_range is None:
            t_high_range = (0.8, 1.0)
        if isinstance(t_high_range, (list, tuple)) and len(t_high_range) == 2:
            high_min = float(t_high_range[0])
            high_max = float(t_high_range[1])
        else:
            high_min, high_max = 0.8, 1.0
        high_min = max(0.0, min(1.0, high_min))
        high_max = max(high_min, min(1.0, high_max))
        self.t_high_min = high_min
        self.t_high_max = high_max

        self.noise = LogLinearNoise()

        self.time_embed = nn.Embedding(num_steps + 1, hidden_dim)

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)

        # Sinusoidal position embeddings for length-aware predictions
        self.register_buffer(
            "position_embeddings",
            _create_sinusoidal_positions(max_seq_len, hidden_dim)
        )

        if use_bidirectional:
            attention_class = BidirectionalAttentionBlockOptimized if use_optimized_attention else BidirectionalAttentionBlock
            self.denoise_blocks = nn.ModuleList([
                attention_class(hidden_dim, num_heads=num_heads)
                for _ in range(num_layers)
            ])
        else:
            self.denoise_blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(num_layers)
            ])

        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        if self.use_prompt_cross_attention:
            if prompt_cross_attention_heads is None:
                prompt_cross_attention_heads = num_heads
            self.prompt_cross_attn = PromptCrossAttention(hidden_dim, prompt_cross_attention_heads)
        else:
            self.prompt_cross_attn = None
        if self.use_field_position:
            self.field_position_emb = nn.Embedding(field_position_max_len, hidden_dim)
        else:
            self.field_position_emb = None
        self.mask_token_id = None
        self.null_token_id = None

    def set_mask_token_id(self, mask_token_id):
        """Set the mask token ID (should be called during initialization)."""
        self.mask_token_id = mask_token_id

    def set_null_token_id(self, null_token_id):
        """Set the NULL token ID for self-adaptive masking."""
        self.null_token_id = null_token_id

    def sample_timesteps(self, batch_size, device):
        """Sample timesteps t in [0, 1] using configured strategy."""
        if self.t_sampling == "sqrt":
            return torch.rand(batch_size, device=device) ** 0.5
        if self.t_sampling == "mixture_high":
            t = torch.rand(batch_size, device=device)
            if self.t_high_prob > 0:
                high_mask = torch.rand(batch_size, device=device) < self.t_high_prob
                if high_mask.any():
                    high_samples = torch.rand(int(high_mask.sum().item()), device=device)
                    t[high_mask] = (
                        high_samples * (self.t_high_max - self.t_high_min) + self.t_high_min
                    )
            return t
        return torch.rand(batch_size, device=device)

    def _compute_field_positions(self, scaffold_mask: torch.Tensor) -> torch.Tensor:
        positions = torch.full_like(scaffold_mask, -1, dtype=torch.long)
        for batch_idx in range(scaffold_mask.size(0)):
            idxs = torch.nonzero(scaffold_mask[batch_idx], as_tuple=False).squeeze(-1)
            if idxs.numel() == 0:
                continue
            start = idxs[0].item()
            rel = 0
            for pos in idxs.tolist():
                if pos != start + rel:
                    start = pos
                    rel = 0
                positions[batch_idx, pos] = rel
                rel += 1
        return positions

    def _add_field_position_embeddings(self, x: torch.Tensor, scaffold_mask: torch.Tensor) -> torch.Tensor:
        if self.field_position_emb is None or scaffold_mask is None:
            return x
        positions = self._compute_field_positions(scaffold_mask)
        positions = positions.clamp(min=0, max=self.field_position_max_len - 1)
        field_emb = self.field_position_emb(positions)
        field_emb = field_emb * scaffold_mask.unsqueeze(-1).to(field_emb.dtype)
        return x + field_emb

    def forward_diffusion(self, tokens, scaffold_mask, t):
        """
        Add noise (mask tokens) based on timestep t.

        From mdlm diffusion.py q_xt method (lines 575-586).

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
        move_chance = 1 - torch.exp(-sigma_t)
        move_chance = move_chance.unsqueeze(-1)

        noisy_tokens = tokens.clone()
        mask_probs = torch.rand_like(tokens.float()) < move_chance
        mask_positions = scaffold_mask & mask_probs.bool()
        noisy_tokens[mask_positions] = self.mask_token_id

        # Ensure at least one masked position per sample when scaffold positions exist
        if scaffold_mask.any():
            mask_counts = mask_positions.sum(dim=1)
            scaffold_any = scaffold_mask.any(dim=1)
            needs_mask = (mask_counts == 0) & scaffold_any
            if needs_mask.any():
                rand = torch.rand_like(tokens.float())
                rand = rand.masked_fill(~scaffold_mask, -1.0)
                choice_idx = rand.argmax(dim=1)
                noisy_tokens[needs_mask, choice_idx[needs_mask]] = self.mask_token_id
                mask_positions[needs_mask, choice_idx[needs_mask]] = True

        return noisy_tokens, mask_positions

    def predict(self, hidden_states, current_tokens, t, attention_mask=None,
                scaffold_mask=None, prompt_hidden_states=None, prompt_mask=None):
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

        Returns:
            logits: [batch, seq_len, vocab_size] predictions
        """
        x = self._compute_hidden(
            hidden_states,
            current_tokens,
            t,
            attention_mask=attention_mask,
            scaffold_mask=scaffold_mask,
            prompt_hidden_states=prompt_hidden_states,
            prompt_mask=prompt_mask,
        )
        logits = self.output_proj(x)
        return logits

    def _compute_hidden(self, hidden_states, current_tokens, t, attention_mask=None,
                        scaffold_mask=None, prompt_hidden_states=None, prompt_mask=None):
        """Compute hidden representations without output projection (memory efficient)."""
        batch_size, seq_len = hidden_states.shape[:2]

        context = self.input_proj(hidden_states)
        safe_tokens = current_tokens.clone()
        safe_tokens[safe_tokens < 0] = 0
        token_emb = self.token_emb(safe_tokens)

        if isinstance(t, (int, float)):
            t_tensor = torch.full((batch_size,), t, device=hidden_states.device, dtype=torch.float)
        else:
            t_tensor = t.float()

        t_indices = (t_tensor * self.num_steps).long().clamp(0, self.num_steps)
        t_emb = self.time_embed(t_indices).unsqueeze(1)
        pos_emb = self.position_embeddings[:seq_len].unsqueeze(0).to(hidden_states.dtype)

        x = context + token_emb + t_emb + pos_emb
        x = self._add_field_position_embeddings(x, scaffold_mask)
        if self.prompt_cross_attn is not None:
            if prompt_hidden_states is None:
                prompt_hidden_states = hidden_states
            if prompt_hidden_states is hidden_states:
                prompt_context = context
            else:
                prompt_context = self.input_proj(prompt_hidden_states)
            x = self.prompt_cross_attn(x, prompt_context, prompt_mask=prompt_mask)

        if self.use_bidirectional:
            for block in self.denoise_blocks:
                x = block(x, attention_mask=attention_mask)
        else:
            for block in self.denoise_blocks:
                x = x + block(x)

        return x

    def training_step(self, tokens, hidden_states, scaffold_mask, attention_mask=None,
                      current_tokens=None, mask_positions=None, t=None,
                      prompt_hidden_states=None, prompt_mask=None):
        """
        Full training forward pass with diffusion loss.

        Memory-optimized: only computes output_proj on masked positions to avoid
        materializing full [batch, seq, vocab] tensor (~3GB with batch=6, seq=2048).

        Args:
            tokens: [batch, seq_len] clean tokens (labels, may contain -100)
            hidden_states: [batch, seq_len, input_dim] from base model
            scaffold_mask: [batch, seq_len] boolean mask

        Returns:
            loss: scalar tensor
        """
        batch_size = tokens.shape[0]

        valid_labels = tokens >= 0
        valid_scaffold_mask = scaffold_mask & valid_labels

        if valid_scaffold_mask.sum() == 0:
            return torch.tensor(0.0, device=tokens.device, requires_grad=True)

        if t is None:
            t = self.sample_timesteps(batch_size, tokens.device)

        noisy_tokens = None
        if current_tokens is None or mask_positions is None:
            noisy_tokens, mask_positions = self.forward_diffusion(
                tokens, scaffold_mask, t
            )
            if current_tokens is None:
                current_tokens = noisy_tokens
        if mask_positions is None:
            mask_positions = (current_tokens == self.mask_token_id) & scaffold_mask

        valid_mask_positions = mask_positions & valid_labels

        if valid_mask_positions.sum() == 0:
            return torch.tensor(0.0, device=tokens.device, requires_grad=True)

        # Memory optimization: compute hidden states, then output_proj only on masked positions
        x = self._compute_hidden(
            hidden_states,
            current_tokens,
            t,
            attention_mask=attention_mask,
            scaffold_mask=scaffold_mask,
            prompt_hidden_states=prompt_hidden_states,
            prompt_mask=prompt_mask,
        )
        active_hidden = x[valid_mask_positions]
        active_logits = self.output_proj(active_hidden)
        active_labels = tokens[valid_mask_positions]

        loss = self._compute_loss(active_logits, active_labels, t.mean())

        return loss

    def training_step_with_outputs(self, tokens, hidden_states, scaffold_mask, attention_mask=None,
                                   current_tokens=None, mask_positions=None, t=None,
                                   prompt_hidden_states=None, prompt_mask=None):
        """Training step that also returns predictions for metrics."""
        batch_size = tokens.shape[0]

        valid_labels = tokens >= 0
        valid_scaffold_mask = scaffold_mask & valid_labels

        if valid_scaffold_mask.sum() == 0:
            loss = torch.tensor(0.0, device=tokens.device, requires_grad=True)
            return {
                "loss": loss,
                "predictions": None,
                "mask_positions": None,
                "noisy_tokens": None,
                "t": None,
            }

        if t is None:
            t = self.sample_timesteps(batch_size, tokens.device)

        noisy_tokens = None
        if current_tokens is None or mask_positions is None:
            noisy_tokens, mask_positions = self.forward_diffusion(
                tokens, scaffold_mask, t
            )
            if current_tokens is None:
                current_tokens = noisy_tokens
        if mask_positions is None:
            mask_positions = (current_tokens == self.mask_token_id) & scaffold_mask

        valid_mask_positions = mask_positions & valid_labels
        if valid_mask_positions.sum() == 0:
            loss = torch.tensor(0.0, device=tokens.device, requires_grad=True)
            return {
                "loss": loss,
                "predictions": None,
                "mask_positions": mask_positions,
                "noisy_tokens": noisy_tokens,
                "t": t,
            }

        # Compute loss efficiently (only output_proj on masked positions)
        x = self._compute_hidden(
            hidden_states,
            current_tokens,
            t,
            attention_mask=attention_mask,
            scaffold_mask=scaffold_mask,
            prompt_hidden_states=prompt_hidden_states,
            prompt_mask=prompt_mask,
        )
        active_hidden = x[valid_mask_positions]
        active_logits = self.output_proj(active_hidden)
        active_labels = tokens[valid_mask_positions]
        loss = self._compute_loss(active_logits, active_labels, t.mean())

        predictions = torch.full_like(tokens, -100)
        predictions[valid_mask_positions] = active_logits.argmax(dim=-1)

        return {
            "loss": loss,
            "predictions": predictions,
            "mask_positions": mask_positions,
            "noisy_tokens": noisy_tokens if noisy_tokens is not None else current_tokens,
            "t": t,
        }

    def _compute_loss(self, active_logits, active_labels, t_mean=None):
        # Apply temperature scaling during training for exploration
        if self.training and self.training_temperature != 1.0:
            scaled_logits = active_logits / self.training_temperature
        else:
            scaled_logits = active_logits

        if self.null_token_id is not None and active_labels.numel() > 0:
            # Dynamic NULL weighting based on timestep
            # At high t (early diffusion): more NULLs expected, higher weight
            # At low t (late diffusion): fewer NULLs, lower weight
            if t_mean is not None:
                try:
                    t_scalar = t_mean.item()
                except (AttributeError, TypeError) as e:
                    logger.debug(f"Could not call .item() on t_mean, using float(): {e}")
                    t_scalar = float(t_mean)
                dynamic_null_weight = self.null_loss_weight * (0.3 + 0.7 * t_scalar)
            else:
                dynamic_null_weight = self.null_loss_weight

            sample_weights = torch.ones_like(active_labels, dtype=torch.float)
            null_mask = active_labels == self.null_token_id
            sample_weights[null_mask] = dynamic_null_weight

            loss_unreduced = F.cross_entropy(
                scaled_logits,
                active_labels,
                label_smoothing=self.label_smoothing,
                reduction="none",
            )
            loss = (loss_unreduced * sample_weights).sum() / sample_weights.sum()
        else:
            loss = F.cross_entropy(
                scaled_logits,
                active_labels,
                label_smoothing=self.label_smoothing,
            )

        # Use scaled logits for all probability-based computations
        probs = F.softmax(scaled_logits, dim=-1)

        # NULL prediction penalty
        if (self.null_token_id is not None
                and self.null_prediction_penalty > 0
                and active_logits.numel() > 0):
            null_probs = probs[:, self.null_token_id]
            non_null_mask = active_labels != self.null_token_id
            if non_null_mask.any():
                penalty = null_probs[non_null_mask].mean() * self.null_prediction_penalty
                loss = loss + penalty

        # Entropy regularization (prevents token collapse/repetition)
        if self.entropy_weight > 0 and active_logits.numel() > 0:
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
            loss = loss - self.entropy_weight * entropy

        # Repetition penalty intentionally disabled for masked-position logits.

        return loss

    def forward(self, hidden_states, current_tokens, step_ids, scaffold_mask=None):
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
        return self.predict(hidden_states, current_tokens, step_ids)
