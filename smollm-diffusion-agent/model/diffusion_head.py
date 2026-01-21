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


class SchemaDiffusionHead(nn.Module):
    def __init__(self, input_dim, vocab_size, hidden_dim=1024, num_layers=2, num_steps=4,
                 label_smoothing=0.1, use_bidirectional=True, num_heads=8,
                 null_loss_weight=0.3, null_prediction_penalty=0.0, entropy_weight=0.05,
                 use_optimized_attention=True, training_temperature=1.0,
                 repetition_penalty=0.0, max_seq_len=2048):
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
        self.mask_token_id = None
        self.null_token_id = None

    def set_mask_token_id(self, mask_token_id):
        """Set the mask token ID (should be called during initialization)."""
        self.mask_token_id = mask_token_id

    def set_null_token_id(self, null_token_id):
        """Set the NULL token ID for self-adaptive masking."""
        self.null_token_id = null_token_id

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

        return noisy_tokens, mask_positions

    def predict(self, hidden_states, current_tokens, t, attention_mask=None):
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
        batch_size, seq_len = hidden_states.shape[:2]

        context = self.input_proj(hidden_states)
        safe_tokens = current_tokens.clone()
        safe_tokens[safe_tokens < 0] = 0
        token_emb = self.token_emb(safe_tokens)

        # Convert timestep to tensor if needed (keep as continuous float)
        if isinstance(t, (int, float)):
            t_tensor = torch.full((batch_size,), t, device=hidden_states.device, dtype=torch.float)
        else:
            t_tensor = t.float()

        # Convert continuous t ∈ [0, 1] to discrete timestep indices for embedding lookup
        # Scale t to [0, num_steps] range and clamp
        t_indices = (t_tensor * self.num_steps).long().clamp(0, self.num_steps)

        t_emb = self.time_embed(t_indices).unsqueeze(1)

        # Add sinusoidal position embeddings for length-aware predictions
        pos_emb = self.position_embeddings[:seq_len].unsqueeze(0).to(hidden_states.dtype)

        x = context + token_emb + t_emb + pos_emb

        if self.use_bidirectional:
            for block in self.denoise_blocks:
                x = block(x, attention_mask=attention_mask)
        else:
            for block in self.denoise_blocks:
                x = x + block(x)

        logits = self.output_proj(x)
        return logits

    def _compute_hidden(self, hidden_states, current_tokens, t):
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

        if self.use_bidirectional:
            for block in self.denoise_blocks:
                x = block(x, attention_mask=None)
        else:
            for block in self.denoise_blocks:
                x = x + block(x)

        return x

    def training_step(self, tokens, hidden_states, scaffold_mask):
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

        t = torch.rand(batch_size, device=tokens.device)

        noisy_tokens, mask_positions = self.forward_diffusion(
            tokens, scaffold_mask, t
        )

        valid_mask_positions = mask_positions & valid_labels

        if valid_mask_positions.sum() == 0:
            return torch.tensor(0.0, device=tokens.device, requires_grad=True)

        # Memory optimization: compute hidden states, then output_proj only on masked positions
        x = self._compute_hidden(hidden_states, noisy_tokens, t)
        active_hidden = x[valid_mask_positions]
        active_logits = self.output_proj(active_hidden)
        active_labels = tokens[valid_mask_positions]

        loss = self._compute_loss(active_logits, active_labels, t.mean())

        return loss

    def training_step_with_outputs(self, tokens, hidden_states, scaffold_mask):
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

        t = torch.rand(batch_size, device=tokens.device)

        noisy_tokens, mask_positions = self.forward_diffusion(
            tokens, scaffold_mask, t
        )

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
        x = self._compute_hidden(hidden_states, noisy_tokens, t)
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
            "noisy_tokens": noisy_tokens,
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

        # Repetition penalty: penalize consecutive identical predictions
        if self.repetition_penalty > 0 and active_logits.numel() > 1:
            predictions = active_logits.argmax(dim=-1)
            consecutive_same = (predictions[1:] == predictions[:-1]).float()
            if consecutive_same.numel() > 0:
                rep_penalty = consecutive_same.mean() * self.repetition_penalty
                loss = loss + rep_penalty

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
