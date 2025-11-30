"""
Schema-constrained diffusion head for function calling.

Combines:
- mdlm diffusion mechanics (forward/reverse diffusion with LogLinearNoise)
- Lightweight architecture (residual blocks instead of full transformer)
- Schema scaffolding support (masks only scaffold positions)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .noise_schedule import LogLinearNoise


class SchemaDiffusionHead(nn.Module):
    def __init__(self, input_dim, vocab_size, hidden_dim=1024, num_layers=2, num_steps=4):
        """
        Args:
            input_dim: Dimension of hidden states from base model
            vocab_size: Size of vocabulary
            hidden_dim: Hidden dimension for diffusion head
            num_layers: Number of residual blocks
            num_steps: Number of diffusion steps for training
        """
        super().__init__()
        self.num_steps = num_steps
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.noise = LogLinearNoise()

        self.time_embed = nn.Embedding(num_steps + 1, hidden_dim)

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)

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

    def set_mask_token_id(self, mask_token_id):
        """Set the mask token ID (should be called during initialization)."""
        self.mask_token_id = mask_token_id

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

    def predict(self, hidden_states, current_tokens, t):
        """
        Predict original tokens from noisy version.

        Args:
            hidden_states: [batch, seq_len, input_dim] from base model
            current_tokens: [batch, seq_len] current noisy state
            t: [batch] or scalar timestep

        Returns:
            logits: [batch, seq_len, vocab_size] predictions
        """
        context = self.input_proj(hidden_states)
        safe_tokens = current_tokens.clone()
        safe_tokens[safe_tokens < 0] = 0
        token_emb = self.token_emb(safe_tokens)

        if isinstance(t, (int, float)):
            t = torch.full((hidden_states.shape[0],), t, device=hidden_states.device, dtype=torch.long)
        elif t.dtype == torch.float:
            t = (t * self.num_steps).long().clamp(0, self.num_steps)

        t_emb = self.time_embed(t).unsqueeze(1)

        x = context + token_emb + t_emb

        for block in self.denoise_blocks:
            x = x + block(x)

        logits = self.output_proj(x)
        return logits

    def training_step(self, tokens, hidden_states, scaffold_mask):
        """
        Full training forward pass with diffusion loss.

        From mdlm diffusion.py _forward_pass_diffusion.

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

        logits = self.predict(hidden_states, noisy_tokens, t)

        valid_mask_positions = mask_positions & valid_labels

        if valid_mask_positions.sum() == 0:
            return torch.tensor(0.0, device=tokens.device, requires_grad=True)

        active_logits = logits[valid_mask_positions]
        active_labels = tokens[valid_mask_positions]
        loss = F.cross_entropy(active_logits, active_labels)

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
