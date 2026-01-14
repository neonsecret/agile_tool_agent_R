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

from .noise_schedule import LogLinearNoise


class BidirectionalAttentionBlock(nn.Module):
    """Transformer encoder block with bidirectional (non-causal) self-attention.
    
    Unlike autoregressive attention, this allows each position to attend to all
    other positions, enabling global constraint verification for structured output.
    """
    
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
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
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, x, attention_mask=None):
        """
        Args:
            x: [batch, seq_len, hidden_dim]
            attention_mask: Optional [batch, seq_len] boolean mask (True = attend)
        
        Returns:
            [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Pre-norm
        normed = self.norm1(x)
        
        # Compute Q, K, V
        q = self.q_proj(normed).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(normed).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(normed).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Bidirectional attention (no causal mask)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask for broadcasting: [batch, 1, 1, seq_len]
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(~mask, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        attn_output = self.out_proj(attn_output)
        
        # Residual connection
        x = x + self.dropout(attn_output)
        
        # MLP with pre-norm and residual
        x = x + self.dropout(self.mlp(self.norm2(x)))
        
        return x


class SchemaDiffusionHead(nn.Module):
    def __init__(self, input_dim, vocab_size, hidden_dim=1024, num_layers=2, num_steps=4,
                 label_smoothing=0.1, use_bidirectional=True, num_heads=8):
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
            self.denoise_blocks = nn.ModuleList([
                BidirectionalAttentionBlock(hidden_dim, num_heads=num_heads)
                for _ in range(num_layers)
            ])
        else:
            # Original residual MLP blocks (faster but no global attention)
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
        context = self.input_proj(hidden_states)
        safe_tokens = current_tokens.clone()
        safe_tokens[safe_tokens < 0] = 0
        token_emb = self.token_emb(safe_tokens)

        # Convert timestep to tensor if needed (keep as continuous float)
        if isinstance(t, (int, float)):
            t_tensor = torch.full((hidden_states.shape[0],), t, device=hidden_states.device, dtype=torch.float)
        else:
            t_tensor = t.float()

        # Convert continuous t ∈ [0, 1] to discrete timestep indices for embedding lookup
        # Scale t to [0, num_steps] range and clamp
        t_indices = (t_tensor * self.num_steps).long().clamp(0, self.num_steps)

        t_emb = self.time_embed(t_indices).unsqueeze(1)

        x = context + token_emb + t_emb

        if self.use_bidirectional:
            # Bidirectional attention blocks
            for block in self.denoise_blocks:
                x = block(x, attention_mask=attention_mask)
        else:
            # Original residual MLP blocks
            for block in self.denoise_blocks:
                x = x + block(x)

        logits = self.output_proj(x)
        return logits

    def training_step(self, tokens, hidden_states, scaffold_mask):
        """
        Full training forward pass with diffusion loss.

        From mdlm diffusion.py _forward_pass_diffusion.
        Uses label smoothing to reduce overconfidence and improve generalization.

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
        
        loss = self._compute_loss(active_logits, active_labels)

        return loss

    def training_step_with_outputs(self, tokens, hidden_states, scaffold_mask):
        batch_size = tokens.shape[0]

        valid_labels = tokens >= 0
        valid_scaffold_mask = scaffold_mask & valid_labels

        if valid_scaffold_mask.sum() == 0:
            loss = torch.tensor(0.0, device=tokens.device, requires_grad=True)
            return {
                "loss": loss,
                "logits": None,
                "mask_positions": None,
                "noisy_tokens": None,
                "t": None,
            }

        t = torch.rand(batch_size, device=tokens.device)

        noisy_tokens, mask_positions = self.forward_diffusion(
            tokens, scaffold_mask, t
        )

        logits = self.predict(hidden_states, noisy_tokens, t)

        valid_mask_positions = mask_positions & valid_labels
        if valid_mask_positions.sum() == 0:
            loss = torch.tensor(0.0, device=tokens.device, requires_grad=True)
            return {
                "loss": loss,
                "logits": logits,
                "mask_positions": mask_positions,
                "noisy_tokens": noisy_tokens,
                "t": t,
            }

        active_logits = logits[valid_mask_positions]
        active_labels = tokens[valid_mask_positions]
        loss = self._compute_loss(active_logits, active_labels)

        return {
            "loss": loss,
            "logits": logits,
            "mask_positions": mask_positions,
            "noisy_tokens": noisy_tokens,
            "t": t,
        }

    def _compute_loss(self, active_logits, active_labels):
        if self.null_token_id is not None and active_labels.numel() > 0:
            sample_weights = torch.ones_like(active_labels, dtype=torch.float)
            null_mask = active_labels == self.null_token_id
            sample_weights[null_mask] = 0.3

            loss_unreduced = F.cross_entropy(
                active_logits,
                active_labels,
                label_smoothing=self.label_smoothing,
                reduction="none",
            )
            return (loss_unreduced * sample_weights).sum() / sample_weights.sum()

        return F.cross_entropy(
            active_logits,
            active_labels,
            label_smoothing=self.label_smoothing,
        )

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
