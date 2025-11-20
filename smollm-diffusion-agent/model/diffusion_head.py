import torch
import torch.nn as nn


class SchemaDiffusionHead(nn.Module):
    def __init__(self, input_dim, vocab_size, hidden_dim=1024, num_layers=2, num_heads=8, num_steps=4):
        super().__init__()
        self.num_steps = num_steps
        self.hidden_dim = hidden_dim

        # 1. Time Embedding
        self.time_embed = nn.Embedding(num_steps, hidden_dim)

        # 2. Input Projection (Base Model Hidden -> Head Hidden)
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # 3. Current Token Embedding (The "Noisy" Tokens we are refining)
        # We need a separate embedding layer for the diffusion process
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)

        # 4. Lightweight Transformer Encoder
        # Allows tokens in the scaffold to attend to each other (co-reference)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

        # 5. Output Projection
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, hidden_states, current_tokens, step_ids, scaffold_mask=None):
        """
        hidden_states: [batch, seq_len, input_dim] (From Frozen Base Model)
        current_tokens: [batch, seq_len] (Current state of generation/noise)
        step_ids: [batch] (Timestep)
        scaffold_mask: [batch, seq_len] (Boolean: True if position is part of the scaffold/mask)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # 1. Embed Inputs
        # Project base context
        context_emb = self.input_proj(hidden_states)  # [batch, seq, hidden]

        # Embed current tokens
        token_emb = self.token_emb(current_tokens)  # [batch, seq, hidden]

        # Time embedding
        t_emb = self.time_embed(step_ids).unsqueeze(1)  # [batch, 1, hidden]

        # 2. Combine
        # We combine Context + Current State + Time
        # Logic: "Given this context (base), and this current guess (token), at this time (t), what is the real token?"
        x = context_emb + token_emb + t_emb

        # 3. Apply Transformer
        # We only want to apply self-attention where scaffold_mask is True?
        # Or we let the whole sequence attend?
        # Better to let whole sequence attend, but current_tokens for non-scaffold parts are just the fixed prompt tokens.
        # Actually, current_tokens input should ideally have the prompt tokens correct, and mask tokens noisy.

        # Create attention mask for Transformer?
        # Transformers usually handle full sequence.
        # But we might want to mask out padding.
        src_key_padding_mask = None
        # If we had a padding mask passed in, we would use it.

        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # 4. Predict
        logits = self.output_proj(x)

        return logits
