import torch
import numpy as np


class DiscreteDiffusionScheduler:
    """
    Discrete Noise Scheduler for Masked Diffusion.
    Inspired by MDLM (Masked Discrete Diffusion) and LaDiR's scheduling concepts.
    
    Handles the logic of:
    1. How many tokens to mask at each step (schedule).
    2. Which tokens to mask (remasking strategy).
    """

    def __init__(self, num_train_timesteps=1000, num_inference_steps=4):
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps

    def get_timesteps(self, num_steps=None, device="cpu"):
        """
        Returns a schedule of timesteps for inference.
        LaDiR uses Flow Matching (1 -> 0). MDLM uses T -> 0.
        We map [0, num_train_timesteps] to [0, num_inference_steps].
        """
        steps = num_steps or self.num_inference_steps
        # Linear schedule from T to 0
        # e.g. 4 steps: [750, 500, 250, 0] if T=1000
        timesteps = np.linspace(self.num_train_timesteps - 1, 0, steps, dtype=int)
        return torch.from_numpy(timesteps).to(device)

    def add_noise(self, original_samples, mask_token_id, t):
        """
        Training forward process: Corrupts the input by masking tokens based on time t.
        Higher t = more noise (more masks).
        
        Args:
            original_samples: [batch, seq_len] (Clean input)
            mask_token_id: int
            t: [batch] (Timestep 0 to T)
            
        Returns:
            noisy_samples: [batch, seq_len]
            mask: [batch, seq_len] (True where masked)
        """
        # Simple linear mask ratio: t / T
        # t=0 -> 0% masked (clean)
        # t=T -> 100% masked (pure noise)

        B, L = original_samples.shape
        ratio = t.float() / self.num_train_timesteps

        # Create random mask based on ratio
        # For each sample in batch, we mask `ratio` fraction of tokens

        # Vectorized implementation
        # Create random matrix [B, L]
        rand = torch.rand(B, L, device=original_samples.device)

        # Expand ratio to [B, 1]
        mask_threshold = ratio.unsqueeze(1)

        # Mask where random < threshold
        mask = rand < mask_threshold

        noisy_samples = original_samples.clone()
        noisy_samples[mask] = mask_token_id

        return noisy_samples, mask

    def step_remask(self, current_tokens, logits, t, prev_t, mask_token_id, scaffold_mask):
        """
        Inference reverse process: Top-K Re-masking.
        Inspired by Discrete Diffusion Forcing / LaDiR.
        
        Args:
            current_tokens: [batch, seq_len] (Current state x_t)
            logits: [batch, seq_len, vocab] (Model prediction)
            t: int (Current timestep)
            prev_t: int (Next timestep, usually t-1 or smaller)
            scaffold_mask: [batch, seq_len] (Where we are ALLOWED to mask/generate)
            
        Returns:
            next_tokens: [batch, seq_len]
        """
        # 1. Predict x_0 (Clean tokens) from logits
        probs = torch.softmax(logits, dim=-1)
        pred_ids = torch.argmax(probs, dim=-1)  # Hard prediction

        # Confidence score (probability of the predicted token)
        confidence = torch.gather(probs, -1, pred_ids.unsqueeze(-1)).squeeze(-1)

        # 2. Determine how many tokens to KEEP based on schedule
        # Linear schedule: we want to keep (1 - ratio) tokens
        # next_ratio = prev_t / T
        ratio = prev_t / self.num_train_timesteps

        # We strictly respect the scaffold_mask (only touch those positions)
        # For each batch, find indices where scaffold_mask is True

        next_tokens = current_tokens.clone()

        batch_size = current_tokens.shape[0]

        for b in range(batch_size):
            mask_indices = torch.nonzero(scaffold_mask[b], as_tuple=True)[0]
            num_mask_slots = len(mask_indices)

            if num_mask_slots == 0:
                continue

            # Calculate how many items should be masked at next step
            # If prev_t=0, num_to_mask = 0 (Fully clean)
            num_to_mask = int(num_mask_slots * ratio)

            # We want to MASK the `num_to_mask` tokens with the LOWEST confidence
            # Conversely, we KEEP the `num_mask_slots - num_to_mask` tokens with HIGHEST confidence

            # Get confidence at mask positions
            mask_confidences = confidence[b, mask_indices]

            # Sort confidences (ascending)
            # The lowest confidence ones should remain masked (or be re-masked)
            # Actually, we take the model's prediction `pred_ids` as the "candidate" x_0
            # And we revert some of them back to <MASK>

            # Fill all positions with predicted IDs first (Full Denoising)
            next_tokens[b, mask_indices] = pred_ids[b, mask_indices]

            if num_to_mask > 0:
                # Find indices of the lowest confidence predictions
                # argsort gives indices into `mask_indices`
                sorted_indices = torch.argsort(mask_confidences)  # Ascending: low -> high

                # The first `num_to_mask` are the ones we are least confident about
                indices_to_remask_local = sorted_indices[:num_to_mask]
                indices_to_remask_global = mask_indices[indices_to_remask_local]

                # Re-mask them
                next_tokens[b, indices_to_remask_global] = mask_token_id

        return next_tokens
