import torch
from .model.hybrid_model import HybridSmolLM
from .data.schema_builder import SchemaTemplate
from .scheduler import DiscreteDiffusionScheduler


class DiffusionInference:
    """
    Inference pipeline for Hybrid Diffusion Agent.
    Implements Top-K Remasking strategy.
    """

    def __init__(self, model: HybridSmolLM, device: torch.device):
        self.model = model
        self.device = device
        self.scheduler = DiscreteDiffusionScheduler()

    def generate(self,
                 prompt_ids: torch.Tensor,
                 template: SchemaTemplate,
                 steps: int = 4,
                 temperature: float = 0.0,
                 initial_guess: torch.Tensor = None):

        self.model.eval()

        # 1. Prepare Input Sequence
        template_tokens = torch.tensor(template.tokens, device=self.device).unsqueeze(0)  # [1, seq]

        # If prompt_ids is None (e.g. continuation), handle it
        if prompt_ids is not None:
            full_input = torch.cat([prompt_ids, template_tokens], dim=1)
            prompt_len = prompt_ids.shape[1]
        else:
            full_input = template_tokens
            prompt_len = 0

        # 2. Base Model Forward (Get Context)
        # We only need to do this ONCE because the context (prompt) is frozen.
        # The diffusion head takes this frozen context + variable noise.
        with torch.no_grad():
            outputs = self.model.base_llm(full_input, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]  # [1, total_seq, hidden]

        # 3. Identify Mask Positions
        mask_token_id = template.mask_token_id
        # Create mask map [1, total_seq]
        # We strictly limit diffusion to the template part that was initialized as masks
        scaffold_mask = torch.zeros_like(full_input, dtype=torch.bool)
        scaffold_mask[:, prompt_len:] = (template_tokens == mask_token_id)

        mask_indices = torch.nonzero(scaffold_mask, as_tuple=True)[1]
        if len(mask_indices) == 0:
            return full_input

            # 4. Initialize State
        # Start from pure noise (all masks) or warm start
        current_tokens = full_input.clone()

        if initial_guess is not None:
            # Warm Start: D2F / Asymmetric Distillation inspiration
            # If we have a guess, we fill it in.
            # But we must ensure it aligns length-wise.
            # For now, simplistic overwrite if lengths match.
            if initial_guess.shape == current_tokens.shape:
                # We only copy into the scaffold region
                current_tokens[:, prompt_len:] = initial_guess[:, prompt_len:]
                # We might want to start with some noise (partial masking) if guess is imperfect?
                # For simplicity, let's assume warm start means "start at t=T/2" or similar.
                # But `current_tokens` here represents x_t.
                pass

        # 5. Diffusion Loop (Reverse Process: T -> 0)
        timesteps = self.scheduler.get_timesteps(steps, device=self.device)

        for i, t in enumerate(timesteps):
            # Determine previous timestep (next in schedule)
            prev_t = timesteps[i + 1] if i < len(timesteps) - 1 else 0

            # t is a tensor [1]
            t_batch = t.unsqueeze(0).expand(full_input.shape[0])

            with torch.no_grad():
                # Head Forward
                logits = self.model.diffusion_head(
                    hidden_states,
                    current_tokens,
                    t_batch,
                    scaffold_mask
                )

                # Top-K Remasking Step
                # Updates current_tokens in-place (or returns new)
                current_tokens = self.scheduler.step_remask(
                    current_tokens,
                    logits,
                    t.item(),
                    prev_t.item(),
                    mask_token_id,
                    scaffold_mask
                )

        return current_tokens
