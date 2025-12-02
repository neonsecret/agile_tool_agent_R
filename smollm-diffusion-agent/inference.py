"""
Self-adaptive generation loop for diffusion LLMs with schema scaffolding.

Based on: dLLM-CtrlGen/decoding/generator.py
Implements the S3 (Schema Scaffolding) denoising loop with top-K remasking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from tqdm import trange

import yaml

from model.hybrid_model import HybridSmolLM
from data.schema import SchemaTemplate, build_schema_template
from data.utils import resolve_mask_token, validate_mask_token_consistency


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@dataclass
class GenerationConfig:
    """Hyperparameters controlling the denoising schedule."""

    steps: int = 4
    temperature: float = 0.0
    cfg_scale: float = 0.0
    topk_remask: Optional[int] = None
    use_fp16: bool = False
    clear_cache: bool = True


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
    """Runs the S3 denoising loop with top-K remasking."""

    def __init__(
            self,
            model: HybridSmolLM,
            tokenizer: AutoTokenizer,
            device: torch.device,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def _build_messages(self, prompt: str) -> List[Dict[str, str]]:
        """Build chat template messages."""
        return [{"role": "user", "content": prompt}]

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        """Encode prompt using chat template."""
        chat_ids = self.tokenizer.apply_chat_template(
            self._build_messages(prompt),
            add_generation_prompt=True,
            tokenize=True,
        )
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

    def _forward(
            self,
            sequence: torch.Tensor,
            prompt_mask: torch.Tensor,
            template: SchemaTemplate,
            cfg_scale: float = 0.0,
    ) -> torch.Tensor:
        """Forward pass through the model."""
        attention_mask = torch.ones_like(sequence)

        with torch.no_grad():
            outputs = self.model.base_llm(
                input_ids=sequence,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            hidden_states = outputs.hidden_states[-1]
            
            # Convert hidden_states to match diffusion head dtype (bfloat16)
            diffusion_head_dtype = next(self.model.diffusion_head.parameters()).dtype
            hidden_states = hidden_states.to(dtype=diffusion_head_dtype)

        t = torch.zeros(sequence.shape[0], device=self.device)
        logits = self.model.diffusion_head.predict(
            hidden_states, sequence, t
        )

        return logits

    def generate(
            self,
            prompt: str,
            template: SchemaTemplate,
            config: Optional[GenerationConfig] = None,
            trace: bool = False,
            callback: Optional[Callable[[int, torch.Tensor, int], None]] = None,
    ) -> GenerationOutput:
        """
        Main inference loop combining AR + Scaffolding + Diffusion.

        Based on dLLM-CtrlGen generator.py lines 151-278.

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

        with torch.no_grad():
            prompt_ids = self._encode_prompt(prompt)
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

            for step in trange(cfg.steps, desc="Diffusion steps"):
                mask_positions.fill_(False)
                mask_positions[sequence == template.mask_token_id] = True
                mask_positions[:, :prompt_length] = False
                mask_indices = torch.nonzero(mask_positions[0], as_tuple=False).squeeze(-1)

                if mask_indices.numel() == 0:
                    executed_steps = step
                    break

                logits = self._forward(sequence, prompt_mask, template, cfg.cfg_scale)
                logits = _apply_gumbel_noise(logits, cfg.temperature)
                predictions = torch.argmax(logits, dim=-1)

                log_probs = F.log_softmax(logits[0, mask_indices], dim=-1)
                del logits

                mask_conf = log_probs.gather(
                    -1, predictions[0, mask_indices].unsqueeze(-1)
                ).squeeze(-1)
                del log_probs

                remaining = mask_indices.numel()
                remaining_steps = cfg.steps - step
                if remaining_steps <= 1:
                    k = remaining
                else:
                    k = min(budget, remaining)

                topk = torch.topk(mask_conf, k)
                del mask_conf

                selected = mask_indices[topk.indices]
                del mask_indices, topk

                sequence[0, selected] = predictions[0, selected]
                del predictions

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
                response_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            del sequence, prompt_mask, mask_positions

            if cfg.clear_cache:
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        return GenerationOutput(
            text=text,
            token_ids=response_tokens.tolist(),
            steps_executed=executed_steps,
            trace=generation_trace,
        )


def demo_inference():
    """Demo function showing how to use the generator."""
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    # Load config and tokenizer first
    config = load_config()
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B")

    # Resolve mask token
    data_cfg = config.get("data", {})
    mask_token_config = data_cfg.get("mask_token", None)
    mask_token_str, mask_token_id = resolve_mask_token(tokenizer, mask_token_config)

    print(f"Using mask token: {mask_token_str} (ID: {mask_token_id})")
    print(f"Tokenizer vocab size: {len(tokenizer)}")

    # Initialize model with correct vocab size
    model = HybridSmolLM(vocab_size=len(tokenizer))
    model.to(device)
    model.eval()
    model.diffusion_head.set_mask_token_id(mask_token_id)

    generator = FunctionCallGenerator(model, tokenizer, device)

    prompt = "What's the weather in London?"

    fields = [
        ("location", 10),
        ("unit", 3),
    ]

    template = build_schema_template(
        tokenizer=tokenizer,
        fields=fields,
        mask_token=mask_token_str,
        include_codeblock=False
    )
    
    validate_mask_token_consistency(
        model.diffusion_head.mask_token_id,
        template.mask_token_id,
        context=" in demo_inference()"
    )

    print(f"Scaffold template: {template.text}")

    config = GenerationConfig(
        steps=4,
        temperature=0.0,
        cfg_scale=0.0,
    )

    output = generator.generate(
        prompt=prompt,
        template=template,
        config=config,
        trace=True
    )

    print(f"\nGenerated text: {output.text}")
    print(f"Steps executed: {output.steps_executed}")


if __name__ == "__main__":
    demo_inference()
