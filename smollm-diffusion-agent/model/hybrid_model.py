import logging

logger = logging.getLogger(__name__)

try:
    import unsloth
    from unsloth import FastLanguageModel

    UNSLOTH_AVAILABLE = True
except ImportError as e:
    UNSLOTH_AVAILABLE = False
    unsloth = None
    FastLanguageModel = None
    logger.debug(f"unsloth not available: {e}")

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import sys
import os
import platform
from .diffusion_head import SchemaDiffusionHead

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.device_utils import get_device, get_device_map_for_quantization


class HybridSmolLM(nn.Module):
    def __init__(self, base_model_id="HuggingFaceTB/SmolLM3-3B", load_in_4bit=False,
                 diffusion_config=None, vocab_size=None, use_unsloth=None,
                 max_seq_length=2048, enable_unsloth_inference_opt=True,
                 device: torch.device | None = None,
                 use_flash_attention=True, use_gradient_checkpointing=False,
                 use_better_transformer=False,
                 unsloth_use_gradient_checkpointing="unsloth",
                 unsloth_rope_scaling=None):
        super().__init__()

        if diffusion_config is None:
            diffusion_config = {}

        device = device or get_device()
        self.base_llm = None
        self.use_unsloth = False
        self.use_flash_attention = use_flash_attention
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_better_transformer = use_better_transformer
        self.unsloth_use_gradient_checkpointing = unsloth_use_gradient_checkpointing
        self.unsloth_rope_scaling = unsloth_rope_scaling

        self._init_torch_model(base_model_id, load_in_4bit, device, use_unsloth, max_seq_length,
                               enable_unsloth_inference_opt)
        hidden_size = self.base_llm.config.hidden_size
        if vocab_size is None:
            vocab_size = self.base_llm.config.vocab_size

        hidden_dim = diffusion_config.get("hidden_dim", 1024)
        num_layers = diffusion_config.get("num_layers", 2)
        num_steps = diffusion_config.get("num_steps", 4)
        label_smoothing = diffusion_config.get("label_smoothing", 0.1)
        use_bidirectional = diffusion_config.get("use_bidirectional", True)
        num_heads = diffusion_config.get("num_heads", 8)
        null_loss_weight = diffusion_config.get("null_loss_weight", 0.3)
        null_prediction_penalty = diffusion_config.get("null_prediction_penalty", 0.0)
        entropy_weight = diffusion_config.get("entropy_weight", 0.05)
        use_optimized_attention = diffusion_config.get("use_optimized_attention", True)
        training_temperature = diffusion_config.get("training_temperature", 1.0)
        repetition_penalty = diffusion_config.get("repetition_penalty", 0.0)
        use_attention_mask = diffusion_config.get("use_attention_mask", False)
        t_sampling = diffusion_config.get("t_sampling", "uniform")
        t_high_prob = diffusion_config.get("t_high_prob", 0.0)
        t_high_range = diffusion_config.get("t_high_range", None)
        init_vocab_from_base = diffusion_config.get("init_vocab_from_base", True)
        use_prompt_cross_attention = diffusion_config.get("use_prompt_cross_attention", False)
        prompt_cross_attention_heads = diffusion_config.get("prompt_cross_attention_heads", None)
        use_field_position = diffusion_config.get("use_field_position", False)
        field_position_max_len = diffusion_config.get("field_position_max_len", 64)

        self.diffusion_head = SchemaDiffusionHead(
            input_dim=hidden_size,
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_steps=num_steps,
            label_smoothing=label_smoothing,
            use_bidirectional=use_bidirectional,
            num_heads=num_heads,
            null_loss_weight=null_loss_weight,
            null_prediction_penalty=null_prediction_penalty,
            entropy_weight=entropy_weight,
            use_optimized_attention=use_optimized_attention,
            training_temperature=training_temperature,
            repetition_penalty=repetition_penalty,
            max_seq_len=max_seq_length,
            use_attention_mask=use_attention_mask,
            t_sampling=t_sampling,
            t_high_prob=t_high_prob,
            t_high_range=t_high_range,
            use_prompt_cross_attention=use_prompt_cross_attention,
            prompt_cross_attention_heads=prompt_cross_attention_heads,
            use_field_position=use_field_position,
            field_position_max_len=field_position_max_len,
        )

        self.diffusion_head = self.diffusion_head.to(device=device, dtype=torch.bfloat16)
        self._maybe_init_vocab_from_base(init_vocab_from_base)

    def _init_torch_model(self, base_model_id, load_in_4bit, device, use_unsloth=None, max_seq_length=2048,
                          enable_unsloth_inference_opt=True):
        """Initialize PyTorch model (supports CUDA with quantization, MPS, and CPU)."""
        cuda_available = torch.cuda.is_available()
        
        # Check if we're in distributed training mode
        is_distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if use_unsloth is None:
            use_unsloth = cuda_available and not is_distributed

        if use_unsloth and not cuda_available:
            print("Warning: unsloth requires CUDA, disabling on non-CUDA device")
            use_unsloth = False

        if use_unsloth and is_distributed:
            print("Warning: unsloth not stable with DDP, disabling in multi-GPU mode")
            use_unsloth = False

        if load_in_4bit and not cuda_available:
            print("Warning: 4-bit quantization requires CUDA, falling back to bfloat16")
            load_in_4bit = False

        if use_unsloth and self.use_flash_attention and cuda_available:
            print("Warning: unsloth has built-in optimizations, disabling FlashAttention to avoid conflicts")
            self.use_flash_attention = False

        if use_unsloth and self.use_gradient_checkpointing and cuda_available:
            print("Warning: unsloth manages memory efficiently, disabling gradient checkpointing")
            self.use_gradient_checkpointing = False

        if use_unsloth and UNSLOTH_AVAILABLE:
            print(f"Loading model with unsloth on CUDA (max_seq_length={max_seq_length})")

            unsloth_kwargs = {
                "model_name": base_model_id,
                "max_seq_length": max_seq_length,
                "dtype": None if load_in_4bit else torch.bfloat16,
                "load_in_4bit": load_in_4bit,
                "load_in_8bit": False,
            }

            if self.unsloth_use_gradient_checkpointing is not None:
                unsloth_kwargs["use_gradient_checkpointing"] = self.unsloth_use_gradient_checkpointing
                if self.unsloth_use_gradient_checkpointing == "unsloth":
                    print("  Using unsloth gradient checkpointing (offloads activations to RAM, saves VRAM)")

            if self.unsloth_rope_scaling is not None:
                unsloth_kwargs["rope_scaling"] = self.unsloth_rope_scaling
                print(f"  RoPE scaling: {self.unsloth_rope_scaling}")

            self.base_llm, _ = FastLanguageModel.from_pretrained(**unsloth_kwargs)

            if enable_unsloth_inference_opt:
                FastLanguageModel.for_inference(self.base_llm)
                print("Unsloth inference optimizations enabled (2x faster)")
            self.use_unsloth = True
        elif use_unsloth and not UNSLOTH_AVAILABLE:
            print("Warning: unsloth requested but not installed, using standard model")
            use_unsloth = False

        if not use_unsloth:
            kwargs = {
                "torch_dtype": torch.bfloat16,
            }

            if self.use_flash_attention and device.type == "cuda":
                kwargs["attn_implementation"] = "flash_attention_2"
                print("Enabling FlashAttention-2 for base model")

            if device.type == "cuda":
                # In DDP mode, use device directly (Accelerate will handle placement)
                # device_map should NOT be "auto" or dict in DDP, let Accelerate handle it
                if is_distributed:
                    print(f"Loading model for DDP on rank {local_rank} (device: {device})")
                    if load_in_4bit:
                        print("  Using 4-bit quantization with DDP")
                        kwargs["quantization_config"] = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.bfloat16,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4"
                        )
                        # For DDP with quantization: no device_map, Accelerate will move to correct device
                        # bitsandbytes will automatically use the CUDA device set by torch.cuda.set_device()
                    else:
                        print("  Using bfloat16 (no quantization) with DDP")
                        # No device_map for DDP, let Accelerate handle device placement
                else:
                    # Single GPU mode: use explicit device_map
                    device_index = device.index if device.index is not None else 0
                    if load_in_4bit:
                        print("Loading model with 4-bit quantization on CUDA")
                        kwargs["quantization_config"] = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.bfloat16,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4"
                        )
                        kwargs["device_map"] = get_device_map_for_quantization(device)
                    else:
                        print("Loading model in bfloat16 on CUDA")
                        kwargs["device_map"] = {"": device_index}
            elif device.type == "mps":
                print("Loading model in bfloat16 on MPS (Apple Silicon)")
                kwargs["device_map"] = "auto"
            else:
                print("Loading model in bfloat16 on CPU")
                kwargs["device_map"] = "auto"

            self.base_llm = AutoModelForCausalLM.from_pretrained(base_model_id, **kwargs)
            self.use_unsloth = False

            if self.use_gradient_checkpointing:
                try:
                    self.base_llm.gradient_checkpointing_enable()
                    logger.info("Gradient checkpointing enabled for base model (saves memory)")
                except AttributeError as e:
                    logger.warning(f"Model does not support gradient_checkpointing_enable: {e}")

            if self.use_better_transformer and device.type == "cuda" and not load_in_4bit:
                try:
                    from optimum.bettertransformer import BetterTransformer
                    self.base_llm = BetterTransformer.transform(self.base_llm)
                    logger.info("BetterTransformer enabled for base model")
                except ImportError as e:
                    logger.warning(f"optimum not installed, skipping BetterTransformer (pip install optimum): {e}")
                except Exception as e:
                    logger.warning(f"Could not enable BetterTransformer: {e}", exc_info=True)

        for param in self.base_llm.parameters():
            param.requires_grad = False

    def _maybe_init_vocab_from_base(self, init_vocab_from_base: bool) -> None:
        if not init_vocab_from_base:
            return

        input_emb = self.base_llm.get_input_embeddings()
        output_emb = self.base_llm.get_output_embeddings()
        if input_emb is None or output_emb is None:
            logger.warning("Skipping vocab init: base model embeddings unavailable.")
            return

        if not hasattr(input_emb, "weight") or not hasattr(output_emb, "weight"):
            logger.warning("Skipping vocab init: embedding weights not accessible.")
            return

        if input_emb.weight.shape != self.diffusion_head.token_emb.weight.shape:
            logger.warning(
                "Skipping vocab init: input embedding shape %s != diffusion token_emb shape %s.",
                tuple(input_emb.weight.shape),
                tuple(self.diffusion_head.token_emb.weight.shape),
            )
            return

        if output_emb.weight.shape != self.diffusion_head.output_proj.weight.shape:
            logger.warning(
                "Skipping vocab init: output embedding shape %s != diffusion output_proj shape %s.",
                tuple(output_emb.weight.shape),
                tuple(self.diffusion_head.output_proj.weight.shape),
            )
            return

        with torch.no_grad():
            self.diffusion_head.token_emb.weight.copy_(
                input_emb.weight.to(
                    device=self.diffusion_head.token_emb.weight.device,
                    dtype=self.diffusion_head.token_emb.weight.dtype,
                )
            )
            self.diffusion_head.output_proj.weight.copy_(
                output_emb.weight.to(
                    device=self.diffusion_head.output_proj.weight.device,
                    dtype=self.diffusion_head.output_proj.weight.dtype,
                )
            )
            if self.diffusion_head.output_proj.bias is not None and getattr(output_emb, "bias", None) is not None:
                self.diffusion_head.output_proj.bias.copy_(
                    output_emb.bias.to(
                        device=self.diffusion_head.output_proj.bias.device,
                        dtype=self.diffusion_head.output_proj.bias.dtype,
                    )
                )

    def get_hidden_states(self, input_ids, attention_mask, output_hidden_states=False,
                          use_cache=False, past_key_values=None, position_ids=None):
        """Get hidden states from base model.
        
        Args:
            output_hidden_states: If True, returns all layer hidden states (memory intensive).
                                 If False, only returns last layer (recommended for training).
        """
        outputs = self.base_llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            position_ids=position_ids
        )
        return outputs

    def get_trainable_state_dict(self):
        """Get only the trainable parameters (diffusion_head)."""
        full_state_dict = self.state_dict()
        return {k: v for k, v in full_state_dict.items() if k.startswith('diffusion_head.')}
    
    def load_trainable_state_dict(self, state_dict, strict=False):
        """Load only trainable parameters, filtering to diffusion_head keys.
        
        Note: Ensure tensors in state_dict are on the correct device before calling.
        """
        filtered_state = {k: v for k, v in state_dict.items() if k.startswith('diffusion_head.')}
        return self.load_state_dict(filtered_state, strict=strict)

    def forward(self, input_ids, attention_mask,
                labels=None, scaffold_mask=None, return_logits=False):
        """
        Forward pass for training.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Ground truth tokens for diffusion loss
            scaffold_mask: Boolean mask indicating which positions to apply diffusion

        Returns:
            dict with 'loss', 'losses'
        """

        device = input_ids.device
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        losses = {}

        predictions = None
        mask_positions = None
        if labels is not None and scaffold_mask is not None:
            valid_labels = labels >= 0
            valid_scaffold_mask = scaffold_mask & valid_labels
            if valid_scaffold_mask.sum() == 0:
                valid_scaffold_mask = None

        if labels is not None and scaffold_mask is not None and valid_scaffold_mask is not None:
            batch_size = labels.shape[0]
            t = self.diffusion_head.sample_timesteps(batch_size, labels.device)
            noisy_tokens, mask_positions = self.diffusion_head.forward_diffusion(
                labels, scaffold_mask, t
            )
            current_tokens = input_ids.clone()
            current_tokens[valid_scaffold_mask] = noisy_tokens[valid_scaffold_mask]
            prompt_mask = None
            if scaffold_mask is not None:
                prompt_mask = ~scaffold_mask
                if attention_mask is not None:
                    prompt_mask = prompt_mask & attention_mask.bool()

            with torch.no_grad():
                outputs = self.get_hidden_states(current_tokens, attention_mask, output_hidden_states=False)
                try:
                    hidden_states = outputs.last_hidden_state
                except AttributeError as e:
                    logger.debug(f"last_hidden_state not available, trying hidden_states: {e}")
                    try:
                        hidden_states = outputs.hidden_states[-1]
                    except (AttributeError, TypeError) as e2:
                        logger.debug(f"hidden_states not available, requesting output_hidden_states=True: {e2}")
                        outputs = self.get_hidden_states(current_tokens, attention_mask, output_hidden_states=True)
                        hidden_states = outputs.hidden_states[-1]

            if return_logits:
                output = self.diffusion_head.training_step_with_outputs(
                    tokens=labels,
                    hidden_states=hidden_states,
                    scaffold_mask=scaffold_mask,
                    attention_mask=(
                        attention_mask if self.diffusion_head.use_bidirectional else None
                    ) if self.diffusion_head.training and self.diffusion_head.use_attention_mask else None,
                    current_tokens=current_tokens,
                    mask_positions=mask_positions,
                    t=t,
                    prompt_mask=prompt_mask,
                )
                diff_loss = output["loss"]
                predictions = output.get("predictions")
            else:
                diff_loss = self.diffusion_head.training_step(
                    tokens=labels,
                    hidden_states=hidden_states,
                    scaffold_mask=scaffold_mask,
                    attention_mask=(
                        attention_mask if self.diffusion_head.use_bidirectional else None
                    ) if self.diffusion_head.training and self.diffusion_head.use_attention_mask else None,
                    current_tokens=current_tokens,
                    mask_positions=mask_positions,
                    t=t,
                    prompt_mask=prompt_mask,
                )
            total_loss = total_loss + diff_loss
            losses["diffusion"] = diff_loss

        has_loss = len(losses) > 0
        output = {
            "loss": total_loss if has_loss else None,
            "losses": losses
        }
        if predictions is not None:
            output["predictions"] = predictions
        if mask_positions is not None:
            output["mask_positions"] = mask_positions
        return output
