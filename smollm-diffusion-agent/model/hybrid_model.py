try:
    import unsloth
    from unsloth import FastLanguageModel

    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    unsloth = None
    FastLanguageModel = None

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
                 device: torch.device | None = None):
        super().__init__()

        if diffusion_config is None:
            diffusion_config = {}

        device = device or get_device()
        self.base_llm = None
        self.use_unsloth = False

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
        )

        self.diffusion_head = self.diffusion_head.to(dtype=torch.bfloat16)

    def _init_torch_model(self, base_model_id, load_in_4bit, device, use_unsloth=None, max_seq_length=2048,
                          enable_unsloth_inference_opt=True):
        """Initialize PyTorch model (supports CUDA with quantization, MPS, and CPU)."""
        cuda_available = torch.cuda.is_available()

        if use_unsloth is None:
            use_unsloth = cuda_available

        if use_unsloth and not cuda_available:
            print("Warning: unsloth requires CUDA, disabling on non-CUDA device")
            use_unsloth = False

        if load_in_4bit and not cuda_available:
            print("Warning: 4-bit quantization requires CUDA, falling back to bfloat16")
            load_in_4bit = False

        if use_unsloth and UNSLOTH_AVAILABLE:
            print(f"Loading model with unsloth on CUDA (max_seq_length={max_seq_length})")
            self.base_llm, _ = FastLanguageModel.from_pretrained(
                model_name=base_model_id,
                max_seq_length=max_seq_length,
                dtype=None if load_in_4bit else torch.bfloat16,
                load_in_4bit=load_in_4bit,
                load_in_8bit=False,
            )
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

            if device.type == "cuda":
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

        for param in self.base_llm.parameters():
            param.requires_grad = False


    def get_hidden_states(self, input_ids, attention_mask, output_hidden_states=True,
                          use_cache=False, past_key_values=None, position_ids=None):
        """Get hidden states from base model."""
        return self.base_llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            position_ids=position_ids
        )

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

        with torch.no_grad():
            outputs = self.get_hidden_states(input_ids, attention_mask)
            hidden_states = outputs.hidden_states[-1]

        device = hidden_states.device
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        losses = {}

        logits = None
        if labels is not None and scaffold_mask is not None and scaffold_mask.sum() > 0:
            if return_logits:
                output = self.diffusion_head.training_step_with_outputs(
                    tokens=labels,
                    hidden_states=hidden_states,
                    scaffold_mask=scaffold_mask,
                )
                diff_loss = output["loss"]
                logits = output["logits"]
            else:
                diff_loss = self.diffusion_head.training_step(
                    tokens=labels,
                    hidden_states=hidden_states,
                    scaffold_mask=scaffold_mask
                )
            total_loss = total_loss + diff_loss
            losses["diffusion"] = diff_loss

        has_loss = len(losses) > 0
        output = {
            "loss": total_loss if has_loss else None,
            "losses": losses
        }
        if logits is not None:
            output["logits"] = logits
        return output
