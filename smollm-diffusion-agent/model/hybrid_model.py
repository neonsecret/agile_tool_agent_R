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


class RouterHead(nn.Module):
    def __init__(self, hidden_size, num_classes=3):
        """
        Mode router for Chat/Think/Tool classification.
        
        Args:
            hidden_size: Size of input hidden states
            num_classes: Number of modes (3: Chat=0, Tool=1, Think=2)
        """
        super().__init__()
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states, attention_mask=None):
        """
        Args:
            hidden_states: [batch, seq_len, hidden]
            attention_mask: Optional [batch, seq_len] with 1 for valid tokens.
        """
        if attention_mask is None:
            pooled = hidden_states[:, -1, :]
            return self.classifier(pooled)

        # Pool the last non-padding token per sample.
        # attention_mask is expected to be 1 for real tokens, 0 for padding.
        lengths = attention_mask.long().sum(dim=1).clamp(min=1) - 1
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        pooled = hidden_states[batch_indices, lengths, :]
        return self.classifier(pooled)


class HybridSmolLM(nn.Module):
    def __init__(self, base_model_id="HuggingFaceTB/SmolLM3-3B", load_in_4bit=False,
                 diffusion_config=None, vocab_size=None, backend=None, mlx_base_model_id=None,
                 use_unsloth=None, max_seq_length=2048, enable_unsloth_inference_opt=True):
        super().__init__()

        if diffusion_config is None:
            diffusion_config = {}

        device = get_device()
        use_mlx = backend is not None and backend.lower() == "mlx"

        self.use_mlx = use_mlx
        self.mlx_model = None
        self.base_llm = None
        self.use_unsloth = False

        if use_mlx:
            print("Using mlx")
            self._init_mlx_model(mlx_base_model_id or base_model_id, load_in_4bit, vocab_size)
            hidden_size = self.mlx_config.hidden_size
            if vocab_size is None:
                vocab_size = self.mlx_config.vocab_size
        else:
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

        self.diffusion_head = SchemaDiffusionHead(
            input_dim=hidden_size,
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_steps=num_steps,
            label_smoothing=label_smoothing,
            use_bidirectional=use_bidirectional,
            num_heads=num_heads
        )
        self.router_head = RouterHead(hidden_size, num_classes=3)  # Chat=0, Tool=1, Think=2

        self.diffusion_head = self.diffusion_head.to(dtype=torch.bfloat16)
        self.router_head = self.router_head.to(dtype=torch.bfloat16)

    def _should_use_mlx(self, backend, device):
        if backend is not None:
            return backend.lower() == "mlx"
        return platform.system() == "Darwin" and platform.machine() == "arm64" and device.type == "mps"

    def _init_torch_model(self, base_model_id, load_in_4bit, device, use_unsloth=None, max_seq_length=2048,
                          enable_unsloth_inference_opt=True):
        cuda_available = torch.cuda.is_available()

        if use_unsloth is None:
            use_unsloth = cuda_available

        if use_unsloth and UNSLOTH_AVAILABLE:
            print("Using unsloth FastModel for faster CUDA training/inference")
            self.base_llm, _ = FastLanguageModel.from_pretrained(
                model_name=base_model_id,
                max_seq_length=max_seq_length,
                dtype=None if load_in_4bit else torch.bfloat16,
                load_in_4bit=load_in_4bit,
                load_in_8bit=False,
            )
            if enable_unsloth_inference_opt:
                FastLanguageModel.for_inference(self.base_llm)
                print("Unsloth inference optimizations enabled (2x faster inference)")
            self.use_unsloth = True
        elif use_unsloth and not UNSLOTH_AVAILABLE:
            print("Warning: unsloth requested but not available, falling back to standard model")
            use_unsloth = False
        else:
            kwargs = {
                "torch_dtype": torch.bfloat16,
                "device_map": "auto" if device.type == "mps" else None
            }

            if load_in_4bit and cuda_available:
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                kwargs["device_map"] = get_device_map_for_quantization(torch.device("cuda"))

            self.base_llm = AutoModelForCausalLM.from_pretrained(base_model_id, **kwargs)
            self.use_unsloth = False

        for param in self.base_llm.parameters():
            param.requires_grad = False

    def _init_mlx_model(self, model_id, quantize, vocab_size):
        try:
            import mlx.core as mx
            from mlx_lm import load
        except ImportError:
            raise ImportError("MLX is required for Apple Silicon inference. Install with: pip install mlx mlx-lm")

        print(f"Loading MLX model: {model_id}")

        if quantize:
            print(f"Using MLX with 4-bit quantization")
        else:
            print(f"Using MLX without quantization")

        self.mlx_model, self.mlx_tokenizer = load(model_id)
        self.mlx_config = self.mlx_model.args

    def _extract_hidden_states_mlx(self, input_ids, attention_mask):
        import mlx.core as mx
        import numpy as np

        input_ids_np = input_ids.cpu().numpy()
        mx_input_ids = mx.array(input_ids_np)

        x = self.mlx_model.model.embed_tokens(mx_input_ids)

        mask = None
        for layer in self.mlx_model.model.layers:
            x = layer(x, mask=mask)

        hidden_states_mx = self.mlx_model.model.norm(x)

        mx.eval(hidden_states_mx)

        hidden_states_mx_float32 = hidden_states_mx.astype(mx.float32)
        hidden_states_np = np.array(hidden_states_mx_float32, copy=False)

        hidden_states_tensor = torch.tensor(hidden_states_np, dtype=torch.bfloat16, device=input_ids.device)

        if hidden_states_tensor.dim() == 2:
            hidden_states_tensor = hidden_states_tensor.unsqueeze(0)

        return hidden_states_tensor

    def get_hidden_states(self, input_ids, attention_mask, output_hidden_states=True,
                          use_cache=False, past_key_values=None, position_ids=None):
        """Unified interface for getting hidden states from either PyTorch or MLX backend."""
        if self.use_mlx:
            hidden_states = self._extract_hidden_states_mlx(input_ids, attention_mask)
            return type('Outputs', (), {
                'hidden_states': (hidden_states,),
                'past_key_values': None
            })()
        else:
            return self.base_llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=output_hidden_states,
                use_cache=use_cache,
                past_key_values=past_key_values,
                position_ids=position_ids
            )

    def forward(self, input_ids, attention_mask,
                labels=None, scaffold_mask=None,
                router_labels=None):
        """
        Forward pass for training.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Ground truth tokens for diffusion loss
            scaffold_mask: Boolean mask indicating which positions to apply diffusion
            router_labels: Ground truth labels for router classification

        Returns:
            dict with 'loss', 'losses', 'router_logits'
        """

        with torch.no_grad():
            outputs = self.get_hidden_states(input_ids, attention_mask)
            hidden_states = outputs.hidden_states[-1]

        router_logits = self.router_head(hidden_states, attention_mask=attention_mask)

        device = hidden_states.device
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        losses = {}

        if labels is not None and scaffold_mask is not None and scaffold_mask.sum() > 0:
            diff_loss = self.diffusion_head.training_step(
                tokens=labels,
                hidden_states=hidden_states,
                scaffold_mask=scaffold_mask
            )
            total_loss = total_loss + diff_loss
            losses["diffusion"] = diff_loss

        if router_labels is not None:
            router_loss = nn.CrossEntropyLoss()(router_logits, router_labels)
            total_loss = total_loss + router_loss
            losses["router"] = router_loss

        has_loss = len(losses) > 0
        return {
            "loss": total_loss if has_loss else None,
            "losses": losses,
            "router_logits": router_logits
        }
