import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from .diffusion_head import SchemaDiffusionHead


class RouterHead(nn.Module):
    def __init__(self, hidden_size, num_classes=2):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states):
        # hidden_states: [batch, seq_len, hidden]
        # We pool the last token
        pooled = hidden_states[:, -1, :]
        return self.classifier(pooled)


class HybridSmolLM(nn.Module):
    def __init__(self, base_model_id="HuggingFaceTB/SmolLM3-3B", load_in_4bit=False,
                 diffusion_config=None, vocab_size=None):
        super().__init__()

        # 1. Load Base Model (Frozen)
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            self.base_llm = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                quantization_config=bnb_config,
                device_map={"": 0}  # Necessary for bitsandbytes
            )
        else:
            self.base_llm = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                torch_dtype=torch.bfloat16
            )

        # FREEZE the base model
        for param in self.base_llm.parameters():
            param.requires_grad = False

        # 2. Initialize Heads
        hidden_size = self.base_llm.config.hidden_size

        # Use tokenizer vocab_size if provided (includes special tokens)
        # Otherwise fall back to model config vocab_size
        if vocab_size is None:
            vocab_size = self.base_llm.config.vocab_size

        # Use diffusion_config if provided, otherwise use defaults
        if diffusion_config is None:
            diffusion_config = {}

        hidden_dim = diffusion_config.get("hidden_dim", 1024)
        num_layers = diffusion_config.get("num_layers", 2)
        num_steps = diffusion_config.get("num_steps", 4)
        label_smoothing = diffusion_config.get("label_smoothing", 0.1)

        self.diffusion_head = SchemaDiffusionHead(
            input_dim=hidden_size,
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_steps=num_steps,
            label_smoothing=label_smoothing
        )
        self.router_head = RouterHead(hidden_size, num_classes=2)

        self.diffusion_head = self.diffusion_head.to(dtype=torch.bfloat16)
        self.router_head = self.router_head.to(dtype=torch.bfloat16)

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

        # 1. Run Base LLM to get Context Embeddings (frozen)
        with torch.no_grad():
            outputs = self.base_llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            hidden_states = outputs.hidden_states[-1]

        # 2. Router Forward
        router_logits = self.router_head(hidden_states)

        device = hidden_states.device
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        losses = {}

        # 3. Diffusion Loss (if applicable)
        if labels is not None and scaffold_mask is not None and scaffold_mask.sum() > 0:
            diff_loss = self.diffusion_head.training_step(
                tokens=labels,
                hidden_states=hidden_states,
                scaffold_mask=scaffold_mask
            )
            total_loss = total_loss + diff_loss
            losses["diffusion"] = diff_loss

        # 4. Router Loss (if training router)
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
