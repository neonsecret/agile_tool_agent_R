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
    def __init__(self, base_model_id="HuggingFaceTB/SmolLM3-3B", load_in_4bit=False):
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
        vocab_size = self.base_llm.config.vocab_size

        self.diffusion_head = SchemaDiffusionHead(hidden_size, vocab_size)
        self.router_head = RouterHead(hidden_size, num_classes=2)  # 0=Chat, 1=Tool

    def forward(self, input_ids, attention_mask,
                labels=None, scaffold_mask=None, diffusion_steps=None,
                router_labels=None):
        """
        scaffold_mask: Boolean mask for diffusion training.
        router_labels: [batch] (0 or 1) for classification training.
        """

        # 1. Run Base LLM to get Context Embeddings
        with torch.no_grad():
            outputs = self.base_llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            hidden_states = outputs.hidden_states[-1]

        # 2. Router Forward
        router_logits = self.router_head(hidden_states)

        # 3. Diffusion Forward
        diffusion_logits = self.diffusion_head(
            hidden_states,
            input_ids,
            diffusion_steps,
            scaffold_mask
        )

        # Initialize total_loss as a tensor to avoid int/tensor type issues
        device = hidden_states.device
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        losses = {}
        
        # Loss 1: Diffusion (Structured Gen)
        if labels is not None and scaffold_mask is not None:
            # Check if we have any masked tokens in this batch
            if scaffold_mask.sum() > 0:
                active_logits = diffusion_logits[scaffold_mask]
                active_labels = labels[scaffold_mask]
                
                diff_loss = nn.CrossEntropyLoss()(active_logits, active_labels)
                total_loss = total_loss + diff_loss
                losses["diffusion"] = diff_loss
            
        # Loss 2: Router (Decision)
        if router_labels is not None:
            router_loss = nn.CrossEntropyLoss()(router_logits, router_labels)
            total_loss = total_loss + router_loss
            losses["router"] = router_loss
        
        # Return None if no losses were computed, otherwise return tensor
        has_loss = len(losses) > 0
        return {
            "loss": total_loss if has_loss else None,
            "losses": losses,
            "diffusion_logits": diffusion_logits,
            "router_logits": router_logits
        }
