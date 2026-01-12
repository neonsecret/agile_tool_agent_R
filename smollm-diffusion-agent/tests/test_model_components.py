"""Tests for model components (HybridSmolLM, RouterHead, DiffusionHead)."""
import pytest
import torch

from transformers import AutoTokenizer


@pytest.fixture(scope="module")
def tokenizer():
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B")
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return tok


@pytest.fixture(scope="module")
def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TestRouterHead:

    def test_router_head_forward(self, tokenizer, device):
        from model.hybrid_model import RouterHead
        
        hidden_size = 3072  # SmolLM3-3B hidden size
        router = RouterHead(hidden_size, num_classes=3).to(device)
        
        batch_size = 2
        seq_len = 128
        hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
        
        logits = router(hidden_states, attention_mask)
        
        assert logits.shape == (batch_size, 3)

    def test_router_pools_last_non_pad(self, tokenizer, device):
        from model.hybrid_model import RouterHead
        
        hidden_size = 3072
        router = RouterHead(hidden_size, num_classes=3).to(device)
        
        batch_size = 2
        seq_len = 10
        hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device)
        
        # Make the hidden states at different positions distinct
        hidden_states[0, 5] = torch.ones(hidden_size, device=device) * 100
        hidden_states[1, 7] = torch.ones(hidden_size, device=device) * 200
        
        # Attention mask: first example has 6 tokens, second has 8
        attention_mask = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        attention_mask[0, :6] = 1  # Last non-pad at position 5
        attention_mask[1, :8] = 1  # Last non-pad at position 7
        
        logits = router(hidden_states, attention_mask)
        
        # Verify it pools from correct positions by checking the output is influenced
        # by the distinct values we set
        assert logits.shape == (batch_size, 3)
        # The outputs should be different because they pool from different positions
        assert not torch.allclose(logits[0], logits[1])

    def test_router_with_all_padding(self, tokenizer, device):
        """Test router handles edge case of all padding gracefully."""
        from model.hybrid_model import RouterHead
        
        hidden_size = 3072
        router = RouterHead(hidden_size, num_classes=3).to(device)
        
        hidden_states = torch.randn(1, 10, hidden_size, device=device)
        attention_mask = torch.zeros(1, 10, dtype=torch.long, device=device)  # All padding
        
        # Should not crash, falls back to last position
        logits = router(hidden_states, attention_mask)
        assert logits.shape == (1, 3)


class TestDiffusionHead:

    def test_diffusion_head_forward(self, tokenizer, device):
        from model.diffusion_head import SchemaDiffusionHead
        
        input_dim = 3072
        vocab_size = len(tokenizer)
        
        head = SchemaDiffusionHead(
            input_dim=input_dim,
            vocab_size=vocab_size,
            hidden_dim=1024,
            num_layers=2,
            num_steps=4,
            use_bidirectional=True,
        ).to(device)
        
        batch_size = 2
        seq_len = 64
        
        hidden_states = torch.randn(batch_size, seq_len, input_dim, device=device)
        current_tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        t = torch.rand(batch_size, device=device)
        
        logits = head.predict(hidden_states, current_tokens, t)
        
        assert logits.shape == (batch_size, seq_len, vocab_size)

    def test_diffusion_head_training_step(self, tokenizer, device):
        from model.diffusion_head import SchemaDiffusionHead
        from data.utils import resolve_mask_token
        
        input_dim = 3072
        vocab_size = len(tokenizer)
        mask_str, mask_id = resolve_mask_token(tokenizer, None)
        
        head = SchemaDiffusionHead(
            input_dim=input_dim,
            vocab_size=vocab_size,
            hidden_dim=1024,
            num_layers=2,
            num_steps=4,
        ).to(device)
        head.set_mask_token_id(mask_id)
        
        batch_size = 2
        seq_len = 32
        
        tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        hidden_states = torch.randn(batch_size, seq_len, input_dim, device=device)
        scaffold_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        scaffold_mask[:, 10:20] = True  # Mark some positions as scaffold
        
        loss = head.training_step(tokens, hidden_states, scaffold_mask)
        
        assert loss.dim() == 0  # Scalar loss
        assert loss.item() >= 0

    def test_forward_diffusion(self, tokenizer, device):
        from model.diffusion_head import SchemaDiffusionHead
        from data.utils import resolve_mask_token
        
        vocab_size = len(tokenizer)
        mask_str, mask_id = resolve_mask_token(tokenizer, None)
        
        head = SchemaDiffusionHead(
            input_dim=3072,
            vocab_size=vocab_size,
            hidden_dim=1024,
            num_layers=2,
        ).to(device)
        head.set_mask_token_id(mask_id)
        
        batch_size = 1
        seq_len = 20
        
        tokens = torch.randint(0, 1000, (batch_size, seq_len), device=device)
        scaffold_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        scaffold_mask[:, 5:15] = True
        
        t = torch.tensor([0.5], device=device)  # 50% noise
        
        noised, _ = head.forward_diffusion(tokens, scaffold_mask, t)
        
        # Some scaffold positions should be masked
        scaffold_positions = scaffold_mask[0].nonzero(as_tuple=True)[0]
        mask_count = (noised[0, scaffold_positions] == mask_id).sum().item()
        
        # With t=0.5, expect roughly half to be masked
        assert mask_count > 0, "Some positions should be masked"
        assert mask_count < len(scaffold_positions), "Not all should be masked at t=0.5"


class TestHybridModelLoading:

    @pytest.mark.slow
    def test_hybrid_model_loads(self, tokenizer, device):
        """Test that HybridSmolLM loads correctly with frozen base."""
        from model.hybrid_model import HybridSmolLM
        
        # Use 4-bit only on CUDA (MPS doesn't support it)
        use_4bit = torch.cuda.is_available()
        
        model = HybridSmolLM(
            base_model_id="HuggingFaceTB/SmolLM3-3B",
            load_in_4bit=use_4bit,
            diffusion_config={
                "hidden_dim": 512,  # Smaller for faster tests
                "num_layers": 1,
                "num_steps": 2,
            },
            vocab_size=len(tokenizer),
            use_unsloth=False,
        )
        
        # Check base model is frozen
        for name, param in model.named_parameters():
            if name.startswith("base_llm"):
                assert not param.requires_grad, f"Base LLM param {name} should be frozen"
            elif name.startswith("diffusion_head") or name.startswith("router_head"):
                assert param.requires_grad, f"Head param {name} should be trainable"

    @pytest.mark.slow
    def test_hybrid_model_forward(self, tokenizer, device):
        """Test full forward pass through hybrid model."""
        from model.hybrid_model import HybridSmolLM
        from data.utils import resolve_mask_token
        
        mask_str, mask_id = resolve_mask_token(tokenizer, None)
        
        # Use 4-bit only on CUDA
        use_4bit = torch.cuda.is_available()
        
        model = HybridSmolLM(
            base_model_id="HuggingFaceTB/SmolLM3-3B",
            load_in_4bit=use_4bit,
            diffusion_config={
                "hidden_dim": 512,
                "num_layers": 1,
                "num_steps": 2,
            },
            vocab_size=len(tokenizer),
            use_unsloth=False,
        )
        model.diffusion_head.set_mask_token_id(mask_id)
        
        # Move to device (model is PyTorch only now)
        if device.type != "cpu":
            model = model.to(device)
        
        model.eval()
        
        batch_size = 1
        seq_len = 32
        
        # Get the actual device where the model is
        model_device = next(model.parameters()).device
        
        input_ids = torch.randint(0, len(tokenizer), (batch_size, seq_len), device=model_device)
        attention_mask = torch.ones_like(input_ids)
        labels = torch.randint(0, len(tokenizer), (batch_size, seq_len), device=model_device)
        scaffold_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        scaffold_mask[:, 16:24] = True
        router_labels = torch.ones(batch_size, dtype=torch.long, device=model_device)
        
        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                scaffold_mask=scaffold_mask,
                router_labels=router_labels,
            )
        
        assert "loss" in output
        assert "losses" in output
        assert "diffusion" in output["losses"]
        assert "router" in output["losses"]
        assert "router_logits" in output
        
        # Verify losses are reasonable (not NaN/inf)
        assert not torch.isnan(output["loss"])
        assert not torch.isinf(output["loss"])
        assert output["loss"].item() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
