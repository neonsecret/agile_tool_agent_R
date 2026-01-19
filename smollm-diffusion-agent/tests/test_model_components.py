"""Tests for model components (HybridSmolLM, DiffusionHead)."""
import pytest
import torch


@pytest.fixture
def tokenizer(shared_tokenizer):
    """Use shared session-scoped tokenizer."""
    return shared_tokenizer


@pytest.fixture
def device(shared_device):
    """Use shared session-scoped device."""
    return shared_device


@pytest.fixture
def hybrid_model(shared_hybrid_model):
    """Use shared session-scoped hybrid model."""
    return shared_hybrid_model


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
    def test_hybrid_model_loads(self, hybrid_model):
        """Test that HybridSmolLM loads correctly with frozen base."""
        # Check base model is frozen
        for name, param in hybrid_model.named_parameters():
            if name.startswith("base_llm"):
                assert not param.requires_grad, f"Base LLM param {name} should be frozen"
            elif name.startswith("diffusion_head"):
                assert param.requires_grad, f"Head param {name} should be trainable"

    @pytest.mark.slow
    def test_hybrid_model_forward(self, tokenizer, hybrid_model):
        """Test full forward pass through hybrid model."""
        batch_size = 1
        seq_len = 32

        model_device = next(hybrid_model.parameters()).device

        input_ids = torch.randint(0, len(tokenizer), (batch_size, seq_len), device=model_device)
        attention_mask = torch.ones_like(input_ids)
        labels = torch.randint(0, len(tokenizer), (batch_size, seq_len), device=model_device)
        scaffold_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        scaffold_mask[:, 16:24] = True

        with torch.no_grad():
            output = hybrid_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                scaffold_mask=scaffold_mask,
            )

        assert "loss" in output
        assert "losses" in output
        assert "diffusion" in output["losses"]

        # Verify losses are reasonable (not NaN/inf)
        assert not torch.isnan(output["loss"])
        assert not torch.isinf(output["loss"])
        assert output["loss"].item() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
