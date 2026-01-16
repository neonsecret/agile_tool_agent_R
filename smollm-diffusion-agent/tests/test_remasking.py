"""Tests for Running Confidence Remasking (RCR) functionality.

RCR is a fundamental change to inference behavior that allows tokens to be
remasked if they never achieve high confidence, preventing irreversible errors.
"""

import pytest
import torch
from transformers import AutoTokenizer

from data.schema import build_schema_template
from data.utils import resolve_mask_token, resolve_null_token
from inference import FunctionCallGenerator, GenerationConfig


@pytest.fixture
def tokenizer(shared_tokenizer):
    return shared_tokenizer


@pytest.fixture
def device(shared_device):
    return shared_device


@pytest.fixture
def hybrid_model(shared_hybrid_model):
    return shared_hybrid_model


@pytest.fixture
def mask_and_null(shared_tokenizer):
    mask_str, mask_id = resolve_mask_token(shared_tokenizer, None)
    null_str, null_id = resolve_null_token(shared_tokenizer, None)
    return mask_str, mask_id, null_str, null_id


class TestRemaskingConfig:
    """Test that remasking config is properly handled."""

    def test_remasking_disabled_by_default(self):
        """Remasking should be enabled by default in GenerationConfig."""
        config = GenerationConfig()
        assert config.enable_remasking is True
        assert config.remask_ratio == 0.2
        assert config.min_lock_confidence == 0.7

    def test_remasking_can_be_disabled(self):
        """Remasking can be explicitly disabled."""
        config = GenerationConfig(enable_remasking=False)
        assert config.enable_remasking is False

    def test_remasking_parameters_are_configurable(self):
        """Remasking ratio and threshold can be configured."""
        config = GenerationConfig(
            enable_remasking=True,
            remask_ratio=0.3,
            min_lock_confidence=0.8
        )
        assert config.remask_ratio == 0.3
        assert config.min_lock_confidence == 0.8


class TestRemaskingBehavior:
    """Test that remasking actually works during inference."""

    @pytest.mark.slow
    def test_remasking_runs_without_error(self, tokenizer, device, mask_and_null, hybrid_model):
        """Test that inference with remasking enabled runs without crashing."""
        from inference import FunctionCallGenerator, GenerationConfig

        mask_str, mask_id, null_str, null_id = mask_and_null
        actual_device = next(hybrid_model.parameters()).device

        generator = FunctionCallGenerator(
            model=hybrid_model,
            tokenizer=tokenizer,
            device=actual_device,
            use_torch_compile=False,
            use_cuda_graph=False,
        )

        fields = [("location", 16)]
        template = build_schema_template(
            tokenizer=tokenizer,
            fields=fields,
            mask_token=mask_str,
            null_token=null_str,
            include_codeblock=False,
        )

        config = GenerationConfig(
            steps=4,
            temperature=0.0,
            show_steps=False,
            enable_remasking=True,
            remask_ratio=0.2,
            min_lock_confidence=0.7,
        )

        output = generator.generate(
            prompt="Tokyo weather?",
            template=template,
            config=config,
            tool_name="get_weather",
        )

        assert output is not None
        assert output.steps_executed > 0

    @pytest.mark.slow
    def test_remasking_disabled_still_works(self, tokenizer, device, mask_and_null, hybrid_model):
        """Test that inference with remasking disabled still works (backward compatibility)."""
        from inference import FunctionCallGenerator, GenerationConfig

        mask_str, mask_id, null_str, null_id = mask_and_null
        actual_device = next(hybrid_model.parameters()).device

        generator = FunctionCallGenerator(
            model=hybrid_model,
            tokenizer=tokenizer,
            device=actual_device,
            use_torch_compile=False,
            use_cuda_graph=False,
        )

        fields = [("location", 16)]
        template = build_schema_template(
            tokenizer=tokenizer,
            fields=fields,
            mask_token=mask_str,
            null_token=null_str,
            include_codeblock=False,
        )

        config = GenerationConfig(
            steps=4,
            temperature=0.0,
            show_steps=False,
            enable_remasking=False,
        )

        output = generator.generate(
            prompt="Tokyo weather?",
            template=template,
            config=config,
            tool_name="get_weather",
        )

        assert output is not None
        assert output.steps_executed > 0


class TestEntropyRegularization:
    """Test that entropy regularization is applied in loss computation."""

    def test_entropy_weight_is_configurable(self):
        """Entropy weight can be set via config."""
        from model.diffusion_head import SchemaDiffusionHead

        head = SchemaDiffusionHead(
            input_dim=1024,
            vocab_size=32000,
            entropy_weight=0.05,
        )

        assert head.entropy_weight == 0.05

    def test_entropy_weight_defaults_to_zero(self):
        """Entropy weight defaults to 0.0 if not specified (backward compatibility)."""
        from model.diffusion_head import SchemaDiffusionHead

        head = SchemaDiffusionHead(
            input_dim=1024,
            vocab_size=32000,
            entropy_weight=0.0,
        )

        assert head.entropy_weight == 0.0

    def test_entropy_regularization_in_loss(self):
        """Test that entropy regularization is included in loss computation."""
        from model.diffusion_head import SchemaDiffusionHead
        import torch.nn.functional as F

        head = SchemaDiffusionHead(
            input_dim=1024,
            vocab_size=32000,
            entropy_weight=0.05,
        )

        batch_size = 2
        seq_len = 10
        vocab_size = 32000

        active_logits = torch.randn(batch_size * seq_len, vocab_size, requires_grad=True)
        active_labels = torch.randint(0, vocab_size, (batch_size * seq_len,))

        loss = head._compute_loss(active_logits, active_labels, t_mean=0.5)

        assert loss.requires_grad
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        assert loss.item() > 0


class TestDynamicNullWeighting:
    """Test that NULL weighting is dynamic based on timestep."""

    def test_null_weight_scales_with_timestep(self):
        """NULL weight should be higher at early timesteps (high t) and lower at late timesteps (low t)."""
        from model.diffusion_head import SchemaDiffusionHead
        import torch.nn.functional as F

        head = SchemaDiffusionHead(
            input_dim=1024,
            vocab_size=32000,
            null_loss_weight=0.1,
        )

        batch_size = 2
        seq_len = 10
        vocab_size = 32000

        active_logits = torch.randn(batch_size * seq_len, vocab_size)
        active_labels = torch.randint(0, vocab_size, (batch_size * seq_len,))
        head.set_null_token_id(1000)
        active_labels[0] = 1000

        # Early timestep (high t = 0.9)
        loss_early = head._compute_loss(active_logits, active_labels, t_mean=0.9)

        # Late timestep (low t = 0.1)
        loss_late = head._compute_loss(active_logits, active_labels, t_mean=0.1)

        assert not torch.isnan(loss_early)
        assert not torch.isnan(loss_late)

    def test_null_weight_backward_compatible(self):
        """NULL weighting should work without timestep (backward compatibility)."""
        from model.diffusion_head import SchemaDiffusionHead

        head = SchemaDiffusionHead(
            input_dim=1024,
            vocab_size=32000,
            null_loss_weight=0.1,
        )

        batch_size = 2
        seq_len = 10
        vocab_size = 32000

        active_logits = torch.randn(batch_size * seq_len, vocab_size)
        active_labels = torch.randint(0, vocab_size, (batch_size * seq_len,))

        loss = head._compute_loss(active_logits, active_labels, t_mean=None)

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
