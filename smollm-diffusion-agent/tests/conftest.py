"""Pytest configuration for smollm-diffusion-agent tests."""
import sys
from pathlib import Path
import pytest
import torch

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


@pytest.fixture(scope="session")
def shared_tokenizer():
    """Session-scoped tokenizer to avoid reloading."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B")
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return tok


@pytest.fixture(scope="session")
def shared_device():
    """Session-scoped device detection."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@pytest.fixture(scope="session")
def shared_base_model(shared_tokenizer, shared_device):
    """Session-scoped base model for slow tests.
    
    Loads once and shares across all test modules.
    """
    from transformers import AutoModelForCausalLM

    if shared_device.type == "mps":
        dtype = torch.float16
    elif shared_device.type == "cuda":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM3-3B",
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model


@pytest.fixture(scope="session")
def shared_hybrid_model(shared_tokenizer, shared_device):
    """Session-scoped hybrid model for slow tests."""
    from model.hybrid_model import HybridSmolLM
    from data.utils import resolve_mask_token, resolve_null_token

    use_4bit = shared_device.type == "cuda"

    model = HybridSmolLM(
        base_model_id="HuggingFaceTB/SmolLM3-3B",
        load_in_4bit=use_4bit,
        diffusion_config={
            "hidden_dim": 512,  # Smaller for faster tests
            "num_layers": 1,
            "num_steps": 2,
        },
        vocab_size=len(shared_tokenizer),
        use_unsloth=False,
    )

    mask_str, mask_id = resolve_mask_token(shared_tokenizer, None)
    model.diffusion_head.set_mask_token_id(mask_id)

    null_str, null_id = resolve_null_token(shared_tokenizer, None)
    if null_id is not None:
        model.diffusion_head.set_null_token_id(null_id)

    # Move diffusion head to same device as base LLM
    base_device = next(model.base_llm.parameters()).device
    model.diffusion_head = model.diffusion_head.to(base_device)

    model.eval()
    return model
