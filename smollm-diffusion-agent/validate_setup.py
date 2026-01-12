"""
Quick validation script to test device configuration and model loading.

Usage:
    python validate_setup.py
"""

import sys
import torch
import yaml


def test_device_detection():
    """Test device detection."""
    print("\n" + "="*60)
    print("TEST 1: Device Detection")
    print("="*60)
    
    from data.device_utils import get_device
    device = get_device()
    print(f"✓ Detected device: {device.type}")
    
    if device.type == "cuda":
        print(f"  - CUDA device: {torch.cuda.get_device_name(0)}")
    elif device.type == "mps":
        print(f"  - MPS (Apple Silicon) detected")
    else:
        print(f"  - CPU fallback")
    
    return device


def test_config_validation(device):
    """Test config validation and adjustment."""
    print("\n" + "="*60)
    print("TEST 2: Config Validation")
    print("="*60)
    
    from data.config_utils import validate_and_adjust_config, print_device_capabilities
    
    print_device_capabilities()
    
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    config_adjusted = validate_and_adjust_config(config, device)
    
    print("\n✓ Config validated and adjusted for device")
    
    quant_enabled = config_adjusted.get("quantization", {}).get("enabled", False)
    print(f"  - Quantization: {'enabled' if quant_enabled else 'disabled'}")
    
    use_unsloth = config_adjusted.get("model", {}).get("use_unsloth")
    print(f"  - Unsloth: {use_unsloth}")
    
    use_cuda_graph = config_adjusted.get("inference", {}).get("use_cuda_graph", False)
    print(f"  - CUDA graphs: {'enabled' if use_cuda_graph else 'disabled'}")
    
    return config_adjusted


def test_model_init(device, config):
    """Test model initialization."""
    print("\n" + "="*60)
    print("TEST 3: Model Initialization")
    print("="*60)
    
    from transformers import AutoTokenizer
    from data.config_utils import get_model_kwargs
    from model.hybrid_model import HybridSmolLM
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B")
    print(f"✓ Tokenizer loaded (vocab size: {len(tokenizer)})")
    
    print("\nPreparing model kwargs...")
    model_kwargs = get_model_kwargs(config, device)
    model_kwargs['vocab_size'] = len(tokenizer)
    
    print(f"  - base_model_id: {model_kwargs['base_model_id']}")
    print(f"  - load_in_4bit: {model_kwargs['load_in_4bit']}")
    print(f"  - use_unsloth: {model_kwargs['use_unsloth']}")
    
    print("\nInitializing model (this may take a minute)...")
    try:
        model = HybridSmolLM(**model_kwargs)
        print("✓ Model initialized successfully")
        
        from data.utils import resolve_mask_token, resolve_null_token
        data_cfg = config.get("data", {})
        mask_token_str, mask_token_id = resolve_mask_token(tokenizer, data_cfg.get("mask_token"))
        null_token_str, null_token_id = resolve_null_token(tokenizer, data_cfg.get("null_token"))
        
        model.diffusion_head.set_mask_token_id(mask_token_id)
        if null_token_id is not None:
            model.diffusion_head.set_null_token_id(null_token_id)
        
        print(f"✓ Mask token set: {mask_token_str} (ID: {mask_token_id})")
        
        # Quick forward pass test
        print("\nTesting forward pass...")
        dummy_input = torch.randint(0, len(tokenizer), (1, 10), device=device)
        dummy_mask = torch.ones_like(dummy_input, dtype=torch.bool)
        
        with torch.no_grad():
            outputs = model.get_hidden_states(dummy_input, dummy_mask)
            hidden = outputs.hidden_states[-1]
        
        print(f"✓ Forward pass successful (hidden shape: {hidden.shape})")
        
        return True
        
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_inference_setup(device, config):
    """Test inference generator setup."""
    print("\n" + "="*60)
    print("TEST 4: Inference Setup")
    print("="*60)
    
    from data.config_utils import get_inference_kwargs
    
    inference_kwargs = get_inference_kwargs(config, device)
    
    print(f"  - use_torch_compile: {inference_kwargs['use_torch_compile']}")
    print(f"  - use_cuda_graph: {inference_kwargs['use_cuda_graph']}")
    print(f"  - max_seq_len: {inference_kwargs['max_seq_len']}")
    
    print("✓ Inference kwargs extracted successfully")
    
    return True


def main():
    print("\n" + "="*60)
    print("SmolLM Diffusion Agent - Setup Validation")
    print("="*60)
    
    try:
        # Test 1: Device detection
        device = test_device_detection()
        
        # Test 2: Config validation
        config = test_config_validation(device)
        
        # Test 3: Model initialization
        model_ok = test_model_init(device, config)
        
        if not model_ok:
            print("\n" + "="*60)
            print("VALIDATION FAILED")
            print("="*60)
            print("\nModel initialization failed. Please check:")
            print("  1. Are all required packages installed?")
            print("  2. Is your device supported?")
            print("  3. Do you have enough memory?")
            sys.exit(1)
        
        # Test 4: Inference setup
        test_inference_setup(device, config)
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED")
        print("="*60)
        print("\nYour setup is working correctly!")
        print(f"You can now:")
        print(f"  - Train: accelerate launch train.py")
        print(f"  - Inference: python inference.py")
        
    except KeyboardInterrupt:
        print("\n\nValidation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
