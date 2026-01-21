#!/usr/bin/env python3
"""
Quick diagnostic script to check FlashAttention installation and compatibility.
"""

import sys
import torch

print("=" * 60)
print("FlashAttention Diagnostic")
print("=" * 60)

print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Compute capability: {torch.cuda.get_device_capability(0)}")

print("\n" + "-" * 60)
print("1. Checking flash-attn import...")
print("-" * 60)

try:
    import flash_attn
    print(f"✓ flash-attn imported successfully")
    print(f"  Version: {flash_attn.__version__}")
except ImportError as e:
    print(f"✗ flash-attn import failed: {e}")
    print("\n  To install: pip install flash-attn --no-build-isolation")
    sys.exit(1)

print("\n" + "-" * 60)
print("2. Testing flash-attn CUDA module...")
print("-" * 60)

try:
    import flash_attn_2_cuda
    print("✓ flash_attn_2_cuda module loaded successfully")
except ImportError as e:
    print(f"✗ flash_attn_2_cuda import failed: {e}")
    print("\n  This usually means:")
    print("  - flash-attn was compiled against a different PyTorch version")
    print("  - CUDA toolkit mismatch")
    print("  - Missing CUDA libraries")
    print("\n  Fix: Reinstall flash-attn against your current torch:")
    print("    pip uninstall -y flash-attn")
    print("    pip install flash-attn --no-build-isolation")
    sys.exit(1)

print("\n" + "-" * 60)
print("3. Testing basic flash-attn forward pass...")
print("-" * 60)

try:
    from flash_attn import flash_attn_func
    
    if not torch.cuda.is_available():
        print("✗ CUDA not available, cannot test forward pass")
        sys.exit(1)
    
    device = torch.device("cuda")
    dtype = torch.bfloat16
    
    batch_size, seq_len, num_heads, head_dim = 2, 128, 8, 64
    
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    
    out = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False)
    
    print(f"✓ Forward pass successful")
    print(f"  Input shape: {q.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Output dtype: {out.dtype}")
    
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "-" * 60)
print("4. Testing with transformers...")
print("-" * 60)

try:
    from transformers import AutoConfig, AutoModelForCausalLM
    
    model_id = "HuggingFaceTB/SmolLM3-3B"
    print(f"Loading config for {model_id}...")
    
    config = AutoConfig.from_pretrained(model_id)
    
    print(f"  Default attn_implementation: {config._attn_implementation}")
    
    if hasattr(config, '_attn_implementation_internal'):
        print(f"  Internal attn_implementation: {config._attn_implementation_internal}")
    
    print("\n  Testing model init with flash_attention_2...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
    )
    
    print("✓ Model loaded with flash_attention_2 successfully")
    
    del model
    torch.cuda.empty_cache()
    
except Exception as e:
    print(f"✗ Transformers integration failed: {e}")
    print("\n  This means flash-attn works but transformers can't use it.")
    print("  Possible causes:")
    print("  - Model architecture doesn't support flash_attention_2")
    print("  - Version mismatch between transformers and flash-attn")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ All checks passed! FlashAttention is working correctly.")
print("=" * 60)
print("\nYou can enable it in config.yaml:")
print("  model:")
print("    use_flash_attention: true")
print()
