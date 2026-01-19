"""Configuration validation and device-specific adjustments."""

import torch
from typing import Dict, Any


def validate_and_adjust_config(config: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """
    Validate configuration and adjust settings based on device capabilities.
    
    Args:
        config: Configuration dictionary
        device: Target device (cuda/mps/cpu)
        
    Returns:
        Adjusted configuration dictionary
    """
    config = config.copy()
    
    device_type = device.type
    is_cuda = device_type == "cuda"
    is_mps = device_type == "mps"
    is_cpu = device_type == "cpu"
    
    print(f"\n{'='*60}")
    print(f"Device Configuration Validation")
    print(f"{'='*60}")
    print(f"Device: {device_type}")
    
    model_cfg = config.get("model", {})
    quant_cfg = config.get("quantization", {})
    train_cfg = config.get("training", {})
    infer_cfg = config.get("inference", {})
    
    if not is_cuda:
        print(f"\nNon-CUDA device detected ({device_type}). Adjusting config...")
        
        if quant_cfg.get("enabled", False):
            print(f"  ⚠️  Disabling 4-bit quantization (CUDA-only via bitsandbytes)")
            quant_cfg["enabled"] = False
        
        if model_cfg.get("use_unsloth"):
            print(f"  ⚠️  Disabling unsloth (CUDA-only)")
            model_cfg["use_unsloth"] = False
        
        compile_cfg = train_cfg.get("compile", {})
        if compile_cfg.get("enabled", False):
            print(f"  ⚠️  Disabling torch.compile for training (best on CUDA)")
            compile_cfg["enabled"] = False
        
        if infer_cfg.get("use_torch_compile", False):
            print(f"  ⚠️  Disabling torch.compile for inference (best on CUDA)")
            infer_cfg["use_torch_compile"] = False
        
        if infer_cfg.get("use_cuda_graph", False):
            print(f"  ⚠️  Disabling CUDA graphs (CUDA-only)")
            infer_cfg["use_cuda_graph"] = False
        
        if is_mps:
            print(f"\n  ℹ️  MPS (Apple Silicon) mode:")
            print(f"     - Using bfloat16 precision")
            print(f"     - Full model will be loaded (no quantization)")
            print(f"     - Standard PyTorch inference (no CUDA optimizations)")
    
    else:
        print(f"\nCUDA device detected. Optimal settings enabled:")
        
        if quant_cfg.get("enabled", False):
            print(f"  ✓ 4-bit quantization enabled")
        
        use_unsloth = model_cfg.get("use_unsloth")
        if use_unsloth or use_unsloth is None:
            print(f"  ✓ Unsloth optimization available")
        
        if model_cfg.get("use_flash_attention", True):
            print(f"  ✓ FlashAttention-2 enabled for base model")
        
        if model_cfg.get("use_gradient_checkpointing", False):
            print(f"  ✓ Gradient checkpointing enabled (memory efficient)")
        
        if model_cfg.get("use_better_transformer", False):
            print(f"  ✓ BetterTransformer enabled")
        
        if infer_cfg.get("use_cuda_graph", False):
            print(f"  ✓ CUDA graphs enabled for inference")
        
        compile_enabled = train_cfg.get("compile", {}).get("enabled", False)
        if compile_enabled:
            print(f"  ✓ torch.compile enabled for training")
    
    config["model"] = model_cfg
    config["quantization"] = quant_cfg
    config["training"] = train_cfg
    config["inference"] = infer_cfg
    
    print(f"{'='*60}\n")
    
    return config


def get_model_kwargs(config: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """
    Extract model initialization kwargs from config.
    
    Args:
        config: Configuration dictionary
        device: Target device
        
    Returns:
        Dictionary of kwargs for HybridSmolLM.__init__
    """
    model_cfg = config.get("model", {})
    quant_cfg = config.get("quantization", {})
    diff_cfg = config.get("diffusion", {})
    train_cfg = config.get("training", {})
    
    is_cuda = device.type == "cuda"
    load_in_4bit = quant_cfg.get("enabled", False) and is_cuda
    
    use_unsloth = model_cfg.get("use_unsloth")
    if use_unsloth is None:
        use_unsloth = is_cuda
    elif use_unsloth and not is_cuda:
        use_unsloth = False
    
    kwargs = {
        "base_model_id": model_cfg.get("base_model_id", "HuggingFaceTB/SmolLM3-3B"),
        "load_in_4bit": load_in_4bit,
        "diffusion_config": diff_cfg,
        "use_unsloth": use_unsloth,
        "max_seq_length": train_cfg.get("max_seq_len", 2048),
        "enable_unsloth_inference_opt": model_cfg.get("enable_unsloth_inference_opt", True),
        "device": device,
        "use_flash_attention": model_cfg.get("use_flash_attention", True) and is_cuda,
        "use_gradient_checkpointing": model_cfg.get("use_gradient_checkpointing", False),
        "use_better_transformer": model_cfg.get("use_better_transformer", False) and is_cuda and not load_in_4bit,
        "unsloth_use_gradient_checkpointing": model_cfg.get("unsloth_use_gradient_checkpointing", "unsloth"),
        "unsloth_rope_scaling": model_cfg.get("unsloth_rope_scaling", None),
    }
    
    return kwargs


def get_inference_kwargs(config: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """
    Extract inference kwargs from config.
    
    Args:
        config: Configuration dictionary
        device: Target device
        
    Returns:
        Dictionary of kwargs for inference
    """
    infer_cfg = config.get("inference", {})
    
    is_cuda = device.type == "cuda"
    
    kwargs = {
        "use_torch_compile": infer_cfg.get("use_torch_compile", False) and is_cuda,
        "use_cuda_graph": infer_cfg.get("use_cuda_graph", False) and is_cuda,
        "max_seq_len": infer_cfg.get("max_seq_len", 2048),
    }
    
    return kwargs


def print_device_capabilities():
    """Print information about available compute devices."""
    print("\n" + "="*60)
    print("Device Capabilities")
    print("="*60)
    
    print(f"PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA available")
        print(f"  - Device count: {torch.cuda.device_count()}")
        print(f"  - Current device: {torch.cuda.current_device()}")
        print(f"  - Device name: {torch.cuda.get_device_name(0)}")
        print(f"  - CUDA version: {torch.version.cuda}")
        
        capability = torch.cuda.get_device_capability(0)
        print(f"  - Compute capability: {capability[0]}.{capability[1]}")
        if capability[0] >= 8:
            print(f"    (Ampere+ GPU: FlashAttention-2 fully supported)")
        elif capability[0] >= 7:
            print(f"    (Volta/Turing GPU: FlashAttention supported with limitations)")
        else:
            print(f"    (Older GPU: FlashAttention may not be available)")
    else:
        print(f"✗ CUDA not available")
    
    if torch.backends.mps.is_available():
        print(f"✓ MPS (Apple Silicon) available")
        print(f"  - Metal acceleration enabled")
    else:
        print(f"✗ MPS not available")
    
    try:
        import bitsandbytes
        print(f"✓ bitsandbytes available (version {bitsandbytes.__version__})")
    except ImportError:
        print(f"✗ bitsandbytes not available (4-bit quantization disabled)")
    
    try:
        from unsloth import FastLanguageModel
        print(f"✓ unsloth available")
    except ImportError:
        print(f"✗ unsloth not available")
    
    try:
        import flash_attn
        print(f"✓ flash-attn available (version {flash_attn.__version__})")
    except ImportError:
        print(f"✗ flash-attn not available (FlashAttention disabled)")
    
    try:
        from optimum.bettertransformer import BetterTransformer
        print(f"✓ optimum available (BetterTransformer enabled)")
    except ImportError:
        print(f"✗ optimum not available (BetterTransformer disabled)")
    
    print("="*60 + "\n")
