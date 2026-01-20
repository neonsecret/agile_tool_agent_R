"""
Quick test script to verify multi-GPU setup with 4-bit quantization.
Run with: accelerate launch test_multi_gpu.py
"""
import torch
import os
from accelerate import Accelerator


def test_multi_gpu():
    accelerator = Accelerator(mixed_precision="bf16")
    
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    print(f"[Rank {accelerator.process_index}/{accelerator.num_processes}] "
          f"Device: {accelerator.device}, "
          f"WORLD_SIZE={world_size}, LOCAL_RANK={local_rank}")
    
    if torch.cuda.is_available():
        print(f"[Rank {local_rank}] GPU: {torch.cuda.get_device_name(accelerator.device)}")
        print(f"[Rank {local_rank}] Memory: "
              f"{torch.cuda.get_device_properties(accelerator.device).total_memory / 1e9:.2f}GB")
    
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        print("\n" + "=" * 60)
        print(f"✓ Multi-GPU setup verified: {accelerator.num_processes} processes")
        print("=" * 60)
        
        if accelerator.num_processes == 1:
            print("\nNote: Running on single GPU/CPU.")
            print("To test multi-GPU, run with:")
            print("  accelerate launch test_multi_gpu.py")
            print("or")
            print("  torchrun --nproc_per_node=2 test_multi_gpu.py")


if __name__ == "__main__":
    test_multi_gpu()
