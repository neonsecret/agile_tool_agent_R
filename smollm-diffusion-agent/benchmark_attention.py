"""
Benchmark script to compare optimized vs non-optimized attention.

Usage:
    python benchmark_attention.py --mode [original|optimized|both]
"""

import torch
import time
import argparse
from model.attention_blocks import BidirectionalAttentionBlock
from model.attention_blocks_optimized import BidirectionalAttentionBlockOptimized


def benchmark_attention(attention_class, batch_size, seq_len, hidden_dim, num_heads, num_iterations=100, warmup=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    block = attention_class(hidden_dim, num_heads=num_heads, dropout=0.1).to(device)
    block.eval()

    x = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=torch.bfloat16)
    attention_mask = torch.ones(batch_size, seq_len, device=device, dtype=torch.bool)

    for _ in range(warmup):
        with torch.no_grad():
            _ = block(x, attention_mask)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = block(x, attention_mask)

    if device.type == "cuda":
        torch.cuda.synchronize()

    end = time.time()

    avg_time = (end - start) / num_iterations * 1000

    if device.type == "cuda":
        memory_allocated = torch.cuda.max_memory_allocated(device) / 1024 ** 2
        torch.cuda.reset_peak_memory_stats(device)
    else:
        memory_allocated = 0

    return avg_time, memory_allocated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="both", choices=["original", "optimized", "both"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--hidden_dim", type=int, default=2048)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"Num heads: {args.num_heads}")
    print(f"Iterations: {args.num_iterations}")
    print("-" * 60)

    if args.mode in ["original", "both"]:
        print("\nBenchmarking ORIGINAL attention...")
        orig_time, orig_mem = benchmark_attention(
            BidirectionalAttentionBlock,
            args.batch_size, args.seq_len, args.hidden_dim, args.num_heads,
            args.num_iterations, args.warmup
        )
        print(f"Original attention: {orig_time:.2f} ms/iter")
        if device.type == "cuda":
            print(f"Peak memory: {orig_mem:.2f} MB")

    if args.mode in ["optimized", "both"]:
        print("\nBenchmarking OPTIMIZED attention (SDPA)...")
        opt_time, opt_mem = benchmark_attention(
            BidirectionalAttentionBlockOptimized,
            args.batch_size, args.seq_len, args.hidden_dim, args.num_heads,
            args.num_iterations, args.warmup
        )
        print(f"Optimized attention: {opt_time:.2f} ms/iter")
        if device.type == "cuda":
            print(f"Peak memory: {opt_mem:.2f} MB")

    if args.mode == "both":
        speedup = orig_time / opt_time
        print("\n" + "=" * 60)
        print(f"SPEEDUP: {speedup:.2f}x")
        if device.type == "cuda":
            memory_reduction = (1 - opt_mem / orig_mem) * 100
            print(f"MEMORY REDUCTION: {memory_reduction:.1f}%")
        print("=" * 60)

        if speedup < 1.5:
            print("\nNote: Speedup is less than 1.5x. Possible reasons:")
            print("- FlashAttention not available on this GPU")
            print("- Sequence length too short (try --seq_len 4096)")
            print("- CPU or MPS device (SDPA optimization limited)")


if __name__ == "__main__":
    main()
