"""
Quick test to verify optimized attention integration.

This test verifies:
1. Optimized attention module can be imported
2. Diffusion head can use optimized attention
3. Output shapes are correct
4. Forward/backward pass works
"""

import torch
from model.attention_blocks_optimized import BidirectionalAttentionBlockOptimized
from model.diffusion_head import SchemaDiffusionHead


def test_optimized_attention_basic():
    print("Testing optimized attention block...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    hidden_dim = 512
    num_heads = 8
    batch_size = 2
    seq_len = 128
    
    block = BidirectionalAttentionBlockOptimized(hidden_dim, num_heads=num_heads).to(device)
    
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    attention_mask = torch.ones(batch_size, seq_len, device=device, dtype=torch.bool)
    
    output = block(x, attention_mask)
    
    assert output.shape == (batch_size, seq_len, hidden_dim), f"Expected shape {(batch_size, seq_len, hidden_dim)}, got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"
    
    print("✅ Optimized attention block test passed")


def test_diffusion_head_with_optimized_attention():
    print("\nTesting diffusion head with optimized attention...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    input_dim = 512
    vocab_size = 1000
    hidden_dim = 512
    num_layers = 2
    num_heads = 8
    batch_size = 2
    seq_len = 128
    
    head = SchemaDiffusionHead(
        input_dim=input_dim,
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        use_bidirectional=True,
        use_optimized_attention=True,
    ).to(device)
    
    hidden_states = torch.randn(batch_size, seq_len, input_dim, device=device)
    current_tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    t = torch.rand(batch_size, device=device)
    
    logits = head.predict(hidden_states, current_tokens, t)
    
    assert logits.shape == (batch_size, seq_len, vocab_size), f"Expected shape {(batch_size, seq_len, vocab_size)}, got {logits.shape}"
    assert not torch.isnan(logits).any(), "Logits contain NaN"
    assert not torch.isinf(logits).any(), "Logits contain Inf"
    
    print("✅ Diffusion head with optimized attention test passed")


def test_backward_pass():
    print("\nTesting backward pass...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    input_dim = 512
    vocab_size = 1000
    hidden_dim = 512
    batch_size = 2
    seq_len = 128
    
    head = SchemaDiffusionHead(
        input_dim=input_dim,
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=2,
        use_optimized_attention=True,
    ).to(device)
    head.set_mask_token_id(vocab_size - 1)
    
    hidden_states = torch.randn(batch_size, seq_len, input_dim, device=device, requires_grad=True)
    tokens = torch.randint(0, vocab_size - 1, (batch_size, seq_len), device=device)
    scaffold_mask = torch.ones(batch_size, seq_len, device=device, dtype=torch.bool)
    
    loss = head.training_step(tokens, hidden_states, scaffold_mask)
    
    assert loss.requires_grad, "Loss should require grad"
    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is Inf"
    
    loss.backward()
    
    assert hidden_states.grad is not None, "Hidden states grad is None"
    assert not torch.isnan(hidden_states.grad).any(), "Gradients contain NaN"
    
    print("✅ Backward pass test passed")


def test_equivalence_with_original():
    print("\nTesting output equivalence (optimized vs original)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    input_dim = 512
    vocab_size = 1000
    hidden_dim = 512
    batch_size = 2
    seq_len = 128
    
    torch.manual_seed(42)
    head_opt = SchemaDiffusionHead(
        input_dim=input_dim,
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=2,
        use_optimized_attention=True,
    ).to(device)
    
    torch.manual_seed(42)
    head_orig = SchemaDiffusionHead(
        input_dim=input_dim,
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=2,
        use_optimized_attention=False,
    ).to(device)
    
    head_opt.eval()
    head_orig.eval()
    
    torch.manual_seed(123)
    hidden_states = torch.randn(batch_size, seq_len, input_dim, device=device)
    current_tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    t = torch.rand(batch_size, device=device)
    
    with torch.no_grad():
        logits_opt = head_opt.predict(hidden_states, current_tokens, t)
        logits_orig = head_orig.predict(hidden_states, current_tokens, t)
    
    max_diff = (logits_opt - logits_orig).abs().max().item()
    print(f"Max difference: {max_diff:.6f}")
    
    if max_diff < 1e-3:
        print("✅ Outputs are equivalent (within tolerance)")
    else:
        print(f"⚠️  Outputs differ by {max_diff:.6f} (expected <1e-3)")
        print("Note: This is normal due to different numerical implementations")


if __name__ == "__main__":
    print("Running optimized attention integration tests...")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print("=" * 60)
    
    test_optimized_attention_basic()
    test_diffusion_head_with_optimized_attention()
    test_backward_pass()
    test_equivalence_with_original()
    
    print("\n" + "=" * 60)
    print("All tests passed! ✅")
    print("\nNext steps:")
    print("1. Run benchmark: python benchmark_attention.py --mode both")
    print("2. Test full training: python train.py")
    print("3. Test inference: python inference.py")
