import torch

from model.diffusion_head import SchemaDiffusionHead


def test_field_positions_scaffold_only():
    head = SchemaDiffusionHead(
        input_dim=8,
        vocab_size=16,
        hidden_dim=8,
        num_layers=1,
        num_steps=2,
        use_bidirectional=False,
        use_field_position=True,
        field_position_max_len=8,
    )

    scaffold_mask = torch.tensor([[0, 1, 1, 0, 1, 1, 1, 0]], dtype=torch.bool)
    positions = head._compute_field_positions(scaffold_mask)

    assert positions[0, 0].item() == -1
    assert positions[0, 1].item() == 0
    assert positions[0, 2].item() == 1
    assert positions[0, 3].item() == -1
    assert positions[0, 4].item() == 0
    assert positions[0, 5].item() == 1
    assert positions[0, 6].item() == 2
    assert positions[0, 7].item() == -1

    x = torch.zeros(1, 8, 8)
    x_out = head._add_field_position_embeddings(x, scaffold_mask)
    non_scaffold = ~scaffold_mask
    assert torch.allclose(x_out[non_scaffold], torch.zeros_like(x_out[non_scaffold]))
