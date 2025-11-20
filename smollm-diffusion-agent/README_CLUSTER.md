# Hybrid Diffusion-Agent Training Guide

## 1. Environment Setup

On your cluster node:

```bash
# Create environment
conda create -n diffusion_agent python=3.11
conda activate diffusion_agent

# Install dependencies
pip install -r requirements.txt
```

## 2. Configure Accelerate (Multi-GPU)

Run the interactive config wizard:

```bash
accelerate config
```

**Recommended Settings for Cluster:**

- Compute environment: `This machine`
- Distributed type: `Multi-GPU`
- Mixed precision: `bf16` (if supported, else `fp16`)
- Use DeepSpeed? `Yes` (Recommended: Zero Stage 2)

## 3. Launch Training

**Crucial:** Before launching, edit `train.py`:

1. Change `limit=20` to `limit=None` in `SmartScaffoldDataset`.
2. Increase `batch_size` (e.g., 4 or 8 per GPU depending on VRAM).

**Run Command:**

```bash
accelerate launch smollm-diffusion-agent/train.py
```

## 4. Architecture Notes

- **Base Model**: `HuggingFaceTB/SmolLM3-3B` (Frozen)
- **Trainable**: `SchemaDiffusionHead` (MLP + Time Embeddings)
- **Objective**: Schema-constrained token diffusion.

## 5. Troubleshooting

- **OOM Errors**: Enable DeepSpeed Zero-3 or reduce batch size.
- **Gradient issues**: Check `gradient_accumulation_steps` in `train.py`.

