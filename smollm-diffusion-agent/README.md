# SmolLM Diffusion Agent

Hybrid diffusion-autoregressive architecture for function calling, combining SmolLM3 base model with a lightweight diffusion head for structured JSON generation.

## Quick Start

### Prerequisites

- Python 3.11+
- CUDA-capable GPU (for training) or Apple Silicon (MPS) for development
- [UV](https://github.com/astral-sh/uv) package manager

### Installation with UV

```bash
# Install UV if not already installed
# Option 1: Official installer (adds to PATH, may need shell restart)
curl -LsSf https://astral.sh/uv/install.sh | sh
# After installation, restart terminal or run: source ~/.zshrc  # (or ~/.bashrc)

# Option 2: Install via pip/pipx (available immediately)
pip install uv
# or: pipx install uv

# Verify UV is installed and in PATH
uv --version

# Navigate to project directory
cd smollm-diffusion-agent

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install core dependencies
uv pip install -r requirements.txt

# For CUDA training (Linux/Windows with NVIDIA GPU), install CUDA-specific packages:
uv pip install "bitsandbytes>=0.41.0" "triton>=2.0.0" unsloth xformers

# FlashAttention-2 (optional, for CUDA only - may need prebuilt wheel)
# Check https://github.com/mjun0812/flash-attention-prebuild-wheels for your PyTorch/CUDA version
# For PyTorch 2.9 + CUDA 12.8:
# uv pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.0/flash_attn-2.8.3%2Bcu128torch2.9-cp311-cp311-linux_x86_64.whl
```

### Alternative: Using Conda (for torch313 environment)

```bash
# Create conda environment
conda create -n torch313 python=3.13
conda activate torch313

# Install PyTorch with CUDA (if needed)
conda install pytorch torchvision torchaudio pytorch-cuda=12.8 -c pytorch -c nvidia

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Edit `config.yaml` to adjust:
- Batch size, learning rate, optimizer (Muon/AdamW)
- Dataset sources and weights
- Diffusion head architecture
- Training hyperparameters

Key settings for 4x RTX 4090 (or 5x RTX 4000 Ada):
- `batch_size: 4` per GPU
- `gradient_accumulation_steps: 4`
- Effective batch size = `4 GPUs × 4 batch × 4 grad_accum = 64`

## Training

### Single GPU

```bash
# Option 1: Using UV (no activation needed)
uv run wandb login
uv run huggingface-cli login
uv run python train.py

# Option 2: Using activated environment
source .venv/bin/activate  # or conda activate torch313
wandb login
huggingface-cli login
python train.py
```

### Multi-GPU (4x RTX 4090 or 5x RTX 4000 Ada)

```bash
# Using accelerate
uv run accelerate launch --config_file accelerate_config_multigpu.yaml train.py

# Or using torchrun
uv run torchrun --nproc_per_node=4 train.py

# Or with activated environment
accelerate launch --config_file accelerate_config_multigpu.yaml train.py
```

### Multi-GPU with custom config

```bash
uv run accelerate launch --num_processes 4 --multi_gpu --mixed_precision bf16 train.py
```

## Inference

```bash
# Using UV
uv run python demo_inference.py

# Or with activated environment
python demo_inference.py
```

## Testing

```bash
# Run all fast tests
uv run pytest tests/ -v -m "not slow"

# Run specific test suite
uv run pytest tests/test_model_components.py -v

# Validate setup
python validate_setup.py
```

## Project Structure

```
smollm-diffusion-agent/
├── data/              # Dataset loading, schema scaffolding, metrics
├── model/             # Hybrid model, diffusion head, attention blocks
├── tests/             # Test suite
├── config.yaml        # Main configuration file
├── train.py           # Training script
├── inference.py       # Inference pipeline
└── requirements.txt   # Dependencies
```

## Key Features

- **Schema Scaffolding**: Python generates JSON structure, model only predicts values (0% syntax errors)
- **Diffusion Head**: Lightweight bidirectional attention for structured generation
- **Memory Optimized**: Chunked logits computation, only processes masked positions
- **Multi-GPU Support**: Distributed training with Accelerate
- **Muon Optimizer**: Matrix-aware optimizer for hidden layers (auto-falls back to AdamW on single GPU)
- **OneCycleLR**: Learning rate scheduling for faster convergence

## Troubleshooting

### CUDA Out of Memory

- Reduce `batch_size` in `config.yaml` (try 3 or 2 per GPU)
- Increase `gradient_accumulation_steps` to maintain effective batch size
- Enable `use_gradient_checkpointing: true` in config
- Check `bucket_sizes` match your sequence length distribution

### Dataset Cache

Processed datasets are cached in `data_cache/` to avoid reprocessing on multi-GPU runs. Clear cache if you change dataset config:

```bash
rm -rf data_cache/
```

### FlashAttention Installation Issues

FlashAttention requires compilation or prebuilt wheels. For CUDA 12.8 + PyTorch 2.9, use prebuilt wheels from:
https://github.com/mjun0812/flash-attention-prebuild-wheels

## Citation

If you use this codebase, please cite the original papers:
- MDLM (Masked Discrete Diffusion Language Models)
- dLLM-CtrlGen (Schema Scaffolding)
- Muon Optimizer
