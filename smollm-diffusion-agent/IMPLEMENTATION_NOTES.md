# Implementation Notes - SmolLM Diffusion Agent

## Overview

This implementation follows the architecture described in `guide.md` and `IMPLEMENTATION_ANALYSIS.md`, combining components from multiple repositories to create a hybrid diffusion-autoregressive model for function calling.

## Changes Made

### 1. Schema Scaffolding (✅ Complete)

**Source:** `dLLM-CtrlGen/scaffolding/schema.py`

**File:** `data/schema.py`

- Copied the complete schema scaffolding implementation from dLLM-CtrlGen
- Provides `SchemaTemplate` and `build_schema_template` functions
- Handles deterministic JSON structure generation with mask tokens
- Note: `data/schema_builder.py` also exists with similar functionality (pre-existing)

**Key Features:**
- Deterministic scaffold building (Python generates structure, not LLM)
- Track mask positions per field with `FieldSegment`
- Guarantee 0% syntax errors in JSON output

### 2. Noise Scheduling (✅ Complete)

**Source:** `mdlm/noise_schedule.py`

**File:** `model/noise_schedule.py`

- Copied `LogLinearNoise` class from mdlm
- Implements proper discrete diffusion noise schedule
- Built such that `1 - 1/e^(n(t))` interpolates smoothly from 0 to ~1

**Key Features:**
- `total_noise(t)`: Returns accumulated noise at timestep t
- `rate_noise(t)`: Returns rate of change of noise (g(t))
- `forward(t)`: Returns both total and rate noise
- Importance sampling transformation for variance reduction

### 3. Diffusion Head (✅ Complete Rewrite)

**Sources:**
- `mdlm/diffusion.py` (lines 575-586 for q_xt, lines 592-637 for _ddpm_update)
- Architecture guidance from `guide.md`

**File:** `model/diffusion_head.py`

**Major Changes:**
1. Added `forward_diffusion()` method for proper noising (from mdlm)
2. Integrated `LogLinearNoise` for scheduling
3. Replaced full Transformer with lightweight residual blocks
4. Added `training_step()` method with D3PM-style loss
5. Direct prediction of clean tokens (not noise)

**Architecture:**
```
Input: Hidden states from base model + Current tokens + Timestep
       ↓
Input Projection + Token Embedding + Time Embedding
       ↓
Residual Blocks (2 layers, GELU activation)
       ↓
Output Projection → Logits
```

**Key Methods:**
- `forward_diffusion(tokens, scaffold_mask, t)`: Add noise to tokens based on timestep
- `predict(hidden_states, current_tokens, t)`: Predict original tokens from noisy version
- `training_step(tokens, hidden_states, scaffold_mask)`: Full training forward pass with diffusion loss
- `set_mask_token_id(mask_token_id)`: Set the mask token ID for proper noising

### 4. Inference Pipeline (✅ Complete Rewrite)

**Source:** `dLLM-CtrlGen/decoding/generator.py` (lines 151-278)

**File:** `inference.py`

**Major Changes:**
1. Complete rewrite following dLLM-CtrlGen's S3 generation loop
2. Implements top-K remasking strategy (70/30 keep/update strategy)
3. Confidence-based token selection
4. Adaptive budget calculation

**Key Components:**

**`FunctionCallGenerator` class:**
- `generate()`: Main inference loop
  - Encodes prompt with chat template
  - Concatenates prompt + scaffold
  - Iterative diffusion denoising (configurable steps, default 4)
  - Top-K remasking: keeps highest confidence predictions per step
  - Early stopping when all masks filled

**Generation Flow:**
```
1. Encode prompt → prompt_ids
2. Build scaffold template → template
3. Initialize sequence: [prompt_ids | scaffold_ids]
4. For each diffusion step (4 → 0):
   a. Find mask positions
   b. Forward through model → logits
   c. Calculate confidence scores (log-probs)
   d. Select top-K highest confidence predictions
   e. Update sequence with selected predictions
5. Decode final sequence → JSON output
```

**`GenerationConfig` dataclass:**
- `steps`: Number of diffusion steps (default 4)
- `temperature`: Sampling temperature (default 0.0 = greedy)
- `cfg_scale`: Classifier-free guidance scale
- `topk_remask`: Optional fixed K for top-K selection

### 5. Training Updates (✅ Complete)

**File:** `train.py`

**Changes:**
- Added `model.diffusion_head.set_mask_token_id(tokenizer.mask_token_id)` to properly initialize the diffusion head
- Updated comments to clarify diffusion loss computation

**File:** `model/hybrid_model.py`

**Changes:**
1. Updated `SchemaDiffusionHead` initialization with proper parameters:
   - `input_dim`, `vocab_size`, `hidden_dim=1024`, `num_layers=2`, `num_steps=4`
2. Modified forward pass to use new `training_step()` method:
   - Calls `self.diffusion_head.training_step()` which handles noising + prediction + loss internally
   - Follows mdlm-style D3PM loss computation

**Training Flow:**
```
For each batch:
  1. Base LLM (frozen) → hidden_states
  2. Diffusion head training_step():
     a. Sample random timestep t ~ U(0,1)
     b. Forward diffusion: add noise to tokens
     c. Predict clean tokens from noisy version
     d. Compute cross-entropy loss on masked positions only
  3. Backprop through diffusion head only (base LLM frozen)
```

### 6. Dataset Loader Updates (✅ Complete)

**File:** `data/dataset_loader.py`

**Changes:**
- Updated import to use `from .schema import build_schema_template`
- Already uses proper schema scaffolding from existing `schema_builder.py`

## Architecture Summary

### Complete Flow

```
User Query + Tools → Base Model (frozen)
                ↓
    Base Model → Decides: Chat or Tool Call?
                ↓
        (If Tool Call)
                ↓
    Base Model → Function Name (via native tool calling)
                ↓
    Python → Build Scaffold Template
                ↓
    Diffusion Head → Fill Masks (2-4 steps)
                ↓
    Python → Merge Scaffold + Predictions
                ↓
    Valid JSON Output

Note: We use the frozen base model's native tool-calling capability
instead of a separate router head. SmolLM3 already understands when
to use tools via its built-in chat template with tool injection.
```

### Key Design Decisions

1. **Schema Scaffolding**: Python generates JSON structure, LLM only predicts values
   - Guarantees 0% syntax errors
   - Prevents schema hallucination

2. **Noise Schedule**: LogLinearNoise from mdlm
   - Proven for discrete text diffusion
   - Smooth interpolation from 0 to ~1

3. **Architecture**: Lightweight diffusion head
   - 2-layer residual blocks (not full transformer)
   - Faster inference, less memory

4. **Training**: D3PM-style loss
   - Direct prediction of clean tokens
   - Loss only on masked positions
   - Random timestep sampling during training

5. **Inference**: Top-K remasking (S3 strategy)
   - Adaptive budget per step
   - Confidence-based selection
   - 2-4 denoising steps

## Code Citations

All borrowed code properly cited in file headers:

- `data/schema.py`: From `dLLM-CtrlGen/scaffolding/schema.py`
- `model/noise_schedule.py`: From `mdlm/noise_schedule.py`
- `model/diffusion_head.py`: Based on `mdlm/diffusion.py` (q_xt, _ddpm_update)
- `inference.py`: Based on `dLLM-CtrlGen/decoding/generator.py`

## Verification Against Guide

### ✅ Matches guide.md specifications:

1. **Base Model**: SmolLM3-3B (frozen) ✅
2. **Diffusion Head**: Lightweight MLP/Residual blocks ✅
3. **Noise Schedule**: LogLinearNoise ✅
4. **Schema Scaffolding**: Python-generated templates ✅
5. **Inference Steps**: 2-4 steps (configurable) ✅
6. **Top-K Remasking**: Implemented ✅
7. **Training Loss**: D3PM-style cross-entropy on masked tokens ✅

### ✅ Matches IMPLEMENTATION_ANALYSIS.md recommendations:

1. **Schema from dLLM-CtrlGen**: ✅ Copied directly
2. **Noise from mdlm**: ✅ LogLinearNoise class
3. **Diffusion mechanics from mdlm**: ✅ q_xt, forward_diffusion
4. **S3 generation from dLLM-CtrlGen**: ✅ Complete generator with top-K
5. **Proper loss functions**: ✅ D3PM loss via training_step

## Recent Updates (Jan 2026)

### Device Support & Configuration Cleanup (✅ Complete)

**Changes:**
1. **Removed MLX integration from main pipeline**
   - MLX code moved to separate `train_mlx.py` (optional)
   - Main code is now PyTorch-only (CUDA + MPS + CPU)

2. **Automatic device configuration** (`data/config_utils.py`)
   - `validate_and_adjust_config()`: Auto-disables unsupported features per device
   - `get_model_kwargs()`: Extracts model init kwargs from config
   - `get_inference_kwargs()`: Extracts inference kwargs from config
   - `print_device_capabilities()`: Shows available hardware/libraries

3. **Platform support:**
   - **CUDA** (Primary): Full speed with 4-bit quantization, unsloth, CUDA graphs
   - **MPS** (Mac): Inference-ready, uses bfloat16, auto-disables CUDA features
   - **CPU**: Supported but slow (testing only)

4. **Config cleanup:**
   - Removed `backend`, `mlx_base_model_id` from config.yaml
   - Removed deprecated `use_kv_cache` setting
   - Fixed `accelerate_config.yaml` (was forcing CPU training)
   - Updated `requirements.txt` (CUDA-only packages now optional)

5. **Simplified model initialization:**
   ```python
   # Old way (many parameters):
   model = HybridSmolLM(base_model_id, mlx_base_model_id, backend, load_in_4bit, use_unsloth, ...)
   
   # New way (via config):
   model_kwargs = get_model_kwargs(config, device)
   model = HybridSmolLM(**model_kwargs)
   ```

### Implemented Features

1. **`<NULL>` token support**: ✅ Complete
   - Variable-length fields with automatic budgeting
   - Self-adaptive masking

2. **Bidirectional attention**: ✅ Complete
   - Added `BidirectionalAttentionBlock` in diffusion head
   - Global constraint verification for structured output
   - Configurable via `use_bidirectional` in config

3. **CUDA graph optimization**: ✅ Complete
   - Implemented in inference.py
   - Auto-disabled on non-CUDA devices
   - Reduces kernel launch overhead

4. **Hidden states caching**: ✅ Complete
   - Cache base model hidden states once per generation
   - Matches training behavior (predict from initial masked scaffold)

### Missing Features (Optional Enhancements)

1. **Block-wise parallel generation**: From Discrete-Diffusion-Forcing
   - Would allow parallel generation of multiple parameters
   - Current implementation is sequential

2. **Decision Token mechanism**: From MediaTek paper
   - Explicit `<|answer|>` vs `<|use_tool|>` tokens
   - Currently using base model's native tool calling instead

## Next Steps

1. **Test the implementation**: ✅ Complete
   - ✅ Test suite passes (54 fast tests)
   - ✅ Setup validation script works
   - ✅ Model loads and runs forward pass on CUDA and MPS

2. **Evaluate**:
   - Run BFCL benchmark for function calling accuracy
   - Measure hallucination rate
   - Profile latency per mode

3. **Optional enhancements**:
   - Implement block-wise parallel generation (from Discrete-Diffusion-Forcing)
   - Add Decision Token mechanism (explicit `<|answer|>` vs `<|use_tool|>` tokens)

## File Structure

```
smollm-diffusion-agent/
├── data/
│   ├── schema.py                 # ✅ From dLLM-CtrlGen
│   ├── schema_builder.py         # Pre-existing (similar to schema.py)
│   ├── dataset_loader.py         # ✅ Updated import
│   ├── generate_scaffolds.py     # Pre-existing
│   ├── config_utils.py           # ✅ NEW: Device config validation
│   ├── device_utils.py           # Device detection and utilities
│   ├── budget_utils.py           # Field budget calculation
│   ├── smollm3_prompting.py      # SmolLM3 chat template utilities
│   └── utils.py                  # General utilities
├── model/
│   ├── __init__.py
│   ├── noise_schedule.py         # ✅ From mdlm (LogLinearNoise)
│   ├── diffusion_head.py         # ✅ REWRITTEN: mdlm-style diffusion
│   └── hybrid_model.py           # ✅ Updated: PyTorch-only, device-aware
├── tests/
│   ├── test_smoke.py             # Quick sanity checks
│   ├── test_model_components.py  # Unit tests for model parts
│   ├── test_dataset.py           # Dataset loading and format tests
│   ├── test_inference_pipeline.py # Inference utilities tests
│   ├── test_smollm3_prompting.py # Template and parsing tests
│   └── conftest.py               # Pytest configuration
├── inference.py                  # ✅ REWRITTEN: S3 generation loop
├── train.py                      # ✅ Updated: device validation
├── validate_setup.py             # ✅ NEW: Setup validation script
├── config.yaml                   # ✅ Updated: platform-agnostic
├── accelerate_config.yaml        # ✅ Fixed: bf16, auto-detect device
├── requirements.txt              # ✅ Updated: optional CUDA packages
└── IMPLEMENTATION_NOTES.md       # ✅ This file
```

## Platform Support

### Supported Devices:

| Device | Training | Inference | Quantization | Optimizations |
|--------|----------|-----------|--------------|---------------|
| **CUDA** | ✅ Fast | ✅ Fast | 4-bit (bitsandbytes) | unsloth, CUDA graphs, torch.compile |
| **MPS (Apple Silicon)** | ⚠️ Slow | ✅ Good | ❌ (uses bfloat16) | torch.compile (2.1+) |
| **CPU** | ❌ Very Slow | ❌ Slow | ❌ | None |

### Automatic Configuration:

The system automatically adjusts settings via `data/config_utils.py`:
- Disables 4-bit quantization on MPS/CPU
- Disables unsloth on non-CUDA devices
- Disables CUDA graphs on MPS/CPU
- Disables torch.compile on unsupported platforms

**Usage:**
```python
from data.device_utils import get_device
from data.config_utils import validate_and_adjust_config, get_model_kwargs

device = get_device()
config = validate_and_adjust_config(config, device)
model_kwargs = get_model_kwargs(config, device)
model = HybridSmolLM(**model_kwargs)
```

## Training Metrics

### WandB Metrics (✅ Complete)

**File:** `data/metrics.py`, integrated into `train.py`

The training pipeline tracks specialized metrics for function calling with diffusion:

#### 1. NULL Token Behavior Metrics
Tracks the model's use of the special NULL token for unused scaffold slots:

- **`train/null_prediction_rate`**: % of predictions that are NULL tokens (logged every 50 steps)
- **`train/null_accuracy`**: Accuracy on positions that should be NULL
- **`train/real_token_accuracy`**: Accuracy on non-NULL positions (most important!)
- **`eval/null_precision`**: Of predicted NULLs, how many are correct (detects false NULL predictions)
- **`eval/null_recall`**: Of actual NULLs, how many are predicted (detects missed NULL slots)

**Why this matters**: Helps detect over-NULLing patterns (e.g., if model predicts 70-95% NULL tokens like in early training).

#### 2. Scaffold Statistics
Track the size and composition of scaffolds during evaluation:

- **`eval/avg_scaffold_size`**: Average number of tokens in scaffolds
- **`eval/avg_mask_count`**: Average masked positions per example
- **`eval/avg_null_ratio`**: Average ratio of NULL to total scaffold tokens
- **`eval/scaffold_size_std`**: Variability in scaffold sizes (detects outliers)
- **`eval/max_scaffold_size`** / **`min_scaffold_size`**: Range of scaffold sizes

**Why this matters**: Validates that `mask_budget` configuration is appropriate for the dataset.

#### 3. Training Loss Components
Standard loss tracking with enhanced detail:

- **`train/total_loss`**: Overall loss (every step)
- **`train/diffusion_loss`**: Diffusion head loss specifically
- **`train/learning_rate`**: LR schedule tracking (every 100 steps)

#### 4. Key Metrics to Watch

| Metric | Good Pattern | Bad Pattern | Action |
|--------|--------------|-------------|--------|
| `real_token_accuracy` | Increasing (>80%) | Stuck <50% | Check data quality, increase capacity |
| `null_prediction_rate` | Decreasing over time | Stays >70% | Reduce `mask_budget`, check loss weights |
| `null_precision` | >0.8 | <0.5 | Model over-predicting NULLs |
| `avg_null_ratio` | Stable 0.2-0.4 | >0.7 | Dataset has too many NULL slots |

#### 5. Additional Metrics (Available in `metrics.py`)

The metrics module also includes functions for:
- **Field-level accuracy**: Track accuracy per argument field (location, units, etc.)
- **Parse metrics**: JSON parse success rate, format validation
- **Confidence metrics**: Track model confidence vs correctness

These can be added to functional evaluation as needed.

**Usage:**
```python
from data.metrics import (
    calculate_null_token_metrics,
    calculate_field_level_metrics,
    calculate_parse_metrics,
    calculate_scaffold_metrics
)
```

All metrics are automatically logged to WandB during training when `use_wandb: true` in config.

## Testing

Run the test suite:
```bash
# Quick smoke tests
pytest tests/test_smoke.py -v

# All fast tests
pytest tests/ -v -m "not slow"

# Metrics tests
pytest tests/test_metrics.py -v

# All tests including slow ones
pytest tests/ -v

# Validate setup
python validate_setup.py
```

**Test Coverage:**
- ✅ Data pipeline (dataset loading, scaffolding)
- ✅ Model components (diffusion head, forward pass)
- ✅ SmolLM3 prompting (chat template, tool injection)
- ✅ Inference pipeline (schema building, parsing)
- ✅ Device configuration (auto-adjustment)
- ✅ Metrics calculation (NULL tokens, fields, parsing)

All 61 fast tests pass on both CUDA and MPS (54 original + 7 metrics tests).
