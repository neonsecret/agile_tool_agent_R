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
  2. Router head → router_logits
  3. Diffusion head training_step():
     a. Sample random timestep t ~ U(0,1)
     b. Forward diffusion: add noise to tokens
     c. Predict clean tokens from noisy version
     d. Compute cross-entropy loss on masked positions only
  4. Combine losses: diffusion_loss + router_loss (if training router)
  5. Backprop through trainable heads only
```

### 6. Dataset Loader Updates (✅ Complete)

**File:** `data/dataset_loader.py`

**Changes:**
- Updated import to use `from .schema import build_schema_template`
- Already uses proper schema scaffolding from existing `schema_builder.py`

## Architecture Summary

### Complete Flow

```
User Query → Router → Mode Selection
                ↓
        [Chat | Think | Tool]
                ↓
        (If Tool selected)
                ↓
    AR Model → Function Name
                ↓
    Python → Build Scaffold Template
                ↓
    Diffusion Head → Fill Masks (2-4 steps)
                ↓
    Python → Merge Scaffold + Predictions
                ↓
    Valid JSON Output
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

## Missing Features (Optional Enhancements)

These are listed in IMPLEMENTATION_ANALYSIS.md but not yet implemented:

1. **`<NULL>` token support**: For variable-length fields
   - Would allow adaptive field lengths
   - Current implementation uses fixed budgets per field

2. **Block-wise parallel generation**: From Discrete-Diffusion-Forcing
   - Would allow parallel generation of multiple parameters
   - Current implementation is sequential

3. **CUDA graph optimization**: From dInfer
   - Would significantly speed up inference
   - Current implementation is standard PyTorch

4. **Decision Token mechanism**: From MediaTek paper
   - Explicit `<|answer|>` vs `<|use_tool|>` tokens
   - Current router head provides classification but not explicit tokens

5. **Bidirectional attention from LLaDA**: From dInfer
   - Would allow global constraint verification
   - Current implementation uses residual blocks

## Next Steps

1. **Test the implementation**:
   - Run training on small dataset
   - Verify diffusion loss converges
   - Test inference pipeline

2. **Evaluate**:
   - BFCL benchmark for function calling accuracy
   - Measure hallucination rate
   - Profile latency per mode

3. **Optional enhancements**:
   - Add `<NULL>` token support for variable-length fields
   - Implement block-wise parallel generation
   - Add CUDA graph optimization

## File Structure

```
smollm-diffusion-agent/
├── data/
│   ├── schema.py                 # ✅ NEW: From dLLM-CtrlGen
│   ├── schema_builder.py         # Pre-existing (similar to schema.py)
│   ├── dataset_loader.py         # ✅ Updated import
│   └── generate_scaffolds.py     # Pre-existing
├── model/
│   ├── __init__.py
│   ├── noise_schedule.py         # ✅ NEW: From mdlm (LogLinearNoise)
│   ├── diffusion_head.py         # ✅ REWRITTEN: mdlm-style diffusion
│   └── hybrid_model.py           # ✅ Updated: proper initialization
├── inference.py                  # ✅ REWRITTEN: S3 generation loop
├── train.py                      # ✅ Updated: set_mask_token_id
└── IMPLEMENTATION_NOTES.md       # ✅ NEW: This file
```
