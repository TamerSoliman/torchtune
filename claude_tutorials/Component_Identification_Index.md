# torchtune Component Identification Index

This document catalogs the key modular components discovered during architecture analysis, organized by type and purpose.

---

## 8 Key Modular Components (Analyzed in Detail)

### 1. LoRA Module
- **File:** `torchtune/modules/peft/lora.py`
- **Class:** `LoRALinear`
- **Purpose:** Parameter-efficient fine-tuning adapter
- **Design Pattern:** Adapter Pattern
- **Key Features:**
  - Wraps nn.Linear with low-rank matrices
  - Freezes base weights, trains adapters
  - Supports quantization (QLoRA)
- **Annotated Version:** `01_annotated_lora_module.py`

### 2. Model Builder
- **File:** `torchtune/models/llama3_1/_model_builders.py`
- **Functions:** `lora_llama3_1_8b()`, `lora_llama3_1_70b()`, etc.
- **Purpose:** Factory functions for creating model instances
- **Design Pattern:** Two-Level Builder Pattern
- **Key Features:**
  - Provides sensible defaults for known configurations
  - Enables easy variant creation (8B, 70B, 405B)
  - Supports LoRA, QLoRA, DoRA variants
- **Annotated Version:** `02_annotated_model_builders.py`

### 3. Dataset Builder
- **File:** `torchtune/datasets/_alpaca.py`
- **Function:** `alpaca_cleaned_dataset()`
- **Purpose:** Factory for creating dataset instances
- **Design Pattern:** Factory Pattern
- **Key Features:**
  - Handles data source loading
  - Applies message transforms
  - Supports packing and batching
- **Referenced in:** Guides and README

### 4. Configuration Instantiation
- **File:** `torchtune/config/_instantiate.py`
- **Function:** `instantiate()`
- **Purpose:** Dynamic object creation from config dictionaries
- **Design Pattern:** Dependency Injection
- **Key Features:**
  - Recursive instantiation
  - Support for positional arguments
  - Config interpolation
  - Import resolution
- **Annotated Version:** `03_annotated_config_instantiation.py`

### 5. Configuration Parsing
- **File:** `torchtune/config/_parse.py`
- **Class:** `TuneRecipeArgumentParser`
- **Decorator:** `@parse`
- **Purpose:** Load YAML configs and merge CLI overrides
- **Design Pattern:** Parser/Decorator
- **Key Features:**
  - YAML file loading
  - CLI override merging
  - OmegaConf integration
- **Referenced in:** `Config_Loading_Guide.md`

### 6. Checkpointer
- **File:** `torchtune/training/checkpointing/_checkpointer.py`
- **Classes:** `FullModelHFCheckpointer`, `FullModelMetaCheckpointer`, etc.
- **Purpose:** Load pretrained weights, save fine-tuned checkpoints
- **Design Pattern:** Strategy Pattern
- **Key Features:**
  - Multiple format support (HF, Meta, torchtune)
  - Adapter-only saving
  - Recipe state management
- **Referenced in:** `Training_Recipe_Lifecycle_Guide.md`

### 7. Transformer Module
- **File:** `torchtune/modules/transformer.py`
- **Classes:** `TransformerSelfAttentionLayer`, `TransformerDecoder`
- **Purpose:** Core transformer architecture building blocks
- **Design Pattern:** Composition
- **Key Features:**
  - Modular layer design
  - Support for cross-attention
  - KV-cache for inference
  - Flexible normalization and scaling
- **Referenced in:** Guides

### 8. Attention Module
- **File:** `torchtune/modules/attention.py`
- **Class:** `MultiHeadAttention`
- **Purpose:** Multi-head attention with GQA support
- **Design Pattern:** Composition
- **Key Features:**
  - Grouped Query Attention (GQA)
  - Positional embeddings support
  - KV-caching
  - Flexible attention dropout
- **Referenced in:** Guides

---

## 7 Training Recipes (Complete YAML Configs)

### 1. Llama 3.1 8B LoRA Single Device
- **File:** `recipes/configs/llama3_1/8B_lora_single_device.yaml`
- **Recipe:** `recipes/lora_finetune_single_device.py`
- **Use Case:** Basic LoRA fine-tuning on single GPU
- **Key Settings:**
  - Model: `lora_llama3_1_8b`
  - Rank: 8, Alpha: 16
  - Dataset: Alpaca Cleaned
  - Device: Single CUDA GPU
- **Annotated Version:** `04_annotated_lora_recipe.yaml`

### 2. Llama 3.1 8B Full Fine-Tuning
- **File:** `recipes/configs/llama3_1/8B_full.yaml`
- **Recipe:** `recipes/full_finetune_distributed.py`
- **Use Case:** Full parameter fine-tuning (distributed)
- **Key Settings:**
  - Model: `llama3_1_8b`
  - All 8B parameters trainable
  - Requires multiple GPUs
  - Distributed training enabled

### 3. Llama 2 7B QLoRA Single Device
- **File:** `recipes/configs/llama2/7B_qlora_single_device.yaml`
- **Recipe:** `recipes/lora_finetune_single_device.py`
- **Use Case:** Quantized LoRA for maximum memory efficiency
- **Key Settings:**
  - Model: `qlora_llama2_7b`
  - Base weights: 4-bit NF4 quantization
  - LoRA adapters: Full precision
  - Memory: ~10GB total

### 4. Llama 3.1 8B LoRA DPO
- **File:** `recipes/configs/llama3_1/8B_lora_dpo_single_device.yaml`
- **Recipe:** `recipes/lora_dpo_single_device.py`
- **Use Case:** Direct Preference Optimization (alignment)
- **Key Settings:**
  - Model: `lora_llama3_1_8b`
  - Loss: DPOLoss
  - Dataset: Preference pairs (chosen/rejected)
  - Requires pre-trained SFT model

### 5. Gemma 2B LoRA
- **File:** `recipes/configs/gemma/2B_lora.yaml`
- **Recipe:** `recipes/lora_finetune_distributed.py`
- **Use Case:** Different model family (Gemma)
- **Key Settings:**
  - Model: `lora_gemma_2b`
  - Tokenizer: `gemma_tokenizer`
  - Demonstrates cross-model compatibility

### 6. Qwen2 7B LoRA
- **File:** `recipes/configs/qwen2/7B_lora.yaml`
- **Recipe:** `recipes/lora_finetune_distributed.py`
- **Use Case:** Another model family (Qwen)
- **Key Settings:**
  - Model: `lora_qwen2_7b`
  - Tokenizer: `qwen2_tokenizer` (BPE-based)
  - Different tokenization approach

### 7. Llama 3.2 Vision 11B LoRA
- **File:** `recipes/configs/llama3_2_vision/11B_lora_single_device.yaml`
- **Recipe:** `recipes/lora_finetune_single_device.py`
- **Use Case:** Multimodal (vision + language) fine-tuning
- **Key Settings:**
  - Model: `lora_llama3_2_vision_11b`
  - Modality: Vision + Text
  - Special tokenizer: `llama3_2_vision_transform`
  - Dataset: Multimodal (images + text)

---

## Additional Important Components

### Model Families Supported

| Family | Location | Sizes | Key Features |
|--------|----------|-------|--------------|
| Llama 2 | `torchtune/models/llama2/` | 7B, 13B, 70B | Original Llama architecture |
| Llama 3 | `torchtune/models/llama3/` | 8B, 70B | Improved tokenizer, GQA |
| Llama 3.1 | `torchtune/models/llama3_1/` | 8B, 70B, 405B | Extended context (128K) |
| Llama 3.2 | `torchtune/models/llama3_2/` | 1B, 3B | Smaller variants |
| Llama 3.2 Vision | `torchtune/models/llama3_2_vision/` | 11B, 90B | Multimodal |
| Gemma | `torchtune/models/gemma/` | 2B, 7B | Google's model |
| Gemma 2 | `torchtune/models/gemma2/` | 2B, 9B, 27B | Improved Gemma |
| Qwen 2 | `torchtune/models/qwen2/` | 0.5B-72B | Alibaba's model |
| Qwen 2.5 | `torchtune/models/qwen2_5/` | Various | Latest Qwen |
| Qwen 3 | `torchtune/models/qwen3/` | Various | Newest Qwen |
| Mistral | `torchtune/models/mistral/` | 7B | Mistral AI model |
| Phi 3 | `torchtune/models/phi3/` | Mini, Small, Medium | Microsoft's small models |
| Phi 4 | `torchtune/models/phi4/` | 14B | Latest Phi |

### PEFT Methods Available

| Method | File | Description |
|--------|------|-------------|
| LoRA | `torchtune/modules/peft/lora.py` | Low-Rank Adaptation |
| QLoRA | `torchtune/modules/peft/lora.py` (with quantization) | Quantized LoRA |
| DoRA | `torchtune/modules/peft/dora.py` | Decomposed LoRA (magnitude + direction) |

### Dataset Types

| Type | Example File | Use Case |
|------|--------------|----------|
| SFT (Supervised) | `torchtune/datasets/_sft.py` | Standard fine-tuning |
| Instruction | `torchtune/datasets/_instruct.py` | Instruction following |
| Chat | `torchtune/datasets/_chat.py` | Conversational |
| Preference | `torchtune/datasets/_preference.py` | DPO/RLHF alignment |
| Packed | `torchtune/datasets/_packed.py` | Efficient packing |
| Multimodal | `torchtune/datasets/multimodal/` | Vision + text |

### Training Infrastructure

| Component | Location | Purpose |
|-----------|----------|---------|
| Checkpointing | `torchtune/training/checkpointing/` | Save/load weights |
| Metric Logging | `torchtune/training/metric_logging.py` | Track training metrics |
| LR Schedulers | `torchtune/training/lr_schedulers.py` | Learning rate schedules |
| Precision | `torchtune/training/precision.py` | Mixed precision utilities |
| Activations | `torchtune/training/activations.py` | Activation checkpointing |
| Memory | `torchtune/training/memory.py` | Memory optimization |
| Distributed | `torchtune/training/_distributed.py` | Multi-GPU support |

### Loss Functions

| Loss | File | Use Case |
|------|------|----------|
| Linear Cross Entropy | `torchtune/modules/loss/linear_ce.py` | Standard language modeling |
| DPO Loss | `torchtune/rlhf/loss/dpo.py` | Preference optimization |
| PPO Loss | `torchtune/rlhf/loss/ppo.py` | Reinforcement learning |

### Utilities

| Utility | Location | Purpose |
|---------|----------|---------|
| Device Management | `torchtune/utils/_device.py` | CUDA/CPU handling |
| Logging | `torchtune/utils/_logging.py` | Logging setup |
| Common Utils | `torchtune/modules/common_utils.py` | Shared utilities |

---

## Recipe Entry Points

### Available Training Recipes

| Recipe | File | Use Case |
|--------|------|----------|
| LoRA Single Device | `recipes/lora_finetune_single_device.py` | Basic LoRA on 1 GPU |
| LoRA Distributed | `recipes/lora_finetune_distributed.py` | LoRA on multiple GPUs |
| Full Single Device | `recipes/full_finetune_single_device.py` | Full fine-tuning, 1 GPU |
| Full Distributed | `recipes/full_finetune_distributed.py` | Full fine-tuning, multi-GPU |
| LoRA DPO Single | `recipes/lora_dpo_single_device.py` | DPO with LoRA, 1 GPU |
| Full DPO Distributed | `recipes/full_dpo_distributed.py` | DPO full model, multi-GPU |
| PPO Full | `recipes/ppo_full_finetune_single_device.py` | PPO training |
| QAT Distributed | `recipes/qat_distributed.py` | Quantization-aware training |
| Knowledge Distillation | `recipes/knowledge_distillation_distributed.py` | Model distillation |
| Generation | `recipes/generate.py` | Text generation |
| Eleuther Eval | `recipes/eleuther_eval.py` | Model evaluation |

---

## Configuration System Files

### Core Config Files

| File | Purpose |
|------|---------|
| `torchtune/config/_parse.py` | YAML loading and parsing |
| `torchtune/config/_instantiate.py` | Dynamic object instantiation |
| `torchtune/config/_utils.py` | Helper functions |
| `torchtune/config/_errors.py` | Custom error types |
| `torchtune/config/_validate.py` | Config validation |

### Registry Files

| File | Purpose |
|------|---------|
| `torchtune/_recipe_registry.py` | Maps recipe names to files |

---

## CLI Entry Points

| Command | File | Purpose |
|---------|------|---------|
| `tune run` | `torchtune/_cli/tune.py` | Run a recipe |
| `tune download` | `torchtune/_cli/tune.py` | Download model weights |
| `tune ls` | `torchtune/_cli/tune.py` | List available recipes |
| `tune cp` | `torchtune/_cli/tune.py` | Copy config to local dir |
| `tune validate` | `torchtune/_cli/tune.py` | Validate a config |
| `tune cat` | `torchtune/_cli/tune.py` | View config contents |

---

## Summary Statistics

- **Model Families:** 10+
- **Model Sizes:** 0.5B to 405B parameters
- **PEFT Methods:** 3 (LoRA, QLoRA, DoRA)
- **Training Recipes:** 11+ Python recipes
- **Config Files:** 50+ YAML configs
- **Dataset Types:** 6+ types
- **Supported Devices:** CUDA, CPU, MPS, XPU

---

## Extension Points

### Easy to Add

1. **New Model Size:** Just create a new builder function
2. **New Dataset:** Implement message transform + builder
3. **New Config:** Copy existing, modify parameters
4. **New Optimizer:** Just reference in YAML config

### Moderate Effort

1. **New Model Family:** Implement component builders + model builders
2. **New PEFT Method:** Implement adapter module + builders
3. **New Recipe:** Implement training logic + checkpointing

### Advanced

1. **New Architecture:** Implement custom transformer blocks
2. **New Training Paradigm:** Implement new recipe type (e.g., new RLHF variant)
3. **New Quantization:** Integrate new quantization backend

---

This index provides a comprehensive catalog of all identified components in the torchtune architecture, organized for easy reference and extension planning.
