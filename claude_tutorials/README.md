# torchtune Modular Architecture & Recipe System - Complete Guide

Welcome to the comprehensive guide for understanding torchtune's modular, component-based architecture and how YAML configurations drive the entire training system.

---

## 📚 Table of Contents

1. [Overview](#overview)
2. [What's Included](#whats-included)
3. [Learning Path](#learning-path)
4. [Quick Reference](#quick-reference)
5. [Key Concepts](#key-concepts)
6. [Component Map](#component-map)

---

## Overview

**torchtune** is built on a highly modular architecture where:
- **YAML configurations** specify all training components (model, optimizer, dataset, etc.)
- **Dynamic instantiation** creates Python objects from these configurations
- **Composable builders** allow mixing and matching components
- **Clear interfaces** enable easy extension and customization

This tutorial package provides heavily annotated source files, comprehensive guides, and real-world examples to help you understand how all the pieces fit together.

---

## What's Included

### 📖 Comprehensive Guides

| Guide | Description | When to Read |
|-------|-------------|--------------|
| **Config_Loading_Guide.md** | Complete explanation of how YAML configs become Python objects, including the instantiation system and 10 critical file paths | First - understand the foundation |
| **Training_Recipe_Lifecycle_Guide.md** | Full walkthrough of a training run from CLI command to final checkpoint, including all phases and component interactions | Second - see it in action |

### 🔍 Annotated Source Files

| File | Original Source | Focus Area |
|------|----------------|------------|
| **01_annotated_lora_module.py** | `torchtune/modules/peft/lora.py` | LoRA implementation, adapter pattern, QLoRA, parameter-efficient fine-tuning |
| **02_annotated_model_builders.py** | `torchtune/models/llama3_1/_model_builders.py` | Two-level builder pattern, factory functions, model size variants (8B/70B/405B) |
| **03_annotated_config_instantiation.py** | `torchtune/config/_instantiate.py` | Dependency injection engine, recursive instantiation, dynamic object creation |
| **04_annotated_lora_recipe.yaml** | `recipes/configs/llama3_1/8B_lora_single_device.yaml` | Complete training configuration with detailed explanations of every field |

### 🗂️ Component Reference

This README also includes:
- **Component Map**: Visual diagram of how all pieces connect
- **Key Concepts**: Core architectural patterns
- **Quick Reference**: Fast lookup for common tasks

---

## Learning Path

### 🎯 Beginner Track: "I want to understand torchtune's architecture"

1. **Start Here:** `Config_Loading_Guide.md`
   - Learn about the config-driven architecture
   - Understand `_component_` and `instantiate()`
   - See the complete flow from YAML to Python objects

2. **Next:** `04_annotated_lora_recipe.yaml`
   - Explore a real training configuration
   - Understand what each section does
   - Learn how to customize configs

3. **Then:** `Training_Recipe_Lifecycle_Guide.md`
   - Follow a complete training run
   - See all components working together
   - Understand the full lifecycle

4. **Deep Dive:** Annotated source files
   - `01_annotated_lora_module.py` - Understand LoRA
   - `02_annotated_model_builders.py` - Understand model construction
   - `03_annotated_config_instantiation.py` - Understand dynamic instantiation

### 🚀 Advanced Track: "I want to extend torchtune"

1. **Quick Review:** `Component_Map` (below) + `Quick Reference`
   - Get the big picture
   - Identify extension points

2. **Study Patterns:**
   - `02_annotated_model_builders.py` - Learn the builder pattern
   - `03_annotated_config_instantiation.py` - Learn dependency injection

3. **Pick Your Extension:**
   - **New Model?** Study `02_annotated_model_builders.py` + `torchtune/models/`
   - **New Dataset?** Study `torchtune/datasets/_alpaca.py`
   - **New PEFT Method?** Study `01_annotated_lora_module.py`
   - **New Training Method?** Study `recipes/lora_finetune_single_device.py`

4. **Implement & Test:**
   - Create your component (following existing patterns)
   - Add builder function
   - Create YAML config
   - Test with `tune run`

### 🎓 Instructor Track: "I want to teach torchtune"

Use these materials as teaching resources:
1. **Lecture 1:** Config system (`Config_Loading_Guide.md`)
2. **Lecture 2:** Model architecture (`02_annotated_model_builders.py` + `01_annotated_lora_module.py`)
3. **Lecture 3:** Training flow (`Training_Recipe_Lifecycle_Guide.md`)
4. **Lab:** Modify `04_annotated_lora_recipe.yaml` and run experiments

---

## Quick Reference

### How to... Find the Right File

| I want to... | Look at... |
|-------------|------------|
| Understand LoRA | `01_annotated_lora_module.py` |
| Create a new model variant | `02_annotated_model_builders.py` |
| Understand config → objects | `03_annotated_config_instantiation.py` |
| Customize training settings | `04_annotated_lora_recipe.yaml` |
| Understand full training flow | `Training_Recipe_Lifecycle_Guide.md` |
| Understand config loading | `Config_Loading_Guide.md` |
| Find critical file paths | `Config_Loading_Guide.md` (Section: "10 Critical File Paths") |
| Understand component interactions | `Training_Recipe_Lifecycle_Guide.md` (Section: "Component Interactions") |

### How to... Customize

| I want to... | Edit this in YAML... |
|-------------|---------------------|
| Use different model size | `model._component_` → `lora_llama3_1_70b` |
| Change LoRA rank | `model.lora_rank: 16` |
| Try QLoRA | `model._component_` → `qlora_llama3_1_8b` |
| Adjust learning rate | `optimizer.lr: 5e-4` |
| Change batch size | `batch_size: 4` |
| Enable gradient checkpointing | `enable_activation_checkpointing: True` |
| Use different dataset | `dataset._component_` → `torchtune.datasets.chat_dataset` |
| Change number of epochs | `epochs: 3` |

### How to... Debug

| Issue | Check... |
|-------|----------|
| Config not loading | `Config_Loading_Guide.md` - Phase 2 |
| Component not instantiating | `03_annotated_config_instantiation.py` - `_get_component_from_path` |
| Model structure wrong | `02_annotated_model_builders.py` - builder functions |
| LoRA not being applied | `01_annotated_lora_module.py` - `adapter_params()` |
| Training not starting | `Training_Recipe_Lifecycle_Guide.md` - Phase 4 & 5 |
| Checkpoints not saving | `Training_Recipe_Lifecycle_Guide.md` - Phase 6 |
| Out of memory | `Training_Recipe_Lifecycle_Guide.md` - "Memory Management" section |

---

## Key Concepts

### 1. **Dependency Injection**
```yaml
# Declare WHAT you want
model:
  _component_: torchtune.models.llama3_1.lora_llama3_1_8b
  lora_rank: 8
```
```python
# System handles HOW to create it
model = config.instantiate(cfg.model)  # Model created automatically
```

**Why it matters:** Change components without changing code. Perfect for experimentation.

---

### 2. **Two-Level Builder Pattern**
```
Level 1: Component Builder (flexible)
  ↓
  lora_llama3_1(vocab_size, num_layers, ..., lora_rank, ...)
  (Accepts ALL parameters)

Level 2: Model Builder (convenient)
  ↓
  lora_llama3_1_8b(lora_rank, ...)
  (8B defaults provided automatically)
```

**Why it matters:** Best of both worlds - flexibility AND convenience.

---

### 3. **Adapter Pattern (LoRA)**
```
Standard Linear Layer:
  nn.Linear(in_dim, out_dim)

LoRA-Enhanced:
  LoRALinear(in_dim, out_dim, rank, alpha)
  ├── base weight (frozen)
  └── lora_a + lora_b (trainable)
```

**Why it matters:** Train 0.1% of parameters, get 95%+ of full fine-tuning quality.

---

### 4. **Config Interpolation**
```yaml
output_dir: /tmp/output

checkpointer:
  output_dir: ${output_dir}  # Auto-resolves to /tmp/output

logger:
  log_dir: ${output_dir}/logs  # Auto-resolves to /tmp/output/logs
```

**Why it matters:** DRY principle - define once, reference everywhere.

---

### 5. **Recursive Instantiation**
```yaml
model:
  _component_: Model
  encoder:
    _component_: Encoder  # Nested!
    num_layers: 6
```
```python
# Automatically creates nested structure:
model = Model(encoder=Encoder(num_layers=6))
```

**Why it matters:** Compose complex architectures declaratively.

---

## Component Map

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER LAYER                               │
├─────────────────────────────────────────────────────────────────┤
│  • CLI Commands (tune run, tune download, etc.)                 │
│  • YAML Configurations (recipes/configs/*.yaml)                 │
│  • Override Syntax (key=value)                                  │
└─────────────────────────────────────────────────────────────────┘
                             ↓ ↓ ↓
┌─────────────────────────────────────────────────────────────────┐
│                      CONFIGURATION LAYER                         │
├─────────────────────────────────────────────────────────────────┤
│  torchtune/config/                                              │
│  ├── _parse.py         → Load YAML + merge CLI overrides       │
│  ├── _instantiate.py   → Create objects from config            │
│  └── _utils.py         → Import resolution helpers             │
└─────────────────────────────────────────────────────────────────┘
                             ↓ ↓ ↓
┌─────────────────────────────────────────────────────────────────┐
│                       RECIPE LAYER                               │
├─────────────────────────────────────────────────────────────────┤
│  recipes/                                                        │
│  ├── lora_finetune_single_device.py   → LoRA training          │
│  ├── full_finetune_distributed.py    → Full fine-tuning        │
│  ├── lora_dpo_single_device.py       → DPO alignment           │
│  └── ... (15+ recipes)                                          │
│                                                                  │
│  Each recipe:                                                    │
│  • Receives DictConfig                                          │
│  • Instantiates components                                      │
│  • Implements training loop                                     │
│  • Handles checkpointing                                        │
└─────────────────────────────────────────────────────────────────┘
                             ↓ ↓ ↓
┌─────────────────────────────────────────────────────────────────┐
│                     COMPONENT LAYER                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌───────────────┐  ┌───────────────┐  ┌──────────────┐        │
│  │    MODELS     │  │   DATASETS    │  │   TRAINING   │        │
│  ├───────────────┤  ├───────────────┤  ├──────────────┤        │
│  │ • Model       │  │ • Dataset     │  │ • Checkpoint │        │
│  │   Builders    │  │   Builders    │  │ • Optimizer  │        │
│  │ • LoRA        │  │ • Transforms  │  │ • LR Sched.  │        │
│  │ • Attention   │  │ • Collation   │  │ • Loss Fns   │        │
│  │ • FFN         │  │ • Tokenizers  │  │ • Logging    │        │
│  │ • Embeddings  │  │               │  │              │        │
│  └───────────────┘  └───────────────┘  └──────────────┘        │
│                                                                  │
│  torchtune/                                                      │
│  ├── models/           → Model architectures & builders         │
│  ├── modules/          → Reusable building blocks              │
│  ├── datasets/         → Dataset implementations               │
│  ├── training/         → Training infrastructure               │
│  └── config/           → Configuration system                  │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
YAML Config
    ↓
[_parse.py] Load & Parse
    ↓
DictConfig
    ↓
[Recipe] Instantiate Components
    ↓
[_instantiate.py] Dynamic Creation
    ↓                    ↓                    ↓
Model Builder      Dataset Builder    Optimizer Class
    ↓                    ↓                    ↓
Model Instance     Dataset Instance   Optimizer Instance
    ↓                    ↓                    ↓
        ┌───────────────────────────────────┐
        │      Training Loop                │
        │  for batch in dataset:            │
        │      logits = model(batch)        │
        │      loss = loss_fn(logits, ...)  │
        │      loss.backward()               │
        │      optimizer.step()              │
        └───────────────────────────────────┘
                    ↓
            Checkpoints Saved
```

---

## Key Files - Mapping Table

### Configuration System (3 files)

| File | Location | Purpose | Annotated Version |
|------|----------|---------|-------------------|
| Config Parser | `torchtune/config/_parse.py` | Load YAML + CLI overrides | See `Config_Loading_Guide.md` |
| Instantiation Engine | `torchtune/config/_instantiate.py` | Create objects from config | `03_annotated_config_instantiation.py` |
| Import Utils | `torchtune/config/_utils.py` | Resolve dotted paths to Python objects | See `Config_Loading_Guide.md` |

### Model Components (8 files identified)

| Component | Location | Purpose | Annotated Version |
|-----------|----------|---------|-------------------|
| 1. LoRA Module | `torchtune/modules/peft/lora.py` | LoRA adapter implementation | `01_annotated_lora_module.py` ✓ |
| 2. Model Builders | `torchtune/models/llama3_1/_model_builders.py` | Factory functions for model variants | `02_annotated_model_builders.py` ✓ |
| 3. Transformer Layers | `torchtune/modules/transformer.py` | Core transformer blocks | Referenced in guides |
| 4. Attention Module | `torchtune/modules/attention.py` | Multi-head attention implementation | Referenced in guides |
| 5. Dataset Builder | `torchtune/datasets/_alpaca.py` | Dataset factory example | Referenced in guides |
| 6. Checkpointer | `torchtune/training/checkpointing/_checkpointer.py` | Load/save checkpoints | See `Training_Recipe_Lifecycle_Guide.md` |
| 7. Config Instantiation | `torchtune/config/_instantiate.py` | Dynamic object creation | `03_annotated_config_instantiation.py` ✓ |
| 8. Config Parsing | `torchtune/config/_parse.py` | YAML loading system | See `Config_Loading_Guide.md` |

### Training Recipes (7 configs identified)

| Recipe Type | Config Location | Use Case | Annotated Version |
|-------------|----------------|----------|-------------------|
| 1. LoRA Single Device | `recipes/configs/llama3_1/8B_lora_single_device.yaml` | Basic LoRA fine-tuning | `04_annotated_lora_recipe.yaml` ✓ |
| 2. Full Fine-Tuning | `recipes/configs/llama3_1/8B_full.yaml` | Full model training | Referenced in guides |
| 3. QLoRA | `recipes/configs/llama2/7B_qlora_single_device.yaml` | Quantized LoRA | Referenced in guides |
| 4. DPO | `recipes/configs/llama3_1/8B_lora_dpo_single_device.yaml` | Preference optimization | Referenced in guides |
| 5. Gemma LoRA | `recipes/configs/gemma/2B_lora.yaml` | Different model family | Referenced in guides |
| 6. Qwen LoRA | `recipes/configs/qwen2/7B_lora.yaml` | Different model family | Referenced in guides |
| 7. Vision LoRA | `recipes/configs/llama3_2_vision/11B_lora_single_device.yaml` | Multimodal training | Referenced in guides |

---

## Extending torchtune

### Adding a New Model

1. **Create component builder** in `torchtune/models/your_model/_component_builders.py`
2. **Create model builder** in `torchtune/models/your_model/_model_builders.py`
3. **Add __init__.py** exports
4. **Create config** in `recipes/configs/your_model/config.yaml`
5. **Test** with `tune run`

**Example:**
```python
# my_model/_model_builders.py
def my_awesome_model_7b(lora_rank=8, ...):
    return my_awesome_model(
        # 7B architecture defaults
        num_layers=32,
        embed_dim=4096,
        # LoRA params
        lora_rank=lora_rank,
        ...
    )
```

```yaml
# configs/my_model/7B_lora.yaml
model:
  _component_: torchtune.models.my_model.my_awesome_model_7b
  lora_rank: 8
```

### Adding a New Dataset

1. **Create dataset builder** in `torchtune/datasets/_my_dataset.py`
2. **Implement message transform** (convert raw data to chat format)
3. **Export** in `torchtune/datasets/__init__.py`
4. **Use in config**

**Example:**
```python
# _my_dataset.py
def my_custom_dataset(tokenizer, **kwargs):
    message_transform = MyTransform()
    return SFTDataset(
        source="my_org/my_dataset",
        message_transform=message_transform,
        model_transform=tokenizer,
        **kwargs
    )
```

### Adding a New PEFT Method

Study `01_annotated_lora_module.py` and follow the same pattern:
1. **Inherit from `nn.Module` and `AdapterModule`**
2. **Implement `adapter_params()` method**
3. **Implement `forward()` with adapter logic**
4. **Add builder function**

---

## Additional Resources

### Original Source Files
All annotated files reference their original sources. You can compare:
- Annotated version (with extensive comments)
- Original version (in `torchtune/`)

### Official Documentation
- torchtune docs: https://pytorch.org/torchtune/
- Config system: https://pytorch.org/torchtune/main/basics/configs.html
- LoRA tutorial: https://pytorch.org/torchtune/main/tutorials/lora_finetune.html

### Papers
- LoRA: https://arxiv.org/abs/2106.09685
- QLoRA: https://arxiv.org/abs/2305.14314
- Llama 2: https://arxiv.org/abs/2307.09288
- Llama 3: https://ai.meta.com/blog/meta-llama-3/

---

## Summary

This tutorial package provides:
- ✅ **4 heavily annotated source files** explaining What, How, and Why
- ✅ **2 comprehensive guides** covering config loading and training lifecycle
- ✅ **Complete component map** showing how everything connects
- ✅ **Quick reference tables** for common tasks
- ✅ **Extension guides** for adding new components

**Start with the guides, dive into annotated sources, then extend the system with your own components!**

---

## Questions?

If something is unclear:
1. Check the relevant annotated file
2. Read the corresponding guide section
3. Look at the Quick Reference
4. Examine the original source file
5. Review the official torchtune docs

Happy learning! 🚀
