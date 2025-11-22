# Training Recipe Lifecycle Guide

## Table of Contents
1. [Overview](#overview)
2. [Complete Training Flow](#complete-training-flow)
3. [Phase-by-Phase Breakdown](#phase-by-phase-breakdown)
4. [Component Interactions](#component-interactions)
5. [Data Flow Through the System](#data-flow-through-the-system)
6. [Checkpointing and Resumption](#checkpointing-and-resumption)
7. [Memory Management](#memory-management)

---

## Overview

This guide walks through the complete lifecycle of a LoRA fine-tuning run in torchtune, from the moment you run the `tune` command to when training completes and checkpoints are saved.

**What You'll Learn:**
- How CLI commands trigger training
- How YAML configs become Python objects
- The full training loop execution
- Dataset preparation and batching
- Model initialization and forward/backward passes
- Checkpointing at epoch boundaries
- How modular components interact

**Example Command:**
```bash
tune run lora_finetune_single_device \
    --config llama3_1/8B_lora_single_device \
    model.lora_rank=16
```

Let's trace exactly what happens!

---

## Complete Training Flow

```
┌────────────────────────────────────────────────────────────────────┐
│                         USER COMMAND                                │
│  $ tune run lora_finetune_single_device --config llama3_1/8B_lora  │
└────────────────────────────────────────────────────────────────────┘
                                 ↓
┌────────────────────────────────────────────────────────────────────┐
│ PHASE 1: CLI & RECIPE DISCOVERY                                    │
│ File: torchtune/_cli/tune.py                                       │
├────────────────────────────────────────────────────────────────────┤
│ 1. Parse command line arguments                                    │
│ 2. Look up recipe in _recipe_registry.py                          │
│ 3. Find recipe file: recipes/lora_finetune_single_device.py       │
│ 4. Find config file: recipes/configs/llama3_1/8B_lora_...yaml    │
│ 5. Execute: python recipes/lora_finetune_single_device.py \       │
│              --config recipes/configs/llama3_1/8B_lora_...yaml    │
└────────────────────────────────────────────────────────────────────┘
                                 ↓
┌────────────────────────────────────────────────────────────────────┐
│ PHASE 2: CONFIGURATION LOADING                                     │
│ File: torchtune/config/_parse.py                                   │
├────────────────────────────────────────────────────────────────────┤
│ @parse decorator wraps recipe main():                              │
│ 1. TuneRecipeArgumentParser created                                │
│ 2. YAML file loaded → OmegaConf DictConfig                         │
│ 3. CLI overrides merged (model.lora_rank=16)                      │
│ 4. DictConfig passed to recipe's main(cfg)                        │
└────────────────────────────────────────────────────────────────────┘
                                 ↓
┌────────────────────────────────────────────────────────────────────┐
│ PHASE 3: RECIPE INITIALIZATION                                     │
│ File: recipes/lora_finetune_single_device.py                       │
├────────────────────────────────────────────────────────────────────┤
│ def main(cfg: DictConfig):                                         │
│     recipe = LoRAFinetuneRecipeSingleDevice(cfg)                  │
│     recipe.setup(cfg)                                              │
│     recipe.train()                                                 │
│     recipe.cleanup()                                               │
└────────────────────────────────────────────────────────────────────┘
                                 ↓
┌────────────────────────────────────────────────────────────────────┐
│ PHASE 4: COMPONENT SETUP                                           │
│ File: recipes/lora_finetune_single_device.py (setup method)        │
├────────────────────────────────────────────────────────────────────┤
│ 1. Set random seeds (reproducibility)                              │
│ 2. Setup device (CUDA/CPU)                                         │
│ 3. Instantiate model:                                              │
│    model = config.instantiate(cfg.model)                          │
│    → LoRALinear layers wrap attention projections                 │
│ 4. Load pretrained checkpoint                                      │
│    → Base weights loaded, LoRA adapters initialized               │
│ 5. Set trainable parameters                                        │
│    → Only LoRA adapters require gradients                         │
│ 6. Setup optimizer:                                                │
│    optimizer = config.instantiate(cfg.optimizer, trainable_params)│
│ 7. Setup learning rate scheduler                                   │
│ 8. Instantiate tokenizer                                           │
│ 9. Setup dataset & dataloader                                      │
│ 10. Setup loss function                                            │
│ 11. Setup metric logger                                            │
│ 12. Enable memory optimizations (activation checkpointing, etc.)  │
└────────────────────────────────────────────────────────────────────┘
                                 ↓
┌────────────────────────────────────────────────────────────────────┐
│ PHASE 5: TRAINING LOOP                                             │
│ File: recipes/lora_finetune_single_device.py (train method)        │
├────────────────────────────────────────────────────────────────────┤
│ for epoch in range(num_epochs):                                    │
│     for step, batch in enumerate(dataloader):                      │
│         # Forward pass                                             │
│         logits = model(batch['input_ids'])                        │
│                                                                     │
│         # Compute loss                                             │
│         loss = loss_fn(logits, batch['labels'])                   │
│                                                                     │
│         # Backward pass                                            │
│         loss.backward()  # Only LoRA params get gradients         │
│                                                                     │
│         # Gradient accumulation check                              │
│         if (step + 1) % gradient_accumulation_steps == 0:         │
│             # Clip gradients (optional)                            │
│             if clip_grad_norm:                                     │
│                 torch.nn.utils.clip_grad_norm_(...)               │
│                                                                     │
│             # Update weights                                       │
│             optimizer.step()                                       │
│             lr_scheduler.step()                                    │
│             optimizer.zero_grad()                                  │
│                                                                     │
│         # Log metrics                                              │
│         if step % log_every_n_steps == 0:                         │
│             logger.log({'loss': loss, 'lr': lr, ...})             │
│                                                                     │
│     # End of epoch                                                 │
│     save_checkpoint(epoch, model, optimizer, ...)                  │
└────────────────────────────────────────────────────────────────────┘
                                 ↓
┌────────────────────────────────────────────────────────────────────┐
│ PHASE 6: CHECKPOINTING                                             │
│ File: recipes/lora_finetune_single_device.py (save_checkpoint)     │
├────────────────────────────────────────────────────────────────────┤
│ After each epoch:                                                   │
│ 1. Extract adapter weights:                                        │
│    adapter_weights = get_adapter_params(model)                    │
│ 2. Save adapter checkpoint:                                        │
│    torch.save(adapter_weights, 'epoch_N/adapter_model.pt')       │
│ 3. Save recipe state:                                              │
│    torch.save({                                                    │
│        'optimizer': optimizer.state_dict(),                       │
│        'lr_scheduler': lr_scheduler.state_dict(),                 │
│        'epoch': epoch,                                             │
│        'seed': seed,                                               │
│        ...                                                          │
│    }, 'epoch_N/recipe_state.pt')                                  │
│ 4. Optionally save full merged model (base + adapters)            │
│ 5. Copy config files for reproducibility                          │
└────────────────────────────────────────────────────────────────────┘
                                 ↓
┌────────────────────────────────────────────────────────────────────┐
│ PHASE 7: CLEANUP & COMPLETION                                      │
├────────────────────────────────────────────────────────────────────┤
│ 1. Save final checkpoint                                           │
│ 2. Close metric logger                                             │
│ 3. Free GPU memory                                                 │
│ 4. Print summary statistics                                        │
│ 5. Exit                                                             │
└────────────────────────────────────────────────────────────────────┘
```

---

## Phase-by-Phase Breakdown

### Phase 1: CLI & Recipe Discovery

**File:** `torchtune/_cli/tune.py`

**What Happens:**
```python
# User runs:
# $ tune run lora_finetune_single_device --config llama3_1/8B_lora

# CLI parses command
recipe_name = "lora_finetune_single_device"
config_name = "llama3_1/8B_lora_single_device"

# Look up in recipe registry
from torchtune._recipe_registry import RECIPES
recipe = RECIPES[recipe_name]

# Get paths
recipe_file = recipe.file_path  # "lora_finetune_single_device.py"
config_file = f"configs/{config_name}.yaml"

# Execute recipe with config
subprocess.run([
    "python",
    f"recipes/{recipe_file}",
    "--config", f"recipes/{config_file}"
])
```

**Key Modularity Point:** Recipe registry decouples CLI from recipes. Adding a new recipe just requires registering it in `_recipe_registry.py`.

---

### Phase 2: Configuration Loading

**File:** `torchtune/config/_parse.py`

**The @parse Decorator:**
```python
@parse
def main(cfg: DictConfig):
    # Recipe code receives parsed config
    ...

# The decorator does:
def parse(recipe_main):
    def wrapper():
        # 1. Create parser
        parser = TuneRecipeArgumentParser()

        # 2. Parse args (gets --config and key=value overrides)
        yaml_args, cli_args = parser.parse_known_args()

        # 3. Merge YAML + CLI
        cfg = _merge_yaml_and_cli_args(yaml_args, cli_args)

        # 4. Call recipe
        sys.exit(recipe_main(cfg))

    return wrapper
```

**Result:** Recipe receives a complete `DictConfig` with all settings.

---

### Phase 3: Recipe Initialization

**File:** `recipes/lora_finetune_single_device.py`

**Recipe Structure:**
```python
class LoRAFinetuneRecipeSingleDevice(FTRecipeInterface):
    """
    Recipe for single-device LoRA fine-tuning.

    Follows the FTRecipeInterface protocol:
    - load_checkpoint()
    - setup()
    - train()
    - save_checkpoint()
    - cleanup()
    """

    def __init__(self, cfg: DictConfig):
        # Store config
        self._device = utils.get_device(cfg.device)
        self._dtype = training.get_dtype(cfg.dtype)
        self._output_dir = cfg.output_dir
        # ... store other config values

    def setup(self, cfg: DictConfig):
        """Initialize all components before training"""
        # Phase 4 happens here...

    def train(self):
        """Main training loop"""
        # Phase 5 happens here...

    def save_checkpoint(self, epoch):
        """Save checkpoint after epoch"""
        # Phase 6 happens here...

    def cleanup(self):
        """Cleanup after training"""
        # Phase 7 happens here...
```

**Entry Point:**
```python
@parse
def main(cfg: DictConfig) -> None:
    recipe = LoRAFinetuneRecipeSingleDevice(cfg)
    recipe.setup(cfg)
    recipe.train()
    recipe.cleanup()
```

---

### Phase 4: Component Setup

**File:** `recipes/lora_finetune_single_device.py` (setup method)

**Detailed Walkthrough:**

#### Step 1: Seeds and Device
```python
def setup(self, cfg: DictConfig):
    # Set random seeds for reproducibility
    if cfg.seed is not None:
        utils.set_seed(cfg.seed)

    # Move to GPU
    self._device = torch.device(cfg.device)
    self._dtype = training.get_dtype(cfg.dtype, self._device)
```

#### Step 2: Model Instantiation
```python
# Instantiate model from config
model = config.instantiate(cfg.model)

# What happens inside instantiate():
# 1. Reads: _component_: torchtune.models.llama3_1.lora_llama3_1_8b
# 2. Imports: lora_llama3_1_8b function
# 3. Calls: lora_llama3_1_8b(
#       lora_attn_modules=['q_proj', 'v_proj', 'output_proj'],
#       lora_rank=8,
#       lora_alpha=16
#   )
# 4. Builder creates TransformerDecoder with LoRA layers
# 5. Returns model instance

# Move to device and dtype
model = model.to(device=self._device, dtype=self._dtype)
```

**Model Structure After Instantiation:**
```
TransformerDecoder
├── tok_embeddings (Embedding)
├── layers (ModuleList)
│   ├── [0] TransformerSelfAttentionLayer
│   │   ├── attn (MultiHeadAttention)
│   │   │   ├── q_proj (LoRALinear)  ← Adapter!
│   │   │   │   ├── weight (frozen)
│   │   │   │   ├── lora_a (trainable)
│   │   │   │   └── lora_b (trainable)
│   │   │   ├── k_proj (Linear)      ← No adapter
│   │   │   ├── v_proj (LoRALinear)  ← Adapter!
│   │   │   └── output_proj (LoRALinear)  ← Adapter!
│   │   └── mlp (FeedForward)
│   │       ├── w1 (LoRALinear)      ← Adapter (if apply_lora_to_mlp)
│   │       └── w2 (LoRALinear)      ← Adapter (if apply_lora_to_mlp)
│   └── ... (32 total layers)
├── norm (RMSNorm)
└── output (Linear)
```

#### Step 3: Load Checkpoint
```python
# Setup checkpointer
checkpointer = config.instantiate(cfg.checkpointer)

# Load pretrained weights
checkpoint_dict = checkpointer.load_checkpoint()

# checkpoint_dict contains:
# {
#     'model': {
#         'tok_embeddings.weight': tensor(...),
#         'layers.0.attn.q_proj.weight': tensor(...),  ← Base weight
#         'layers.0.attn.q_proj.lora_a.weight': tensor(...),  ← Random init
#         'layers.0.attn.q_proj.lora_b.weight': tensor(...),  ← Zero init
#         ...
#     }
# }

# Load weights into model
model.load_state_dict(checkpoint_dict['model'], strict=True)
```

#### Step 4: Set Trainable Parameters
```python
# Get LoRA adapter parameter names
adapter_params = get_adapter_params(model)

# adapter_params = [
#     'layers.0.attn.q_proj.lora_a.weight',
#     'layers.0.attn.q_proj.lora_b.weight',
#     'layers.0.attn.v_proj.lora_a.weight',
#     'layers.0.attn.v_proj.lora_b.weight',
#     ...
# ]

# Freeze all parameters
for param in model.parameters():
    param.requires_grad = False

# Unfreeze only adapter parameters
for name, param in model.named_parameters():
    if name in adapter_params:
        param.requires_grad = True

# Log parameter counts
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
# Output: Trainable: 6,291,456 / 8,030,261,248 (0.08%)
```

#### Step 5: Setup Optimizer
```python
# Get trainable parameters
trainable_params = [p for p in model.parameters() if p.requires_grad]

# Instantiate optimizer with trainable parameters
optimizer = config.instantiate(
    cfg.optimizer,
    trainable_params  # Positional argument!
)

# Result: AdamW(trainable_params, lr=3e-4, weight_decay=0.01, fused=True)
```

#### Step 6: Setup Learning Rate Scheduler
```python
lr_scheduler = config.instantiate(
    cfg.lr_scheduler,
    optimizer,  # Positional argument
    num_training_steps=total_steps
)

# Result: CosineAnnealingLR with warmup
```

#### Step 7: Setup Dataset & DataLoader
```python
# Instantiate tokenizer
tokenizer = config.instantiate(cfg.tokenizer)

# Instantiate dataset (automatically downloads if needed)
dataset = config.instantiate(
    cfg.dataset,
    tokenizer=tokenizer  # Passed as kwarg
)

# Setup dataloader
sampler = DistributedSampler(dataset, shuffle=cfg.shuffle)
dataloader = DataLoader(
    dataset,
    batch_size=cfg.batch_size,
    sampler=sampler,
    collate_fn=dataset.collate_fn
)
```

#### Step 8: Setup Loss Function
```python
loss_fn = config.instantiate(cfg.loss)
# Result: LinearCrossEntropyLoss instance
```

#### Step 9: Enable Memory Optimizations
```python
# Activation checkpointing (saves memory, costs compute)
if cfg.enable_activation_checkpointing:
    utils.set_activation_checkpointing(
        model,
        auto_wrap_policy={TransformerSelfAttentionLayer}
    )

# Activation offloading (saves more memory, costs more time)
if cfg.enable_activation_offloading:
    utils.set_activation_offloading(model)

# torch.compile (faster execution)
if cfg.compile:
    model = torch.compile(model)
    loss_fn = torch.compile(loss_fn)
```

---

### Phase 5: Training Loop

**The Main Training Loop:**
```python
def train(self):
    for epoch in range(self._epochs):
        # Epoch-level tracking
        epoch_loss = 0.0
        num_batches = 0

        # Batch-level loop
        for step, batch in enumerate(self._dataloader):
            # =========================================================
            # FORWARD PASS
            # =========================================================
            # Move batch to device
            batch = {k: v.to(self._device) for k, v in batch.items()}

            # Forward pass through model
            # batch['tokens']: [batch_size, seq_len]
            # logits: [batch_size, seq_len, vocab_size]
            logits = self._model(batch['tokens'])

            # =========================================================
            # COMPUTE LOSS
            # =========================================================
            # loss_fn combines output projection + cross entropy
            # labels: [batch_size, seq_len]
            # Computes: -log(softmax(logits)[labels])
            loss = self._loss_fn(logits, batch['labels'])

            # Normalize by gradient accumulation steps
            loss = loss / self._gradient_accumulation_steps

            # =========================================================
            # BACKWARD PASS
            # =========================================================
            # Compute gradients (only for LoRA parameters!)
            loss.backward()

            # Accumulate loss for logging
            epoch_loss += loss.item()
            num_batches += 1

            # =========================================================
            # OPTIMIZER STEP (if accumulated enough gradients)
            # =========================================================
            if (step + 1) % self._gradient_accumulation_steps == 0:
                # Optional: Clip gradients
                if self._clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self._model.parameters(),
                        self._clip_grad_norm
                    )

                # Update weights
                self._optimizer.step()

                # Update learning rate
                self._lr_scheduler.step()

                # Zero gradients for next accumulation
                self._optimizer.zero_grad()

            # =========================================================
            # LOGGING
            # =========================================================
            if step % self._log_every_n_steps == 0:
                current_lr = self._lr_scheduler.get_last_lr()[0]
                self._metric_logger.log({
                    'loss': loss.item() * self._gradient_accumulation_steps,
                    'lr': current_lr,
                    'epoch': epoch,
                    'step': step,
                })

            # =========================================================
            # EARLY STOPPING (if max_steps_per_epoch set)
            # =========================================================
            if (self._max_steps_per_epoch is not None and
                step >= self._max_steps_per_epoch):
                break

        # =========================================================
        # END OF EPOCH
        # =========================================================
        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch {epoch} completed. Avg loss: {avg_epoch_loss:.4f}")

        # Save checkpoint
        self.save_checkpoint(epoch, intermediate_checkpoint=True)
```

**What Makes This Modular:**
1. Model doesn't know about LoRA (just calls forward())
2. Loss function doesn't know about the model
3. Optimizer doesn't know what it's optimizing (just parameters)
4. Logger doesn't know about training logic (just receives metrics)

---

### Phase 6: Checkpointing

**Checkpoint Structure:**
```
output_dir/
├── epoch_0/
│   ├── adapter_model.pt          # LoRA adapter weights (~10 MB)
│   │   {
│   │       'layers.0.attn.q_proj.lora_a.weight': tensor(...),
│   │       'layers.0.attn.q_proj.lora_b.weight': tensor(...),
│   │       ...
│   │   }
│   │
│   ├── recipe_state.pt           # Optimizer + scheduler state
│   │   {
│   │       'optimizer': {...},   # Optimizer state_dict
│   │       'lr_scheduler': {...},# Scheduler state_dict
│   │       'epoch': 0,
│   │       'seed': 42,
│   │       'total_epochs': 3,
│   │       'max_steps_per_epoch': None
│   │   }
│   │
│   └── model-*.safetensors       # Full merged model (optional, ~16 GB)
│       (base weights + adapters merged)
│
└── logs/
    └── metrics_*.json            # Training metrics
```

**Checkpointing Code:**
```python
def save_checkpoint(self, epoch, intermediate_checkpoint=False):
    # Prepare state dict
    state_dict = {
        'model': self._model.state_dict(),
        'optimizer': self._optimizer.state_dict(),
        'lr_scheduler': self._lr_scheduler.state_dict(),
        'epoch': epoch,
        'seed': self._seed,
    }

    # Save through checkpointer
    self._checkpointer.save_checkpoint(
        state_dict=state_dict,
        epoch=epoch,
        intermediate_checkpoint=intermediate_checkpoint,
        adapter_only=self._save_adapter_weights_only
    )

# Checkpointer extracts adapter weights:
# adapter_state_dict = {
#     k: v for k, v in state_dict['model'].items()
#     if 'lora_a' in k or 'lora_b' in k
# }
```

---

## Component Interactions

### Model + Tokenizer
```
Raw Text
    ↓
Tokenizer.encode()
    ↓
Token IDs [batch_size, seq_len]
    ↓
Model.forward()
    ↓
Logits [batch_size, seq_len, vocab_size]
    ↓
Loss Function
    ↓
Scalar Loss
```

### Model + Optimizer + LR Scheduler
```
Loss.backward()
    ↓
Gradients computed (only for LoRA params)
    ↓
Optimizer.step()
    ↓
Weights updated (LoRA params only)
    ↓
LR Scheduler.step()
    ↓
Learning rate adjusted
    ↓
Optimizer.zero_grad()
    ↓
Ready for next batch
```

### Model + Checkpointer
```
Training Complete
    ↓
Extract state_dict()
    ↓
Checkpointer.save_checkpoint()
    ├→ Separate adapter weights
    ├→ Separate recipe state
    └→ Optionally merge and save full model
```

---

## Data Flow Through the System

### Batch Preparation
```python
# Raw data from HuggingFace dataset
raw_sample = {
    'instruction': "What is machine learning?",
    'input': "",
    'output': "Machine learning is..."
}

# ↓ Dataset transform (AlpacaToMessages)
message_sample = [
    {'role': 'user', 'content': 'What is machine learning?'},
    {'role': 'assistant', 'content': 'Machine learning is...'}
]

# ↓ Tokenizer
tokenized_sample = {
    'tokens': [1, 5618, 374, 5780, 6975, 30, ...],  # Token IDs
    'labels': [-100, -100, -100, -100, -100, 21539, ...],  # Labels (prompt masked)
}

# ↓ DataLoader collation
batch = {
    'tokens': tensor([[1, 5618, ...], [1, 2564, ...]]),  # [batch_size, max_len]
    'labels': tensor([[-100, -100, ...], [-100, 5618, ...]]),
}

# ↓ Model forward
logits = model(batch['tokens'])  # [batch_size, seq_len, vocab_size]

# ↓ Loss computation
loss = loss_fn(logits, batch['labels'])  # Scalar

# ↓ Backward
loss.backward()  # Gradients for LoRA params
```

---

## Checkpointing and Resumption

### Saving During Training
```python
# After each epoch
save_checkpoint(epoch=0, intermediate_checkpoint=True)
# Saves:
#   - output_dir/epoch_0/adapter_model.pt
#   - output_dir/epoch_0/recipe_state.pt
```

### Resuming Interrupted Training
```yaml
# Set in config
resume_from_checkpoint: True
```

```python
# In setup():
if cfg.resume_from_checkpoint:
    # Load recipe state
    recipe_state = torch.load('output_dir/epoch_2/recipe_state.pt')

    # Restore optimizer
    optimizer.load_state_dict(recipe_state['optimizer'])

    # Restore LR scheduler
    lr_scheduler.load_state_dict(recipe_state['lr_scheduler'])

    # Restore epoch counter
    start_epoch = recipe_state['epoch'] + 1

    # Restore random seed
    utils.set_seed(recipe_state['seed'])

# Training continues from epoch 3
```

---

## Memory Management

### Memory Optimization Techniques

#### 1. LoRA (Primary Memory Saving)
```
Full Fine-Tuning: Train 8B parameters
LoRA: Train 6M parameters (99.92% reduction!)

Memory breakdown:
- Model weights: 16 GB (no change)
- Gradients: 16 GB → 0.012 GB (for adapters only)
- Optimizer states: 32 GB → 0.024 GB
Total: ~64 GB → ~16 GB
```

#### 2. Activation Checkpointing
```python
# Enabled in config
enable_activation_checkpointing: True

# Recomputes activations in backward instead of storing
# Memory: -30 to -50%
# Speed: -20 to -30%
```

#### 3. Gradient Accumulation
```python
# Simulates larger batch size without memory cost
batch_size: 2
gradient_accumulation_steps: 8
# Effective batch size: 16
# Memory cost: Same as batch_size=2
```

#### 4. Mixed Precision (bf16)
```python
# Enabled in config
dtype: bf16

# Activations and gradients in bf16
# Memory: ~50% reduction vs fp32
# Speed: ~2× faster on modern GPUs
```

---

## Summary

**Key Takeaways:**

1. **Modular Pipeline:** Each phase is independent
   - CLI → Config Loading → Setup → Training → Checkpointing
   - Swap any component without breaking others

2. **Config-Driven:** YAML specifies everything
   - Model, optimizer, dataset, training params
   - Change config, not code

3. **Dependency Injection:** Components created dynamically
   - `instantiate()` creates objects from config
   - Recipe receives ready-to-use components

4. **Clear Interfaces:** Each component has well-defined role
   - Model: forward()
   - Optimizer: step()
   - Dataset: __getitem__()
   - Checkpointer: load/save

5. **Memory Efficient:** Multiple optimization techniques
   - LoRA: 99%+ parameter reduction
   - Activation checkpointing: 30-50% memory savings
   - Gradient accumulation: Simulate large batches
   - Mixed precision: 2× memory reduction

**The training lifecycle demonstrates torchtune's modularity at every step, from config loading to final checkpointing!**
