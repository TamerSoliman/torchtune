# ==============================================================================
# ANNOTATED: torchtune Checkpointer - State Persistence and Model Serialization
# ==============================================================================
# Source: torchtune/training/checkpointing/_checkpointer.py
#
# **WHAT**: Checkpointers manage loading pretrained weights and saving fine-tuned
#           models. They handle multiple checkpoint formats (HuggingFace, Meta,
#           torchtune) and ensure compatibility across ecosystems.
#
# **WHY**:  Training requires:
#           1. Loading pretrained weights (often 8-405B parameters!)
#           2. Saving progress during training (resume if interrupted)
#           3. Saving final results (adapters only or full merged model)
#           4. Format conversion (torchtune ↔ HuggingFace ↔ Meta)
#           5. State management (optimizer, lr_scheduler, epoch counter)
#
# **HOW**:  Strategy Pattern with multiple implementations:
#           - FullModelHFCheckpointer: HuggingFace format (most common)
#           - FullModelMetaCheckpointer: Meta's format
#           - FullModelTorchTuneCheckpointer: torchtune native
#           - DistributedCheckpointer: Multi-GPU training
#
# **KEY DESIGN PATTERNS**:
#   1. **Strategy Pattern**: Multiple checkpointer implementations,  same interface
#   2. **Protocol/Interface**: _CheckpointerInterface defines contract
#   3. **State-Dict Invariant**: Output format matches input format
#   4. **Separation of Concerns**: Model weights vs. recipe state
# ==============================================================================

import os
from pathlib import Path
from typing import Any, Optional, Protocol

import torch
from safetensors.torch import save as save_safetensors

from torchtune import training
from torchtune.models import convert_weights
from torchtune.training.checkpointing._utils import (
    ModelType,
    safe_torch_load,
)


# ==============================================================================
# PROTOCOL: _CheckpointerInterface
# ==============================================================================
# **WHAT**: Defines the contract that all checkpointers must implement
# **WHY**: Ensures consistency across different checkpointer implementations
# **HOW**: Two required methods: load_checkpoint() and save_checkpoint()
#
# **PROTOCOL PATTERN**:
# Protocol in Python (PEP 544) is like an interface in other languages.
# It defines "shape" without requiring inheritance.
#
# Benefits:
# - Structural subtyping (duck typing with type checking)
# - No inheritance needed
# - Easy to extend
# - Clear contract
# ==============================================================================
class _CheckpointerInterface(Protocol):
    """
    Interface for all torchtune checkpointers.

    **DESIGN PHILOSOPHY**:

    1. **Composable**: Checkpointers are pluggable components
       - Recipe doesn't care which checkpointer is used
       - Swap checkpointers via config
       - Each checkpointer handles specific formats

    2. **State-Dict Invariant**: Output format == Input format
       - Load HF checkpoint → Save HF checkpoint
       - Load Meta checkpoint → Save Meta checkpoint
       - Ensures compatibility with post-training tools

    3. **Two Checkpointing Modes**:

       a) End-of-Training Checkpointing:
          - Save final model weights
          - Maintain original format
          - Compatible with inference engines

       b) Mid-Training Checkpointing:
          - Save model weights + recipe state
          - Enables resuming interrupted training
          - Separate files: model + recipe_state.pt

    **STATE DICTIONARY FORMATS**:

    End-of-Training (model only):
    ```python
    {
        "key_1": tensor(...),  # Model weights
        "key_2": tensor(...),
        ...
    }
    ```

    Mid-Training (model + recipe state):
    ```python
    # Returned by load_checkpoint():
    {
        "model": {
            "key_1": tensor(...),  # Model weights
            ...
        },
        "optimizer": {...},        # Optimizer state
        "lr_scheduler": {...},     # LR scheduler state
        "epoch": 2,                # Current epoch
        "seed": 42,                # Random seed
        ...
    }
    ```

    Saved as separate files:
    - `epoch_2/model-*.safetensors`: Model weights
    - `epoch_2/recipe_state.pt`: Optimizer + metadata

    **WHY SEPARATE FILES**:
    - Model weights: Large, format-specific
    - Recipe state: Small, torchtune-specific
    - Allows using model with any tool
    - Recipe state only for resuming in torchtune

    **CHECKPOINTER TYPES**:

    | Checkpointer | Format | Use Case |
    |--------------|--------|----------|
    | FullModelHFCheckpointer | HuggingFace | Most common, wide compatibility |
    | FullModelMetaCheckpointer | Meta | Original Llama checkpoints |
    | FullModelTorchTuneCheckpointer | torchtune | Native format |
    | DistributedCheckpointer | Sharded | Multi-GPU training |
    """

    def load_checkpoint(self, **kwargs) -> dict[str, Any]:
        """
        Load checkpoint from disk.

        Returns:
            dict[str, Any]: State dictionary with:
                - "model": Model weights
                - "optimizer": Optimizer state (if resuming)
                - "lr_scheduler": LR scheduler state (if resuming)
                - Other recipe-specific state
        """
        ...

    def save_checkpoint(self, state_dict: dict[str, Any], **kwargs) -> None:
        """
        Save checkpoint to disk.

        Args:
            state_dict: Complete state including model and recipe state
            **kwargs: Checkpointer-specific arguments
        """
        ...


# ==============================================================================
# CONCRETE IMPLEMENTATION: FullModelHFCheckpointer
# ==============================================================================
# **WHAT**: Checkpointer for HuggingFace format checkpoints
# **WHY**: HF is the most common format in open-source LLM ecosystem
# **HOW**: Uses convert_weights to translate between formats
#
# **KEY FEATURES**:
# 1. Automatic weight conversion (HF ↔ torchtune)
# 2. Multi-file checkpoint support
# 3. Adapter-only saving (for LoRA)
# 4. Recipe state management
# 5. Safetensors support
# ==============================================================================
class FullModelHFCheckpointer(_CheckpointerInterface):
    """
    Checkpointer for HuggingFace format.

    **WHAT IS HUGGINGFACE FORMAT**:
    Standard format used by transformers library:
    - Weights in model-*.safetensors or pytorch_model-*.bin
    - Config in config.json
    - Tokenizer in tokenizer.model / tokenizer.json
    - Often split across multiple files (shards)

    Example HF checkpoint structure:
    ```
    Meta-Llama-3.1-8B-Instruct/
    ├── config.json
    ├── tokenizer.model
    ├── model-00001-of-00004.safetensors
    ├── model-00002-of-00004.safetensors
    ├── model-00003-of-00004.safetensors
    └── model-00004-of-00004.safetensors
    ```

    **WEIGHT CONVERSION**:
    HF and torchtune use different key names:

    HuggingFace:
    - "model.layers.0.self_attn.q_proj.weight"
    - "model.layers.0.self_attn.k_proj.weight"

    torchtune:
    - "layers.0.attn.q_proj.weight"
    - "layers.0.attn.k_proj.weight"

    Checkpointer automatically converts between these!

    **USAGE IN CONFIG**:
    ```yaml
    checkpointer:
      _component_: torchtune.training.FullModelHFCheckpointer
      checkpoint_dir: /tmp/Meta-Llama-3.1-8B-Instruct/
      checkpoint_files: [
        model-00001-of-00004.safetensors,
        model-00002-of-00004.safetensors,
        model-00003-of-00004.safetensors,
        model-00004-of-00004.safetensors
      ]
      output_dir: ${output_dir}
      model_type: LLAMA3
    ```

    Args:
        checkpoint_dir: Directory with pretrained weights
        checkpoint_files: List of weight files to load
        model_type: Model family (LLAMA2, LLAMA3, GEMMA, etc.)
        output_dir: Where to save fine-tuned checkpoints
        adapter_checkpoint: Path to adapter weights (for LoRA)
        recipe_checkpoint: Path to recipe state (for resuming)
        should_load_recipe_state: Whether to load optimizer/scheduler state

    **EXAMPLE FLOW**:

    1. Loading (start training):
    ```python
    checkpointer = FullModelHFCheckpointer(
        checkpoint_dir="/tmp/Llama-3.1-8B/",
        checkpoint_files=["model-*.safetensors"],
        model_type="LLAMA3",
        output_dir="/tmp/output"
    )

    # Load pretrained weights
    ckpt_dict = checkpointer.load_checkpoint()
    # Returns: {"model": {HF weights converted to torchtune format}}

    # Load into model
    model.load_state_dict(ckpt_dict["model"])
    ```

    2. Saving (end of epoch):
    ```python
    # Prepare state dict
    state_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "epoch": 0,
    }

    # Save checkpoint
    checkpointer.save_checkpoint(
        state_dict=state_dict,
        epoch=0,
        intermediate_checkpoint=True,
        adapter_only=False
    )
    ```

    Output:
    ```
    /tmp/output/
    └── epoch_0/
        ├── model-00001-of-00004.safetensors  # Model weights (HF format)
        ├── model-00002-of-00004.safetensors
        ├── model-00003-of-00004.safetensors
        ├── model-00004-of-00004.safetensors
        ├── adapter_model.pt                  # LoRA adapters (if present)
        └── recipe_state.pt                   # Optimizer + metadata
    ```

    3. Resuming (interrupted training):
    ```python
    checkpointer = FullModelHFCheckpointer(
        checkpoint_dir="/tmp/Llama-3.1-8B/",
        checkpoint_files=["model-*.safetensors"],
        model_type="LLAMA3",
        output_dir="/tmp/output",
        should_load_recipe_state=True  # Resume flag!
    )

    # Load everything
    ckpt_dict = checkpointer.load_checkpoint()
    # Returns:
    # {
    #     "model": {...},
    #     "optimizer": {...},  # Restored!
    #     "lr_scheduler": {...},  # Restored!
    #     "epoch": 0,  # Continue from epoch 1
    # }

    # Restore all state
    model.load_state_dict(ckpt_dict["model"])
    optimizer.load_state_dict(ckpt_dict["optimizer"])
    lr_scheduler.load_state_dict(ckpt_dict["lr_scheduler"])
    start_epoch = ckpt_dict["epoch"] + 1
    ```
    """

    def __init__(
        self,
        checkpoint_dir: str,
        checkpoint_files: list[str],
        model_type: str,
        output_dir: str,
        adapter_checkpoint: Optional[str] = None,
        recipe_checkpoint: Optional[str] = None,
        should_load_recipe_state: bool = False,
    ) -> None:
        """
        Initialize checkpointer.

        **INITIALIZATION STEPS**:
        1. Validate and store paths
        2. Convert model_type string → ModelType enum
        3. Set up output directory
        4. Locate checkpoint files
        5. Locate adapter/recipe checkpoints (if resuming)
        """
        # Store configuration
        self._checkpoint_dir = Path(checkpoint_dir)
        self._checkpoint_files = checkpoint_files
        self._model_type = ModelType[model_type]  # e.g., "LLAMA3" → ModelType.LLAMA3
        self._output_dir = Path(output_dir)
        self._should_load_recipe_state = should_load_recipe_state

        # Create output directory
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Locate checkpoint files
        # (Implementation details abstracted for clarity)
        # In reality: sorts files, validates they exist, etc.

    def load_checkpoint(self, weights_only: bool = True) -> dict[str, Any]:
        """
        Load checkpoint from HuggingFace format.

        **LOADING PROCESS**:

        1. Load weight files (possibly sharded across multiple files)
        2. Convert from HF format → torchtune format
        3. Optionally load adapter weights (for LoRA)
        4. Optionally load recipe state (for resuming)

        **WEIGHT CONVERSION EXAMPLE**:
        ```python
        # HF checkpoint (loaded)
        hf_state_dict = {
            "model.layers.0.self_attn.q_proj.weight": tensor(...),
            "model.layers.0.self_attn.k_proj.weight": tensor(...),
        }

        # Converted to torchtune format
        tt_state_dict = convert_weights.hf_to_tune(
            hf_state_dict,
            model_type=ModelType.LLAMA3
        )
        # Result:
        # {
        #     "layers.0.attn.q_proj.weight": tensor(...),
        #     "layers.0.attn.k_proj.weight": tensor(...),
        # }
        ```

        Args:
            weights_only: Whether to load only weights (safer, default: True)
                         Set to False for quantized models

        Returns:
            dict with structure:
            ```python
            {
                "model": {converted torchtune weights},
                "optimizer": {...},      # Only if should_load_recipe_state=True
                "lr_scheduler": {...},   # Only if should_load_recipe_state=True
                "epoch": N,              # Only if should_load_recipe_state=True
            }
            ```
        """
        # Load HF weights
        hf_state_dict = {}
        for file in self._checkpoint_files:
            file_path = self._checkpoint_dir / file
            # Load individual shard
            shard = safe_torch_load(file_path, weights_only=weights_only)
            hf_state_dict.update(shard)

        # Convert HF → torchtune format
        state_dict = {}
        state_dict[training.MODEL_KEY] = convert_weights.hf_to_tune(
            hf_state_dict,
            num_heads=32,  # Model-specific (from model_type)
            num_kv_heads=8,
            dim=4096,
        )

        # Load adapter weights if present (for LoRA)
        if self._adapter_checkpoint:
            adapter_state_dict = safe_torch_load(self._adapter_checkpoint)
            state_dict[training.ADAPTER_KEY] = adapter_state_dict

        # Load recipe state if resuming
        if self._should_load_recipe_state:
            recipe_state = safe_torch_load(self._recipe_checkpoint)
            state_dict.update(recipe_state)

        return state_dict

    def save_checkpoint(
        self,
        state_dict: dict[str, Any],
        epoch: int,
        intermediate_checkpoint: bool = False,
        adapter_only: bool = False,
        **kwargs,
    ) -> None:
        """
        Save checkpoint in HuggingFace format.

        **SAVING PROCESS**:

        1. Convert torchtune format → HF format
        2. Shard weights across multiple files (if needed)
        3. Save in safetensors format
        4. Optionally save adapter weights separately
        5. Optionally save recipe state

        **STATE-DICT INVARIANT**:
        If you loaded from 4 HF files, we save to 4 HF files.
        Key names and file structure match original checkpoint.

        This ensures compatibility with:
        - HuggingFace transformers
        - vLLM
        - llama.cpp
        - Other inference engines

        Args:
            state_dict: Complete state with model + recipe state
            epoch: Current epoch (used in output filename)
            intermediate_checkpoint: If True, save recipe_state.pt
            adapter_only: If True, only save adapter weights (for LoRA)

        **OUTPUT STRUCTURE**:

        Without LoRA (full model):
        ```
        output_dir/epoch_0/
        ├── model-00001-of-00004.safetensors
        ├── model-00002-of-00004.safetensors
        ├── model-00003-of-00004.safetensors
        ├── model-00004-of-00004.safetensors
        ├── config.json  # Copied from input
        ├── tokenizer.model  # Copied from input
        └── recipe_state.pt  # Only if intermediate_checkpoint=True
        ```

        With LoRA (adapter_only=True):
        ```
        output_dir/epoch_0/
        ├── adapter_model.pt  # LoRA adapters (~10MB)
        └── recipe_state.pt  # Optimizer + metadata
        ```

        With LoRA (adapter_only=False):
        ```
        output_dir/epoch_0/
        ├── model-*.safetensors  # Full merged model (base + adapters)
        ├── adapter_model.pt  # LoRA adapters (for easy swapping)
        ├── config.json
        ├── tokenizer.model
        └── recipe_state.pt
        ```
        """
        # Create output directory
        output_path = self._output_dir / f"epoch_{epoch}"
        output_path.mkdir(parents=True, exist_ok=True)

        # Save model weights (if not adapter_only)
        if not adapter_only:
            # Convert torchtune → HF format
            hf_state_dict = convert_weights.tune_to_hf(
                state_dict[training.MODEL_KEY],
                num_heads=32,
                num_kv_heads=8,
                dim=4096,
            )

            # Save sharded weights
            for idx, shard in enumerate(self._shard_state_dict(hf_state_dict)):
                shard_path = output_path / f"model-{idx+1:05d}-of-{len(self._checkpoint_files):05d}.safetensors"
                save_safetensors(shard, str(shard_path))

        # Save adapter weights if present
        if training.ADAPTER_KEY in state_dict:
            adapter_path = output_path / "adapter_model.pt"
            torch.save(state_dict[training.ADAPTER_KEY], adapter_path)

        # Save recipe state if intermediate checkpoint
        if intermediate_checkpoint:
            # Extract recipe state (everything except model/adapter)
            recipe_state = {
                k: v for k, v in state_dict.items()
                if k not in [training.MODEL_KEY, training.ADAPTER_KEY]
            }
            recipe_path = output_path / "recipe_state.pt"
            torch.save(recipe_state, recipe_path)

    def _shard_state_dict(self, state_dict):
        """Helper to shard state dict across multiple files"""
        # Simplified - actual implementation handles even distribution
        num_shards = len(self._checkpoint_files)
        keys_per_shard = len(state_dict) // num_shards

        shards = []
        items = list(state_dict.items())
        for i in range(num_shards):
            start = i * keys_per_shard
            end = start + keys_per_shard if i < num_shards - 1 else len(items)
            shard = dict(items[start:end])
            shards.append(shard)

        return shards


# ==============================================================================
# HOW CHECKPOINTING ACHIEVES MODULARITY
# ==============================================================================
"""
**STRATEGY PATTERN IN ACTION**:

Recipe doesn't care about checkpoint format:
```python
# In recipe
checkpointer = config.instantiate(cfg.checkpointer)  # Any checkpointer!
state_dict = checkpointer.load_checkpoint()         # Same interface
model.load_state_dict(state_dict["model"])          # Works with all formats
```

```yaml
# Switch formats just by changing config
checkpointer:
  _component_: torchtune.training.FullModelHFCheckpointer  # or FullModelMetaCheckpointer
```

**BENEFITS**:

1. **Format Independence**:
   - Recipe doesn't know or care about format
   - Same training code works with HF, Meta, torchtune formats
   - Easy to add new format support

2. **State-Dict Invariant**:
   - Output format matches input format
   - Ensures compatibility with inference tools
   - No format conversion needed post-training

3. **Separation of Concerns**:
   - Checkpointer: Format conversion and I/O
   - Recipe: Training logic
   - Model: Architecture and forward pass
   - Never mixed together

4. **Resumability**:
   - Recipe state separate from model weights
   - Can resume any training run
   - Optimizer state preserved exactly

5. **Flexibility**:
   - Save adapters only (LoRA)
   - Save full merged model
   - Save both for maximum flexibility

**EXAMPLE USE CASES**:

1. Start training from HF checkpoint:
```yaml
checkpointer:
  _component_: torchtune.training.FullModelHFCheckpointer
  checkpoint_dir: /path/to/hf/model
  should_load_recipe_state: False  # Fresh training
```

2. Resume interrupted training:
```yaml
checkpointer:
  _component_: torchtune.training.FullModelHFCheckpointer
  checkpoint_dir: /path/to/hf/model
  output_dir: /path/to/output
  should_load_recipe_state: True  # Resume!
```

3. Save LoRA adapters only:
```python
checkpointer.save_checkpoint(
    state_dict=state_dict,
    epoch=epoch,
    adapter_only=True  # Just adapters, not full model
)
```

**KEY INSIGHT**: Checkpointing is completely decoupled from training.
Recipe just calls load/save, checkpointer handles all format complexity.
This is the essence of the Strategy Pattern and why torchtune is so modular!
"""


# ==============================================================================
# RELATIONSHIP TO OTHER COMPONENTS
# ==============================================================================
"""
**HOW CHECKPOINTER CONNECTS TO THE SYSTEM**:

1. **Config System** (_instantiate.py):
   ```yaml
   checkpointer:
     _component_: torchtune.training.FullModelHFCheckpointer
     checkpoint_dir: /path
   ```
   → config.instantiate(cfg.checkpointer)
   → FullModelHFCheckpointer instance

2. **Recipe** (lora_finetune_single_device.py):
   ```python
   # Setup phase
   checkpointer = config.instantiate(cfg.checkpointer)
   ckpt_dict = checkpointer.load_checkpoint()
   model.load_state_dict(ckpt_dict["model"])

   # Training loop
   for epoch in range(epochs):
       train()
       checkpointer.save_checkpoint(state_dict, epoch)
   ```

3. **Weight Converters** (torchtune/models/*/):
   - convert_weights.hf_to_tune()
   - convert_weights.tune_to_hf()
   - convert_weights.meta_to_tune()
   - Model-specific conversion logic

4. **Model** (TransformerDecoder):
   - Provides state_dict() for saving
   - Loads from state_dict for initialization
   - Agnostic to checkpoint format

**FLOW DIAGRAM**:

Config
  ↓
instantiate(cfg.checkpointer)
  ↓
Checkpointer Instance
  ↓
load_checkpoint()
  ├→ Load files
  ├→ Convert format
  └→ Return state_dict
      ↓
Recipe
  ↓
model.load_state_dict()
  ↓
Training
  ↓
save_checkpoint()
  ├→ Convert format
  ├→ Shard if needed
  └→ Save files

**NEXT**: See Training_Recipe_Lifecycle_Guide.md for complete checkpointing flow!
"""
