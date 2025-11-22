# ==============================================================================
# ANNOTATED: torchtune Configuration Instantiation - Dependency Injection Engine
# ==============================================================================
# Source: torchtune/config/_instantiate.py
#
# **WHAT**: A dynamic object instantiation system that creates Python objects
#           (models, optimizers, datasets) from configuration dictionaries.
#
# **WHY**:  This is THE KEY to torchtune's modularity! Instead of hardcoding:
#           ```python
#           model = LlamaModel(num_layers=32, embed_dim=4096, ...)
#           optimizer = AdamW(model.parameters(), lr=3e-4)
#           ```
#
#           We write declarative config:
#           ```yaml
#           model:
#             _component_: torchtune.models.llama3_1.lora_llama3_1_8b
#             lora_rank: 8
#           optimizer:
#             _component_: torch.optim.AdamW
#             lr: 3e-4
#           ```
#
#           And instantiation happens automatically! Benefits:
#           1. Swap implementations without code changes
#           2. Share configs across experiments
#           3. Version control configurations
#           4. Reproduce experiments exactly
#
# **HOW**:  Based on Hydra's instantiate utility. Key mechanism:
#           - `_component_`: Dotted path to class/function (e.g., torch.optim.AdamW)
#           - Remaining fields: kwargs for that class/function
#           - Recursive: Nested configs → nested object construction
#
# **KEY DESIGN PATTERN**: Dependency Injection / Inversion of Control
#                         - Config specifies *what* to create
#                         - This module handles *how* to create it
#                         - Recipe receives fully constructed objects
# ==============================================================================

import copy
import inspect
import os
import sys
from typing import Any, Callable, Optional, Union

from omegaconf import DictConfig, OmegaConf
from torchtune.config._errors import InstantiationError
from torchtune.config._utils import _get_component_from_path


# ==============================================================================
# HELPER: _create_component - The actual instantiation
# ==============================================================================
# **WHAT**: Given a callable and args, creates an instance
# **WHY**: Separated for clarity and testing
# **HOW**: Simple function call with unpacking
# ==============================================================================
def _create_component(
    _component_: Callable[..., Any],  # The class or function to instantiate
    args: tuple[Any, ...],            # Positional arguments
    kwargs: dict[str, Any],           # Keyword arguments
) -> Any:
    """
    Create an instance by calling the component with given arguments.

    Example:
        _create_component(
            _component_=torch.optim.AdamW,
            args=(),
            kwargs={'lr': 3e-4, 'weight_decay': 0.01}
        )
        → Returns: AdamW optimizer instance
    """
    return _component_(*args, **kwargs)


# ==============================================================================
# CORE RECURSIVE FUNCTION: _instantiate_node
# ==============================================================================
# **WHAT**: Recursively processes configuration structures, instantiating
#           components when it finds `_component_` keys
#
# **WHY RECURSIVE**: Configs are nested! Example:
#           ```yaml
#           model:
#             _component_: Model
#             optimizer:
#               _component_: AdamW  ← Nested component!
#               lr: 3e-4
#           ```
#
# **HOW IT WORKS**: Three cases:
#   1. Dict with _component_: Instantiate the component
#   2. Dict without _component_: Recursively process values
#   3. List: Recursively process each item
#   4. Other: Return as-is
# ==============================================================================
def _instantiate_node(
    obj: Any,                                      # Object to process
    *args: Any,                                    # Positional args (for top level)
    caller_globals: Optional[dict[str, Any]] = None,  # Caller's namespace
) -> Any:
    """
    Recursively instantiate objects from configuration.

    **ALGORITHM**:
    ```
    if obj is dict with '_component_':
        # CASE 1: Instantiation node
        component = import(_component_ path)
        kwargs = {k: _instantiate_node(v) for k, v in obj.items() if k != '_component_'}
        return component(**kwargs)

    elif obj is dict without '_component_':
        # CASE 2: Plain dictionary (keep structure)
        return {k: _instantiate_node(v) for k, v in obj.items()}

    elif obj is list:
        # CASE 3: List (process each element)
        return [_instantiate_node(item) for item in obj]

    else:
        # CASE 4: Primitive (string, int, etc.)
        return obj
    ```

    **EXAMPLE WALKTHROUGH**:

    Input config:
    ```python
    {
        'model': {
            '_component_': 'torch.nn.Linear',
            'in_features': 128,
            'out_features': 10
        },
        'optimizer': {
            '_component_': 'torch.optim.AdamW',
            'lr': 0.001
        }
    }
    ```

    Execution trace:
    1. _instantiate_node(entire_config)
       → dict without _component_, process values
    2. _instantiate_node(model_config)
       → dict with _component_
       → Import torch.nn.Linear
       → Recursively process: in_features=128, out_features=10 (primitives)
       → Call Linear(in_features=128, out_features=10)
       → Return Linear instance
    3. _instantiate_node(optimizer_config)
       → dict with _component_
       → Import torch.optim.AdamW
       → Recursively process: lr=0.001 (primitive)
       → Call AdamW(lr=0.001)
       → Return AdamW instance
    4. Return: {'model': <Linear>, 'optimizer': <AdamW>}
    """

    # =========================================================================
    # CASE 1 & 2: Dictionary (with or without _component_)
    # =========================================================================
    if isinstance(obj, dict) or isinstance(obj, DictConfig):
        # ---------------------------------------------------------------------
        # CASE 2: Dict without _component_ → Process recursively, keep structure
        # ---------------------------------------------------------------------
        if "_component_" not in obj:
            return {
                k: _instantiate_node(v, caller_globals=caller_globals)
                for k, v in obj.items()
            }

        # ---------------------------------------------------------------------
        # CASE 1: Dict with _component_ → INSTANTIATE!
        # ---------------------------------------------------------------------
        else:
            # Step 1: Resolve the component (import the class/function)
            # Example: "_component_: torch.optim.AdamW"
            #          → _component_ = <class 'torch.optim.adamw.AdamW'>
            _component_ = _get_component_from_path(
                obj["_component_"], caller_globals=caller_globals
            )

            # Step 2: Recursively process all kwargs (excluding _component_)
            # This handles nested instantiation!
            # Example: {'lr': 0.001, 'betas': [0.9, 0.999]}
            #          → {'lr': 0.001, 'betas': [0.9, 0.999]} (primitives pass through)
            kwargs = {
                k: _instantiate_node(v, caller_globals=caller_globals)
                for k, v in obj.items()
                if k != "_component_"
            }

            # Step 3: Create the instance!
            return _create_component(_component_, args, kwargs)

    # =========================================================================
    # CASE 3: List → Process each element recursively
    # =========================================================================
    elif isinstance(obj, list):
        return [_instantiate_node(item, caller_globals=caller_globals) for item in obj]

    # =========================================================================
    # CASE 4: Primitive (str, int, float, etc.) → Return as-is
    # =========================================================================
    else:
        return obj


# ==============================================================================
# PUBLIC API: instantiate - Entry point for users
# ==============================================================================
# **WHAT**: Main function called by recipes to create objects from config
# **WHY**: Provides user-friendly interface with validation and preprocessing
# **HOW**: Validates input, prepares config, calls _instantiate_node
# ==============================================================================
def instantiate(
    config: Union[dict[str, Any], DictConfig],     # Configuration dict/DictConfig
    *args: Any,                                     # Optional positional args
    caller_globals: Optional[dict[str, Any]] = None,  # Caller namespace
    **kwargs: Any,                                  # Override kwargs
) -> Any:
    """
    Instantiate an object from configuration.

    **THE HEART OF TORCHTUNE'S CONFIG SYSTEM**

    This function enables the entire config-driven architecture. Here's how
    it's used throughout torchtune:

    **USAGE IN RECIPES**:
    ```python
    from torchtune import config

    def recipe_main(cfg: DictConfig):
        # Instantiate model
        model = config.instantiate(cfg.model)  # → TransformerDecoder

        # Instantiate optimizer (needs model params)
        optimizer = config.instantiate(
            cfg.optimizer,
            model.parameters()  # Passed as positional arg
        )

        # Instantiate dataset
        dataset = config.instantiate(cfg.dataset)  # → SFTDataset

        # Instantiate loss
        loss_fn = config.instantiate(cfg.loss)  # → LinearCrossEntropyLoss

        # Now train!
        for batch in dataloader:
            ...
    ```

    **CORRESPONDING YAML CONFIG**:
    ```yaml
    model:
      _component_: torchtune.models.llama3_1.lora_llama3_1_8b
      lora_rank: 8
      lora_alpha: 16

    optimizer:
      _component_: torch.optim.AdamW
      lr: 3e-4
      weight_decay: 0.01

    dataset:
      _component_: torchtune.datasets.alpaca_cleaned_dataset
      packed: False

    loss:
      _component_: torchtune.modules.loss.LinearCrossEntropyLoss
    ```

    **WHAT HAPPENS**:
    1. User runs: `tune run lora_finetune_single_device --config llama3_1/8B_lora`
    2. Config parser loads YAML → cfg (DictConfig)
    3. Recipe calls: config.instantiate(cfg.model)
    4. This function:
       - Validates cfg.model has _component_
       - Resolves 'torchtune.models.llama3_1.lora_llama3_1_8b'
       - Calls lora_llama3_1_8b(lora_rank=8, lora_alpha=16)
       - Returns TransformerDecoder instance
    5. Model is ready to train!

    **KEY FEATURES**:

    1. **Nested Instantiation**:
       ```yaml
       model:
         _component_: Model
         attention:
           _component_: Attention  # Nested!
           num_heads: 32
       ```
       → Model with Attention instance as attribute

    2. **Positional Arguments**:
       ```python
       optimizer = config.instantiate(
           cfg.optimizer,
           model.parameters()  # Positional arg
       )
       ```
       → AdamW(model.parameters(), lr=3e-4)

    3. **Kwargs Override**:
       ```python
       model = config.instantiate(
           cfg.model,
           lora_rank=16  # Override config value
       )
       ```

    4. **Interpolation** (OmegaConf feature):
       ```yaml
       output_dir: /tmp/output
       logger:
         _component_: DiskLogger
         log_dir: ${output_dir}/logs  # References output_dir!
       ```
       → Automatically resolves to /tmp/output/logs

    Args:
        config: Configuration dict with '_component_' key
        *args: Positional arguments to pass to the component
        caller_globals: Caller's global namespace (for local imports)
        **kwargs: Keyword arguments to override config values

    Returns:
        Instantiated object

    Raises:
        ValueError: If config is not a dict/DictConfig
        InstantiationError: If '_component_' key is missing

    **EXAMPLE**:
    ```python
    # Simple example
    config_dict = {
        '_component_': 'torch.nn.Linear',
        'in_features': 128,
        'out_features': 10,
        'bias': True
    }
    layer = instantiate(config_dict)
    # → Returns: Linear(in_features=128, out_features=10, bias=True)

    # Override example
    layer = instantiate(config_dict, bias=False)
    # → Returns: Linear(in_features=128, out_features=10, bias=False)

    # Nested example
    config_dict = {
        '_component_': 'torch.optim.Adam',
        'lr': 0.001,
        'betas': [0.9, 0.999]  # List of primitives
    }
    # Assume params is a list of parameters
    optimizer = instantiate(config_dict, params)
    # → Returns: Adam(params, lr=0.001, betas=[0.9, 0.999])
    ```
    """

    # =========================================================================
    # STEP 1: Early return for None (allows optional configs)
    # =========================================================================
    if config is None:
        return None

    # =========================================================================
    # STEP 2: Convert plain dict to DictConfig
    # WHY: DictConfig provides features like interpolation, dot-notation access
    # =========================================================================
    if isinstance(config, dict):
        config = OmegaConf.create(config)

    elif not OmegaConf.is_dict(config):
        raise ValueError(
            f"instantiate only supports DictConfigs or dicts, got {type(config)}"
        )

    # =========================================================================
    # STEP 3: Validate _component_ exists
    # WHY: Without _component_, we don't know what to instantiate!
    # =========================================================================
    if "_component_" not in config:
        raise InstantiationError(
            "Cannot instantiate specified object."
            + "\nMake sure you've specified a _component_ field with a valid dotpath."
            + f"\nGot {config=}."
        )

    # =========================================================================
    # STEP 4: Ensure current directory in sys.path (for local imports)
    # WHY: Allows instantiating local classes (e.g., custom datasets)
    # =========================================================================
    if os.getcwd() not in sys.path:
        sys.path.append(os.getcwd())

    # =========================================================================
    # STEP 5: Prepare config for instantiation
    # WHY: Need to enable object mode, disable struct mode for modification
    # =========================================================================
    config_copy = copy.deepcopy(config)
    config_copy._set_flag(
        flags=["allow_objects", "struct", "readonly"],
        values=[True, False, False]
    )
    config_copy._set_parent(config._get_parent())
    config = config_copy

    # =========================================================================
    # STEP 6: Merge kwargs (overrides)
    # WHY: Allow programmatic override of config values
    # EXAMPLE: instantiate(cfg, lr=0.01) overrides cfg.lr
    # =========================================================================
    if kwargs:
        config = OmegaConf.merge(config, kwargs)

    # =========================================================================
    # STEP 7: Resolve interpolations
    # WHAT: ${var} references in config are resolved to actual values
    # EXAMPLE: log_dir: ${output_dir}/logs → log_dir: /tmp/output/logs
    # =========================================================================
    OmegaConf.resolve(config)

    # =========================================================================
    # STEP 8: Get caller's globals (for local class instantiation)
    # WHY: If user defines custom class in recipe, need access to it
    # HOW: Walk up call stack to get caller's global namespace
    # =========================================================================
    if caller_globals is None:
        current_frame = inspect.currentframe()
        if current_frame and current_frame.f_back:
            caller_globals = current_frame.f_back.f_globals

    # =========================================================================
    # STEP 9: Actually instantiate! (recursive magic happens here)
    # =========================================================================
    return _instantiate_node(
        OmegaConf.to_container(config, resolve=True),
        caller_globals=caller_globals,
        *args,
    )


# ==============================================================================
# HOW THIS ACHIEVES MODULARITY
# ==============================================================================
"""
**DEPENDENCY INJECTION IN ACTION**:

Traditional approach (tightly coupled):
```python
def train():
    model = LlamaModel(num_layers=32, ...)  # Hardcoded!
    optimizer = AdamW(model.parameters(), lr=3e-4)  # Hardcoded!
    dataset = AlpacaDataset()  # Hardcoded!

    # Changing model requires code changes
    # Experimenting with different optimizers requires code changes
    # Can't easily share configurations
```

Torchtune approach (loosely coupled):
```python
def train(cfg: DictConfig):
    model = config.instantiate(cfg.model)  # Any model!
    optimizer = config.instantiate(cfg.optimizer, model.parameters())  # Any optimizer!
    dataset = config.instantiate(cfg.dataset)  # Any dataset!

    # Change model: Edit YAML, not code
    # Try different optimizer: Edit YAML, not code
    # Share configs: Just share YAML file
```

**BENEFITS**:

1. **Separation of Concerns**:
   - Recipe: Training logic (how to train)
   - Config: Component selection (what to train)
   - Implementation: Component code (model, optimizer, etc.)

2. **Flexibility**:
   - Want to try QLoRA instead of LoRA?
     Change: `_component_: ...lora_llama3_1_8b`
     To: `_component_: ...qlora_llama3_1_8b`
   - Want to try AdamW instead of SGD?
     Change: `_component_: torch.optim.SGD`
     To: `_component_: torch.optim.AdamW`

3. **Reproducibility**:
   - YAML config fully specifies experiment
   - Share config = share exact setup
   - Version control configs = track experiments

4. **Testability**:
   - Test instantiation logic independently
   - Mock configs for unit tests
   - Validate configs without running training

5. **Extensibility**:
   - Add new component: Just create class/function
   - Use in training: Add _component_ to config
   - No changes to recipe code needed!

**EXAMPLE FLOW**:

User command:
```bash
tune run lora_finetune_single_device --config llama3_1/8B_lora
```

What happens:
1. Parse YAML config
2. Recipe receives DictConfig
3. Recipe calls: model = config.instantiate(cfg.model)
4. instantiate() function:
   - Reads _component_: torchtune.models.llama3_1.lora_llama3_1_8b
   - Imports that function
   - Extracts kwargs: {lora_rank: 8, lora_alpha: 16, ...}
   - Calls: lora_llama3_1_8b(lora_rank=8, lora_alpha=16, ...)
5. Model builder creates model
6. Returns to recipe
7. Training begins!

**KEY INSIGHT**: The recipe never knows or cares what specific model,
                 optimizer, or dataset is being used. It just calls
                 config.instantiate() and gets the right object!

This is the ESSENCE of dependency injection and why torchtune is so modular.
"""


# ==============================================================================
# RELATIONSHIP TO OTHER FILES
# ==============================================================================
"""
**HOW THIS FILE CONNECTS TO THE SYSTEM**:

1. **Config Parsing** (_parse.py):
   - Loads YAML → DictConfig
   - Passes to recipe
   - Recipe uses THIS file to instantiate components

2. **Model Builders** (_model_builders.py):
   - Specify _component_ paths that point to builders
   - instantiate() imports and calls these builders
   - Builders return model instances

3. **Recipes** (e.g., lora_finetune_single_device.py):
   - Receive DictConfig from parser
   - Call config.instantiate() for each component
   - Use instantiated objects for training

4. **Component Implementations** (models, datasets, etc.):
   - Implement the actual functionality
   - Never know about configs or instantiation
   - Just receive constructor arguments

**FLOW DIAGRAM**:

YAML File
    ↓
_parse.py (load & validate)
    ↓
DictConfig
    ↓
Recipe (train logic)
    ↓
_instantiate.py (THIS FILE) ← Creates objects
    ↓
Component Builders
    ↓
Actual Components (Model, Optimizer, Dataset)
    ↓
Training Loop

**NEXT**: See annotated YAML configs to understand what goes IN to this
          instantiation system!
"""
