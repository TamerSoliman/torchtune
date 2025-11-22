# ==============================================================================
# ANNOTATED: torchtune Dataset Builder - Factory Pattern for Data Pipelines
# ==============================================================================
# Source: torchtune/datasets/_alpaca.py
#
# **WHAT**: Dataset factory functions that create configured dataset instances
#           for training. These builders handle data loading, transformation,
#           and preparation for model consumption.
#
# **WHY**:  Training requires different datasets (Alpaca, Stack Exchange, custom).
#           Dataset builders provide:
#           1. Consistent interface for all dataset types
#           2. Automatic data downloading from HuggingFace
#           3. Flexible transformations (instruction → chat format)
#           4. Optional packing for efficiency
#           5. Config-driven instantiation (YAML → Dataset)
#
# **HOW**:  Multi-stage pipeline architecture:
#
#           Raw Data (HuggingFace/local)
#                ↓
#           Message Transform (format conversion)
#                ↓
#           Model Transform (tokenization)
#                ↓
#           Optional Packing (efficiency)
#                ↓
#           Dataset Ready for Training
#
# **KEY DESIGN PATTERNS**:
#   1. **Factory Pattern**: Functions that create dataset instances
#   2. **Pipeline Pattern**: Data flows through transformation stages
#   3. **Partial Application**: Create specialized variants (alpaca_cleaned)
#   4. **Composition**: Combine transforms, dataset, packing
# ==============================================================================

from functools import partial
from typing import Any, Callable, Optional, Union

# ==============================================================================
# IMPORTS: The Data Pipeline Components
# ==============================================================================
# WHY THESE IMPORTS: Each represents a stage in the data pipeline
# ==============================================================================

# Message Transform: Converts raw data to standardized message format
from torchtune.data._messages import AlpacaToMessages

# Dataset Wrapper: Handles packing multiple examples into sequences
from torchtune.datasets._packed import PackedDataset

# Base Dataset: Supervised Fine-Tuning dataset class
from torchtune.datasets._sft import SFTDataset

# Tokenizer Interface: Converts text to token IDs
from torchtune.modules.transforms.tokenizers import ModelTokenizer


# ==============================================================================
# MAIN FACTORY FUNCTION: alpaca_dataset
# ==============================================================================
# **WHAT**: Creates a dataset instance configured for Alpaca-style data
#
# **WHY ALPACA FORMAT**: A popular instruction-following format:
#           {
#               "instruction": "What is machine learning?",
#               "input": "",  # Optional context
#               "output": "Machine learning is..."
#           }
#
# **HOW IT WORKS**:
#   1. Create message transform (Alpaca → chat format)
#   2. Create SFTDataset (handles loading + transformation)
#   3. Optionally wrap in PackedDataset (for efficiency)
#   4. Return configured dataset
# ==============================================================================
def alpaca_dataset(
    tokenizer: ModelTokenizer,           # Tokenizer for converting text → IDs
    *,
    source: str = "tatsu-lab/alpaca",   # HuggingFace dataset or local path
    column_map: Optional[dict[str, str]] = None,  # Custom column names
    train_on_input: bool = True,         # Whether to train on prompt
    packed: bool = False,                # Whether to pack sequences
    filter_fn: Optional[Callable] = None,  # Optional data filtering
    split: str = "train",                # Dataset split to use
    **load_dataset_kwargs: dict[str, Any],  # Extra HF load_dataset args
) -> Union[SFTDataset, PackedDataset]:
    """
    Factory for Alpaca-style datasets.

    **THE ALPACA FORMAT**:
    Original paper: https://crfm.stanford.edu/2023/03/13/alpaca.html

    Data structure:
    ```json
    {
        "instruction": "Give three tips for staying healthy.",
        "input": "",
        "output": "1. Eat a balanced diet..."
    }
    ```

    Becomes this prompt:
    ```
    Below is an instruction that describes a task...

    ### Instruction:
    Give three tips for staying healthy.

    ### Response:
    1. Eat a balanced diet...
    ```

    **THE DATA PIPELINE**:
    ```
    HuggingFace Dataset
        ↓
    AlpacaToMessages (convert to chat format)
        ↓
    [
        {"role": "user", "content": "Give three tips..."},
        {"role": "assistant", "content": "1. Eat a balanced..."}
    ]
        ↓
    Tokenizer (convert to token IDs)
        ↓
    {
        "tokens": [1, 5618, 2380, ...],
        "labels": [-100, -100, ..., 16, 2469, ...]  # -100 = masked
    }
        ↓
    Optional Packing (combine multiple examples)
        ↓
    Ready for DataLoader!
    ```

    **KEY PARAMETERS EXPLAINED**:

    tokenizer: ModelTokenizer
        - Must implement tokenize_messages() method
        - Converts chat messages → token IDs
        - Example: Llama3Tokenizer, GemmaTokenizer

    source: str = "tatsu-lab/alpaca"
        - HuggingFace dataset repo (auto-downloads)
        - Or local file type: "json", "csv", "text"
        - Examples:
          * "tatsu-lab/alpaca" → Original Alpaca
          * "yahma/alpaca-cleaned" → Cleaned version
          * "json" + data_files="path/to/data.json" → Local

    column_map: Optional[dict[str, str]]
        - Maps standard names to custom column names
        - Default expects: "instruction", "input", "output"
        - Example: {"instruction": "prompt", "output": "completion"}
        - Useful when your data has different column names

    train_on_input: bool = True
        - **CRITICAL PARAMETER** for instruction tuning!
        - True: Model learns to generate the prompt too (NOT what you want!)
        - False: Prompt masked out (labels=-100), only learn response

        Example with train_on_input=False:
        ```
        Input:  "What is AI?" "AI is..."
        Tokens: [1, 5618, ...]  [16, 2469, ...]
        Labels: [-100, -100, ...] [16, 2469, ...]  ← Prompt masked!
        ```

        This ensures the model only learns to generate answers, not prompts.

    packed: bool = False
        - Whether to pack multiple short examples into one sequence
        - **WHY**: GPU efficiency! Padding wastes computation.

        Without packing:
        ```
        Batch 1: [Example 1 (50 tokens) + padding (1974 tokens)] = 2024 tokens
        Batch 2: [Example 2 (30 tokens) + padding (1994 tokens)] = 2024 tokens
        Wasted: 3968 tokens of padding!
        ```

        With packing:
        ```
        Batch 1: [Example 1 (50) + Example 2 (30) + Example 3 (40) + ...]
                 = 2024 tokens, NO padding!
        Wasted: 0 tokens!
        ```

        **TRADEOFF**:
        - Pros: Better GPU utilization, faster training
        - Cons: More complex, slight implementation overhead
        - Typical: False for simplicity, True for production

    filter_fn: Optional[Callable]
        - Function to filter dataset before processing
        - Example: lambda x: len(x['output']) > 10
        - Useful for removing empty/invalid examples

    split: str = "train"
        - Which dataset split to load
        - Can load subset: "train[:10%]" → 10% of training data
        - Examples:
          * "train" → Full training set
          * "train[:1000]" → First 1000 examples
          * "train[80%:]" → Last 20% of training data
          * "validation" → Validation split

    **load_dataset_kwargs: Additional arguments
        - Passed directly to datasets.load_dataset()
        - Examples:
          * data_files="path/to/data.json" → For local files
          * cache_dir="/path/to/cache" → Custom cache location
          * trust_remote_code=True → For datasets with custom code

    Returns:
        Union[SFTDataset, PackedDataset]: Configured dataset ready for training

    Raises:
        ValueError: If packed=True but tokenizer.max_seq_len is None

    **USAGE IN CONFIG**:
    ```yaml
    dataset:
      _component_: torchtune.datasets.alpaca_dataset
      source: tatsu-lab/alpaca
      train_on_input: False  # Don't train on prompts!
      packed: False
      split: train
    ```

    **USAGE IN CODE**:
    ```python
    from torchtune.datasets import alpaca_dataset
    from torchtune.models.llama3 import llama3_tokenizer

    tokenizer = llama3_tokenizer(path="path/to/tokenizer.model")
    dataset = alpaca_dataset(
        tokenizer=tokenizer,
        train_on_input=False,  # Mask prompts
        packed=True            # Enable packing
    )

    # Use with DataLoader
    dataloader = DataLoader(dataset, batch_size=8, collate_fn=dataset.collate)
    for batch in dataloader:
        # batch['tokens']: [batch_size, seq_len]
        # batch['labels']: [batch_size, seq_len]
        ...
    ```
    """

    # =========================================================================
    # STEP 1: Create Message Transform
    # =========================================================================
    # **WHAT**: AlpacaToMessages converts raw Alpaca format → chat messages
    # **WHY**: Standardizes format for tokenizer (all data → same structure)
    #
    # TRANSFORMATION:
    # Input (Alpaca):
    #   {"instruction": "What is AI?", "input": "", "output": "AI is..."}
    #
    # Output (Messages):
    #   [
    #       {"role": "user", "content": "What is AI?"},
    #       {"role": "assistant", "content": "AI is..."}
    #   ]
    #
    # This standardization allows:
    # - Same tokenizer works for all dataset types
    # - Easy to add system prompts
    # - Consistent with chat model conventions
    # =========================================================================
    message_transform = AlpacaToMessages(
        train_on_input=train_on_input,  # Controls prompt masking
        column_map=column_map           # Custom column mapping
    )

    # =========================================================================
    # STEP 2: Create SFTDataset
    # =========================================================================
    # **WHAT**: SFTDataset handles:
    #   1. Loading data from source (HuggingFace or local)
    #   2. Applying message transform
    #   3. Applying model transform (tokenization)
    #   4. Optional filtering
    #
    # **HOW IT WORKS**:
    # 1. datasets.load_dataset(source, split=split, **kwargs)
    # 2. For each example:
    #    a. Apply filter_fn (if provided)
    #    b. Apply message_transform → messages
    #    c. Apply model_transform (tokenizer) → tokens
    # 3. Return dataset ready for DataLoader
    #
    # **KEY METHODS**:
    # - __getitem__(idx): Returns tokenized example
    # - __len__(): Returns dataset size
    # - collate(batch): Collates batch for DataLoader
    # =========================================================================
    ds = SFTDataset(
        source=source,                    # Where to load data from
        message_transform=message_transform,  # Alpaca → messages
        model_transform=tokenizer,        # Messages → tokens
        filter_fn=filter_fn,              # Optional filtering
        split=split,                      # Which split to load
        **load_dataset_kwargs,            # Extra load_dataset args
    )

    # =========================================================================
    # STEP 3: Optional Packing
    # =========================================================================
    # **WHAT**: PackedDataset wraps SFTDataset to pack multiple examples
    # **WHY**: Eliminates padding, improves GPU utilization
    # **HOW**: Concatenates examples until reaching max_seq_len
    #
    # EXAMPLE:
    # Input: [Example1 (50 tokens), Example2 (30 tokens), Example3 (40 tokens)]
    # Output: [Example1 + Example2 + Example3] (120 tokens in one sequence)
    #
    # **IMPLEMENTATION**:
    # PackedDataset:
    # 1. Iterates through base dataset
    # 2. Accumulates tokens until max_seq_len
    # 3. Tracks boundaries between examples (for loss masking)
    # 4. Returns packed sequences
    #
    # **VALIDATION**: Requires max_seq_len to be set on tokenizer
    # =========================================================================
    if packed:
        # Validate max_seq_len is set
        if tokenizer.max_seq_len is None:
            raise ValueError(
                "PackedDataset requires a max_seq_len to be set on the tokenizer."
            )
        # Wrap in PackedDataset
        return PackedDataset(ds, max_seq_len=tokenizer.max_seq_len)

    # Return unpacked dataset
    return ds


# ==============================================================================
# VARIANT FACTORY: alpaca_cleaned_dataset
# ==============================================================================
# **WHAT**: A pre-configured variant using the cleaned Alpaca dataset
# **HOW**: Uses functools.partial to create specialized factory
# **WHY**: Convenience - commonly used variant gets its own function
#
# **PARTIAL APPLICATION PATTERN**:
# Instead of:
#   def alpaca_cleaned_dataset(tokenizer, **kwargs):
#       return alpaca_dataset(tokenizer, source="yahma/alpaca-cleaned", **kwargs)
#
# We use partial:
#   alpaca_cleaned_dataset = partial(alpaca_dataset, source="yahma/alpaca-cleaned")
#
# This creates a new function with source pre-filled!
#
# **USAGE**:
# ```python
# # These are equivalent:
# ds1 = alpaca_dataset(tokenizer, source="yahma/alpaca-cleaned")
# ds2 = alpaca_cleaned_dataset(tokenizer)
# ```
# ==============================================================================
alpaca_cleaned_dataset = partial(alpaca_dataset, source="yahma/alpaca-cleaned")

# Add documentation to the partial function
alpaca_cleaned_dataset.__doc__ = """
Builder for the cleaned Alpaca dataset variant.

**WHAT IS ALPACA-CLEANED**:
The original Alpaca dataset (tatsu-lab/alpaca) contains some issues:
- Duplicate examples
- Malformed outputs
- Inconsistent formatting

yahma/alpaca-cleaned fixes these issues, providing higher quality data.

**SOURCE**: https://huggingface.co/datasets/yahma/alpaca-cleaned

**USAGE IN CONFIG**:
```yaml
dataset:
  _component_: torchtune.datasets.alpaca_cleaned_dataset
  train_on_input: False
  packed: False
```

**DIFFERENCE FROM alpaca_dataset**:
Just the source is different:
- alpaca_dataset: Uses "tatsu-lab/alpaca" by default
- alpaca_cleaned_dataset: Uses "yahma/alpaca-cleaned" by default

All other parameters are identical. See alpaca_dataset() for full docs.
"""


# ==============================================================================
# HOW THIS ACHIEVES MODULARITY
# ==============================================================================
"""
**FACTORY PATTERN IN ACTION**:

Traditional approach (tightly coupled):
```python
# Recipe hardcodes dataset
from datasets import load_dataset
raw_data = load_dataset("tatsu-lab/alpaca")
# ... manual transformation code ...
# ... manual tokenization code ...
# Painful to switch datasets!
```

torchtune approach (loosely coupled):
```yaml
# Config specifies dataset
dataset:
  _component_: torchtune.datasets.alpaca_cleaned_dataset
  train_on_input: False
```

```python
# Recipe just instantiates
dataset = config.instantiate(cfg.dataset, tokenizer=tokenizer)
# Works with ANY dataset builder!
```

**BENEFITS**:

1. **Separation of Concerns**:
   - Dataset builder: Data loading and transformation
   - Recipe: Training logic
   - Never mixed together

2. **Flexibility**:
   - Want different dataset? Change _component_ in config
   - Want custom columns? Set column_map
   - Want to pack? Set packed=True

3. **Reusability**:
   - Same builder works across all recipes
   - Same transformation logic for all Alpaca variants
   - Easy to create new variants (partial application)

4. **Testability**:
   - Test dataset builders independently
   - Mock tokenizers for unit tests
   - Validate transformations in isolation

5. **Extensibility**:
   - Add new dataset: Create new builder function
   - Add new transform: Create new message transform
   - Use in config: Just update _component_

**EXAMPLE EXTENSION - Custom Dataset**:
```python
def my_custom_dataset(
    tokenizer: ModelTokenizer,
    source: str = "my_org/my_data",
    **kwargs
) -> SFTDataset:
    # Custom message transform
    message_transform = MyCustomToMessages()

    # Use same SFTDataset infrastructure
    return SFTDataset(
        source=source,
        message_transform=message_transform,
        model_transform=tokenizer,
        **kwargs
    )
```

```yaml
# Use in config
dataset:
  _component_: path.to.my_custom_dataset
  source: my_org/my_data
```

**THE PIPELINE ARCHITECTURE**:

Each stage is modular and composable:

Stage 1: Data Loading
  - Handled by datasets.load_dataset()
  - Supports HuggingFace, local files, custom sources
  - Abstracted away by SFTDataset

Stage 2: Message Transform
  - AlpacaToMessages, ChatToMessages, etc.
  - Protocol: raw_data → List[Message]
  - Easy to add new transforms

Stage 3: Model Transform (Tokenization)
  - Tokenizer implements protocol
  - Protocol: List[Message] → Dict[str, Tensor]
  - Same interface for all tokenizers

Stage 4: Optional Packing
  - PackedDataset wraps any dataset
  - Transparent to the recipe
  - Can be toggled via config

Result: Highly modular, easy to customize at any stage!

**KEY INSIGHT**: The factory pattern + pipeline architecture allows:
- Mix and match: Any dataset + any tokenizer + any packing strategy
- Config-driven: All controlled via YAML
- Recipe-agnostic: Same recipe works with all datasets

This is the essence of torchtune's modularity for data!
"""


# ==============================================================================
# RELATIONSHIP TO OTHER COMPONENTS
# ==============================================================================
"""
**HOW DATASET BUILDERS CONNECT TO THE SYSTEM**:

1. **Config System** (_instantiate.py):
   ```yaml
   dataset:
     _component_: torchtune.datasets.alpaca_cleaned_dataset
     train_on_input: False
   ```

   ```python
   dataset = config.instantiate(cfg.dataset, tokenizer=tokenizer)
   # Calls alpaca_cleaned_dataset(tokenizer, train_on_input=False)
   ```

2. **Tokenizer** (ModelTokenizer):
   - Receives messages from message_transform
   - Converts to tokens and labels
   - Returns dict ready for model

3. **Recipe** (lora_finetune_single_device.py):
   - Creates tokenizer
   - Instantiates dataset with tokenizer
   - Creates DataLoader from dataset
   - Trains on batches

4. **DataLoader** (PyTorch):
   - Calls dataset[idx] for each sample
   - Uses dataset.collate() to batch samples
   - Yields batches to training loop

**FLOW DIAGRAM**:

YAML Config
    ↓
config.instantiate()
    ↓
alpaca_cleaned_dataset(tokenizer)
    ↓
AlpacaToMessages + SFTDataset
    ↓
Load & Transform Data
    ↓
PackedDataset (optional)
    ↓
DataLoader
    ↓
Training Loop

**NEXT**: See Training_Recipe_Lifecycle_Guide.md for how datasets fit into
          the complete training flow!
"""
