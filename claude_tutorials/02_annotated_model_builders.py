# ==============================================================================
# ANNOTATED: torchtune Model Builder Pattern - Two-Level Builder Architecture
# ==============================================================================
# Source: torchtune/models/llama3_1/_model_builders.py
#
# **WHAT**: Builder functions that construct complete model instances with
#           specific architectures and configurations (e.g., Llama 3.1 8B with LoRA)
#
# **WHY**:  Models like Llama have many hyperparameters (num_layers, embed_dim,
#           num_heads, etc.). Model builders provide:
#           1. Sensible defaults for known configurations (8B, 70B, 405B)
#           2. Easy LoRA/QLoRA variants without rewriting model code
#           3. Config-driven instantiation (YAML → Python objects)
#           4. Consistency across experiments
#
# **HOW**:  Two-level architecture for maximum flexibility:
#
#           LEVEL 1 - Component Builders (in _component_builders.py):
#                     Flexible functions accepting ALL parameters
#                     Example: llama3_1(vocab_size, num_layers, ..., lora_rank, ...)
#
#           LEVEL 2 - Model Builders (this file):
#                     Convenience functions with fixed architecture defaults
#                     Example: lora_llama3_1_8b(lora_rank=8, ...)
#                              → calls llama3_1(..., num_layers=32, embed_dim=4096, ...)
#
# **KEY DESIGN PATTERNS**:
#   1. **Factory Pattern**: Functions that construct complex objects
#   2. **Builder Pattern**: Separate construction from representation
#   3. **Partial Application**: QLoRA builders use functools.partial
#   4. **Composition**: Builders compose component builders
# ==============================================================================

from functools import partial

# ==============================================================================
# Import component builders - the flexible, underlying implementation
# ==============================================================================
# WHY SEPARATE FILES:
#   - _component_builders.py: Implementation (how to build the model)
#   - _model_builders.py: Interface (what models are available)
#
# BENEFIT: Component builders can be reused across model families
#          (e.g., llama2, llama3, llama3_1 might share components)
# ==============================================================================
from torchtune.models.llama3_1._component_builders import (
    llama3_1,        # Component builder for base Llama 3.1
    lora_llama3_1    # Component builder for LoRA-enabled Llama 3.1
)

from torchtune.modules import TransformerDecoder  # Return type
from torchtune.modules.peft import LORA_ATTN_MODULES  # Type hint for LoRA config


# ==============================================================================
# DOCUMENTATION STRING: Explains the two-level pattern
# ==============================================================================
"""
Model builders build specific instantiations using component builders. For example
the llama3_1_8b model builder uses the llama3 component builder to create the
Llama3.1 8B model.

**ARCHITECTURE OVERVIEW**:

    YAML Config
        ↓
    model:
      _component_: torchtune.models.llama3_1.lora_llama3_1_8b  ← Model Builder
      lora_rank: 8                                              ← User params
      lora_alpha: 16
        ↓
    lora_llama3_1_8b(lora_rank=8, lora_alpha=16)              ← Model Builder
        ↓
    lora_llama3_1(                                            ← Component Builder
        vocab_size=128_256,    # ← 8B Architecture defaults
        num_layers=32,
        num_heads=32,
        ...,
        lora_rank=8,           # ← User params passed through
        lora_alpha=16
    )
        ↓
    TransformerDecoder instance (ready to train!)
"""


# ==============================================================================
# BASE MODEL BUILDERS: Standard (non-LoRA) Models
# ==============================================================================
# WHEN USED: Full fine-tuning scenarios where all parameters are trainable
# ==============================================================================

def llama3_1_8b() -> TransformerDecoder:
    """
    Builder for Llama 3.1 8B base model.

    **WHAT**: Constructs a standard Llama 3.1 8B model without adapters
    **WHEN**: Full fine-tuning (all 8 billion parameters trainable)
    **HOW**: Calls component builder with 8B architecture defaults

    **ARCHITECTURE SPECS** (8B model):
    - vocab_size: 128,256 tokens (larger than Llama 2's 32K)
    - num_layers: 32 transformer blocks
    - num_heads: 32 attention heads (standard MHA)
    - num_kv_heads: 8 (Grouped Query Attention for efficiency)
    - embed_dim: 4096 (hidden dimension)
    - max_seq_len: 131,072 (128K context - huge!)
    - intermediate_dim: 14,336 (FFN dimension, ~3.5× embed_dim)
    - rope_base: 500,000 (RoPE frequency base, tuned for long context)

    **WHY THESE NUMBERS**: Determined by Meta through extensive pretraining
                           experiments to balance capacity, speed, memory
    """
    return llama3_1(
        vocab_size=128_256,      # Vocabulary size
        num_layers=32,           # Number of transformer blocks
        num_heads=32,            # Query heads (Multi-Head Attention)
        num_kv_heads=8,          # Key/Value heads (Grouped Query Attention)
        embed_dim=4096,          # Model dimension (d_model)
        max_seq_len=131072,      # Maximum sequence length (128K tokens!)
        intermediate_dim=14336,  # FFN intermediate dimension
        attn_dropout=0.0,        # Attention dropout (0 for Llama)
        norm_eps=1e-5,           # RMSNorm epsilon
        rope_base=500_000,       # RoPE base frequency
    )


def llama3_1_70b() -> TransformerDecoder:
    """
    Builder for Llama 3.1 70B base model.

    **SCALING FROM 8B TO 70B**:
    - More layers: 32 → 80 (2.5×)
    - Wider: embed_dim 4096 → 8192 (2×)
    - More heads: 32 → 64 (2×)
    - Deeper FFN: 14336 → 28672 (2×)

    **MEMORY REQUIREMENTS**: ~140GB for weights alone (fp32)
                            ~70GB with bf16
                            ~35GB with int4 quantization
    """
    return llama3_1(
        vocab_size=128_256,
        num_layers=80,           # Much deeper!
        num_heads=64,            # More attention heads
        num_kv_heads=8,          # KV heads stay same (more sharing)
        embed_dim=8192,          # Wider model
        max_seq_len=131072,
        intermediate_dim=28672,  # Proportionally wider FFN
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
    )


def llama3_1_405b() -> TransformerDecoder:
    """
    Builder for Llama 3.1 405B base model (LARGEST OPEN MODEL).

    **MASSIVE SCALE**:
    - 126 layers (almost 4× the 8B model)
    - embed_dim: 16,384 (4× the 8B model)
    - 128 attention heads

    **DEPLOYMENT**: Requires multiple GPUs (typically 8× A100 80GB minimum)
                   Usually used with:
                   - Tensor parallelism (split across GPUs)
                   - Pipeline parallelism (layers on different GPUs)
                   - Quantization (int8 or int4)
    """
    return llama3_1(
        vocab_size=128_256,
        num_layers=126,
        num_heads=128,
        num_kv_heads=8,          # Still 8 KV heads (extreme GQA)
        embed_dim=16384,
        max_seq_len=131072,
        intermediate_dim=53248,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
    )


# ==============================================================================
# LORA MODEL BUILDERS: Memory-Efficient Adaptation
# ==============================================================================
# **CRITICAL DIFFERENCE**: These builders:
#   1. Create the same base architecture as above
#   2. Wrap specific layers (attention, MLP) with LoRALinear
#   3. Freeze base weights, only train LoRA adapters
#
# **MEMORY IMPACT**: Training 8B LoRA uses ~24GB vs. ~80GB for full fine-tuning
# ==============================================================================

def lora_llama3_1_8b(
    lora_attn_modules: list[LORA_ATTN_MODULES],  # Which attention layers get LoRA
    apply_lora_to_mlp: bool = False,              # Apply LoRA to FFN?
    apply_lora_to_output: bool = False,           # Apply LoRA to output projection?
    lora_rank: int = 8,                           # THE KEY PARAMETER
    lora_alpha: float = 16,                       # Scaling factor
    lora_dropout: float = 0.0,                    # Adapter dropout
    use_dora: bool = False,                       # DoRA variant (magnitude + direction)
    quantize_base: bool = False,                  # QLoRA: quantize base weights?
) -> TransformerDecoder:
    """
    Builder for LoRA-enhanced Llama 3.1 8B.

    **HOW LORA IS APPLIED**:

    1. Start with base Llama 3.1 8B architecture
    2. Replace specified linear layers with LoRALinear:

       Base Model:
       ┌─────────────────────────────────────┐
       │  Attention Block                    │
       │  ┌───────────┐  ┌───────────┐      │
       │  │  q_proj   │  │  k_proj   │      │  ← Standard nn.Linear
       │  │ (frozen)  │  │ (frozen)  │      │
       │  └───────────┘  └───────────┘      │
       └─────────────────────────────────────┘

       With LoRA:
       ┌─────────────────────────────────────┐
       │  Attention Block                    │
       │  ┌───────────────────────────┐      │
       │  │  q_proj (LoRALinear)      │      │
       │  │  ┌─────────┐ ┌─────────┐  │      │
       │  │  │ frozen  │ │ lora_a  │  │      │  ← Adapter added!
       │  │  │ W₀      │ │ (train) │  │      │
       │  │  └─────────┘ │ lora_b  │  │      │
       │  │              │ (train) │  │      │
       │  │              └─────────┘  │      │
       │  └───────────────────────────┘      │
       └─────────────────────────────────────┘

    **PARAMETERS EXPLAINED**:

    lora_attn_modules: Which attention projections to adapt
        - Options: ['q_proj', 'k_proj', 'v_proj', 'output_proj']
        - Common: ['q_proj', 'v_proj', 'output_proj'] (default in configs)
        - Why not k_proj?: Empirically works well without it

    apply_lora_to_mlp: Whether to add LoRA to feed-forward layers
        - True: More capacity, slower, more memory
        - False: Faster, less memory (often sufficient)

    apply_lora_to_output: Whether to add LoRA to final output projection
        - Usually False (output projection is relatively small)

    lora_rank: THE CRITICAL HYPERPARAMETER
        - Controls trainable parameters and model capacity
        - Typical range: 4-64
        - 8: Good default, ~0.1% of base model params
        - 16: More capacity, ~0.2% of base params
        - Higher rank = more params, slower training, better quality (diminishing returns)

    lora_alpha: Scaling factor
        - Controls magnitude of adapter updates
        - Rule of thumb: alpha = 2 × rank
        - Can be tuned without retraining (scales existing adapters)

    **PARAMETER COUNT EXAMPLE**:
    Base Llama 3.1 8B: 8,030,261,248 parameters

    With LoRA (rank=8, on q_proj, v_proj, output_proj per layer):
    Per layer: 3 projections × (4096×8 + 8×4096) = 196,608 params
    Total: 32 layers × 196,608 = 6,291,456 params
    Reduction: 8B → 6.3M trainable (0.08%!)
    """
    # =========================================================================
    # Call component builder with architecture defaults + LoRA params
    # =========================================================================
    # PATTERN: Model builder (specific config) → Component builder (general)
    # =========================================================================
    return lora_llama3_1(
        # LoRA configuration (user-specified or defaults)
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,

        # Architecture defaults (8B-specific)
        vocab_size=128_256,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=4096,
        max_seq_len=131072,
        intermediate_dim=14336,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,

        # LoRA hyperparameters (passed through)
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
    )


def lora_llama3_1_70b(
    lora_attn_modules: list[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    apply_lora_to_output: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    quantize_base: bool = False,
) -> TransformerDecoder:
    """
    Builder for LoRA-enhanced Llama 3.1 70B.

    **SCALING LORA TO LARGER MODELS**:

    Key insight: LoRA rank typically DOESN'T need to scale with model size!
    - 8B model with rank=8: Great results
    - 70B model with rank=8: Still great results
    - 405B model with rank=8: Still works!

    WHY: The rank controls the dimensionality of the adaptation space,
         not the base model size. Larger models might benefit from
         slightly higher ranks (e.g., 16 or 32) but not proportionally.

    **MEMORY SAVINGS**:
    70B full fine-tuning: ~140GB (bf16)
    70B LoRA (rank=8):    ~80GB (base) + ~0.1GB (adapters) = ~80GB
    Reduction: ~43% memory savings!
    """
    return lora_llama3_1(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,

        # 70B architecture defaults
        vocab_size=128_256,
        num_layers=80,        # More layers than 8B
        num_heads=64,         # More heads
        num_kv_heads=8,
        embed_dim=8192,       # Wider
        max_seq_len=131072,
        intermediate_dim=28672,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,

        # LoRA hyperparameters (same as 8B!)
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
    )


def lora_llama3_1_405b(
    lora_attn_modules: list[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    apply_lora_to_output: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    lora_dropout: float = 0.0,
    quantize_base: bool = False,  # Note: use_dora not available for 405B
) -> TransformerDecoder:
    """
    Builder for LoRA-enhanced Llama 3.1 405B.

    **405B + LORA: MAKING THE IMPOSSIBLE POSSIBLE**

    Full 405B fine-tuning: ~800GB memory (impossible on single node!)
    405B with LoRA + int4: ~200GB (possible on 8× A100 80GB)

    **TYPICAL SETUP**:
    - Quantize base to int4 (quantize_base=True)
    - Use rank=8 or 16
    - Distributed across 8 GPUs
    - Gradient checkpointing enabled

    Result: Can fine-tune the world's largest open model!
    """
    return lora_llama3_1(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,

        # 405B architecture defaults
        vocab_size=128_256,
        num_layers=126,       # Massive depth
        num_heads=128,        # Many heads
        num_kv_heads=8,
        embed_dim=16384,      # Very wide
        max_seq_len=131072,
        intermediate_dim=53248,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,

        # LoRA hyperparameters
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        quantize_base=quantize_base,
    )


# ==============================================================================
# QLORA BUILDERS: Quantization + LoRA
# ==============================================================================
# **WHAT**: QLoRA = Quantized base weights + LoRA adapters
# **HOW**: Uses functools.partial to create variants with quantize_base=True
# **WHY**: Maximum memory efficiency for largest models
#
# **TECHNIQUE**: partial application - create new function with some args fixed
# ==============================================================================

# Create QLoRA variant by "partially applying" quantize_base=True
# This is equivalent to:
#   def qlora_llama3_1_8b(**kwargs):
#       return lora_llama3_1_8b(**kwargs, quantize_base=True)
qlora_llama3_1_8b = partial(lora_llama3_1_8b, quantize_base=True)

# Document the partial function
qlora_llama3_1_8b.__doc__ = """
Builder for QLoRA Llama 3.1 8B.

**QLORA = QUANTIZED LORA**:
Combines two memory-saving techniques:
1. Base weights quantized to 4-bit NF4 format (8× compression)
2. LoRA adapters for parameter-efficient training

**MEMORY SAVINGS**:
Full fine-tuning (bf16):  ~16GB (model) + ~64GB (optimizer + gradients) = 80GB
LoRA (bf16):              ~16GB (model) + ~8GB (adapters + optimizer) = 24GB
QLoRA (int4 + bf16):       ~2GB (model) + ~8GB (adapters + optimizer) = 10GB

Result: Can fine-tune 8B model on single 16GB GPU!

**QUALITY**:
Remarkably, QLoRA maintains nearly the same quality as full LoRA despite:
- 4-bit base weights (huge compression!)
- Tiny adapter (<<1% of params)

See: https://arxiv.org/abs/2305.14314 for the paper

**PARAMETERS**: Same as lora_llama3_1_8b, but base weights auto-quantized
"""

qlora_llama3_1_70b = partial(lora_llama3_1_70b, quantize_base=True)
qlora_llama3_1_70b.__doc__ = """
Builder for QLoRA Llama 3.1 70B.

**MEMORY MATH**:
70B model in bf16: ~140GB
70B model in int4: ~17.5GB (8× compression!)
70B QLoRA total:   ~17.5GB (base) + ~10GB (adapters/opt) = ~27.5GB

AMAZING: Can fine-tune 70B model on single A100 40GB!

Please see `lora_llama3_1_70b` for full API arguments.
"""

qlora_llama3_1_405b = partial(lora_llama3_1_405b, quantize_base=True)
qlora_llama3_1_405b.__doc__ = """
Builder for QLoRA Llama 3.1 405B.

**THE ULTIMATE COMPRESSION**:
405B full fine-tuning: 800GB+ (impossible!)
405B LoRA: ~810GB (still very hard)
405B QLoRA: ~100GB (possible on 4× A100 40GB or 2× H100 80GB)

This makes the world's largest open model actually trainable!

Please see `lora_llama3_1_405b` for full API arguments.
"""


# ==============================================================================
# CONFIGURATION FLOW: How YAML Config Becomes a Model
# ==============================================================================
"""
**COMPLETE FLOW FROM YAML TO MODEL**:

1. USER WRITES CONFIG (llama3_1/8B_lora_single_device.yaml):
   ```yaml
   model:
     _component_: torchtune.models.llama3_1.lora_llama3_1_8b
     lora_attn_modules: ['q_proj', 'v_proj', 'output_proj']
     lora_rank: 8
     lora_alpha: 16
   ```

2. CONFIG PARSER (torchtune.config._parse.py):
   - Loads YAML file
   - Merges with CLI overrides
   - Creates DictConfig object

3. INSTANTIATION ENGINE (torchtune.config._instantiate.py):
   - Sees _component_ field
   - Imports: torchtune.models.llama3_1.lora_llama3_1_8b
   - Calls: lora_llama3_1_8b(
               lora_attn_modules=['q_proj', 'v_proj', 'output_proj'],
               lora_rank=8,
               lora_alpha=16
            )

4. MODEL BUILDER (this file - lora_llama3_1_8b):
   - Adds 8B architecture defaults
   - Calls component builder: lora_llama3_1(
       # User params
       lora_attn_modules=['q_proj', 'v_proj', 'output_proj'],
       lora_rank=8,
       lora_alpha=16,
       # Architecture defaults
       vocab_size=128_256,
       num_layers=32,
       ...
     )

5. COMPONENT BUILDER (_component_builders.py - lora_llama3_1):
   - Creates base Llama 3.1 architecture
   - Identifies attention modules to replace
   - Wraps specified nn.Linear with LoRALinear
   - Returns TransformerDecoder instance

6. RESULT: Fully constructed model ready for training!

**KEY INSIGHT**: User only specifies:
- Which model (via _component_)
- LoRA-specific params (rank, alpha, modules)

Everything else (architecture) is automatic!
"""


# ==============================================================================
# MODULARITY BENEFITS
# ==============================================================================
"""
**HOW THIS DESIGN ACHIEVES MODULARITY**:

1. **Separation of Concerns**:
   - User config: High-level decisions (LoRA rank, which modules)
   - Model builders: Architecture presets (8B/70B/405B specs)
   - Component builders: Implementation (how to construct model)

2. **Easy Extensibility**:
   Want a new variant?
   ```python
   # In 5 lines, add a new variant:
   dora_llama3_1_8b = partial(lora_llama3_1_8b, use_dora=True)
   ```

3. **Composability**:
   - Same component builder reused for all sizes
   - Same LoRA implementation works across all models
   - Mix and match: any size + any adapter type

4. **Config-Driven**:
   - Change model size: Edit one line in YAML (_component_: ...llama3_1_70b)
   - Try different rank: Edit one line (lora_rank: 16)
   - No code changes needed!

5. **Testability**:
   - Model builders are pure functions
   - Easy to test each size independently
   - Mock component builders for unit testing

6. **Discoverability**:
   - `tune ls` shows all available builders
   - Each builder has clear documentation
   - Type hints show expected parameters
"""


# ==============================================================================
# SUMMARY: The Builder Pattern in Action
# ==============================================================================
"""
**KEY TAKEAWAYS**:

1. **Two Levels** = Flexibility + Convenience
   - Component builders: Maximum flexibility
   - Model builders: Common use cases

2. **Pure Functions** = Predictable, Testable
   - No global state
   - Same inputs → same outputs
   - Easy to reason about

3. **Partial Application** = Code Reuse
   - QLoRA builders reuse LoRA builders
   - No code duplication

4. **Config-Driven** = User-Friendly
   - YAML → Builders → Models
   - No manual instantiation needed

5. **Composable** = Extensible
   - New sizes: Add builder function
   - New techniques: Extend component builders
   - New config options: Add parameters

**NEXT**: See annotated config files to understand how YAML configs
          specify and instantiate these model builders!
"""
