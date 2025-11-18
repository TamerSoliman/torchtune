# ==============================================================================
# ANNOTATED: torchtune LoRA Module - Parameter-Efficient Fine-Tuning (PEFT)
# ==============================================================================
# Source: torchtune/modules/peft/lora.py
#
# **WHAT**: This module implements LoRA (Low-Rank Adaptation), a PEFT technique
#           that dramatically reduces memory and training time by only training
#           a small number of additional parameters while keeping the base model
#           frozen.
#
# **WHY**:  Fine-tuning large language models (8B+ parameters) requires massive
#           GPU memory. LoRA makes fine-tuning accessible by:
#           - Freezing the original model weights (W₀)
#           - Adding trainable low-rank matrices (A and B)
#           - Computing: output = W₀x + (α/r)BAx
#           This reduces trainable parameters from billions to millions!
#
# **HOW**:  LoRA achieves modularity through:
#           1. Inheritance from nn.Module for PyTorch integration
#           2. Inheritance from AdapterModule for PEFT protocol compliance
#           3. Parameter encapsulation: base weights vs. adapter weights
#           4. Optional quantization for even more memory savings (QLoRA)
#
# **KEY DESIGN PATTERN**: Adapter Pattern - LoRALinear "wraps" a standard
#                         linear layer, adding behavior without modifying it
# ==============================================================================

import math
from enum import Enum
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

# ============================================================================
# WHY THIS IMPORT: torchao provides efficient 4-bit NF4 quantization
# HOW IT WORKS: The base weights can be quantized to 4-bit to save memory
#               while the LoRA adapters remain in full precision
# ============================================================================
from torchao.dtypes.nf4tensor import linear_nf4, to_nf4
from torchtune.modules.low_precision import _register_nf4_dispatch_ops
from torchtune.modules.peft import AdapterModule


# ============================================================================
# ENUM: TrainableParams - Defines training modes for layers
# ============================================================================
# WHY: Different fine-tuning scenarios require different parameter training
# WHAT: Three modes:
#   - FULL: All parameters trainable (standard fine-tuning)
#   - LORA: Only LoRA adapter parameters trainable (memory efficient)
#   - FROZEN: No parameters trainable (useful for evaluation)
# ============================================================================
class TrainableParams(Enum):
    FULL = "full"
    LORA = "lora"
    FROZEN = "frozen"


# ============================================================================
# CLASS: LoRALinear - Core LoRA Implementation
# ============================================================================
# **WHAT**: A drop-in replacement for nn.Linear that adds LoRA adapters
#
# **WHY THE MATH**: Instead of fine-tuning all weights in W₀ (e.g., 4096×4096),
#                   LoRA learns two small matrices:
#                   - A: input_dim × rank (e.g., 4096 × 8)
#                   - B: rank × output_dim (e.g., 8 × 4096)
#                   Total params: (4096 + 4096) × 8 = 65K instead of 16M!
#
# **KEY PARAMETERS**:
#   - rank (r): Size of the bottleneck. Smaller = fewer params, faster training
#   - alpha (α): Scaling factor. Usually α = 2×rank for stable training
#   - dropout: Applied before LoRA to prevent overfitting
#
# **HOW IT'S MODULAR**:
#   1. Inherits from both nn.Module and AdapterModule (multiple inheritance)
#   2. AdapterModule provides a protocol for identifying adapter parameters
#   3. Can be swapped in/out of any model that uses nn.Linear
#   4. Self-contained: manages its own weights, initialization, forward pass
# ============================================================================
class LoRALinear(nn.Module, AdapterModule):
    """
    LoRA linear layer from https://arxiv.org/abs/2106.09685

    **MATHEMATICAL FORMULATION**:
    Standard linear layer: y = W₀x + b
    LoRA-enhanced layer:   y = W₀x + (α/r)BAx + b

    Where:
    - W₀: Frozen pre-trained weights [out_dim × in_dim]
    - B: Trainable down-projection [out_dim × rank]
    - A: Trainable up-projection [rank × in_dim]
    - α: Scaling factor (controls magnitude of adaptation)
    - r: Rank (controls number of trainable parameters)

    **MEMORY SAVINGS EXAMPLE** (for a 4096 → 4096 layer):
    - Standard fine-tuning: 16,777,216 parameters
    - LoRA with rank=8:     65,536 parameters (256× reduction!)
    """

    def __init__(
        self,
        in_dim: int,              # Input dimension (e.g., 4096 for Llama)
        out_dim: int,             # Output dimension (e.g., 4096 for Llama)
        rank: int,                # LoRA rank - THE KEY HYPERPARAMETER
        alpha: float,             # Scaling factor - usually 2×rank
        dropout: float = 0.0,     # Dropout before LoRA (prevent overfitting)
        use_bias: bool = False,   # Whether to include bias (usually False for LLMs)
        quantize_base: bool = False,  # Enable QLoRA (4-bit base weights)
        **quantization_kwargs,    # Block size, scaler options for quantization
    ):
        super().__init__()

        # ====================================================================
        # STEP 1: Store configuration
        # WHY: Needed for checkpoint saving/loading and introspection
        # ====================================================================
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank          # Critical: determines memory/quality tradeoff
        self.alpha = alpha        # Critical: controls adapter strength
        self.use_bias = use_bias
        self._quantize_base = quantize_base

        # ====================================================================
        # VALIDATION: Ensure quantization args only provided when needed
        # WHY: Prevents user confusion - if quantize_base=False, kwargs ignored
        # ====================================================================
        if not self._quantize_base and any([v for v in quantization_kwargs.values()]):
            raise ValueError(
                f"``quantize_base`` is False, but received quantization arguments: {quantization_kwargs}"
            )

        # ====================================================================
        # STEP 2: Create base linear layer (FROZEN during training)
        # WHY: We need the original weights W₀ for the forward pass
        # HOW: Can optionally quantize to 4-bit NF4 format (QLoRA)
        # ====================================================================
        linear = nn.Linear(in_features=in_dim, out_features=out_dim, bias=self.use_bias)

        # QLORA QUANTIZATION: Convert base weights to 4-bit
        # WHY: Reduces memory by 8× (32-bit → 4-bit) for base weights
        # WHAT: Uses NF4 (Normal Float 4) - optimized for neural net weights
        weight = (
            linear.weight
            if not self._quantize_base
            else to_nf4(linear.weight, **quantization_kwargs)  # Quantize!
        )
        bias = linear.bias if self.use_bias else None

        # ====================================================================
        # STEP 3: Register base weights as parameters
        # WHY: PyTorch needs to know about these for state_dict, device movement
        # NOTE: These will be frozen during training (requires_grad controlled elsewhere)
        # ====================================================================
        self.disabled = False  # Flag for DPO: disable adapter to use base model
        self.register_parameter("weight", nn.Parameter(weight))
        self.register_parameter(
            "bias", nn.Parameter(bias) if bias is not None else None
        )

        # ====================================================================
        # STEP 4: Create dropout layer (optional regularization)
        # WHY: Prevents overfitting of LoRA adapters
        # HOW: Applied to input BEFORE LoRA transformation
        # ====================================================================
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # ====================================================================
        # STEP 5: Create LoRA adapter matrices A and B
        # **THIS IS THE CORE OF LORA** - Only these are trainable!
        #
        # ARCHITECTURE:
        #   x [batch, seq, in_dim]
        #     → dropout
        #     → lora_a [in_dim → rank]      # Dimensionality reduction
        #     → lora_b [rank → out_dim]     # Dimensionality expansion
        #     → scale by (α/r)
        #     → add to base output
        #
        # WHY TWO LAYERS: Creates a low-rank factorization
        #   - Rank controls capacity: low rank = fewer params, less expressive
        #   - Mathematical intuition: Any matrix can be approximated by low-rank factors
        # ====================================================================
        self.lora_a = nn.Linear(in_features=in_dim, out_features=rank, bias=False)
        self.lora_b = nn.Linear(in_features=rank, out_features=out_dim, bias=False)

        # ====================================================================
        # STEP 6: Initialize adapter parameters
        # WHY IMPORTANT: Proper initialization ensures:
        #   1. Training stability (no exploding/vanishing gradients)
        #   2. Initial behavior matches base model (B initialized to zero)
        # HOW: See _lora_a_init_params() and _lora_b_init_params() below
        # ====================================================================
        self.merged = False  # For weight merging (not used during training)
        self.initialize_parameters()

    def to_empty(
        self, *, device: Optional[Union[str, torch.device, int]], recurse: bool = True
    ):
        """
        Move LoRA parameters to device without initializing memory.

        WHY: Efficient for distributed training - allocate on device directly
        WHEN USED: During model initialization in FSDP (Fully Sharded Data Parallel)
        """
        self.lora_a.to_empty(device=device, recurse=recurse)
        self.lora_b.to_empty(device=device, recurse=recurse)

    def initialize_parameters(self):
        """
        Initialize LoRA adapter weights.

        WHY CRITICAL: Zero-initialized lora_b ensures that initially:
                      output = W₀x + (α/r) × B × A × x
                            = W₀x + (α/r) × 0 × A × x
                            = W₀x
                      So the model starts identical to the base model!

        INITIALIZATION STRATEGY (from LoRA paper):
        - lora_a: Kaiming uniform (standard init, creates random projections)
        - lora_b: Zero initialization (ensures no change to base model initially)
        """
        _lora_a_init_params(self.lora_a)  # Random init
        _lora_b_init_params(self.lora_b)  # Zero init

    def adapter_params(self) -> list[str]:
        """
        Return names of trainable adapter parameters.

        WHY: The recipe needs to know which parameters to:
             1. Optimize (set requires_grad=True)
             2. Save separately (adapter checkpoints)
             3. Report in logs (trainable param count)

        WHAT: Returns ["lora_a.weight", "lora_b.weight"]
              These are the ONLY trainable params in LoRA!

        HOW USED: In recipes, this enables:
              adapter_params = {n for n, p in model.named_parameters()
                                if any(key in n for key in module.adapter_params())}
        """
        adapter_params = ["lora_a.weight", "lora_b.weight"]
        return adapter_params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoRA adaptation.

        **THE CORE LORA COMPUTATION**:

        Args:
            x: Input tensor [..., in_dim]

        Returns:
            Output tensor [..., out_dim]

        STEP-BY-STEP:
        1. Compute base model output: out = W₀x + b
        2. If not disabled, compute LoRA adaptation:
           a. Apply dropout to input
           b. Project down: h = A × dropout(x)    [... × rank]
           c. Project up:   Δ = B × h             [... × out_dim]
           d. Scale:        Δ = (α/r) × Δ
        3. Add adaptation to base: final = out + Δ

        WHY THE SCALING (α/r):
        - r (rank): Smaller rank → smaller magnitude updates needed
        - α: Allows controlling adapter strength without retraining
        - Typical: α=16, r=8, so scaling = 2.0
        """
        # ====================================================================
        # STEP 1: Compute base model output
        # HOW: Use quantized forward if base is quantized (QLoRA)
        # ====================================================================
        if self._quantize_base:
            # QLoRA path: Use specialized 4-bit matmul
            out = linear_nf4(input=x, weight=self.weight)
            if self.use_bias:
                out = out + self.bias
        else:
            # Standard path: Regular matmul
            out = F.linear(x, self.weight, self.bias)

        # ====================================================================
        # STEP 2: Early return if adapter disabled (for DPO reference model)
        # WHY: In DPO, we need both adapted model (policy) and base model (reference)
        # HOW: Set self.disabled=True to get base model behavior
        # ====================================================================
        if self.disabled:
            return out

        # ====================================================================
        # STEP 3: Compute LoRA adaptation
        # THIS IS THE KEY OPERATION: Δout = (α/r) × B × A × x
        # ====================================================================
        lora_out = self.lora_a(self.dropout(x))  # Project to low dimension
        lora_out = (self.alpha / self.rank) * self.lora_b(lora_out)  # Project back & scale

        # ====================================================================
        # STEP 4: Add adaptation to base output
        # FINAL FORMULA: y = W₀x + (α/r)BAx
        # ====================================================================
        return out + lora_out


# ==============================================================================
# QLoRA Extension: Quantization-Aware Training with LoRA
# ==============================================================================
# WHAT: Extends LoRA with fake quantization during training
# WHY: Improves final quantized model accuracy by simulating quantization
# HOW: Applies fake quantization to activations and/or weights before adapters
#
# KEY DIFFERENCE FROM LORA:
# - LoRA: base weights frozen (optionally quantized), adapters full precision
# - QAT LoRA: fake quantization applied during training to improve real quantization
# ==============================================================================
class QATLoRALinear(LoRALinear):
    """
    LoRA with Quantization-Aware Training (QAT).

    **WHAT IS QAT**: Training technique that simulates quantization numerics
                     without actually reducing precision. This improves the
                     accuracy of the final quantized model.

    **HOW IT WORKS**:
    1. During training: Apply "fake quantization" (quantize then dequantize)
    2. After training: Apply real quantization (much better accuracy!)

    **WHEN TO USE**: When you plan to deploy with quantization (int8/int4).
                     The accuracy loss from quantization will be much smaller.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        activation_qat_config: Optional["FakeQuantizeConfig"] = None,
        weight_qat_config: Optional["FakeQuantizeConfig"] = None,
    ):
        # Initialize parent (LoRA base functionality)
        # Note: quantize_base=False because QAT uses fake quantization
        super().__init__(
            in_dim,
            out_dim,
            rank,
            alpha,
            dropout,
            use_bias=False,
            quantize_base=False,
        )

        # Import QAT components (requires torchao 0.7+)
        try:
            from torchao.quantization.qat.api import FakeQuantizeConfig
            from torchao.quantization.qat.fake_quantizer import FakeQuantizer
        except ImportError as err:
            raise ValueError(
                "QATLoRALinear requires torchao 0.7+"
            ) from err

        # ====================================================================
        # Setup fake quantization for activations
        # WHY: Simulates int8 activation quantization during training
        # WHEN: If you plan to use int8 activations in deployment
        # ====================================================================
        if activation_qat_config is not None:
            self.activation_fake_quantizer = FakeQuantizer(activation_qat_config)
        else:
            self.activation_fake_quantizer = nn.Identity()

        # ====================================================================
        # Setup fake quantization for weights
        # WHY: Simulates int4/int8 weight quantization during training
        # WHEN: If you plan to use quantized weights in deployment
        # ====================================================================
        if weight_qat_config is not None:
            group_size = weight_qat_config.group_size
            if group_size is not None and in_dim % group_size != 0:
                raise ValueError(
                    f"in_dim ({in_dim}) must be divisible by group_size ({group_size})"
                )
            self.weight_fake_quantizer = FakeQuantizer(weight_qat_config)
        else:
            self.weight_fake_quantizer = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with fake quantization.

        DIFFERENCE FROM REGULAR LORA:
        Instead of: out = W×x
        We compute: out = fake_quant(W) × fake_quant(x)

        This simulates quantization without actually reducing precision!
        """
        # Apply fake quantization to activations
        _x = self.activation_fake_quantizer(x)
        # Apply fake quantization to weights
        w = self.weight_fake_quantizer(self.weight)
        # Compute base output with "quantized" inputs
        out = F.linear(_x, w)

        # Early return if disabled
        if self.disabled:
            return out

        # Compute LoRA adaptation (same as regular LoRA)
        lora_out = self.lora_a(self.dropout(x))
        lora_out = (self.alpha / self.rank) * self.lora_b(lora_out)
        return out + lora_out


# ==============================================================================
# Helper Functions: Parameter Initialization
# ==============================================================================
# WHY SEPARATE FUNCTIONS: Initialization logic may be reused, easier to test
# ==============================================================================

def _lora_a_init_params(x: nn.Linear) -> None:
    """
    Initialize LoRA A matrix with Kaiming uniform.

    WHY KAIMING: Designed for ReLU activations, maintains variance across layers
    EFFECT: Creates random projections from input space to low-rank space
    """
    nn.init.kaiming_uniform_(x.weight, a=math.sqrt(5))


def _lora_b_init_params(x: nn.Linear) -> None:
    """
    Initialize LoRA B matrix to zeros.

    **CRITICAL FOR LORA**: This ensures that initially, the LoRA adapter
    produces zero output, so the model behaves exactly like the base model.
    As training progresses, B learns to produce useful adaptations.

    WHY ZEROS: Makes LoRA a "residual adapter":
        output = base_output + 0  (initially)
        output = base_output + learned_adaptation  (after training)
    """
    nn.init.zeros_(x.weight)


# ==============================================================================
# SUMMARY: How LoRA Achieves Modularity
# ==============================================================================
# 1. **Adapter Pattern**: Wraps nn.Linear without modifying it
# 2. **Protocol Compliance**: Inherits from AdapterModule interface
# 3. **Composable**: Can be dropped into any model using nn.Linear
# 4. **Configurable**: rank, alpha, dropout control behavior
# 5. **Checkpoint-friendly**: adapter_params() identifies what to save
# 6. **Extensible**: QATLoRALinear shows how to extend base behavior
#
# **HOW IT'S USED IN TORCHTUNE**:
# 1. Model builders replace specific nn.Linear with LoRALinear
#    (e.g., in attention: q_proj, k_proj, v_proj)
# 2. Recipe sets requires_grad=True only for adapter_params()
# 3. Checkpointer saves adapter weights separately
# 4. Config system instantiates with proper rank/alpha from YAML
#
# **CONFIGURATION FLOW** (see annotated config files):
# YAML config specifies:
#   model:
#     _component_: torchtune.models.llama3_1.lora_llama3_1_8b
#     lora_rank: 8
#     lora_alpha: 16
#     lora_attn_modules: ['q_proj', 'v_proj', 'output_proj']
#
# This calls the model builder, which:
# 1. Creates base model with nn.Linear layers
# 2. Replaces specified layers with LoRALinear(rank=8, alpha=16)
# 3. Returns model ready for training!
# ==============================================================================
