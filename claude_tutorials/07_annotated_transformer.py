# ==============================================================================
# ANNOTATED: torchtune Transformer Layers - Composable Architecture Building Blocks
# ==============================================================================
# Source: torchtune/modules/transformer.py
#
# **WHAT**: Core transformer layer implementations that form the building blocks
#           of all LLMs in torchtune. These layers implement the standard
#           transformer architecture with modern improvements.
#
# **WHY**:  All modern LLMs (Llama, Gemma, Qwen) use transformer architecture:
#           - Self-attention for sequence processing
#           - Feed-forward networks for transformation
#           - Residual connections for gradient flow
#           - Layer normalization for stability
#
# **HOW**:  Composition pattern - layers built from smaller modules:
#
#           TransformerSelfAttentionLayer
#           ├── sa_norm (RMSNorm)
#           ├── attn (MultiHeadAttention)
#           │   ├── q_proj, k_proj, v_proj
#           │   └── output_proj
#           ├── mlp_norm (RMSNorm)
#           └── mlp (FeedForward)
#               ├── w1, w2, w3 (SwiGLU)
#               └── activation
#
# **KEY DESIGN PATTERNS**:
#   1. **Composition**: Layer composed of attention + FFN + norms
#   2. **Protocol/Interface**: TransformerDecoder works with any layer type
#   3. **Pre-Normalization**: Norm before attention/FFN (Llama-style)
#   4. **Residual Connections**: x + layer(norm(x))
#   5. **Modularity**: Easy to swap attention mechanisms or FFNs
# ==============================================================================

from typing import Optional, Callable

import torch
from torch import nn

# Placeholder imports (in real code, these come from torchtune.modules)
from torchtune.modules import MultiHeadAttention
from torchtune.modules.attention_utils import _MaskType


# ==============================================================================
# CORE BUILDING BLOCK: TransformerSelfAttentionLayer
# ==============================================================================
# **WHAT**: A single transformer layer with self-attention and feed-forward
# **WHY**: Reusable block - stack 32 of these → 8B model, 80 → 70B model
# **HOW**: Pre-norm architecture: norm → attention/FFN → residual add
#
# **ARCHITECTURE** (Llama-style):
# ```
# Input (x)
#   ├→ LayerNorm → MultiHeadAttention → scale → (+)───┐
#   │                                                   ↓
#   └────────────────────────────────────────────────→ (+) → h
#                                                       ↓
#   ┌──────────────────────────────────────────────────┘
#   ├→ LayerNorm → FeedForward → scale → (+)──────────┐
#   │                                                   ↓
#   └────────────────────────────────────────────────→ (+) → output
# ```
#
# **vs. POST-NORM (original Transformer)**:
# Post-norm: x → layer → norm → residual add
# Pre-norm: x → norm → layer → residual add
#
# Pre-norm advantages:
# - Better gradient flow
# - Training stability
# - Easier to scale to larger models
# ==============================================================================
class TransformerSelfAttentionLayer(nn.Module):
    """
    Single transformer layer with self-attention and feed-forward network.

    **WHAT MAKES THIS "SELF-ATTENTION"**:
    Q, K, V all come from the same input sequence:
    - Query: What am I looking for?
    - Key: What do I contain?
    - Value: What information do I have?

    Token attends to all other tokens in the sequence (including itself).

    **KEY COMPONENTS**:

    1. **Self-Attention Block**:
       - sa_norm: RMSNorm before attention
       - attn: MultiHeadAttention (Q, K, V projections + attention)
       - sa_scale: Optional scaling (for training stability)

    2. **Feed-Forward Block**:
       - mlp_norm: RMSNorm before FFN
       - mlp: FeedForward network (usually SwiGLU)
       - mlp_scale: Optional scaling

    3. **Residual Connections**:
       - First residual: after attention
       - Second residual: after FFN
       - Enables gradient flow through deep networks

    **MODERN IMPROVEMENTS** (vs. original Transformer):
    - RMSNorm instead of LayerNorm (simpler, faster)
    - Pre-norm instead of post-norm (more stable)
    - SwiGLU activation instead of ReLU (better performance)
    - Rotary embeddings (RoPE) instead of absolute (better extrapolation)
    - Grouped Query Attention (GQA) for efficiency

    Args:
        attn (MultiHeadAttention): Attention mechanism
        mlp (nn.Module): Feed-forward network
        sa_norm (Optional[nn.Module]): Pre-attention normalization
        mlp_norm (Optional[nn.Module]): Pre-FFN normalization
        sa_scale (Optional[nn.Module]): Post-attention scaling
        mlp_scale (Optional[nn.Module]): Post-FFN scaling
        mask_mod (Optional[Callable]): Mask modification function

    **USAGE IN MODEL BUILDING**:
    ```python
    # Create components
    attn = MultiHeadAttention(
        embed_dim=4096,
        num_heads=32,
        num_kv_heads=8,  # GQA
        ...
    )
    mlp = FeedForward(dim=4096, hidden_dim=14336)
    sa_norm = RMSNorm(dim=4096)
    mlp_norm = RMSNorm(dim=4096)

    # Compose into layer
    layer = TransformerSelfAttentionLayer(
        attn=attn,
        mlp=mlp,
        sa_norm=sa_norm,
        mlp_norm=mlp_norm
    )

    # Stack layers
    num_layers = 32  # For 8B model
    layers = nn.ModuleList([
        TransformerSelfAttentionLayer(...)
        for _ in range(num_layers)
    ])
    ```
    """

    def __init__(
        self,
        attn: MultiHeadAttention,           # Attention mechanism
        mlp: nn.Module,                     # Feed-forward network
        *,
        sa_norm: Optional[nn.Module] = None,      # Pre-attention norm
        mlp_norm: Optional[nn.Module] = None,     # Pre-FFN norm
        sa_scale: Optional[nn.Module] = None,     # Post-attention scale
        mlp_scale: Optional[nn.Module] = None,    # Post-FFN scale
        mask_mod: Optional[Callable[[_MaskType, int, int, int], _MaskType]] = None,
    ) -> None:
        super().__init__()

        # ====================================================================
        # Store components
        # WHY: Composition over inheritance - layer is composed of modules
        # ====================================================================
        self.attn = attn
        self.mlp = mlp

        # ====================================================================
        # Normalization layers (default to Identity if not provided)
        # WHY: Allows layers without explicit norms (for experimentation)
        # ====================================================================
        self.sa_norm = sa_norm or nn.Identity()
        self.mlp_norm = mlp_norm or nn.Identity()

        # ====================================================================
        # Optional scaling layers
        # WHY: Can help with training stability for very deep networks
        # USAGE: Rare, mostly for research/experimentation
        # ====================================================================
        self.sa_scale = sa_scale or nn.Identity()
        self.mlp_scale = mlp_scale or nn.Identity()

        # ====================================================================
        # Optional mask modification
        # WHY: For advanced attention patterns (e.g., chunked attention)
        # ====================================================================
        self.mask_mod = mask_mod

    def setup_caches(
        self,
        batch_size: int,
        dtype: torch.dtype,
        *,
        encoder_max_seq_len: int,
        decoder_max_seq_len: int,
    ) -> None:
        """
        Setup KV-caches for inference.

        **WHAT IS KV-CACHING**:
        During autoregressive generation, we recompute the same K and V
        for all previous tokens at each step. KV-caching stores these
        to avoid redundant computation.

        Without caching:
        ```
        Step 1: Compute K,V for token 1
        Step 2: Compute K,V for tokens 1,2  ← Recomputes token 1!
        Step 3: Compute K,V for tokens 1,2,3  ← Recomputes tokens 1,2!
        ```

        With caching:
        ```
        Step 1: Compute K,V for token 1 → Store in cache
        Step 2: Retrieve token 1 from cache, compute token 2 → Store
        Step 3: Retrieve tokens 1,2 from cache, compute token 3 → Store
        ```

        **MEMORY TRADEOFF**:
        - Storage: batch_size × max_seq_len × num_kv_heads × head_dim × 2
        - Speedup: ~10-100× faster for long sequences

        Args:
            batch_size: Batch size for generation
            dtype: Data type for cache (usually same as model)
            encoder_max_seq_len: Ignored for self-attention
            decoder_max_seq_len: Maximum sequence length
        """
        self.attn.setup_cache(batch_size, dtype, max_seq_len=decoder_max_seq_len)

    def caches_are_setup(self) -> bool:
        """Check if KV-caches are initialized"""
        return self.attn.kv_cache is not None

    def caches_are_enabled(self) -> bool:
        """Check if KV-caches are currently active"""
        return self.attn.cache_enabled

    def reset_cache(self):
        """Reset cache to empty (but keep allocated tensors)"""
        self.attn.reset_cache()

    def forward(
        self,
        x: torch.Tensor,
        *,
        mask: Optional[_MaskType] = None,
        input_pos: Optional[torch.Tensor] = None,
        **kwargs: dict,
    ) -> torch.Tensor:
        """
        Forward pass through the transformer layer.

        **MATHEMATICAL FORMULATION**:

        Pre-norm transformer layer:
        ```
        h = x + sa_scale(Attention(sa_norm(x)))
        out = h + mlp_scale(MLP(mlp_norm(h)))
        ```

        Detailed:
        ```
        # Self-Attention Block
        x_norm = sa_norm(x)                    # RMSNorm
        attn_out = attn(x_norm, x_norm)        # Multi-head attention
        h = x + sa_scale(attn_out)             # Residual connection

        # Feed-Forward Block
        h_norm = mlp_norm(h)                   # RMSNorm
        mlp_out = mlp(h_norm)                  # SwiGLU FFN
        out = h + mlp_scale(mlp_out)           # Residual connection
        ```

        **WHY RESIDUAL CONNECTIONS**:
        Without residuals, gradients vanish in deep networks:
        - 32 layers: gradient ≈ 0.9^32 ≈ 0.03 (vanished!)

        With residuals, gradients flow directly:
        - Gradient has direct path: ∂out/∂x = 1 + ∂layer/∂x
        - Even if ∂layer/∂x ≈ 0, gradient still ≈ 1

        Args:
            x: Input tensor [batch_size, seq_len, embed_dim]
            mask: Attention mask (optional)
            input_pos: Position IDs for RoPE and caching
            **kwargs: Additional arguments (for cross-attention layers)

        Returns:
            Output tensor [batch_size, seq_len, embed_dim]

        **SHAPE FLOW**:
        ```
        x:           [batch_size, seq_len, embed_dim]
            ↓ sa_norm
        x_norm:      [batch_size, seq_len, embed_dim]
            ↓ attn (Q,K,V projections + attention + output proj)
        attn_out:    [batch_size, seq_len, embed_dim]
            ↓ sa_scale + residual
        h:           [batch_size, seq_len, embed_dim]
            ↓ mlp_norm
        h_norm:      [batch_size, seq_len, embed_dim]
            ↓ mlp (w1, w2, w3 projections + SwiGLU)
        mlp_out:     [batch_size, seq_len, embed_dim]
            ↓ mlp_scale + residual
        out:         [batch_size, seq_len, embed_dim]
        ```
        """
        # ====================================================================
        # SELF-ATTENTION BLOCK
        # ====================================================================
        # Step 1: Normalize input
        h = self.sa_norm(x)

        # Step 2: Optional mask modification (for advanced patterns)
        if self.mask_mod is not None:
            bsz, seq_len, *_ = h.shape
            mask = self.mask_mod(mask=mask, bsz=bsz, seq_len=seq_len)

        # Step 3: Self-attention (Q, K, V all from same input)
        # Note: attn(h, h) means Q=K=V=h
        attn_out = self.attn(h, h, mask=mask, input_pos=input_pos)

        # Step 4: Scale and add residual
        # WHY: Residual connection allows gradients to flow through
        h = self.sa_scale(attn_out) + x

        # ====================================================================
        # FEED-FORWARD BLOCK
        # ====================================================================
        # Step 5: Normalize
        mlp_out = self.mlp(self.mlp_norm(h))

        # Step 6: Scale and add residual
        out = h + self.mlp_scale(mlp_out)

        return out


# ==============================================================================
# FULL MODEL: TransformerDecoder
# ==============================================================================
# **WHAT**: Complete transformer model - stack of layers + embeddings + output
# **WHY**: Provides the full LLM architecture
# **HOW**: Composes layers, embeddings, and output projection
#
# **ARCHITECTURE**:
# ```
# Input token IDs [batch_size, seq_len]
#   ↓
# tok_embeddings [batch_size, seq_len, embed_dim]
#   ↓
# Layer 0 (TransformerSelfAttentionLayer)
#   ↓
# Layer 1 (TransformerSelfAttentionLayer)
#   ↓
# ...
#   ↓
# Layer N-1 (TransformerSelfAttentionLayer)
#   ↓
# norm (final RMSNorm)
#   ↓
# output (Linear: embed_dim → vocab_size)
#   ↓
# Logits [batch_size, seq_len, vocab_size]
# ```
# ==============================================================================
class TransformerDecoder(nn.Module):
    """
    Full transformer decoder model.

    **WHAT**: Complete LLM architecture
    - Token embeddings (vocab → vectors)
    - Stack of transformer layers
    - Final normalization
    - Output projection (vectors → logits)

    **KEY FEATURES**:
    1. Flexible layer composition (can mix different layer types)
    2. KV-cache support for fast inference
    3. Multiple output modes (logits, hidden states, etc.)
    4. Support for very long contexts (up to 128K tokens)

    **MODEL SIZES** (by layer count):
    - Llama 3.1 8B: 32 layers
    - Llama 3.1 70B: 80 layers
    - Llama 3.1 405B: 126 layers

    Args:
        tok_embeddings: Token embedding layer
        layers: List/ModuleList of transformer layers
        max_seq_len: Maximum sequence length
        num_heads: Number of attention heads (for cache setup)
        head_dim: Dimension per head (for cache setup)
        norm: Final normalization layer
        output: Output projection (embed_dim → vocab_size)
        num_layers: Number of layers (if layers is a single module to clone)
        output_hidden_states: Which layer outputs to return

    **USAGE**:
    ```python
    # Create model
    model = TransformerDecoder(
        tok_embeddings=nn.Embedding(128256, 4096),
        layers=nn.ModuleList([
            TransformerSelfAttentionLayer(...)
            for _ in range(32)
        ]),
        max_seq_len=131072,
        num_heads=32,
        head_dim=128,
        norm=RMSNorm(4096),
        output=nn.Linear(4096, 128256)
    )

    # Forward pass
    logits = model(tokens)  # [batch_size, seq_len, vocab_size]

    # Generate
    model.setup_caches(batch_size=1, dtype=torch.bfloat16)
    for i in range(max_new_tokens):
        logits = model(next_token, input_pos=current_pos)
        next_token = logits.argmax(dim=-1)
    ```
    """

    def __init__(
        self,
        *,
        tok_embeddings: nn.Embedding,
        layers: nn.ModuleList,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        norm: nn.Module,
        output: nn.Module,
        num_layers: Optional[int] = None,
        output_hidden_states: Optional[list[int]] = None,
    ) -> None:
        super().__init__()

        # Store components
        self.tok_embeddings = tok_embeddings
        self.layers = layers
        self.norm = norm
        self.output = output
        self.output_hidden_states = output_hidden_states or []
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Flags for various modes
        self.skip_output_layer = False  # For memory-efficient loss computation

    def forward(
        self,
        tokens: Optional[torch.Tensor],
        *,
        mask: Optional[_MaskType] = None,
        input_pos: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass through the complete model.

        **COMPUTATION FLOW**:
        ```
        tokens: [2, 10] (batch_size=2, seq_len=10)
            ↓ tok_embeddings
        h: [2, 10, 4096] (embed_dim=4096)
            ↓ layers[0]
        h: [2, 10, 4096]
            ↓ layers[1]
        ...
            ↓ layers[31]
        h: [2, 10, 4096]
            ↓ norm
        h: [2, 10, 4096]
            ↓ output
        logits: [2, 10, 128256] (vocab_size=128256)
        ```

        **MEMORY USAGE**:
        For batch_size=2, seq_len=10, embed_dim=4096, vocab_size=128256:
        - Embeddings: 2 × 10 × 4096 × 2 bytes ≈ 160 KB
        - Hidden states (32 layers): 32 × 160 KB ≈ 5 MB
        - Output logits: 2 × 10 × 128256 × 4 bytes ≈ 10 MB
        - Total activations: ~15 MB (tiny for modern GPUs!)

        Most memory goes to:
        - Model weights: ~16 GB (for 8B model in bf16)
        - Optimizer states: ~32 GB (Adam with 2 states per param)
        - Gradients: ~16 GB

        Args:
            tokens: Input token IDs [batch_size, seq_len]
            mask: Attention mask (optional)
            input_pos: Position IDs for RoPE/caching (optional)

        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        # Embed tokens → continuous vectors
        h = self.tok_embeddings(tokens)  # [B, S] → [B, S, D]

        # Pass through each transformer layer
        hidden = []
        for i, layer in enumerate(self.layers):
            # Optionally collect hidden states
            if i in self.output_hidden_states:
                hidden.append(h)

            # Transform through layer
            h = layer(h, mask=mask, input_pos=input_pos, **kwargs)

        # Final layer hidden state
        if len(self.layers) in self.output_hidden_states:
            hidden.append(h)

        # Final normalization + output projection
        h = self.norm(h)
        if not self.skip_output_layer:
            output = self.output(h).float()  # [B, S, D] → [B, S, V]
        else:
            output = h

        # Return output (and hidden states if requested)
        return output if not hidden else [*hidden, output]


# ==============================================================================
# HOW TRANSFORMER LAYERS ACHIEVE MODULARITY
# ==============================================================================
"""
**COMPOSITION PATTERN IN ACTION**:

Instead of monolithic classes, torchtune uses composition:

```python
# BAD (monolithic):
class TransformerLayer(nn.Module):
    def __init__(self):
        # Hardcoded attention implementation
        self.q_proj = nn.Linear(...)
        self.k_proj = nn.Linear(...)
        # Hardcoded FFN implementation
        self.w1 = nn.Linear(...)
        # Can't easily swap components!
```

```python
# GOOD (compositional):
class TransformerSelfAttentionLayer(nn.Module):
    def __init__(self, attn, mlp, sa_norm, mlp_norm):
        self.attn = attn  # Any attention implementation!
        self.mlp = mlp    # Any FFN implementation!
        self.sa_norm = sa_norm  # Any norm implementation!
        self.mlp_norm = mlp_norm
```

**BENEFITS**:

1. **Flexible Attention**:
   ```python
   # Use standard MHA
   layer = TransformerSelfAttentionLayer(
       attn=MultiHeadAttention(...),
       mlp=FeedForward(...)
   )

   # Or use flash attention
   layer = TransformerSelfAttentionLayer(
       attn=FlashMultiHeadAttention(...),
       mlp=FeedForward(...)
   )

   # Or use custom attention
   layer = TransformerSelfAttentionLayer(
       attn=MyCustomAttention(...),
       mlp=FeedForward(...)
   )
   ```

2. **Flexible FFN**:
   ```python
   # Use SwiGLU (Llama-style)
   layer = TransformerSelfAttentionLayer(
       attn=attention,
       mlp=SwiGLU(...)
   )

   # Or use GELU (BERT-style)
   layer = TransformerSelfAttentionLayer(
       attn=attention,
       mlp=GELUFeedForward(...)
   )
   ```

3. **Easy LoRA Integration**:
   ```python
   # Wrap attention projections with LoRA
   attn = MultiHeadAttention(
       q_proj=LoRALinear(...),  # LoRA-enhanced!
       k_proj=nn.Linear(...),
       v_proj=LoRALinear(...),  # LoRA-enhanced!
       ...
   )

   # Layer doesn't know or care
   layer = TransformerSelfAttentionLayer(
       attn=attn,  # Works seamlessly
       mlp=mlp
   )
   ```

4. **Mix and Match Layers**:
   ```python
   # Vision-language model with different layer types
   layers = nn.ModuleList([
       # Vision encoder layers
       *[TransformerSelfAttentionLayer(...) for _ in range(24)],
       # Fusion layers (cross-attention)
       *[TransformerCrossAttentionLayer(...) for _ in range(4)],
       # Text decoder layers
       *[TransformerSelfAttentionLayer(...) for _ in range(32)]
   ])

   model = TransformerDecoder(layers=layers, ...)
   ```

**KEY INSIGHT**: Composition allows mixing and matching components without
changing the core layer or model code. This is the essence of modularity!
"""
