# ==============================================================================
# ANNOTATED: torchtune Attention Module - The Heart of Transformers
# ==============================================================================
# Source: torchtune/modules/attention.py
#
# **WHAT**: Multi-head attention implementation with support for Grouped Query
#           Attention (GQA), KV-caching, and rotary positional embeddings (RoPE).
#           This is THE core mechanism that makes transformers work.
#
# **WHY**:  Attention is the breakthrough that enabled modern LLMs:
#           - Allows model to focus on relevant parts of input
#           - Enables long-range dependencies
#           - Parallelizable (unlike RNNs)
#           - Scales to very long sequences
#
# **HOW**:  Query-Key-Value mechanism:
#
#           For each query token:
#           1. Compute attention scores with all key tokens
#           2. Softmax scores → attention weights
#           3. Weighted sum of value tokens → output
#
#           Mathematically: Attention(Q, K, V) = softmax(QK^T / √d) V
#
# **KEY DESIGN PATTERNS**:
#   1. **Multi-Head Attention**: Multiple parallel attention computations
#   2. **Grouped Query Attention (GQA)**: Fewer KV heads than Q heads
#   3. **KV-Caching**: Store past keys/values for fast generation
#   4. **Rotary Embeddings**: Position info encoded in attention computation
# ==============================================================================

import torch
from torch import nn
from typing import Optional

# Simplified imports (in real code, these come from torchtune.modules)
from torchtune.modules.kv_cache import KVCache
from torchtune.modules.attention_utils import _MaskType


# ==============================================================================
# CORE CLASS: MultiHeadAttention
# ==============================================================================
# **WHAT**: Implements multi-head attention with GQA support
#
# **ATTENTION EVOLUTION**:
#
# 1. **Multi-Head Attention (MHA)** - Original Transformer
#    - Each head has its own Q, K, V
#    - num_heads = num_kv_heads (e.g., 32 = 32)
#    - Memory: High (all heads independent)
#
# 2. **Multi-Query Attention (MQA)** - Extreme sharing
#    - All heads share single K, V
#    - num_kv_heads = 1
#    - Memory: Very low, but quality loss
#
# 3. **Grouped Query Attention (GQA)** - Llama 3/3.1
#    - Groups of Q heads share K, V
#    - num_kv_heads = num_heads // group_size
#    - Sweet spot: Good quality + memory savings
#
# **GQA EXAMPLE** (num_heads=32, num_kv_heads=8):
# ```
# Q heads: 32 (organized in 8 groups of 4)
# K heads: 8 (shared across groups)
# V heads: 8 (shared across groups)
#
# Group 0: Q heads 0-3 → K head 0, V head 0
# Group 1: Q heads 4-7 → K head 1, V head 1
# ...
# Group 7: Q heads 28-31 → K head 7, V head 7
# ```
#
# **MEMORY SAVINGS**:
# For Llama 3.1 8B (32 Q heads, 8 KV heads):
# - MHA: 32 × (K + V) = 64 tensors
# - GQA: 32 Q + 8 K + 8 V = 48 tensors (25% savings!)
# - MQA: 32 Q + 1 K + 1 V = 34 tensors (but worse quality)
# ==============================================================================
class MultiHeadAttention(nn.Module):
    """
    Multi-headed attention with Grouped Query Attention (GQA) support.

    **MATHEMATICAL FORMULATION**:

    For each attention head:
    1. Project input to Q, K, V:
       ```
       Q = x @ W_q    # [B, S, D] @ [D, H×D_h] → [B, S, H×D_h]
       K = x @ W_k    # [B, S, D] @ [D, KV×D_h] → [B, S, KV×D_h]
       V = x @ W_v    # [B, S, D] @ [D, KV×D_h] → [B, S, KV×D_h]
       ```

    2. Reshape to separate heads:
       ```
       Q = Q.view(B, S, H, D_h).transpose(1, 2)   # [B, H, S, D_h]
       K = K.view(B, S, KV, D_h).transpose(1, 2)  # [B, KV, S, D_h]
       V = V.view(B, S, KV, D_h).transpose(1, 2)  # [B, KV, S, D_h]
       ```

    3. Apply RoPE (rotary embeddings):
       ```
       Q, K = apply_rotary_emb(Q, K, position_ids)
       ```

    4. Expand KV heads to match Q heads (for GQA):
       ```
       # If num_heads=32, num_kv_heads=8, expand 8 → 32
       K = K.repeat_interleave(4, dim=1)  # [B, 32, S, D_h]
       V = V.repeat_interleave(4, dim=1)  # [B, 32, S, D_h]
       ```

    5. Compute attention scores:
       ```
       scores = Q @ K^T / √d_h   # [B, H, S_q, S_k]
       ```

    6. Apply mask + softmax:
       ```
       scores = scores.masked_fill(mask == 0, -inf)
       attn_weights = softmax(scores, dim=-1)
       ```

    7. Weighted sum of values:
       ```
       output = attn_weights @ V  # [B, H, S_q, D_h]
       ```

    8. Concatenate heads + output projection:
       ```
       output = output.transpose(1, 2).reshape(B, S_q, H×D_h)
       output = output @ W_o
       ```

    **VISUAL EXAMPLE** (1 head, S=4):
    ```
    Query:  [q1, q2, q3, q4]
    Key:    [k1, k2, k3, k4]
    Value:  [v1, v2, v3, v4]

    Scores = Q @ K^T / √d:
            k1   k2   k3   k4
       q1  [0.8, 0.1, 0.05, 0.05]  ← q1 attends mostly to k1
       q2  [0.1, 0.7, 0.15, 0.05]  ← q2 attends mostly to k2
       q3  [0.1, 0.2, 0.6,  0.1 ]  ← q3 attends mostly to k3
       q4  [0.05,0.1, 0.25, 0.6 ]  ← q4 attends to k3, k4

    Output = Scores @ V:
       out1 = 0.8×v1 + 0.1×v2 + 0.05×v3 + 0.05×v4
       out2 = 0.1×v1 + 0.7×v2 + 0.15×v3 + 0.05×v4
       ...
    ```

    **WHY THIS WORKS**:
    - High attention scores → token is relevant
    - Low attention scores → token is not relevant
    - Output is context-aware mixture of all tokens
    - Different heads learn different patterns

    Args:
        embed_dim (int): Model dimension (e.g., 4096)
        num_heads (int): Number of query heads (e.g., 32)
        num_kv_heads (int): Number of key/value heads (e.g., 8 for GQA)
        head_dim (int): Dimension per head (e.g., 128)
        q_proj (nn.Module): Query projection (Linear or LoRALinear)
        k_proj (nn.Module): Key projection
        v_proj (nn.Module): Value projection
        output_proj (nn.Module): Output projection
        pos_embeddings (Optional[nn.Module]): RoPE embeddings
        q_norm (Optional[nn.Module]): Query normalization
        k_norm (Optional[nn.Module]): Key normalization
        kv_cache (Optional[KVCache]): Cache for fast generation
        max_seq_len (int): Maximum sequence length
        is_causal (bool): Whether to use causal masking
        attn_dropout (float): Dropout probability for attention weights

    **USAGE**:
    ```python
    # Standard MHA (32 heads)
    attn = MultiHeadAttention(
        embed_dim=4096,
        num_heads=32,
        num_kv_heads=32,  # Same as num_heads → MHA
        head_dim=128,
        q_proj=nn.Linear(4096, 32*128),
        k_proj=nn.Linear(4096, 32*128),
        v_proj=nn.Linear(4096, 32*128),
        output_proj=nn.Linear(32*128, 4096)
    )

    # GQA (32 query heads, 8 kv heads)
    attn = MultiHeadAttention(
        embed_dim=4096,
        num_heads=32,
        num_kv_heads=8,   # Fewer KV heads → GQA
        head_dim=128,
        q_proj=nn.Linear(4096, 32*128),
        k_proj=nn.Linear(4096, 8*128),  # Smaller!
        v_proj=nn.Linear(4096, 8*128),  # Smaller!
        output_proj=nn.Linear(32*128, 4096)
    )

    # With LoRA
    attn = MultiHeadAttention(
        embed_dim=4096,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        q_proj=LoRALinear(4096, 32*128, rank=8),  # LoRA on Q!
        k_proj=nn.Linear(4096, 8*128),
        v_proj=LoRALinear(4096, 8*128, rank=8),   # LoRA on V!
        output_proj=LoRALinear(32*128, 4096, rank=8)  # LoRA on output!
    )
    ```
    """

    def __init__(
        self,
        *,
        embed_dim: int,              # Model dimension (D)
        num_heads: int,              # Number of query heads (H)
        num_kv_heads: int,           # Number of KV heads (KV)
        head_dim: int,               # Dimension per head (D_h)
        q_proj: nn.Module,           # Query projection
        k_proj: nn.Module,           # Key projection
        v_proj: nn.Module,           # Value projection
        output_proj: nn.Module,      # Output projection
        pos_embeddings: Optional[nn.Module] = None,  # RoPE
        q_norm: Optional[nn.Module] = None,  # Query norm
        k_norm: Optional[nn.Module] = None,  # Key norm
        kv_cache: Optional[KVCache] = None,  # For caching
        max_seq_len: int = 4096,
        is_causal: bool = True,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # ====================================================================
        # VALIDATION: Ensure configuration is valid
        # ====================================================================
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by "
                f"num_kv_heads ({num_kv_heads})"
            )

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        if attn_dropout < 0 or attn_dropout > 1:
            raise ValueError(f"attn_dropout must be between 0.0 and 1.0")

        if bool(q_norm) ^ bool(k_norm):
            raise ValueError("q_norm and k_norm must both be set or both be None")

        # ====================================================================
        # STORE CONFIGURATION
        # ====================================================================
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.embed_dim = embed_dim
        self.attn_dropout = attn_dropout
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.is_causal = is_causal

        # ====================================================================
        # STORE PROJECTION LAYERS
        # WHY: These can be nn.Linear or LoRALinear - module doesn't care!
        # ====================================================================
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.output_proj = output_proj

        # ====================================================================
        # OPTIONAL COMPONENTS
        # ====================================================================
        self.q_norm = q_norm  # For training stability
        self.k_norm = k_norm
        self.pos_embeddings = pos_embeddings  # RoPE
        self.kv_cache = kv_cache  # For fast generation

        # Cache control flag
        self.cache_enabled = False

    def setup_cache(
        self, batch_size: int, dtype: torch.dtype, max_seq_len: int
    ) -> None:
        """
        Setup KV-cache for fast autoregressive generation.

        **WHY KV-CACHING IS CRITICAL**:

        Without caching (generating 100 tokens):
        - Step 1: Compute K,V for 1 token
        - Step 2: Compute K,V for 2 tokens (1 redundant)
        - Step 3: Compute K,V for 3 tokens (2 redundant)
        - ...
        - Step 100: Compute K,V for 100 tokens (99 redundant!)
        Total: 1 + 2 + 3 + ... + 100 = 5,050 computations

        With caching:
        - Step 1: Compute K,V for 1 token, cache it
        - Step 2: Reuse cached token 1, compute token 2
        - Step 3: Reuse cached tokens 1-2, compute token 3
        - ...
        - Step 100: Reuse cached tokens 1-99, compute token 100
        Total: 100 computations (50× faster!)

        **MEMORY COST**:
        Cache size = batch_size × max_seq_len × num_kv_heads × head_dim × 2
        Example: 1 × 2048 × 8 × 128 × 2 bytes = 4 MB (tiny!)

        Args:
            batch_size: Batch size for generation
            dtype: Cache dtype (usually bf16 or fp16)
            max_seq_len: Maximum sequence length to cache
        """
        if self.kv_cache is not None:
            # Already set up, skip
            return

        self.kv_cache = KVCache(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            dtype=dtype,
        )
        self.cache_enabled = True

    def reset_cache(self):
        """Reset cache to empty (keep allocated tensors)"""
        if self.kv_cache is None:
            raise RuntimeError("Cache not set up. Call setup_cache() first.")
        self.kv_cache.reset()

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        *,
        mask: Optional[_MaskType] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through multi-head attention.

        **TWO MODES**:

        1. **Self-Attention** (y=None or y=x):
           - Q, K, V all from same input
           - Used in decoder-only models (Llama, GPT)
           - Each token attends to all tokens

        2. **Cross-Attention** (y!=None and y!=x):
           - Q from x, K and V from y
           - Used in encoder-decoder models
           - Decoder tokens attend to encoder tokens

        **COMPUTATION FLOW**:
        ```
        Input x: [B, S, D]
            ↓
        Project to Q, K, V
            ↓
        Q: [B, S, H, D_h]
        K: [B, S, KV, D_h]
        V: [B, S, KV, D_h]
            ↓
        Apply RoPE (rotary embeddings)
            ↓
        Reshape to [B, H, S, D_h]
            ↓
        Optional: Update KV cache
            ↓
        Expand KV heads (if GQA)
            ↓
        K, V: [B, H, S, D_h]
            ↓
        Compute attention
            ↓
        scores = Q @ K^T / √D_h: [B, H, S_q, S_k]
        attn_weights = softmax(scores): [B, H, S_q, S_k]
        output = attn_weights @ V: [B, H, S_q, D_h]
            ↓
        Reshape: [B, S_q, H×D_h]
            ↓
        Output projection
            ↓
        Output: [B, S_q, D]
        ```

        Args:
            x: Query input [batch_size, seq_len, embed_dim]
            y: Key/Value input (optional, defaults to x)
            mask: Attention mask (optional)
            input_pos: Position IDs for RoPE and caching

        Returns:
            Output tensor [batch_size, seq_len, embed_dim]

        **EXAMPLE**:
        ```python
        # Self-attention
        output = attn(x)  # Q=K=V=x

        # Cross-attention
        output = attn(decoder_hidden, encoder_hidden)  # Q=decoder, K=V=encoder

        # With caching (generation)
        attn.setup_cache(batch_size=1, dtype=torch.bfloat16)
        for i in range(100):
            output = attn(next_token, input_pos=torch.tensor([i]))
        ```
        """
        # ====================================================================
        # STEP 1: Get dimensions
        # ====================================================================
        b, s_x, _ = x.shape  # batch, seq_len for query
        s_y = y.shape[1] if y is not None else 0  # seq_len for key/value

        # ====================================================================
        # STEP 2: Project to Q, K, V
        # ====================================================================
        # Query projection
        q = self.q_proj(x)  # [B, S, H×D_h]

        # Reshape to separate heads
        q_per_kv = self.num_heads // self.num_kv_heads  # Heads per KV head
        q = q.view(b, s_x, self.num_kv_heads * q_per_kv, self.head_dim)

        # ====================================================================
        # STEP 3: Apply positional embeddings (RoPE)
        # ====================================================================
        # RoPE encodes position info into Q and K
        # This allows attention to be position-aware without explicit position embeddings
        if self.pos_embeddings is not None:
            q = self.pos_embeddings(q, input_pos=input_pos)

        # ====================================================================
        # STEP 4: Transpose to [B, H, S, D_h] format for attention
        # ====================================================================
        q = q.transpose(1, 2)  # [B, H, S, D_h]

        # ====================================================================
        # STEP 5: Normalize Q (optional, for training stability)
        # ====================================================================
        if self.q_norm is not None:
            q = self.q_norm(q)

        # ====================================================================
        # STEP 6: Handle K and V
        # ====================================================================
        if y is None:
            # Self-attention mode: Use cached K, V (during generation)
            if self.kv_cache is None or not self.cache_enabled:
                raise ValueError("Must provide y or enable kv_cache")
            k = self.kv_cache.k_cache
            v = self.kv_cache.v_cache
        else:
            # Normal mode: Compute K, V from y
            k = self.k_proj(y)  # [B, S, KV×D_h]
            v = self.v_proj(y)

            # Reshape
            k = k.view(b, s_y, -1, self.head_dim)
            v = v.view(b, s_y, -1, self.head_dim)

            # Apply RoPE to K
            if self.pos_embeddings is not None:
                k = self.pos_embeddings(k, input_pos=input_pos)

            # Transpose
            k = k.transpose(1, 2)  # [B, KV, S, D_h]
            v = v.transpose(1, 2)

            # Normalize K (optional)
            if self.k_norm is not None:
                k = self.k_norm(k)

            # Update cache if enabled
            if self.kv_cache is not None and self.cache_enabled:
                k, v = self.kv_cache.update(k, v)

        # ====================================================================
        # STEP 7: Expand KV heads to match Q heads (for GQA)
        # ====================================================================
        # If num_heads=32 and num_kv_heads=8, expand 8 → 32
        # Each KV head is shared by 4 Q heads
        if self.num_heads != self.num_kv_heads:
            # Expand shape: [B, KV, S, D_h] → [B, KV, q_per_kv, S, D_h]
            expand_shape = (b, self.num_kv_heads, q_per_kv, -1, self.head_dim)
            k = k.unsqueeze(2).expand(expand_shape).flatten(1, 2)
            v = v.unsqueeze(2).expand(expand_shape).flatten(1, 2)
            # Result: [B, H, S, D_h]

        # ====================================================================
        # STEP 8: Compute attention
        # ====================================================================
        # Use PyTorch's scaled_dot_product_attention (fused, optimized)
        # Automatically handles:
        # - Scaling by 1/√d_h
        # - Masking (causal or custom)
        # - Softmax
        # - Dropout
        output = torch.nn.functional.scaled_dot_product_attention(
            q,  # [B, H, S_q, D_h]
            k,  # [B, H, S_k, D_h]
            v,  # [B, H, S_k, D_h]
            attn_mask=mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=self.kv_cache is None and mask is None and self.is_causal,
        )
        # Output: [B, H, S_q, D_h]

        # ====================================================================
        # STEP 9: Reshape and project output
        # ====================================================================
        # Concatenate heads: [B, H, S, D_h] → [B, S, H×D_h]
        output = output.transpose(1, 2).contiguous().view(b, s_x, -1)

        # Final output projection
        return self.output_proj(output)  # [B, S, H×D_h] → [B, S, D]


# ==============================================================================
# HOW ATTENTION ACHIEVES MODULARITY
# ==============================================================================
"""
**COMPOSITION + DEPENDENCY INJECTION**:

Attention module doesn't hardcode projections:
```python
# BAD (hardcoded):
class Attention(nn.Module):
    def __init__(self, dim):
        self.q_proj = nn.Linear(dim, dim)  # Hardcoded!
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
```

```python
# GOOD (injectable):
class MultiHeadAttention(nn.Module):
    def __init__(self, q_proj, k_proj, v_proj, output_proj):
        self.q_proj = q_proj  # Any projection! Linear or LoRALinear
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.output_proj = output_proj
```

**BENEFITS**:

1. **Easy LoRA Integration**:
   ```python
   # Without changing attention code!
   attn = MultiHeadAttention(
       q_proj=LoRALinear(4096, 4096, rank=8),
       k_proj=nn.Linear(4096, 4096),
       v_proj=LoRALinear(4096, 4096, rank=8),
       output_proj=LoRALinear(4096, 4096, rank=8)
   )
   ```

2. **Different Projection Types**:
   ```python
   # Quantized projections
   attn = MultiHeadAttention(
       q_proj=Int8Linear(4096, 4096),
       ...
   )

   # Custom projections
   attn = MultiHeadAttention(
       q_proj=MyCustomProjection(4096, 4096),
       ...
   )
   ```

3. **Flexible Configuration**:
   - MHA: num_kv_heads = num_heads
   - GQA: num_kv_heads < num_heads
   - MQA: num_kv_heads = 1
   All with same class!

4. **Optional Components**:
   - RoPE: pos_embeddings parameter
   - QK Normalization: q_norm, k_norm parameters
   - KV-Cache: kv_cache parameter
   Everything optional, compose as needed!

**KEY INSIGHT**: By accepting modules as parameters rather than creating them
internally, MultiHeadAttention becomes highly flexible and composable. This
is dependency injection in action!
"""
