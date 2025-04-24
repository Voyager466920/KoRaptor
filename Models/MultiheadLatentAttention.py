import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Rotary helpers
# -----------------------------

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Helper used by RoPE – rotate every pair (x1,x2) → (-x2,x1)."""
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)

def apply_rope(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)

# -----------------------------
# Multi‑Head Latent Attention
# DeepSeek‑style implementation
# -----------------------------

class MultiHeadLatentAttention(nn.Module):
    """Multi‑Head Latent Attention (MLA)

    • Projects the input to Q and to *latent* D_K / D_V (lower dimensional).
    • Caches the latents; during forward each step restores full‑dimensional
      K,V through *separate* up‑projections (W_UK, W_UV).
    • Uses Rotary Position Embedding (RoPE).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        latent_dim: int,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.latent_dim = latent_dim
        self.scale = self.head_dim ** -0.5

        # Q projection (same dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        # Shared projection to *latent* D_K | D_V (2 * latent_dim)
        self.dkv_proj = nn.Linear(dim, 2 * latent_dim, bias=False)

        # --- DeepSeek difference: **separate** up‑projections
        self.up_proj_k = nn.Linear(latent_dim, dim, bias=False)
        self.up_proj_v = nn.Linear(latent_dim, dim, bias=False)

        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Pre‑compute inverse frequencies for RoPE
        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    # ---------------------------------------------------------
    # Rotary cache (sin,cos)
    # ---------------------------------------------------------
    def _build_rope_cache(self, seq_len: int, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # [seq_len, head_dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # duplicate to full head_dim
        return emb.sin().to(dtype), emb.cos().to(dtype)

    # ---------------------------------------------------------
    # Forward
    # ---------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        kv_cache: tuple | None = None,
        *,
        attn_mask: torch.Tensor | None = None,
        use_cache: bool = False,
    ):
        """Args:
            x: [B, S, dim]
            kv_cache: optional tuple(K,V) with shapes [B, H, S_cached, Hd]
            attn_mask: [B|1, 1, S, S_total]
            use_cache: if True returns new_cache (K,V) for incremental decoding
        Returns:
            out: [B, S, dim]
            (optional) new_cache: tuple(K,V)
        """
        b, s, _ = x.size()

        # -----------------------------
        # 1. Project to Q, D_K, D_V
        # -----------------------------
        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        dkv = (
            self.dkv_proj(x)
            .view(b, s, 2, self.latent_dim)
            .permute(2, 0, 1, 3)
        )  # [2, B, S, latent_dim]
        d_k, d_v = dkv[0], dkv[1]

        # -----------------------------
        # 2. Restore full‑dimensional K,V
        # -----------------------------
        k = (
            self.up_proj_k(d_k)
            .view(b, s, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )  # [B, H, S, Hd]
        v = (
            self.up_proj_v(d_v)
            .view(b, s, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # -----------------------------
        # 3. RoPE positional encoding
        # -----------------------------
        sin, cos = self._build_rope_cache(s, x.device, x.dtype)
        q = apply_rope(q, sin, cos)
        k = apply_rope(k, sin, cos)

        # -----------------------------
        # 4. Append KV‑cache
        # -----------------------------
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)

        # -----------------------------
        # 5. Scaled dot‑product attention
        # -----------------------------
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, S, S_total]
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, -1e9)

        attn_prob = F.softmax(attn_scores, dim=-1)
        attn_prob = self.dropout(attn_prob)

        out = attn_prob @ v  # [B,H,S,Hd]
        out = out.transpose(1, 2).contiguous().view(b, s, self.dim)
        out = self.out_proj(out)

        # -----------------------------
        new_cache = (k, v) if use_cache else None
        return (out, new_cache) if use_cache else (out, None)

class MLPBlock(nn.Module):
    def __init__(self,
                 embedding_dim: int = 768,
                 mlp_size: int = 3072,
                 dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim,
                      out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size,
                      out_features=embedding_dim),
            nn.Dropout(p=dropout))

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x

