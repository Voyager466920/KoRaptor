import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_half(x):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope(x, sin, cos):
    return (x * cos) + (rotate_half(x) * sin)


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, dim, num_heads, latent_dim, rope_theta=10000.0, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.latent_dim = latent_dim
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.dkv_proj = nn.Linear(dim, 2 * latent_dim, bias=False)
        self.up_proj_k = nn.Linear(latent_dim, dim, bias=False)
        self.up_proj_v = nn.Linear(latent_dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._rope_cache = None

    def _build_rope_cache(self, seq_len, device, dtype):
        if (
            self._rope_cache is None
            or self._rope_cache[0].size(0) < seq_len
            or self._rope_cache[0].dtype != dtype
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._rope_cache = (emb.sin().to(dtype), emb.cos().to(dtype))
        return self._rope_cache

    def forward(self, x, kv_cache=None, *, attn_mask=None, use_cache=False):
        b, s, _ = x.size()
        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        d_k, d_v = self.dkv_proj(x).view(b, s, 2, self.latent_dim).permute(2, 0, 1, 3)
        k = self.up_proj_k(d_k).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.up_proj_v(d_v).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        sin, cos = self._build_rope_cache(k.size(-2), x.device, x.dtype)
        q = apply_rope(q, sin, cos)
        k = apply_rope(k, sin, cos)
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        out = F.scaled_dot_product_attention(q, k, v,attn_mask = attn_mask, dropout_p = self.dropout.p if self.training else 0.0,)
        out = out.transpose(1, 2).contiguous().view(b, s, self.dim)
        out = self.out_proj(out)
        if use_cache:
            return out, (k.detach(), v.detach())
        else:
            return out, None


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class MLPBlock(nn.Module):
    def __init__(self, embedding_dim, mlp_size, dropout=0.1):
        super().__init__()
        self.norm = RMSNorm(embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, mlp_size)
        self.fc2 = nn.Linear(mlp_size, embedding_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
