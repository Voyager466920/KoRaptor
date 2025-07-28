# raptor_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from huggingface_hub import PyTorchModelHubMixin
from transformers import PreTrainedModel, GenerationMixin

from configuration_raptor import RaptorConfig


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


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(
            x.pow(2).mean(-1, keepdim=True) + self.eps
        )


def subsequent_mask(seq_len: int, device=None) -> torch.Tensor:
    mask = torch.tril(
        torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
    )
    return mask.unsqueeze(0).unsqueeze(1)


class Expert(nn.Module):
    def __init__(self, embed_dim: int, mlp_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(mlp_dim, embed_dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MoEBlock(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            mlp_dim: int,
            num_experts: int,
            experts_per_token: int,
            dropout: float = 0.1,
    ):
        super().__init__()
        self.norm = RMSNorm(embed_dim)
        self.experts = nn.ModuleList(
            [Expert(embed_dim, mlp_dim, dropout) for _ in range(num_experts)]
        )
        self.gate = nn.Linear(embed_dim, num_experts)
        self.gate_dropout = nn.Dropout(dropout)
        self.experts_per_token = experts_per_token
        self.num_experts = num_experts

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_norm = self.norm(x)
        if torch.is_autocast_enabled() and x_norm.dtype == torch.float32:
            x_norm = x_norm.to(torch.float16)

        logits = self.gate_dropout(self.gate(x_norm))
        gate_probs = F.softmax(logits, dim=-1)
        topk_p, topk_i = torch.topk(gate_probs, self.experts_per_token, dim=-1)

        B, S, D = x_norm.shape
        K = self.experts_per_token
        x_flat = x_norm.view(-1, D)
        topk_i_flat = topk_i.view(-1, K)
        topk_p_flat = topk_p.view(-1, K)
        out_flat = torch.zeros_like(x_flat)

        for eid, expert in enumerate(self.experts):
            mask = (topk_i_flat == eid)
            if not mask.any():
                continue
            rows = mask.any(dim=1).nonzero(as_tuple=False).squeeze(-1)
            inp = x_flat.index_select(0, rows)
            out = expert(inp)
            probs = (topk_p_flat[rows] * mask[rows].float()).sum(dim=1)
            out.mul_(probs.unsqueeze(-1))
            out_flat.index_add_(0, rows, out)

        out = out_flat.view(B, S, D)
        importance = gate_probs.sum(dim=(0, 1))
        load = torch.zeros(self.num_experts, device=x.device)
        load.scatter_add_(0, topk_i_flat.reshape(-1),
                          torch.ones_like(topk_i_flat.reshape(-1), dtype=x.dtype))
        balance_loss = 0.5 * (
                torch.std(importance / importance.sum()) +
                torch.std(load / load.sum())
        )

        return out, balance_loss


class LatentGPTBlock(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            latent_dim: int,
            mlp_dim: int,
            dropout: float = 0.1,
            num_heads: int = 16,
            num_experts: int = 6,
            experts_per_token: int = 2
    ):
        super().__init__()
        self.norm1 = RMSNorm(embed_dim)
        self.attn = MultiHeadLatentAttention(
            dim=embed_dim,
            num_heads=num_heads,
            latent_dim=latent_dim,
            dropout=dropout
        )
        self.norm2 = RMSNorm(embed_dim)
        self.moe = MoEBlock(embed_dim, mlp_dim, num_experts, experts_per_token, dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y, _ = self.attn(self.norm1(x), attn_mask=mask)
        x = x + y
        moe_out, loss = self.moe(self.norm2(x))
        x = x + moe_out
        return x, loss


class RaptorModel(PreTrainedModel, PyTorchModelHubMixin, GenerationMixin):
    config_class = RaptorConfig
    base_model_prefix = "raptor"

    def __init__(self, config: RaptorConfig):
        super().__init__(config)
        self.config = config
        c = config

        self.token_embedding = nn.Embedding(c.vocab_size, c.embed_dim)
        self.positional_embedding = nn.Parameter(
            torch.randn(1, c.max_seq_len, c.embed_dim)
        )
        self.dropout = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList([
            LatentGPTBlock(
                c.embed_dim,
                c.latent_dim,
                c.mlp_dim,
                c.dropout,
                c.num_heads,
                c.num_experts,
                c.experts_per_token
            )
            for _ in range(c.num_layers)
        ])
        self.norm = RMSNorm(c.embed_dim)
        self.lm_head = nn.Linear(c.embed_dim, c.vocab_size, bias=False)
        self.balance_loss_weight = c.balance_loss_weight

    def forward(self, input_ids: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, s = input_ids.size()
        x = self.token_embedding(input_ids) + self.positional_embedding[:, :s]
        x = self.dropout(x)
        mask = subsequent_mask(s, device=input_ids.device)

        total_loss = 0.0
        for blk in self.blocks:
            x, loss = blk(x, mask)
            total_loss += loss * self.balance_loss_weight

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, total_loss / len(self.blocks)

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        return {"input_ids": input_ids, "attention_mask": attention_mask}
