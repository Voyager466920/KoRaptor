# raptor_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from typing import Tuple

from huggingface_hub import PyTorchModelHubMixin
from Raptor_config import RaptorConfig
from Models.MultiheadLatentAttention import MultiHeadLatentAttention


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


class LatentMoE(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config: RaptorConfig):
        super().__init__()
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
