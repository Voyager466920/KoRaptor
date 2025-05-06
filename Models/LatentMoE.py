import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from typing import Tuple

from Models.MultiheadLatentAttention import MultiHeadLatentAttention

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

def subsequent_mask(seq_len: int, device=None) -> torch.Tensor:
    mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))
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

        # Expert MLP 모음
        self.experts = nn.ModuleList(
            [Expert(embed_dim, mlp_dim, dropout) for _ in range(num_experts)]
        )

        # 게이트 & 게이트 드롭아웃
        self.gate = nn.Linear(embed_dim, num_experts)
        self.gate_dropout = nn.Dropout(p=0.1)

        self.experts_per_token = experts_per_token
        self.num_experts = num_experts


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_norm = self.norm(x)
        gate_logits = self.gate_dropout(self.gate(x_norm))
        gate_probs  = F.softmax(gate_logits, dim=-1)
        topk_probs, topk_idx = torch.topk(
            gate_probs, self.experts_per_token, dim=-1
        )

        expert_outputs = torch.stack([e(x_norm) for e in self.experts], dim=2)
        combine = torch.zeros_like(gate_probs)
        combine.scatter_(2, topk_idx, topk_probs)
        output = torch.einsum("bse,bsed->bsd", combine, expert_outputs)
        importance = gate_probs.sum(dim=(0, 1))
        load       = combine.sum(dim=(0, 1))
        balance_loss = 0.5 * (
            torch.std(importance / importance.sum()) +
            torch.std(load / load.sum())
        )

        return output, balance_loss


class LatentGPTBlock(nn.Module):
    def __init__(self, embed_dim: int, latent_dim: int, mlp_dim: int, dropout: float = 0.1, num_heads: int = 16, num_experts: int = 6, experts_per_token: int = 2):
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
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        y, _ = self.attn(self.norm1(x), attn_mask=mask)
        x = x + y
        moe_output, balance_loss = self.moe(self.norm2(x))
        x = x + moe_output
        return x, balance_loss

class LatentMoE(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32768,
        embed_dim: int = 1024,
        latent_dim: int = 256,
        mlp_dim: int = 4096,
        num_layers: int = 16,
        dropout: float = 0.1,
        num_heads: int = 16,
        num_experts: int = 6,
        experts_per_token: int = 2,
        balance_loss_weight: float = 0.01
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            LatentGPTBlock(
                embed_dim,
                latent_dim,
                mlp_dim,
                dropout,
                num_heads,
                num_experts,
                experts_per_token
            )
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.balance_loss_weight = balance_loss_weight

    def forward(self, input_ids: torch.LongTensor) -> (torch.Tensor, torch.Tensor):
        b, s = input_ids.size()
        x = self.token_embedding(input_ids) + self.positional_embedding[:, :s]
        x = self.dropout(x)
        mask = subsequent_mask(s, device=input_ids.device)
        total_balance_loss = 0.0
        for blk in self.blocks:
            x, balance_loss = blk(x, mask)
            total_balance_loss += balance_loss
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, total_balance_loss / len(self.blocks)