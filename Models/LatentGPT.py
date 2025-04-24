import torch
import torch.nn as nn
import torch.nn.functional as F

from MultiheadLatentAttention import MultiHeadLatentAttention

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

class MLPBlock(nn.Module):
    def __init__(self, embed_dim: int, mlp_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = RMSNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(mlp_dim, embed_dim)
        self.drop2 = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = self.fc1(y)
        y = self.act(y)
        y = self.drop1(y)
        y = self.fc2(y)
        y = self.drop2(y)
        return y

class LatentGPTBlock(nn.Module):
    def __init__(self, embed_dim: int, latent_dim: int, mlp_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = RMSNorm(embed_dim)
        self.attn = MultiHeadLatentAttention(dim=embed_dim,
                                             num_heads=embed_dim // 56,
                                             latent_dim=latent_dim,
                                             dropout=dropout)
        self.norm2 = RMSNorm(embed_dim)
        self.mlp = MLPBlock(embed_dim, mlp_dim, dropout)
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        y, _ = self.attn(self.norm1(x), attn_mask=mask)
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x

class LatentGPT(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 max_seq_len: int = 256,
                 embed_dim: int = 2688,
                 latent_dim: int = 336,
                 mlp_dim: int = 6912,
                 num_layers: int = 24,
                 dropout: float = 0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            LatentGPTBlock(embed_dim, latent_dim, mlp_dim, dropout) for _ in range(num_layers)
        ])
        self.norm = RMSNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        b, s = input_ids.size()
        x = self.token_embedding(input_ids) + self.positional_embedding[:, :s]
        x = self.dropout(x)
        mask = subsequent_mask(s, device=input_ids.device)
        for blk in self.blocks:
            x = blk(x, mask)
        x = self.norm(x)
        return self.lm_head(x)
