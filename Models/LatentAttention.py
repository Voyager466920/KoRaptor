import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentAttention(nn.Module):
    def __init__(self, embed_dim, latent_dim, dropout=0.1):
        super(LatentAttention, self).__init__()
        #W_QW_{UK}^T
        self.proj_q = nn.Linear(embed_dim, latent_dim)
        # W_{DKV}
        self.proj_kv = nn.Linear(embed_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = latent_dim ** 0.5

    def forward(self, x, mask=None):
        Q = self.proj_q(x)
        L_kv = self.proj_kv(x)

        attn_scores = torch.matmul(Q, L_kv.transpose(-2, -1)) / self.scale

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        AL_kv = torch.matmul(attn_weights, L_kv)
        return AL_kv
