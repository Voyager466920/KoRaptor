import torch
from Models.LatentMoE import LatentMoE
import sentencepiece as spm

tokenizer = spm.SentencePieceProcessor()
tokenizer.Load("C:/junha/Git/BFG_2B/Tokenizer/spm_kowiki.model")
vocab_size = tokenizer.GetPieceSize()

model = LatentMoE(
    vocab_size=vocab_size,
    max_seq_len=256,
    embed_dim=640,
    latent_dim=160,
    mlp_dim=1536,
    num_layers=8,
    dropout=0.1,
    num_heads=8,
    num_experts=6,
    experts_per_token=2,
    balance_loss_weight=0.01
)

for name, module in model.named_modules():
    print(name, module.__class__.__name__)
