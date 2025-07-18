# raptor_config.py
from transformers import PretrainedConfig

class RaptorConfig(PretrainedConfig):
    model_type = "latent_moe"

    def __init__(
        self,
        vocab_size: int = None,
        max_seq_len: int = 4096,
        embed_dim: int = 1024,
        latent_dim: int = 256,
        mlp_dim: int = 4096,
        num_layers: int = 16,
        dropout: float = 0.1,
        num_heads: int = 16,
        num_experts: int = 6,
        experts_per_token: int = 2,
        balance_loss_weight: float = 1e-3,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.mlp_dim = mlp_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_heads = num_heads
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token
        self.balance_loss_weight = balance_loss_weight
