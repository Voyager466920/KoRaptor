import torch

from Models.LatentGPT import LatentGPT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def param_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def main():
    vocab_size = 32_000
    seq_len = 128

    model = LatentGPT(
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        embed_dim=2688,
        latent_dim=336,
        mlp_dim=6912,
        num_layers=30,
    ).to(device)

    model.eval()
    batch_size = 2
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    with torch.no_grad():
        logits = model(input_ids)
    assert logits.shape == (batch_size, seq_len, vocab_size)
    print("Forward OK · logits", logits.shape)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    targets = input_ids.clone()
    logits = model(input_ids)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, vocab_size), targets.view(-1)
    )
    loss.backward()
    optimizer.step()

    print(f"Backward OK · loss {loss.item():.4f}")
    print(f"Param count {param_count(model)/1e6:.1f}M")

if __name__ == "__main__":
    main()
