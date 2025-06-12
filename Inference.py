import torch
import os
from transformers import AutoTokenizer
from Models.LatentMoE import LatentMoE


def load_model_and_tokenizer(checkpoint_path, device, max_seq_len=4096, vocab_size=50257, embed_dim=1024,
                             latent_dim=256, mlp_dim=4096, num_layers=16, dropout=0.1, num_heads=16, num_experts=6,
                             experts_per_token=2, balance_loss_weight=0.001):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.model_max_length = max_seq_len
    tokenizer.init_kwargs["model_max_length"] = max_seq_len

    # Initialize model
    model = LatentMoE(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        embed_dim=embed_dim,
        latent_dim=latent_dim,
        mlp_dim=mlp_dim,
        num_layers=num_layers,
        dropout=dropout,
        num_heads=num_heads,
        num_experts=num_experts,
        experts_per_token=experts_per_token,
        balance_loss_weight=balance_loss_weight,
    ).to(device)
    model.lm_head.weight = model.token_embedding.weight  # Tie weights

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    return model, tokenizer


def generate_text(model, tokenizer, prompt, device, max_length=100, top_k=50, temperature=1.0):
    # Tokenize input
    input_ids = tokenizer.encode(
        prompt,
        add_special_tokens=False,
        max_length=4096,
        truncation=True,
        return_tensors="pt"
    ).to(device)  # Shape: [1, seq_len]

    generated_ids = input_ids.clone()

    with torch.no_grad():
        for _ in range(max_length):
            # Forward pass
            outputs = model(generated_ids)
            logits = outputs[0][:, -1, :]  # Get logits for last token, shape: [1, vocab_size]

            # Apply temperature
            logits = logits / temperature

            # Top-k sampling
            top_k_probs, top_k_indices = torch.topk(logits, top_k, dim=-1)
            top_k_probs = torch.softmax(top_k_probs, dim=-1)
            next_token = torch.multinomial(top_k_probs, num_samples=1)
            next_token = top_k_indices.gather(-1, next_token)  # Shape: [1, 1]

            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            # Stop if sequence exceeds max_seq_len
            if generated_ids.size(1) >= 4096:
                break

    # Decode generated tokens
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text


def main():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = r"C:\junha\Git\BFG_2B\Checkpoints\model_epoch_5.pt"
    max_seq_len = 4096
    vocab_size = 50257  # GPT-2 vocab size
    embed_dim = 1024
    latent_dim = 256
    mlp_dim = 4096
    num_layers = 16
    dropout = 0.1
    num_heads = 16
    num_experts = 6
    experts_per_token = 2
    balance_loss_weight = 0.001

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        checkpoint_path,
        device,
        max_seq_len,
        vocab_size,
        embed_dim,
        latent_dim,
        mlp_dim,
        num_layers,
        dropout,
        num_heads,
        num_experts,
        experts_per_token,
        balance_loss_weight
    )

    # Example prompt
    prompt = "Once upon a time in a distant land, there was a"
    max_length = 100  # Number of tokens to generate
    top_k = 50
    temperature = 1.0

    # Generate text
    generated_text = generate_text(model, tokenizer, prompt, device, max_length, top_k, temperature)
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")


if __name__ == "__main__":
    main()
