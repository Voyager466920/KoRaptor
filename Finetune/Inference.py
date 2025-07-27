import os
import torch
import torch.nn.functional as F
import sentencepiece as spm
from peft import PeftModel
from LatentMoE import LatentMoE, LatentMoEShim



def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    batch_size, vocab_size = logits.size()
    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
        min_values = values[:, -1].unsqueeze(1)
        logits = torch.where(logits < min_values, filter_value, logits)
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        cutoff = cumulative_probs > top_p
        cutoff[..., 1:] = cutoff[..., :-1].clone()
        cutoff[..., 0] = False
        sorted_logits[cutoff] = filter_value
        logits = logits.scatter(1, sorted_indices, sorted_logits)
    return logits

@torch.no_grad()
def custom_generate(model, tokenizer, prompt,
                    max_new_tokens=50,
                    temperature=1.0,
                    top_k=50,
                    top_p=0.9,
                    device='cuda'):
    model.to(device).eval()
    ids = tokenizer.EncodeAsIds(prompt)
    input_ids = torch.tensor([ids], device=device)
    for _ in range(max_new_tokens):
        logits, _ = model(input_ids)
        next_logits = logits[:, -1, :] / temperature
        filtered = top_k_top_p_filtering(next_logits, top_k, top_p)
        probs = F.softmax(filtered, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        if next_token.item() == tokenizer.eos_id():
            break
    output_ids = input_ids[0].tolist()
    init_len = len(ids)
    gen_ids = output_ids[init_len:]
    if tokenizer.eos_id() in gen_ids:
        gen_ids = gen_ids[:gen_ids.index(tokenizer.eos_id())]
    decoded = tokenizer.DecodeIds(gen_ids)
    return decoded.split("</s>", 1)[0]


def load_tokenizer(model_path: str):
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(model_path)
    return tokenizer


def load_model(
        base_ckpt_path: str,
        lora_ckpt_dir: str,
        tokenizer: spm.SentencePieceProcessor,
        device: torch.device,
        # LatentMoE 생성 인자 (파인튜닝 때와 동일해야 합니다)
        vocab_size: int,
        max_seq_len: int,
        embed_dim: int,
        latent_dim: int,
        mlp_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        num_experts: int,
        experts_per_token: int,
        balance_loss_weight: float,
):
    # 1) 원본 LatentMoE 모델 불러오기
    base_model = LatentMoE(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        embed_dim=embed_dim,
        latent_dim=latent_dim,
        mlp_dim=mlp_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        num_experts=num_experts,
        experts_per_token=experts_per_token,
        balance_loss_weight=balance_loss_weight,
    )
    base_model.load_state_dict(torch.load(base_ckpt_path, map_location="cpu"))
    base_model.to(device)

    # 2) Shim 래핑
    shim = LatentMoEShim(base_model)
    shim.to(device)

    # 3) LoRA 어댑터 로드
    model = PeftModel.from_pretrained(shim, lora_ckpt_dir, device_map={"": device})
    model.eval()
    return model


@torch.no_grad()
def generate(
        model: torch.nn.Module,
        tokenizer: spm.SentencePieceProcessor,
        prompt: str,
        device: torch.device,
        max_new_tokens: int = 128,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 0.95,
        temperature: float = 0.8,
):
    # 1) 토크나이즈
    ids = tokenizer.EncodeAsIds(prompt)
    input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    # 2) 생성
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        pad_token_id=tokenizer.pad_id(),  # pad token id (보통 0)
        eos_token_id=tokenizer.eos_id(),  # EOS 토큰 id
    )

    # 3) 디코딩
    gen_ids = outputs[0].cpu().tolist()
    text = tokenizer.DecodeIds(gen_ids)
    return text


if __name__ == "__main__":
    # 환경 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TOKENIZER_PATH = r"C:\junha\Git\BFG_2B\Tokenizer\spm_kowiki.model"
    BASE_CKPT_PATH = r"C:\junha\Git\BFG_2B\Checkpoints\KoRapter150M_Kowiki_AIHub_lr_1e_3\model_epoch_4.pt"
    LORA_CKPT_DIR = r"C:\junha\Git\BFG_2B\Checkpoints\lora_checkpoint\epoch_10"

    # 파인튜닝 때 썼던 설정과 동일하게
    MAX_SEQ_LEN = 256
    EMBED_DIM = 640
    LATENT_DIM = 160
    MLP_DIM = 1536
    NUM_LAYERS = 8
    NUM_HEADS = 8
    DROPOUT = 0.1
    NUM_EXPERTS = 6
    EXPERTS_PER_TOKEN = 2
    BALANCE_LOSS_WEIGHT = 0.01

    # 로드
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    model = load_model(
        BASE_CKPT_PATH, LORA_CKPT_DIR, tokenizer, device,
        vocab_size=tokenizer.GetPieceSize(),
        max_seq_len=MAX_SEQ_LEN,
        embed_dim=EMBED_DIM,
        latent_dim=LATENT_DIM,
        mlp_dim=MLP_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        num_experts=NUM_EXPERTS,
        experts_per_token=EXPERTS_PER_TOKEN,
        balance_loss_weight=BALANCE_LOSS_WEIGHT,
    )

    # 대화 루프
    print("=== 모델 인퍼런스 시작 (종료하려면 Ctrl+C) ===")
    while True:
        prompt = input("User: ")
        output = custom_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=50,
            temperature=1.0,
            top_k=50,
            top_p=0.9,
            device=device
        )
        print(f"Model: {output}\n")
