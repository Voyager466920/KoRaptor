import math
import torch
from typing import Tuple


def train_step(
        model,
        dataloader,
        loss_fn,
        optimizer,
        device,
        accumulation_steps: int = 4,
        use_amp: bool = False,
) -> Tuple[float, float]:
    model.train()
    scaler = torch.amp.GradScaler(enabled=use_amp)
    optimizer.zero_grad(set_to_none=True)

    loss_tokens = 0.0
    total_tok = 0
    correct_tok = 0

    for step, batch in enumerate(dataloader, start=1):
        # StreamingDataset의 배치 형식 처리
        inputs = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        with torch.amp.autocast(enabled=use_amp, device_type=device.type):
            out = model(inputs)
            logits, balance_loss = (out if isinstance(out, tuple) else (out, 0.0))
            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = labels.view(-1)

            # 손실 계산 (pad_id=0에 맞게 ignore_index=0)
            ce_loss = loss_fn(logits_flat, labels_flat, ignore_index=0)
            num_tok = (labels_flat != 0).sum().item()

            # 유효 토큰 없음 경고
            if num_tok == 0:
                print(f"Warning: Step {step} has no valid tokens (all labels are pad_id=0)")
                continue

            ce_sum = ce_loss * num_tok
            balance_loss_weight = getattr(model, "balance_loss_weight", 0.001)
            loss = (ce_loss + balance_loss * balance_loss_weight) / accumulation_steps

        scaler.scale(loss).backward()
        if step % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        loss_tokens += ce_sum.item()
        total_tok += num_tok
        preds_flat = logits_flat.argmax(dim=1)
        correct_tok += (preds_flat == labels_flat).sum().item()

        # 디버깅 로그
        if step % 10 == 0:
            valid_ratio = num_tok / labels_flat.numel()
            print(f"Step {step}:")
            print(f"  CE Loss: {ce_loss.item():.4f}, Balance Loss: {balance_loss.item():.4f}")
            print(f"  Logits Sample: {logits_flat[0, :5].detach().cpu().tolist()}")
            print(f"  Predictions: {preds_flat[:5].detach().cpu().tolist()}")
            print(f"  Labels: {labels_flat[:5].detach().cpu().tolist()}")
            print(f"  Num Tokens: {num_tok} (Valid Ratio: {valid_ratio:.4f}), Grad Norm: {grad_norm:.4f}")
            if ce_loss.isnan() or ce_loss.isinf():
                print("Warning: CE Loss is NaN or Inf")

    if total_tok == 0:
        print("Warning: No valid tokens processed in this epoch")
        return 1.0, 0.0  # Perplexity=1.0, Accuracy=0.0 반환

    avg_ce = loss_tokens / total_tok
    ppl = math.exp(avg_ce)
    acc = correct_tok / total_tok

    print(f"Train Step Summary: Avg CE = {avg_ce:.4f}, PPL = {ppl:.4f}, Acc = {acc:.4f}")
    return ppl, acc