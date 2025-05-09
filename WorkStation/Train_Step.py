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

    for step, (inputs, labels) in enumerate(dataloader, start=1):
        inputs, labels = inputs.to(device), labels.to(device)

        with torch.amp.autocast(enabled=use_amp, device_type=device.type):
            out = model(inputs)
            logits, balance_loss = (out if isinstance(out, tuple) else (out, 0.0))
            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = labels.view(-1)

            ce_loss = loss_fn(logits_flat, labels_flat)  # Mean loss
            num_tok = (labels_flat != -100).sum().item()
            ce_sum = ce_loss * num_tok  # Scale to sum for accumulation

            loss = (ce_loss + balance_loss * getattr(model, "balance_loss_weight", 0.0)) / accumulation_steps

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

        if step % 100 == 0:
            print(f"Step {step}: CE Loss = {ce_loss.item():.4f}, Balance Loss = {balance_loss.item():.4f}, Grad Norm = {grad_norm:.4f}")

    avg_ce = loss_tokens / max(total_tok, 1)
    ppl = math.exp(avg_ce)
    acc = correct_tok / max(total_tok, 1)

    return ppl, acc