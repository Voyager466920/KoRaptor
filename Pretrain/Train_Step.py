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
) -> float:
    model.train()
    scaler = torch.amp.GradScaler(enabled=use_amp)
    optimizer.zero_grad(set_to_none=True)

    total_ce_sum = 0.0
    total_tok = 0

    ignore_index = getattr(loss_fn, "ignore_index", -100)

    for step, batch in enumerate(dataloader, start=1):
        inputs = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        with torch.amp.autocast(enabled=use_amp, device_type=device.type):
            out = model(inputs)
            logits, balance_loss = out if isinstance(out, tuple) else (out, 0.0)
            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = labels.view(-1)

            ce_loss = loss_fn(logits_flat, labels_flat)

            loss = (ce_loss + (balance_loss if isinstance(balance_loss, torch.Tensor) else 0.0)) / accumulation_steps

        scaler.scale(loss).backward()
        if step % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        valid = labels_flat != ignore_index
        num_tok = valid.sum().item()
        if num_tok == 0:
            continue

        total_ce_sum += ce_loss.item() * num_tok
        total_tok += num_tok

    if total_tok == 0:
        return math.exp(1.0)

    avg_ce = total_ce_sum / total_tok
    ppl = math.exp(avg_ce)
    print(f"Train Step Summary: Avg CE = {avg_ce:.4f}, PPL = {ppl:.4f}")
    return ppl
