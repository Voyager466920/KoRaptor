import torch
from torch import nn
from typing import Tuple

def train_step(model, dataloader, loss_fn, optimizer, device, accumulation_steps=4, use_amp: bool = False) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    scaler = torch.amp.GradScaler(enabled=use_amp)
    optimizer.zero_grad(set_to_none=True)

    for step, (images, labels) in enumerate(dataloader, start=1):
        images, labels = images.to(device), labels.to(device)

        with torch.amp.autocast(enabled=use_amp, device_type=device.type):
            out = model(images)
            if isinstance(out, tuple):
                logits, balance_loss = out
            else:
                logits, balance_loss = out, 0.0
            B, S, V = logits.size()
            logits_flat = logits.view(-1, V)
            labels_flat = labels.view(-1)
            ce_loss = loss_fn(logits_flat, labels_flat)
            loss = (ce_loss + balance_loss * getattr(model, "balance_loss_weight", 0.0)) \
                                / accumulation_steps

        scaler.scale(loss).backward()
        if step % accumulation_steps == 0 or step == len(dataloader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * accumulation_steps

        preds_flat = logits_flat.argmax(dim=1)
        correct += (preds_flat == labels_flat).sum().item()
        total += labels_flat.numel()

    avg_loss = running_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy
