import torch
from torch import nn
from typing import Tuple

def train_step(model, dataloader, loss_fn, optimizer, device, accumulation_steps=4,use_amp: bool = False,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    optimizer.zero_grad(set_to_none=True)

    for step, (images, labels) in enumerate(dataloader, start=1):
        images, labels = images.to(device), labels.to(device)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            loss = loss_fn(logits, labels) / accumulation_steps

        scaler.scale(loss).backward()

        if step % accumulation_steps == 0 or step == len(dataloader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * accumulation_steps
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy
