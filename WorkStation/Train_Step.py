import torch
from typing import Tuple
import time

try:
    import pynvml
    pynvml.nvmlInit()
    _HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    _HANDLE = None

_TEMP_LIMIT = 83
_COOL_TIME  = 30
_STEP_INTERVAL = 200
_STEP_COOL   = 10

def _maybe_cool_down(step: int) -> None:
    if step % _STEP_INTERVAL == 0:
        torch.cuda.empty_cache()
        #print(f"[Train] {step} steps → {_STEP_COOL}s 휴식")
        time.sleep(_STEP_COOL)
    if _HANDLE is None:
        return

    temp = pynvml.nvmlDeviceGetTemperature(
        _HANDLE, pynvml.NVML_TEMPERATURE_GPU
    )
    if temp >= _TEMP_LIMIT:
        torch.cuda.empty_cache()
        print(f"[GPU] 온도 {temp}°C ‑> {_COOL_TIME}s 휴식")
        time.sleep(_COOL_TIME)

def train_step(model, dataloader, loss_fn, optimizer, device, accumulation_steps=4, use_amp: bool = False) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    batch_count = 0

    scaler = torch.amp.GradScaler(enabled=use_amp)
    optimizer.zero_grad(set_to_none=True)

    for step, (images, labels) in enumerate(dataloader, start=1):
        _maybe_cool_down(step)
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
        if step % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * accumulation_steps
        batch_count += 1

        preds_flat = logits_flat.argmax(dim=1)
        correct += (preds_flat == labels_flat).sum().item()
        total += labels_flat.numel()

    avg_loss = running_loss / batch_count
    accuracy = correct / total
    return avg_loss, accuracy
