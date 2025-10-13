import torch
import math

def train_step(model, dataloader, loss_fn, optimizer, device, acc_steps=1, amp=True):
    model.train()
    scaler = torch.amp.GradScaler(enabled=(amp and device.type == "cuda"))
    optimizer.zero_grad(set_to_none=True)
    ce_sum, tok_sum = 0.0, 0
    for i, batch in enumerate(dataloader, 1):
        x = batch["input_ids"].to(device)
        y = batch["labels"].to(device)
        with torch.amp.autocast(device_type="cuda", enabled=(amp and device.type == "cuda")):
            outputs = model(x)
            if isinstance(outputs, tuple):
                logits, aux = outputs
            else:
                logits, aux = outputs, 0.0
            ce = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            loss = (ce + (aux if isinstance(aux, torch.Tensor) else 0.0)) / acc_steps
        scaler.scale(loss).backward()
        if i % acc_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        m = (y.view(-1) != -100)
        ce_sum += ce.item() * m.sum().item()
        tok_sum += m.sum().item()
    return math.exp(ce_sum / max(1, tok_sum))
