import torch
import math


def train_step(model, loader, loss_fn, opt, dev, acc_steps=1, amp=True):
    model.train()
    sc = torch.amp.GradScaler(enabled=amp)
    opt.zero_grad(set_to_none=True)
    ce_sum, tok_sum = 0.0, 0
    for i, b in enumerate(loader, 1):
        x = b["input_ids"].to(dev)
        y = b["labels"].to(dev)
        with torch.amp.autocast(device_type="cuda", enabled=(amp and dev.type == "cuda")):
            o = model(x)
            l = o[0] if isinstance(o, tuple) else o
            ce = loss_fn(l.view(-1, l.size(-1)), y.view(-1))
            loss = ce / acc_steps
        sc.scale(loss).backward()
        if i % acc_steps == 0:
            sc.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            sc.step(opt)
            sc.update()
            opt.zero_grad(set_to_none=True)
        m = (y.view(-1) != -100)
        ce_sum += ce.item() * m.sum().item()
        tok_sum += m.sum().item()
    return math.exp(ce_sum / max(1, tok_sum))