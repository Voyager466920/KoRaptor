import math
import torch

def test_step(model, loader, loss_fn, dev, amp=True):
    model.eval()
    ce_sum, tok_sum, cor = 0.0, 0, 0
    with torch.no_grad():
        for b in loader:
            x = b["input_ids"].to(dev)
            y = b["labels"].to(dev)
            with torch.amp.autocast(device_type="cuda", enabled=(amp and dev.type == "cuda")):
                o = model(x)
                l = o[0] if isinstance(o, tuple) else o
                ce = loss_fn(l.view(-1, l.size(-1)), y.view(-1))
            m = (y.view(-1) != -100)
            ce_sum += ce.item() * m.sum().item()
            tok_sum += m.sum().item()
            p = l.argmax(dim=-1)
            cor += (p.view(-1)[m] == y.view(-1)[m]).sum().item()
    return math.exp(ce_sum / max(1, tok_sum)), cor / max(1, tok_sum)