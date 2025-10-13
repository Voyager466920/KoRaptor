import math
import torch

def test_step(model, dataloader, loss_fn, device, amp=True):
    model.eval()
    ce_sum, tok_sum, correct = 0.0, 0, 0
    with torch.no_grad():
        for batch in dataloader:
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)
            with torch.amp.autocast(device_type="cuda", enabled=(amp and device.type == "cuda")):
                outputs = model(x)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            m = (y.view(-1) != -100)
            ce_sum += loss.item() * m.sum().item()
            tok_sum += m.sum().item()
            p = logits.argmax(dim=-1)
            correct += (p.view(-1)[m] == y.view(-1)[m]).sum().item()
    return math.exp(ce_sum / max(1, tok_sum)), correct / max(1, tok_sum)
