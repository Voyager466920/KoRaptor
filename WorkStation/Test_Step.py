import math
import torch
from typing import Tuple

def test_step(
    model,
    dataloader,
    loss_fn,
    device,
    use_amp: bool = False,
) -> Tuple[float, float]:
    model.eval()
    total_ce_sum = 0.0
    total_tok = 0
    correct_tok = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(enabled=use_amp, device_type=device.type):
                out = model(inputs)
                logits = out[0] if isinstance(out, tuple) else out

            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = labels.view(-1)

            batch_ce = loss_fn(logits_flat, labels_flat)
            num_tok = (labels_flat != 0).sum().item()
            if num_tok == 0:
                continue

            total_ce_sum += batch_ce.item() * num_tok
            total_tok += num_tok

            preds_flat = logits_flat.argmax(dim=1)
            mask = labels_flat != 0
            correct_tok += (preds_flat[mask] == labels_flat[mask]).sum().item()

    if total_tok == 0:
        return math.exp(1.0), 0.0

    avg_ce = total_ce_sum / total_tok
    ppl = math.exp(avg_ce)
    acc = correct_tok / total_tok
    print(f"Test Step Summary: Avg CE = {avg_ce:.4f}, PPL = {ppl:.4f}, Acc = {acc:.4f}")
    return ppl, acc
