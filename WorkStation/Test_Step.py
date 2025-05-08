import math
import torch
from sklearn.metrics import f1_score
from typing import Tuple


def test_step(
        model,
        dataloader,
        loss_fn,
        device,
        use_amp: bool = False,
) -> Tuple[float, float, float]:
    model.eval()

    loss_tokens = 0.0
    total_tok = 0
    correct_tok = 0
    all_lbls, all_preds = [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.amp.autocast(enabled=use_amp, device_type=device.type):
                out = model(inputs)
                logits = out[0] if isinstance(out, tuple) else out

            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = labels.view(-1)

            ce_sum = loss_fn(logits_flat, labels_flat)
            num_tok = (labels_flat != -100).sum().item()

            loss_tokens += ce_sum.item()
            total_tok += num_tok

            preds_flat = logits_flat.argmax(dim=1)
            mask = labels_flat != -100
            preds_masked, labels_masked = preds_flat[mask], labels_flat[mask]

            correct_tok += (preds_masked == labels_masked).sum().item()
            all_lbls.extend(labels_masked.cpu().tolist())
            all_preds.extend(preds_masked.cpu().tolist())

    avg_ce = loss_tokens / max(total_tok, 1)
    ppl = math.exp(avg_ce)
    acc = correct_tok / max(total_tok, 1)
    f1 = f1_score(all_lbls, all_preds, average="weighted") if total_tok else 0.0

    return ppl, acc, f1
