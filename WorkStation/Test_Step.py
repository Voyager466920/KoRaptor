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

    total_ce_sum = 0.0
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

            batch_ce = loss_fn(logits_flat, labels_flat)  # Mean loss
            num_tok = (labels_flat != -100).sum().item()
            batch_ce_sum = batch_ce.item() * num_tok

            total_ce_sum += batch_ce_sum  # Fixed: Removed redundant line
            total_tok += num_tok

            preds_flat = logits_flat.argmax(dim=1)
            mask = labels_flat != -100
            preds_masked = preds_flat[mask]
            labels_masked = labels_flat[mask]

            correct_tok += (preds_masked == labels_masked).sum().item()
            all_lbls.extend(labels_masked.cpu().tolist())
            all_preds.extend(preds_masked.cpu().tolist())

            # Debug logging
            print(f"Batch CE: {batch_ce.item():.4f}, Num Tokens: {num_tok}")
            print(f"Sample Predictions: {preds_masked[:5].tolist()}")
            print(f"Sample Labels: {labels_masked[:5].tolist()}")

    # Final metrics
    avg_ce = total_ce_sum / max(total_tok, 1)
    ppl = math.exp(avg_ce)
    acc = correct_tok / max(total_tok, 1)
    f1 = f1_score(all_lbls, all_preds, average="weighted") if total_tok > 0 else 0.0

    return ppl, acc, f1