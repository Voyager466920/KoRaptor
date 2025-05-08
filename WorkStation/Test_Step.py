import torch
from sklearn.metrics import f1_score

from sklearn.metrics import f1_score

def test_step(
    model,
    dataloader,
    device,
    use_amp: bool = False,
):
    model.eval()
    total_correct = 0
    total_samples = 0
    all_labels, all_preds = [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.amp.autocast(enabled=use_amp, device_type=device.type):
                out = model(inputs)
                logits = out[0] if isinstance(out, tuple) else out

            preds = logits.view(-1, logits.size(-1)).argmax(dim=1)
            lbls = labels.view(-1)

            mask = lbls != -100            # ignore_index가 있다면
            preds, lbls = preds[mask], lbls[mask]

            total_correct += (preds == lbls).sum().item()
            total_samples += lbls.numel()

            all_labels.extend(lbls.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    f1 = f1_score(all_labels, all_preds, average="weighted") if total_samples > 0 else 0.0
    return accuracy, f1
