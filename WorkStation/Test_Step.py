import torch
from sklearn.metrics import f1_score

def test_step(
    model,
    dataloader,
    loss_fn,
    device,
    use_amp: bool = False,
):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_labels, all_preds = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            with torch.amp.autocast(enabled=use_amp,device_type=device.type):
                out = model(images)
                if isinstance(out, tuple):
                    logits, _ = out
                else:
                    logits = out
                B, S, V = logits.size()
                logits_flat = logits.view(-1, V)         # → [B*S, V]
                labels_flat = labels.view(-1)            # → [B*S]
                loss = loss_fn(logits_flat, labels_flat) # CrossEntropyLoss(expect [N, C] vs [N])

            total_loss += loss.item()
            preds_flat = logits_flat.argmax(dim=1)      # [B*S]
            total_correct += (preds_flat == labels_flat).sum().item()
            total_samples += labels_flat.numel()

            all_labels.extend(labels_flat.cpu().tolist())
            all_preds.extend(preds_flat.cpu().tolist())

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return avg_loss, accuracy, f1
