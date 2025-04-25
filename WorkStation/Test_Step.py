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

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images)
                loss = loss_fn(logits, labels)

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return avg_loss, accuracy, f1
