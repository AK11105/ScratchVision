import torch
from ..metrics import compute_topk_error

def evaluate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_top1_error = 0.0
    total_top3_error = 0.0
    total_top5_error = 0.0
    num_batches = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)

            if features.dim() == 5:
                B, NCROP, C, H, W = features.shape
                features = features.view(B * NCROP, C, H, W)

                outputs = model(features)                      # (B*10, num_classes)
                outputs = outputs.view(B, NCROP, -1)          # (B, 10, num_classes)
                outputs = outputs.mean(dim=1)                 # (B, num_classes)

            else:
                outputs = model(features)

            loss = criterion(outputs, labels)

            errors = compute_topk_error(outputs, labels, topk=(1, 3, 5))
            top1_error = errors["top1_error"]
            top3_error = errors["top3_error"]
            top5_error = errors["top5_error"]

            total_loss += loss.item()
            total_top1_error += top1_error
            total_top3_error += top3_error
            total_top5_error += top5_error
            num_batches += 1

            # Predictions
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / num_batches
    avg_top1_error = total_top1_error / num_batches
    avg_top3_error = total_top3_error / num_batches
    avg_top5_error = total_top5_error / num_batches

    return avg_loss, avg_top1_error, avg_top3_error, avg_top5_error, all_predictions, all_labels
