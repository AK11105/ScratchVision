from tqdm import tqdm
from ..metrics import compute_topk_error

def evaluate(model, dataloader, criterion, device):
    model.eval()
    pbar = tqdm(dataloader, desc="Eval", leave=False)

    total_loss = 0.0
    total_top1_error = 0.0
    total_top3_error = 0.0
    total_top5_error = 0.0
    num_batches = 0

    with torch.no_grad():
        for features, labels in pbar:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            errors = compute_topk_error(outputs, labels, topk=(1,3,5))

            total_loss += loss.item()
            total_top1_error += errors["top1_error"]
            total_top3_error += errors["top3_error"]
            total_top5_error += errors["top5_error"]
            num_batches += 1

            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    pbar.close()

    if num_batches == 0:
        return 0, 0, 0, 0

    return (
        total_loss / num_batches,
        total_top1_error / num_batches,
        total_top3_error / num_batches,
        total_top5_error / num_batches,
    )