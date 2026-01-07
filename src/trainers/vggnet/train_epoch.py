from tqdm.auto import tqdm
import torch

from src.evaluators.metrics import compute_topk_error

def train_epoch(model, dataloader, criterion, optimizer, device, update_freq=10):
    model.train()
    pbar = tqdm(dataloader, desc="Train", leave=False)
    
    total_loss = 0.0
    total_top1_error = 0.0
    total_top3_error = 0.0
    total_top5_error = 0.0
    num_batches = 0

    for batch_idx, (features, labels) in enumerate(pbar):
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            errors = compute_topk_error(outputs, labels, topk=(1,3,5))

        total_loss += loss.item()
        total_top1_error += errors["top1_error"]
        total_top3_error += errors["top3_error"]
        total_top5_error += errors["top5_error"]
        num_batches += 1

        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'T1E': f'{errors["top1_error"]:.2f}',
            'T3E': f'{errors["top3_error"]:.2f}'
        })

    pbar.close()
    if num_batches == 0:
        return 0, 0, 0, 0
    return (total_loss / num_batches, total_top1_error / num_batches, 
            total_top3_error / num_batches, total_top5_error / num_batches)