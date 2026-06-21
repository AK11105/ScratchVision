from tqdm import tqdm
import time
from .train_epoch import train_epoch
from ...evaluators.eval.evaluate_googlenet import evaluate
import torch.optim as optim

def train(num_epochs, model, train_dataloader, test_dataloader, criterion, optimizer, device, update_freq=10):
    train_losses, train_top1_errors, train_top3_errors, train_top5_errors = [], [], [], []
    test_losses, test_top1_errors, test_top3_errors, test_top5_errors = [], [], [], []

    start_time = time.time()
    epoch_pbar = tqdm(range(num_epochs), desc="Epochs")

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.96)

    for epoch in epoch_pbar:
        epoch_start_time = time.time()
        
        # Train
        train_loss, train_top1, train_top3, train_top5 = train_epoch(
            model, train_dataloader, criterion, optimizer, device, update_freq
        )
        scheduler.step()

        # Evaluate
        test_loss, test_top1, test_top3, test_top5 = evaluate(
            model, test_dataloader, criterion, device
        )

        # Store
        train_losses.append(train_loss); train_top1_errors.append(train_top1)
        train_top3_errors.append(train_top3); train_top5_errors.append(train_top5)
        test_losses.append(test_loss); test_top1_errors.append(test_top1)
        test_top3_errors.append(test_top3); test_top5_errors.append(test_top5)

        epoch_time = time.time() - epoch_start_time
        epoch_pbar.set_postfix({
            'Val T1E': f'{test_top1:.2f}', 
            'Time': f'{epoch_time:.0f}s'
        })

    total_time = time.time() - start_time
    print(f"\n✅ Done! Total: {total_time/60:.1f}min | Final Val T1E: {test_top1_errors[-1]:.2f}%")

    return (train_losses, train_top1_errors, train_top3_errors, train_top5_errors,
            test_losses, test_top1_errors, test_top3_errors, test_top5_errors)