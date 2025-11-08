import torch
import time

from .train_lenet_epoch import train_epoch
from ...evaluators import evaluate
from ...utils import adjust_learning_rate

def train_lenet(num_epochs, model, train_loader, test_loader, criterion, optimizer, target_accuracy, device, update_freq, lr_schedule):
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print("-" * 50)
        
        # Adjust LR according to schedule
        adjust_learning_rate(optimizer, epoch + 1, lr_schedule)
        print(f"INFO - Learning rate set to: {optimizer.param_groups[0]['lr']}")

        # Train
        print("INFO - Training...")
        train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer, device, update_freq=update_freq)
        
        # Eval
        print("INFO - Evaluating...")
        test_loss, test_accuracy, _, _ = evaluate(model, test_loader, criterion, device)
        
        # Store metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        
        epoch_time = time.time() - epoch_start_time
        
        if test_accuracy > target_accuracy:
            print(f"🎯 Target accuracy reached! Stopping early.")
            break
    
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("✅ Training completed!")
    print(f"🕒 Total training time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    print(f"🎯 Final test accuracy: {test_accuracies[-1]:.2f}%")
    
    return train_losses, train_accuracies, test_losses, test_accuracies
