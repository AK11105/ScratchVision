import torch 
import time 

from .train_alexnet_epoch import train_epoch
from src.evaluators.eval.evaluate_alexnet import evaluate

def train(num_epochs, model, train_dataloader, test_dataloader, criterion, optimizer, target_top1_error, target_top3_error, target_top5_error, device, update_freq):
    train_losses = []
    train_top1_errors = []
    train_top3_errors = []
    train_top5_errors = []
    test_losses = []
    test_top1_errors = []
    test_top3_errors = []
    test_top5_errors = []

    start_time = time.time()
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,       # divide LR by 10
        patience=1,       # if error doesn't improve for 1 epoch
    )
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print("-" * 50)
        
        #Train
        print("INFO - Training...")
        train_loss, train_top1_error, train_top3_error, train_top5_error = train_epoch(model, train_dataloader, criterion, optimizer, device, update_freq=update_freq)
        
        #Eval
        print("INFO - Evaluating...")
        test_loss, test_top1_error, test_top3_error, test_top5_error, _, _= evaluate(model, test_dataloader, criterion, device)
        
        # Store metrics
        train_losses.append(train_loss)
        train_top1_errors.append(train_top1_error)
        train_top3_errors.append(train_top3_error)
        train_top5_errors.append(train_top5_error)
        
        test_losses.append(test_loss)
        test_top1_errors.append(test_top1_error)
        test_top3_errors.append(test_top3_error)
        test_top5_errors.append(test_top5_error)
        
        scheduler.step(test_top1_error)
        
        epoch_time = time.time() - epoch_start_time
        
        if test_top1_error <= target_top1_error and test_top3_error <= target_top3_error and test_top5_error <= target_top5_error:
            print(f"🎯 Target Error rate reached! Stopping early.")
            break
    
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("✅ Training completed!")
    print(f"🕒 Total training time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    print(f"🎯 Final top-1 error rate: {test_top1_errors[-1]:.2f}%")
    print(f"🎯 Final top-3 error rate: {test_top3_errors[-1]:.2f}%")
    print(f"🎯 Final top-5 error rate: {test_top5_errors[-1]:.2f}%")
    
    return train_losses, train_top1_errors, train_top3_errors, train_top5_errors, test_losses, test_top1_errors, test_top3_errors, test_top5_errors