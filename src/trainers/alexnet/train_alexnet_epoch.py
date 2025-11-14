import torch 
import torch.nn as nn 
import torch.optim as optim

from ...evaluators import compute_topk_error

def train_epoch(model, dataloader, criterion, optimizer, device, update_freq=10):
    model.train()
    
    total_loss = 0.0
    total_top1_error = 0.0
    total_top3_error = 0.0
    total_top5_error = 0.0
    num_batches = 0
    
    for batch_idx, (features, labels) in enumerate(dataloader):
        # Move to device
        features, labels = features.to(device), labels.to(device)
        
        #Zero the gradients
        optimizer.zero_grad()
        
        #Forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        #Backpropagtion
        loss.backward()
        optimizer.step()
        
        #Calculate error rates
        errors = compute_topk_error(outputs, labels, topk=(1,3,5))
        top1_error = errors["top1_error"]
        top3_error = errors["top3_error"]
        top5_error = errors["top5_error"]
        
        #Accumulate
        total_loss+=loss.item()
        total_top1_error+=top1_error
        total_top3_error+=top3_error
        total_top5_error+=top5_error
        num_batches+=1
        
        if(batch_idx+1)%update_freq == 0:
            print(f'   Batch [{batch_idx+1}/{len(dataloader)}] - 'f'Loss: {loss.item():.4f}, Top 1 Error: {top1_error:.2f} , Top 3 Error: {top3_error:.2f} , Top 5 Error: {top5_error:.2f}')
    
    avg_loss = total_loss/num_batches
    avg_top1_error = total_top1_error/num_batches
    avg_top3_error = total_top3_error/num_batches
    avg_top5_error = total_top5_error/num_batches

    return avg_loss, avg_top1_error, avg_top3_error, avg_top5_error
            
        