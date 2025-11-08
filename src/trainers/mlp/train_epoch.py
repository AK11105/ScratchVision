import torch
import torch.nn as nn
import torch.optim as optim

from ...evaluators import calculate_accuracy

def train_classifier_epoch(model, dataloader, criterion, optimizer, device, update_freq=10):
    
    #Set model to training mode
    model.train()
    
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0
    
    for batch_idx, (features, labels) in enumerate(dataloader):
        #Move to device
        features, labels = features.to(device), labels.to(device)
        
        #Zero the gradients
        optimizer.zero_grad()
        
        #Forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        #Backpropagation
        loss.backward()
        optimizer.step()
        
        #Calculate metrics
        accuracy = calculate_accuracy(outputs, labels)
        total_loss+=loss.item()
        total_accuracy+=accuracy
        num_batches+=1
        
        if(batch_idx+1)%update_freq == 0:
            print(f'   Batch [{batch_idx+1}/{len(dataloader)}] - 'f'Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%')
            
    avg_loss = total_loss/num_batches
    avg_accuracy = total_accuracy/num_batches

    return avg_loss, avg_accuracy
        