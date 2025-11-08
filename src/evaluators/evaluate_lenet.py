import torch
from .metrics.accuracy import calculate_accuracy

def evaluate(model, dataloader, criterion, device):
    #Set to eval mode
    model.eval()
    
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            
            outputs, features = model(features)
            loss = criterion(features, labels)
            
            accuracy = calculate_accuracy(outputs, labels)
            total_loss+=loss.item()
            total_accuracy+=accuracy
            num_batches+=1
            
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches

    return avg_loss, avg_accuracy, all_predictions, all_labels