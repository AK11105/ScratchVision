import sys
import os
from multiprocessing import freeze_support

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if __name__ == "__main__":
    freeze_support()  # Only needed on Windows when using multiprocessing

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from data import download_FashionMNIST
    from src.utils.loaders import create_DataLoaders
    from src.models.MLP import MultiLayerPerceptron
    from src.trainers.train_classifier import train_classifier

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 20
    weight_decay = 1e-4
    target_accuracy = 97.0

    # Load datasets and create DataLoaders
    train_dataset, test_dataset = download_FashionMNIST()
    train_loader, test_loader = create_DataLoaders(
        train_dataset, test_dataset, batch_size,
        shuffle_train=True,
        num_workers=0  # Safe for Windows; increase only after confirming stability
    )

    # Model, loss, optimizer
    model = MultiLayerPerceptron().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Train
    train_losses, train_accuracies, test_losses, test_accuracies = train_classifier(
        num_epochs, model, train_loader, test_loader,
        criterion, optimizer, target_accuracy, device,
        update_freq=200
    )

    # Save weights
    os.makedirs("../experiments/MLP", exist_ok=True)
    torch.save(model.state_dict(), "../experiments/MLP/mlp_weights.pth")

    # Save full checkpoint
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_losses': test_losses,
        'test_accuracies': test_accuracies
    }, "../experiments/MLP/mlp_detailed.pth")
