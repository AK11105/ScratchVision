import torchvision
from torchvision.transforms import transforms


def download_FashionMNIST():
    # Define data transforms for preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),                          # Convert PIL Image to tensor (0-255 → 0-1)
        transforms.Normalize((0.2860,), (0.3530,))     # Normalize with Fashion-MNIST statistics
        # 0.2860 is MEAN and 0.3530 is STANDARD DEVIATION for normalization purposes
    ])
    
    # Load training dataset
    train_dataset = torchvision.datasets.FashionMNIST(
        root='./processed',           # Directory to store data
        train=True,              # Load training set
        download=True,           # Download if not already present
        transform=transform      # Apply preprocessing transforms
    )
    
    # Load test dataset
    test_dataset = torchvision.datasets.FashionMNIST(
        root='../data',
        train=False,             # Load test set
        download=True,
        transform=transform
    )
    
    return train_dataset, test_dataset

def download_MNIST():
    # Define data transforms for preprocessing
    transform = transforms.Compose([
        transforms.Pad(2),  
        transforms.ToTensor(),                        # Convert PIL Image to tensor (0-255 → 0-1)
        transforms.Normalize((0.1307,), (0.3081,))    # Normalize with MNIST statistics
        # 0.1307 is MEAN and 0.3081 is STANDARD DEVIATION for normalization purposes
    ])
    
    # Load training dataset
    train_dataset = torchvision.datasets.MNIST(
        root='../data/processed',         # Directory to store data
        train=True,                 # Load training set
        download=True,              # Download if not already present
        transform=transform         # Apply preprocessing transforms
    )
    
    # Load test dataset
    test_dataset = torchvision.datasets.MNIST(
        root='../data',             # Separate directory for test data
        train=False,                # Load test set
        download=True,
        transform=transform
    )
    
    return train_dataset, test_dataset