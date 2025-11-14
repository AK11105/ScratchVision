import torchvision
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder

from src.utils.transforms import *

def download_FashionMNIST():
    # Define data transforms for preprocessing
    transform = fashion_mnist_transform
    
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
    transform = mnist_transform
    
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

def download_Imagenette():
    root_dir = "data/processed/imagenette2/imagenette2"
    train_dataset = ImageFolder(
        root=f"{root_dir}/train",
        transform=imagenette_train_transform
    )
    
    val_dataset = ImageFolder(
        root=f"{root_dir}/val",
        transform=imagenette_test_transform
    )
    
    return train_dataset, val_dataset

