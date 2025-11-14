import torch 
import torchvision
import torchvision.transforms as transforms


# Define data transforms for preprocessing
fashion_mnist_transform = transforms.Compose([
    transforms.ToTensor(),                          # Convert PIL Image to tensor (0-255 → 0-1)
    transforms.Normalize((0.2860,), (0.3530,))     # Normalize with Fashion-MNIST statistics
    # 0.2860 is MEAN and 0.3530 is STANDARD DEVIATION for normalization purposes
])