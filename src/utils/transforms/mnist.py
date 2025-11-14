import torch 
import torchvision
import torchvision.transforms as transforms

# Define data transforms for preprocessing
mnist_transform = transforms.Compose([
    transforms.Pad(2),  
    transforms.ToTensor(),                        # Convert PIL Image to tensor (0-255 → 0-1)
    transforms.Normalize((0.1307,), (0.3081,))    # Normalize with MNIST statistics
    # 0.1307 is MEAN and 0.3081 is STANDARD DEVIATION for normalization purposes
])