import torch
import torchvision
import torchvision.transforms as transforms
from src.utils.misc.LightingPCA import LightingPCA

def ten_crop_to_tensor(crops):
    return torch.stack([transforms.ToTensor()(crop) for crop in crops])

imagenette_train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    LightingPCA(alpha_std=0.01)
])

imagenette_test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.TenCrop(224),
    transforms.Lambda(ten_crop_to_tensor)
])
