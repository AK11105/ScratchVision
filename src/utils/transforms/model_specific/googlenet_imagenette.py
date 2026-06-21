import torchvision.transforms as transforms
from torchvision.transforms import v2 as transformsv2

train_transform_googlenet = transforms.Compose([
    transforms.RandomResizedCrop(
        size=(224,224),
        scale=(0.08, 1.0),
        ratio=(3/4, 4/3)
    ),
    transformsv2.RandomPhotometricDistort(),  # optional, labeled
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transform_googlenet = transforms.Compose([
    transforms.Resize(256),          # resize shorter side
    transforms.CenterCrop(224),      # enforce spatial contract
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])