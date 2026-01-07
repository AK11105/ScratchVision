import torch
import torch.nn as nn

from src.utils.misc.kaiming_init import init_kaiming;

class VGGNet(nn.Module):
    def __init__(self, output_classes):
        super(VGGNet, self).__init__()

        self.output_classes = output_classes
        
        self.activation = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.2)

        #First Conv Layer
        self.conv1a = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        #Second Conv Layer
        self.conv2a = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)

        #Third Conv Layer
        self.conv3a = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3c = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)

        #Fourth Conv Layer
        self.conv4a = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv4c = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)

        #Fifth Conv Layer
        self.conv5a = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv5b = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv5c = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)

        # Fully Connected Layers
        self.fc1 = nn.Linear(512 * 7 * 7, 4096) ## Size becomes out_channel * W/2 * H/2
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, self.output_classes)

        #Sequential Block
        self.network = nn.Sequential(
            #First Conv Block
            self.conv1a,
            self.activation,
            self.conv1b,
            self.activation,
            self.pool,
            #Second Conv Block
            self.conv2a,
            self.activation,
            self.conv2b,
            self.activation,
            self.pool,
            #Third Conv Block
            self.conv3a,
            self.activation,
            self.conv3b,
            self.activation,
            self.conv3c,
            self.activation,
            self.pool,
            #Fourth Conv Block
            self.conv4a,
            self.activation,
            self.conv4b,
            self.activation,
            self.conv4c,
            self.activation,
            self.pool,
            #Fifth Conv Block
            self.conv5a,
            self.activation,
            self.conv5b,
            self.activation,
            self.conv5c,
            self.activation,
            self.pool,
        )

        self.apply(init_kaiming)

    def forward(self, X):
        X = self.network(X)
        X = torch.flatten(X, 1)
        X = self.fc1(X)
        X = self.dropout(X)
        X = self.fc2(X)
        X = self.dropout(X)
        X = self.fc3(X)
        return X