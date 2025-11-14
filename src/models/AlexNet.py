import torch 
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self,
                activation,
                pool,
                norm,
                dropout,
                C1_in=3, C1_out=96, C1_kernel=11, C1_stride=4, C1_padding=2,
                C2_in=96, C2_out=256, C2_kernel=5, C2_stride=1, C2_padding=2,
                C3_in=256, C3_out=384, C3_kernel=3, C3_stride=1, C3_padding=1,
                C4_in=384, C4_out=384, C4_kernel=3, C4_stride=1, C4_padding=1,
                C5_in=384, C5_out=256, C5_kernel=3, C5_stride=1, C5_padding=1,
                FC1_in=9216, FC1_out=4096,
                FC2_in=4096, FC2_out=4096,
                FC3_in=4096, FC3_out=1000):
        super().__init__()
        self.activation = activation or nn.ReLU()
        self.pool = pool or nn.MaxPool2d(kernel_size=3, stride=2)
        self.norm = norm or nn.LocalResponseNorm(k=2, alpha=1e-4, beta=0.75, size=5)
        self.dropout = dropout or nn.Dropout(p=0.5)
        self.conv1 = nn.Conv2d(
            in_channels=C1_in,
            out_channels=C1_out,
            kernel_size=C1_kernel,
            stride=C1_stride,
            padding=C1_padding
        )
        self.conv2 = nn.Conv2d(
            in_channels=C2_in,
            out_channels=C2_out,
            kernel_size=C2_kernel,
            stride=C2_stride,
            padding=C2_padding
        )
        self.conv3 = nn.Conv2d(
            in_channels=C3_in,
            out_channels=C3_out,
            kernel_size=C3_kernel,
            stride=C3_stride,
            padding=C3_padding
        )
        self.conv4 = nn.Conv2d(
            in_channels=C4_in,
            out_channels=C4_out,
            kernel_size=C4_kernel,
            stride=C4_stride,
            padding=C4_padding
        )
        self.conv5 = nn.Conv2d(
            in_channels=C5_in,
            out_channels=C5_out,
            kernel_size=C5_kernel,
            stride=C5_stride,
            padding=C5_padding
        )
        self.fc1 = nn.Linear(in_features=FC1_in, out_features=FC1_out)
        self.fc2 = nn.Linear(in_features=FC2_in, out_features=FC2_out)
        self.fc3 = nn.Linear(in_features=FC3_in, out_features=FC3_out)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    # conv1 and conv3 get bias = 0
                    if m is self.conv1 or m is self.conv3:
                        nn.init.constant_(m.bias, 0)
                    else:
                        nn.init.constant_(m.bias, 1)
    
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 1)
        
    def forward(self, X):
        X = self.conv1(X)
        X = self.activation(X)
        X = self.norm(X)
        X = self.pool(X)

        X = self.conv2(X)
        X = self.activation(X)
        X = self.norm(X)
        X = self.pool(X)

        X = self.conv3(X)
        X = self.activation(X)

        X = self.conv4(X)
        X = self.activation(X)

        X = self.conv5(X)
        X = self.activation(X)
        X = self.pool(X)

        X = X.view(X.size(0), -1)

        X = self.fc1(X)
        X = self.activation(X)
        X = self.dropout(X)

        X = self.fc2(X)
        X = self.activation(X)
        X = self.dropout(X)

        X = self.fc3(X)
        return X