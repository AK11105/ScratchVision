import torch 
import torch.nn as nn 

from ..components.lenet.SquashingFunction import ScaledTanH
from ..components.lenet.C3Layer import C3 
from ..components.lenet.RBFOutput import RBFOutput

class LeNet5(nn.Module):
    def __init__(
                self,
                activation,
                C1_in, C1_out, C1_filter, C1_stride, C1_padding,
                S2_filter, S2_stride,
                S4_filter, S4_stride,
                C5_in, C5_out, C5_filter, C5_stride, C5_padding,
                F6_in, F6_out,
                output_in, output_out
                ):
        super(LeNet5, self).__init__()

        #Activation after each layer
        self.activation = activation or ScaledTanH()

        #C1
        self.C1 = nn.Conv2d(
            in_channels=C1_in or 1, #Grayscale input
            out_channels =C1_out or 6,
            kernel_size=C1_filter or 5,
            stride=C1_stride or 1,
            padding=C1_padding or 0
        )

        #S2
        self.S2_pool = nn.AvgPool2d(kernel_size=S2_filter or 2, stride=S2_stride or 2)
        self.S2_weight = nn.Parameter(torch.ones(6))
        self.S2_bias = nn.Parameter(torch.zeros(6))

        self.C3_connections = torch.tensor([
            [1,0,0,0,1,1,1,0,0,1,1,1,1,0,1,1],
            [1,1,0,0,0,1,1,1,0,0,1,1,1,1,0,1],
            [1,1,1,0,0,0,1,1,1,0,0,1,0,1,1,1],
            [0,1,1,1,0,0,1,1,1,1,0,0,1,0,1,1],
            [0,0,1,1,1,0,0,1,1,1,1,0,1,1,0,1],
            [0,0,0,1,1,1,0,0,1,1,1,1,0,1,1,1]
        ], dtype=torch.uint8)

        #C3
        self.C3 = C3(self.C3_connections)

        #S4
        self.S4_pool = nn.AvgPool2d(kernel_size=S4_filter or 2, stride=S4_stride or 2)
        self.S4_weight = nn.Parameter(torch.ones(16))
        self.S4_bias = nn.Parameter(torch.zeros(16))

        #C5
        self.C5 = nn.Conv2d(
            in_channels=C5_in or 16,
            out_channels=C5_out or 120,
            kernel_size=C5_filter or 5,
            stride=C5_stride or 1,
            padding=C5_padding or 0
        )

        #F6
        self.F6 = nn.Linear(in_features=F6_in or 120, out_features=F6_out or 84)

        #Output
        self.output = RBFOutput(num_classes=output_out, in_features=output_in)

    def forward(self, X):
        #C1
        X = self.C1(X)
        X = self.activation(X)

        #S2
        X = self.S2_pool(X)
        X = self.S2_weight.view(1,-1,1,1) * X + self.S2_bias.view(1,-1,1,1)
        X = self.activation(X)

        #C3
        X = self.C3(X)
        X = self.activation(X)

        #S4
        X = self.S4_pool(X)
        X = self.S4_weight.view(1,-1,1,1) * X + self.S4_bias.view(1,-1,1,1)
        X = self.activation(X)

        #C5
        X = self.C5(X)
        X = self.activation(X)
        X = X.view(X.size(0), -1)

        #F6
        X = self.F6(X)
        X = self.activation(X)

        dists = self.output(X)
        
        return dists, X

    def info(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        info = {
            'Total Parameters': f'{total_params:,}',
            'Trainable Parameters': f'{trainable_params:,}',
            'Model Size (MB)': f'{total_params * 4 / (1024**2):.2f}',  # Assuming float32
        }
        return info