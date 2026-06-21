import torch.nn as nn
import torch

class Inception(nn.Module):
    def __init__(self, input_dim, x1, x3, x5, dim3, dim5, pp):
        super(Inception, self).__init__()
        #Save details
        self.input_dim = input_dim
        self.x1 = x1
        self.x3 = x3
        self.x5 = x5
        self.dim3 = dim3
        self.dim5 = dim5
        self.pp = pp
        
        self.convx1 = nn.Conv2d(in_channels=self.input_dim, out_channels=self.x1, stride=1, padding=0, kernel_size=1)
        self.convx31 = nn.Conv2d(in_channels=self.input_dim, out_channels=self.dim3, kernel_size=1, stride=1, padding=0)
        self.convx3 = nn.Conv2d(in_channels=self.dim3, out_channels=self.x3, kernel_size=3, stride=1, padding=1)
        self.convx51 = nn.Conv2d(in_channels=self.input_dim, out_channels=self.dim5, kernel_size=1, stride=1, padding=0)
        self.convx5 = nn.Conv2d(in_channels=self.dim5, out_channels=self.x5, kernel_size=5, stride=1, padding=2)
        self.convxp1 = nn.Conv2d(in_channels=self.input_dim, out_channels=self.pp, kernel_size=1, stride=1, padding=0)

        self.pool = nn.MaxPool2d(stride=1, padding=1, kernel_size=3)
        self.activation = nn.ReLU()

    def forward(self, X):
        #1x1 convs
        op_x1 = self.convx1(X)
        op_x1 = self.activation(op_x1)
        #3x3 convs
        mid_x3 = self.convx31(X)
        mid_x3 = self.activation(mid_x3)
        op_x3 = self.convx3(mid_x3)
        op_x3 = self.activation(op_x3)
        #5x5 convs
        mid_x5 = self.convx51(X)
        mid_x5 = self.activation(mid_x5)
        op_x5 = self.convx5(mid_x5)
        op_x5 = self.activation(op_x5)
        #parallel pool
        mid_pp = self.pool(X)
        op_pp = self.convxp1(mid_pp)
        op_pp = self.activation(op_pp)
        #final depth concat
        op = torch.cat((op_x1, op_x3, op_x5, op_pp), dim=1)
        return op