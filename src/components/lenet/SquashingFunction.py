import torch 
import torch.nn as nn 

class ScaledTanH(nn.Module):
    """Scaled Hyperbolic Tangent : f(a) = A * tanh(S * a)"""
    def __init__(self, A=1.7159, S=2/3):
        super(ScaledTanH, self).__init__()
        self.A = A
        self.S = S
    def forward(self, X):
        return self.A * torch.tanh(self.S * X)