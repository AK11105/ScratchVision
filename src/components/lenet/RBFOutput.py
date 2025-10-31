import torch 
import torch.nn as nn

class RBFOutput(nn.Module):
    """
    Computes squared Euclidean distance from F6 activations
    to fixed prototype code vectors (one per class).
    """
    def __init__(self, num_classes=10, in_features=84, codebook=None):
        super(RBFOutput, self).__init__()
        if codebook is None:
            # Default: random ±1 code vectors
            codebook = torch.sign(torch.randn(num_classes, in_features))
        self.register_buffer("codebook", codebook)  # non-trainable

    def forward(self, x):
        # x: (batch, in_features)
        # Return squared Euclidean distances (batch, num_classes)
        dists = torch.cdist(x, self.codebook, p=2) ** 2
        return dists