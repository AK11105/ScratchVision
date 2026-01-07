import torch 
import torch.nn as nn

def init_kaiming(module):
    """
    Kaiming (He) initialization for VGG-style networks.
    - Conv layers: Kaiming normal (fan_out, ReLU)
    - Linear layers: Kaiming normal
    - Biases: zero
    """

    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(
            module.weight,
            mode="fan_out",
            nonlinearity="relu"
        )
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(
            module.weight,
            nonlinearity="relu"
        )
        if module.bias is not None:
            nn.init.zeros_(module.bias)