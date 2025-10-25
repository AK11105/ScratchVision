import torch
import torch.nn.functional as F

#Forming the input vector for each S-cell's receptive field
def patch_extract(x: torch.Tensor, kernel: int, stride: int = 1, padding: int = 0):
    """_summary_

    Args:
        x (torch.Tensor): (B (batch-size), C (no. of channels), H (height of image), W(weight of image)) 
        kernel (int): size of receptive field(rf) / filter (eg. 5x5)
        stride (int, optional): step size. Defaults to 1.
        padding (int, optional): add extra pixels around border. Defaults to 0.
        
    Description:
        Uses torch.nn.functional.unfold ==> converts 2D input image into collection of local patches
        Each patch corresponds to receptive field of neuron in S layer
    
    Returns:
        output (torch.Tensor): (B, C*rf*rf, L) where L = H_out * W_out is the number of  receptive field positions
        Each column of the unfolded result corresponds to one small patch of the input image, flattened into a vector.
    """
    return F.unfold(x, kernel_size=kernel, stride=stride, padding=padding)

#Example
    """
    If input x = (1, 1, 16, 16) and rf=5,
    patch_extract(x, 5) produces shape (1, 25, 144),
    where 25 = 1 × 5 × 5 (flattened 5×5 region per cell) and 144 = 12×12 (number of receptive field positions).
    (12x12 instead of 16x16 : visualize a 5x5 filter moving across 16x16 cell with stride = 1; output is 12x12)
    Each column of this (25×144) matrix corresponds to the “view” of one S-cell in the layer.
    """ 
    