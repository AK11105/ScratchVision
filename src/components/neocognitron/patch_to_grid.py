import torch

#Convert unfolded patches back to grid form
#Forms the output of a S-plane or C-plane
def patch_to_grid(patches: torch.Tensor, out_h: int, out_w: int):
    """_summary_

    Args:
        patches (torch.Tensor): output of patch_extract
        out_h (int): reshape height
        out_w (int): reshape width
    
    Description: 
        Simply reshapes the unfolded patches (which are 1D columns) back into a 2D spatial grid
        Inverse "view" of unfold, not a mathematical inversion
    """
    B, Ck, L = patches.shape
    return patches.view(B, Ck, out_h, out_w)