import torch 
from typing import Tuple, List 

Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Complex Cells ==> POOLING LAYER
class CLayer:
    """
    C-layer: pooling / tolerance-building layer with saturation.
    Function:
        - Pool local S-cell responses -> build spatial tolerance
        - Suppress noise via lateral inhibition
        - Outpu generalized feature maps that apply translational invariance
    """
    def __init__(self, in_planes: int, out_planes: int, rf: int, in_map_size: Tuple[int, int], out_map_size: Tuple[int, int], alpha: float=1.0, device = Device):
        """_summary_

        Args:
            in_planes (int): Number of input  S planes ; theoretically K_{S_{l}} : number of feature maps from S-layer
            out_planes (int): Number of output C planes ; theoretically K_{C_{l}} : number of feature maps in this C-layer
            rf (int): receptive field sixe; theoretically size of connecting reigon D_{l} [area from which each C-cell recieves input]
            in_map_size (Tuple[int, int]): size of input maps; theoretically size of S-layer output maps feeding this layer
            out_map_size (Tuple[int, int]): size of output maps; theoretically number of C-cells per plane after pooling
            alpha (float, optional): Saturation constant. Defaults to 1.0.
            device (_type_, optional): system device. Defaults to Device.
        """
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.rf = rf
        self.in_map_size = in_map_size
        self.out_map_size = out_map_size
        self.alpha = alpha
        self.device = device
        
        # Fixed spatial weighting mask (d_l(v)), Gaussian-like
        center = (rf-1)/2.0
        coords = torch.stack(torch.meshgrid(torch.arange(rf), torch.arange(rf), indexing='ij'), dim=-1).float()
        dist = torch.sqrt(((coords[..., 0]-center)**2 + (coords[..., 1]-center)**2))
        sigma = rf/3.0
        mask = torch.exp(-(dist**2)/(2*sigma**2))
        self.spatial_mask = (mask / mask.sum()).view(1, 1, rf * rf).to(device)
        """
            Creates a 2D gaussian mask centered on receptive field, normalized so all weights sum=1
            
            Theoretically, 
                This is d_l(v), the efficacy of unmodifiable excitatory synapses from S to C layer
                Decreases with spatial distance |v| -> Nearby S-cells influence a C-cell more strongly
        """
        
        #Creates simple 1-1 connection between S and C planes i.e each C-plane recieves input from corresponding S-plane
        self.conn = torch.eye(min(in_planes, out_planes), device=self.device)
        """
            Theoretically, Each C plane corresponds to one S plane -> same feature type, just more tolerant to shifts
        """
        
    def forward(self, s_patches: torch.Tensor, stride: int = 1, padding: int = 0):
        """

        Args:
            s_patches (torch.Tensor): (B, in_planes * rf * rf, L)
        
        Returns:
            out: (B, out_planes, H_out, W_out)
            
        Description:
            Recieves flattened input patches from previous S-layer
            Each patch corresponds to a neighborhood of S-cells (rf)
            L = H_out*W_out -> no. of receptive field positions for C-cells
        """
        B, _, L = s_patches.shape
        in_planes = self.in_planes
        rf = self.rf
        #Reshape
        s_resh = s_patches.view(B, in_planes, rf * rf, L)
        weighted = (s_resh * self.spatial_mask.view(1, 1, rf * rf, 1)).sum(dim=2)
        """
            Multiply each local patch by the spatial Gaussian mask → weighted average
            Sum over spatial neighborhood → pooled feature per S-plane
            
            Theoretically:
                This is local averaging (pooling) within each S-plane’s receptive field.
        """
        comb = torch.einsum('op,bpl->bol', self.conn, weighted)
        """
            'op,bpl->bol':
                    o: output plane index
                    p: input plane index
                    l: spatial position index
            This applies the connection matrix between planes.
            Here, it’s identity → one-to-one mapping, but still flexible for multi-plane pooling.
            
            Theoretically,
            u'{C_l}(k_l, n) may be combination of several S-planes based on the connections
            Here: one-to-one → each C-plane pools from its corresponding S-plane.
        """
        
        out = comb / (1.0 + self.alpha * comb)
        """Saturation, Non Linearity : Prevents excessive activation
            Models the saturation characteristic of C-cells
            As input increases, output approaches 1/alpha
            For small input values, behaves nearly linearly.
            
            Theoretically:
                alpha controls how quickly response saturates
                    - large alpha : slow saturation (more sensitive)
                    - small alpha : fast saturation (less sensitive)
        """
        Hc = Wc = int(L**0.5)
        if Hc * Wc != L:  # non-square case
            Hc = L // Wc
            Wc = L // Hc
        out = out.view(B, self.out_planes, Hc, Wc)
        return out