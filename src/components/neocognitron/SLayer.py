import torch 
import torch.nn.functional as F
from typing import Tuple, List

from .patch_extract import patch_extract

Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Simple Cells ==> Effectively Feature detection ==> CONVOLUTIONAL LAYER
class SLayer:
    def __init__(self, in_channels: int, n_planes: int, rf: int, in_map_size: Tuple[int, int], out_map_size: Tuple[int, int], device=Device, init_scale=0.01):
        """S-layer: Feature extraction layer with shunting inhibition.

        Args:
            in_channels (int): Number of feature maps from previous layer; theoretically number of UC(l-1) maps
            n_planes (int): Number of feature types to detect; theoretically number of S planes in layer l
            rf (int): receptive field size; theoretically local region of previous layer connected to each S-cell
            in_map_size (int, int): spatial size of input feature map; theoretically size of set of feature maps from previous C-layer feature maps
            out_map_size (int, int): spatial size of output featur map; theoretically size of S-layer output maps after applying rf over input maps
            device (_type_, optional): operating device. Defaults to Device.
            init_scale (float, optional): _description_. Defaults to 0.01.
        """
        self.in_ch = in_channels
        self.n_planes = n_planes
        self.rf = rf
        self.in_map_size = in_map_size
        self.out_map_size = out_map_size
        self.device = device
        
        #Excitatory synapses a_l(k_l-1,v,k_l); each S-plane has its own vector of weights (Excitatory Efficiency Coefficients)
        self.weights =  torch.randn(n_planes, in_channels*rf*rf, device=device) * init_scale
        #Inhibitory Efficacy b_l(k_l); Controls how strongly inhibitory signals suppress excitation
        self.b =  torch.zeros(n_planes, device=device)
        # Shunting inhibition coefficient r_l ; Controls selectivity (high r ==> more selective)
        self.r = 1.0
        
    def forward(self, x_prev: torch.Tensor, stride: int = 1, padding: int = 0):
        """Forward pass: compute S-cell outputs

        Args:
            x_prev (torch.Tensor): (B, in_ch, H_in, W_in) ; Input (previous layer); theoretically this is U_C_l-1(k_l-1, n) if l>1 or Uo(n) if l=1
            stride (int, optional): step size. Defaults to 1.
            padding (int, optional): adding extra pixels to borders. Defaults to 0.
            
        Description: 
            x_prev is the output of previous C-layers or the input (if l=1); each channel corresponds to a "plane" from previous layer
        """
        B = x_prev.shape[0]
        # Extract local reigons of size rf*rf from input map
        patches = patch_extract(x_prev, kernel=self.rf, stride=stride, padding=padding)
        _, Ck, L = patches.shape
        
        #Weighted sum (excitatory synapses)
        proj = torch.einsum('pc, bcl->bpl', self.weights, patches)
        """
            torch.einsum: Einstein Summation => Way to specify tensor operations concisely using labels
            ``` torch.einsum('indices_input1, indices_input2 -> indices_output', tensor1, tensor2) ```
            
            self.weights -> shape is (p, c) i.e pc [p: no. of S planes (n_planes); c: no. of input channels * receptive field size]
            patches -> shape is (B, c, L) i.e bcl [B: batch_size; c: same as above; L: number of patches per image (no. of S-cells per plane {H_out*W_out})]
            
            Thus, 'pc,bcl->bpl'
                p: planes (from weights); become output plane dimension
                b: batch size preserved
                l: patch-index / spatial location preserved
                c: appears in both -> summed over (dot product)
            
            Intuition:
                Each S-plane detects a particular feature
                weights[p, :] => "template" for that feature
                einsum applies this template to every patch across all images efficiently
            
                This is like Conv2d 
            
            Theoretically:
                Each S-cell computes dot product between input patch and learnt excitatory synapses
                Same for all S-cells in the same plane (weights shared across positions).
                Produces output (B, n_planes, L) = excitation before inhibition.
                
                - Corresponds to numerator in S-cell equation
            
            Analogy:
                Each neuron checks how similar its preferred pattern (weights) is to the local patch of the image
        """
        
        #Shunting Inhibition
        patch_mag = torch.sqrt(torch.sum(patches*patches, dim=1)+1e-8)
        denom = 1.0 + self.r * (self.b.view(1,-1,1) * patch_mag.view(B,1,L))
        """Shunting Inhibition Term

            patch_mag: RMS energy of local inputs (how strong the overall input is)
            b and r: scale the inhibition
            Thus strong input ==> higher inhibition ==> lesser output
            
            denom: reduces the  S-cell's activation if input magnitude is large => Selective suppression of weak patterns
            
            Theoretically:
                Implements the shunting inhibition, making S-cells selective for strong feature matches

            Analogy:
                Like your brain ignoring very bright light: inhibition prevents the neurons from "firing too much" and keeps them
                focused on strong matches, not just strong intensity
        """
        #Implement final output
        out = F.relu(proj/denom)
        """
        Represents Q(x)
        ReLU mimics non negative firing rate: neurons can't have negative activtiy
        
        Analogy:
            The neuron only activates(fires) when it's input pattern matches strongly enough and isn't suppressed too much by inhibition
        """            
        
        #Reshape to spatial map
        H_out = W_out = int(L**0.5)
        if H_out * W_out != L:  # non-square case
            H_out = L // W_out
            W_out = L // H_out
        out = out.view(B, self.n_planes, H_out, W_out)
        """
        H_out x W_out: number of S-cells per plane i.e  spatial positions over which receptive fields are applied
        n_planes: number of S-planes detecting different features
        """
        return out, patches

    def reinforce(self, rep_positions: List[Tuple[int, int, torch.Tensor]], u_cl_prev: torch.Tensor, v_cl_prev: torch.Tensor, c_l_minus_1: torch.Tensor, q_l: float = 1.0):
        """Unsupervised Weight Update : self-organization rule
        Updates excitatory weights a_l and inhibitory efficacies b_l for "representative" S-cells

        Args:
            rep_positions (List[Tuple[int, int, torch.Tensor]]): List of representative neurons: (plane_idx, patch_pos, patch_vec)
                - plane_idx: which S-plane this representative belongs to (k̂_l)
                - patch_pos: linear index of S-cell in output map n̂
                - patch_vec: unfolded patch vector corresponding to input reigon (u_{C_{l-1}}(k_{l-1}, n̂ + v))
            u_cl_prev (torch.Tensor): output of previous C-layer U_{C_{l-1}}
            v_cl_prev (torch.Tensor): Inhibitory input V_{C_{l-1}}
            c_l_minus_1 (torch.Tensor): Fixed Spatial Gaussian/Distance based weighting mask
            q_l (float, optional): Learning rate(how quickly synapses are updated). Defaults to 1.0.
            
        Description:
            Implements Fukushima’s reinforcement rule:
            Δa_l(k_{l-1}, v, k̂_l) = q_l * c_{l-1}(v) * u_{C_{l-1}}(k_{l-1}, n̂ + v)
            Δb_l(k̂_l) = (q_l / 2) * v_{C_{l-1}}(n̂)
            rep_positions: list of (plane_idx, pos, patch_vec)
            u_cl_prev: (planes_prev, H, W)
            v_cl_prev: (H, W)
            c_l_minus_1: fixed spatial weighting mask (e.g., Gaussian decreasing with |v|)
        """
        rf = self.rf
        """
            This will multiply the input patch element wise reflecting spatial influence
                - closer to centre: stronger weight
                - farther from centre: weaker reinforcement
            
            Theoretically:
                corresponds to spatially decreasing contribution of S-cell’s input patch.
        """ 
        
        # Representative selection
        for (plane_idx, patch_pos, patch_vec) in rep_positions:
            """
                Only representative S-cells are reinforced — those that had strongest activation in their S-plane for this pattern.
                Analogy: Only neurons that “noticed” a pattern get rewarded.
            """
            in_ch = patch_vec.numel() // (rf*rf)
            c_mask = c_l_minus_1.view(1, rf*rf).repeat(in_ch, 1).view(-1).to(self.device)
            with torch.no_grad():
                assert patch_vec.numel() == c_mask.numel(), \
                f"Patch size {patch_vec.numel()} does not match mask size {c_mask.numel()}"
                delta_a = q_l * c_mask * patch_vec.to(self.device)
                self.weights[plane_idx] += delta_a
                """
                    For each representative cell: 
                        - Multiply input patch by c_mask -> weighted contribution (spatial weighting)
                        - Multiply by q_l -> reinforcement speed
                        - Update S-plane weights -> strengthening excitatory synapses
                    
                    Theoretically:
                        Reinforces the synapses of the S-plane that responded most strongly at the representative position.
                        Ensures that future presentations of similar input will excite this S-cell more strongly.
                        
                    Analogy:
                        Neuron saw a pattern it likes → adjust its internal template slightly to match that input better.
                """
                # For inhibitory update, take corresponding position’s v_cl_prev
                h, w = divmod(patch_pos, u_cl_prev.shape[-1])
                delta_b = (q_l / 2.0) * v_cl_prev[h, w]
                self.b[plane_idx] += delta_b
                """
                    Convert linear postion to 2D coordinates
                    Update inhibitory synapse (b_l) proportional to local inhibitory cell activity (v_cl_prev)
                    Multiplied by q_l / 2 → small adjustment
                    
                    Theoretically:
                        Shunting inhibition is adjusted based on V-cell activity → keeps S-cell selectivity in check.
                    
                    Analogy:
                        Neuron increases its self-damping slightly to prevent overreaction — like turning down the volume if it’s already too loud.
                """
                
            #Normalization
            with torch.no_grad():
                norms = self.weights.norm(dim=1, keepdim=True)
                self.weights /= (norms + 1e-8)
                
        