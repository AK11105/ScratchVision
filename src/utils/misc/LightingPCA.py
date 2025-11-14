import torch 

class LightingPCA(object):
    def __init__(self, alpha_std=0.1):
        self.alpha_std = alpha_std
        self.eigvals = torch.tensor([0.2175, 0.0188, 0.0045])
        self.eigvecs = torch.tensor([
            [-0.5675,  0.7192,  0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948,  0.4203]
        ])

    def __call__(self, img):
        """
        img: Tensor assumed to be shape (C, H, W), float in [0, 1].
        """
        if self.alpha_std == 0:
            return img

        # Sample random alpha from N(0, 0.1)
        alpha = torch.normal(mean=0.0, std=self.alpha_std, size=(3,))

        # Compute RGB noise
        rgb = (self.eigvecs @ (alpha * self.eigvals)).view(3, 1, 1)

        return img + rgb