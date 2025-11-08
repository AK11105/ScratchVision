import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscriminativeRBFLoss(nn.Module):
    """
    Discriminative RBF loss used in LeNet-5:
    E = d_correct + λ * log( exp(-margin) + Σ_j exp(-d_j) )

    where:
      - d_correct = squared distance to correct class centroid
      - d_j = squared distances to all centroids
      - λ controls influence of incorrect-class terms
    """

    def __init__(self, num_classes=10, feature_dim=84, margin=1.0, lambda_factor=0.02, trainable_centroids=False):
        super().__init__()
        self.margin = margin
        self.lambda_factor = lambda_factor

        centroids = torch.sign(torch.randn(num_classes, feature_dim))
        if trainable_centroids:
            self.centroids = nn.Parameter(centroids)
        else:
            self.register_buffer("centroids", centroids)

    def forward(self, features, targets):
        # features: (batch, feature_dim)
        # targets:  (batch,)
        dists = torch.cdist(features, self.centroids, p=2) ** 2  # (batch, num_classes)

        # distance to the correct class centroid
        d_correct = dists[torch.arange(features.size(0)), targets]

        # log-sum-exp for numerical stability
        # log(exp(-margin) + Σ exp(-d_j))
        log_term = torch.logsumexp(-dists, dim=1)
        log_term = torch.log(torch.exp(torch.tensor(-self.margin, device=log_term.device, dtype=log_term.dtype))+ torch.exp(log_term))

        # total loss
        loss = (d_correct + self.lambda_factor * log_term).mean()
        return loss
