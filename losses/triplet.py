from logging import getLogger
from typing import Tuple
import torch
from torch.nn import Linear, Module, Parameter
import torch.nn.functional as F
import torch.nn as nn


def batch_hard_triplet_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.3,
    squared: bool = False,
) -> torch.Tensor:
    """
    Batch-hard triplet loss (Hermans et al., 2017).

    For each anchor in the batch, picks the HARDEST positive (same label,
    farthest) and the HARDEST negative (different label, closest), then
    applies the triplet margin.

    Args:
        embeddings: (B, D) tensor. Assumed L2-normalized if you want
            distances bounded in [0, 2].
        labels:     (B,) integer class ids.
        margin:     triplet margin (paper uses 0.3).
        squared:    if True use squared Euclidean distance (slightly more
            stable gradients near 0); if False use Euclidean.

    Returns:
        Scalar loss.
    """
    # Pairwise squared Euclidean distances: ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    dot = embeddings @ embeddings.t()
    sq_norms = torch.diagonal(dot)
    dist_sq = sq_norms.unsqueeze(0) + sq_norms.unsqueeze(1) - 2.0 * dot
    dist_sq = dist_sq.clamp(min=0.0)  # numerical floor

    if squared:
        dist = dist_sq
    else:
        # add tiny epsilon where exactly zero so sqrt has a gradient
        mask_zero = dist_sq.eq(0).float()
        dist = torch.sqrt(dist_sq + mask_zero * 1e-16)
        dist = dist * (1.0 - mask_zero)  # restore exact zeros on the diagonal

    # Masks
    labels_eq = labels.unsqueeze(0).eq(labels.unsqueeze(1))  # (B, B) same class
    eye = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
    pos_mask = labels_eq & ~eye  # same class, not self
    neg_mask = ~labels_eq  # different class

    # Hardest positive: max distance among same-class pairs (per row)
    # Replace invalid entries with -inf so they lose the max.
    dist_pos = dist.masked_fill(~pos_mask, float("-inf"))
    hardest_pos, _ = dist_pos.max(dim=1)

    # Hardest negative: min distance among different-class pairs (per row).
    # Replace invalid entries with +inf so they lose the min.
    dist_neg = dist.masked_fill(~neg_mask, float("inf"))
    hardest_neg, _ = dist_neg.min(dim=1)

    # Some anchors may have no positives (shouldn't happen with PK sampler
    # where K >= 2, but guard anyway).
    valid = torch.isfinite(hardest_pos) & torch.isfinite(hardest_neg)
    if valid.sum() == 0:
        return embeddings.sum() * 0.0  # zero loss with grad path preserved

    losses = F.relu(hardest_pos[valid] - hardest_neg[valid] + margin)
    return losses.mean()


class CenterLoss(nn.Module):
    """
    Center loss (Wen et al., ECCV 2016), adapted for L2-normalized embeddings.

    Maintains a learnable center per class. Loss = (1/2) * mean ||z_i - c_{y_i}||^2.

    Notes for the hyperspherical setting (paper uses L2-normalized 512-d
    embeddings): we re-normalize the centers in the forward pass so the
    pull is along the sphere rather than off it. If you'd rather keep the
    classical formulation, set `normalize_centers=False`.
    """

    def __init__(
        self, num_classes: int, feat_dim: int = 512, normalize_centers: bool = True
    ):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.normalize_centers = normalize_centers
        # Init on the sphere so the early-training distance scale is sane.
        centers = torch.randn(num_classes, feat_dim)
        centers = F.normalize(centers, p=2, dim=1)
        self.centers = nn.Parameter(centers)

    def forward(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # embeddings expected L2-normalized
        centers = self.centers
        if self.normalize_centers:
            centers = F.normalize(centers, p=2, dim=1)

        c = centers[labels]  # (B, D) center for each sample
        loss = 0.5 * (embeddings - c).pow(2).sum(dim=1).mean()
        preds = embeddings @ centers.T
        return loss, preds


class Triplet(Module):
    def __init__(self, config, log, **kwargs):
        super(Triplet, self).__init__()
        self.log = log
        self.name = "Triplet Loss"
        self.num_classes = config["num_classes"]
        self.device = config["device"]
        self.margin = 0.3  # In Chen et al.
        self.lambda_center = 1e-3  # In Wen et al
        self.squared = False
        self.center_loss = CenterLoss(
            num_classes=self.num_classes,
            feat_dim=512,  # In Chen et al.
            normalize_centers=True,
        )
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, embds, labels, freeze_centroids=False, **kwargs):
        if labels.dim() == 2:
            labels = labels.argmax(dim=1)
        labels = labels.long()
        l_tri = batch_hard_triplet_loss(
            embds, labels, margin=self.margin, squared=self.squared
        )
        l_cen, preds = self.center_loss(embds, labels)
        return l_tri, self.lambda_center * l_cen, preds


if __name__ == "__main__":
    config = {
        "num_classes": 301,
        "device": "cpu",
    }
    log = getLogger("Proposed")
    log.setLevel("DEBUG")
    loss = Triplet(config, log)
    embds = torch.randn(10, 512)
    labels = torch.zeros(10, 301)
    for i in range(10):
        labels[i, i] = 1

    x1, x2, preds = loss(embds, labels)
    print(x1.shape, x2.shape, preds.shape)
