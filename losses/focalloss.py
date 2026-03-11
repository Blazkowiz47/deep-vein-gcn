import torch
import torch.nn as nn
import torch.nn.functional as F
from logging import getLogger


def focal_loss(logits, targets, alpha=1.0, gamma=2.0, reduction="mean"):
    """
    Focal Loss implementation

    Args:
        logits: Predicted logits (N, C) where C is the number of classes
        targets: Ground truth labels (N,) or (N, C) for one-hot encoding
        alpha: Weighting factor for rare class (default: 1.0)
        gamma: Focusing parameter (default: 2.0)
        reduction: Reduction method ('none', 'mean', 'sum')

    Returns:
        Focal loss value
    """
    # Convert targets to one-hot if needed
    if targets.dim() == 1:
        targets = F.one_hot(targets, num_classes=logits.size(-1)).float()

    # Apply softmax to get probabilities
    probs = F.softmax(logits, dim=-1)

    # Get the probability of the correct class
    pt = (targets * probs).sum(dim=-1)

    # Calculate focal loss
    focal_weight = (1 - pt) ** gamma
    loss = -alpha * focal_weight * torch.log(pt + 1e-7)

    # Apply reduction
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        raise ValueError(f"Reduction '{reduction}' not supported")


class FocalLoss(nn.Module):
    def __init__(self, config, log, **kwargs):
        super(FocalLoss, self).__init__()
        self.log = log
        self.name = "FocalLoss"
        self.num_classes = config["num_classes"]
        self.fc = nn.Linear(config["embedding_size"], self.num_classes)

        # Initialize focal loss with configurable parameters
        self.alpha = config.get("focal_alpha", 1.0)
        self.gamma = config.get("focal_gamma", 2.0)
        self.reduction = config.get("focal_reduction", "mean")

    def forward(self, embds, labels, **kwargs):
        logits = self.fc(embds)
        self.log.debug(f"Logits shape: {logits.shape}, Labels shape: {labels.shape}")
        loss = focal_loss(logits, labels, self.alpha, self.gamma, self.reduction)
        return loss, torch.tensor(0.0).to(logits.device), logits


if __name__ == "__main__":
    # Test the focal loss implementation
    config = {"num_classes": 10, "focal_alpha": 1.0, "focal_gamma": 2.0}
    log = getLogger("FocalLoss")
    log.setLevel("DEBUG")

    # Test with the wrapper
    loss_wrapper = FocalLoss(config, log)
    embds = torch.randn(10, 512)
    labels = torch.randint(0, 10, (10,))

    x1, x2 = loss_wrapper(embds, labels)
    print(f"Loss shape: {x1.shape}, Aux shape: {x2.shape}")
    print(f"Loss value: {x1.item()}")

    # Test the focal loss function directly
    logits = torch.randn(10, 10)
    targets = torch.randint(0, 10, (10,))

    loss_value = focal_loss(logits, targets)
    print(f"Direct focal loss: {loss_value.item()}")

    # Test with one-hot targets
    targets_onehot = F.one_hot(targets, num_classes=10).float()
    loss_value_onehot = focal_loss(logits, targets_onehot)
    print(f"Focal loss with one-hot: {loss_value_onehot.item()}")
