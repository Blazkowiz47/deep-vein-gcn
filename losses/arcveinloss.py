from typing import Any, Dict
from logging import Logger
import torch
from torch.nn import (
    Linear,
    Module,
    NLLLoss,
    Parameter,
    LogSoftmax,
)


class ArcCosineLoss(Module):
    def __init__(self, config: Dict[str, Any], log: Logger, **kwargs) -> None:
        super(ArcCosineLoss, self).__init__()
        self.log = log
        self.fine_tune = config.get("fine_tune", False)
        self.lmbda = config.get("lmbda", 0.01)
        self.softmax = LogSoftmax(dim=1)
        self.ls = NLLLoss()
        self.fc = Linear(512, config["num_classes"])
        self.centroids = Parameter(torch.randn((config["num_classes"], 512)))

    def forward(self, embds, label, freeze_centroids=False, **kwargs):
        sfmx = self.fc(embds)
        sfmx = self.softmax(sfmx)
        label = torch.argmax(label, dim=1)
        ls = self.ls(sfmx, label)

        if self.fine_tune:
            return ls, sfmx

        nembds = embds / torch.sqrt(torch.sum(embds**2, dim=1, keepdim=True))
        self.centroids.requires_grad = not freeze_centroids
        ncent = self.centroids / torch.sqrt(
            torch.sum(self.centroids**2, dim=1, keepdim=True)
        )
        ncent = ncent.transpose(0, 1)
        lac = torch.matmul(nembds, ncent)
        lac = torch.acos(lac)
        lac = torch.sum(lac)
        return ls, self.lmbda * lac
