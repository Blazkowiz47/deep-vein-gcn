import math
from logging import getLogger

import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter


class MagFace(Module):
    def __init__(self, config, log, **kwargs):
        super(MagFace, self).__init__()
        self.log = log
        self.name = "MagFace"
        self.num_classes = config["num_classes"]
        self.embedding_size = config["embedding_size"]
        self.scale = config.get("scale", 64.0)
        self.l_a = config.get("magface_l_a", 1.0)
        self.u_a = config.get("magface_u_a", 51.0)
        self.l_margin = config.get("magface_l_margin", 0.45)
        self.u_margin = config.get("magface_u_margin", 1.0)
        self.lambda_g = config.get("magface_lambda_g", 5.0)
        self.eps = 1e-6

        self.weight = Parameter(torch.empty(self.num_classes, self.embedding_size))
        torch.nn.init.xavier_uniform_(self.weight)

    def calc_margin(self, norms: torch.Tensor) -> torch.Tensor:
        clipped = norms.clamp(self.l_a, self.u_a)
        scale = (clipped - self.l_a) / (self.u_a - self.l_a + self.eps)
        return self.l_margin + scale * (self.u_margin - self.l_margin)

    def regularizer(self, norms: torch.Tensor) -> torch.Tensor:
        clipped = norms.clamp(self.l_a, self.u_a)
        return (1.0 / (self.u_a**2)) * clipped + 1.0 / clipped

    def forward(self, embds, labels, **kwargs):
        """Compute MagFace logits and magnitude regularization."""
        if labels.dim() == 2:
            labels = torch.argmax(labels, dim=1)

        norms = torch.norm(embds, p=2, dim=1, keepdim=True).clamp_min(self.eps)
        normalized_embds = embds / norms
        weight = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(normalized_embds, weight)
        cosine = cosine.clamp(-1.0 + self.eps, 1.0 - self.eps)

        margins = self.calc_margin(norms)
        theta = torch.acos(cosine)
        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        target_theta = torch.clamp(
            theta + one_hot * margins,
            min=self.eps,
            max=math.pi - self.eps,
        )
        logits = torch.cos(target_theta) * one_hot + cosine * (1.0 - one_hot)
        logits = logits * self.scale

        loss = F.cross_entropy(logits, labels)
        reg = self.lambda_g * self.regularizer(norms).mean()

        return loss, reg, logits


if __name__ == "__main__":
    config = {
        "num_classes": 301,
        "embedding_size": 512,
        "scale": 64.0,
    }
    log = getLogger("MagFace")
    log.setLevel("DEBUG")
    loss = MagFace(config, log)
    embds = torch.randn(10, 512)
    labels = torch.zeros(10, 301)
    labels[torch.arange(10), torch.arange(10)] = 1

    x1, x2, preds = loss(embds, labels)
    print(x1.shape, x2.shape, preds.shape)
