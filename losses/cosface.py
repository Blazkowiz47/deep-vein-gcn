from logging import getLogger

import torch
import torch.nn.functional as F
from torch.nn import Linear, Module, Parameter


class CosFace(Module):
    def __init__(self, config, log, **kwargs):
        super(CosFace, self).__init__()
        self.log = log
        self.name = "CosFace"
        self.num_classes = config["num_classes"]
        self.scale = config.get("scale", 64.0)
        self.margin = config.get("margin", 0.35)
        self.weight = Parameter(
            torch.empty(self.num_classes, config["embedding_size"])
        )
        self.fc = Linear(config["embedding_size"], self.num_classes)
        self.model_init()

    def model_init(self):
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.kaiming_normal_(self.fc.weight)
        torch.nn.init.zeros_(self.fc.bias)

    def forward(self, embds, labels, **kwargs):
        if labels.dim() == 2:
            labels = torch.argmax(labels, dim=1)

        embds = F.normalize(embds, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(embds, weight)
        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        logits = (cosine - one_hot * self.margin) * self.scale
        loss = F.cross_entropy(logits, labels)

        return loss, torch.tensor(0.0, device=logits.device), logits


if __name__ == "__main__":
    config = {
        "num_classes": 301,
        "embedding_size": 512,
        "scale": 64.0,
        "margin": 0.35,
    }
    log = getLogger("CosFace")
    log.setLevel("DEBUG")
    loss = CosFace(config, log)
    embds = torch.randn(10, 512)
    labels = torch.zeros(10, 301)
    for i in range(10):
        labels[i, i] = 1

    x1, x2, preds = loss(embds, labels)
    print(x1.shape, x2.shape, preds.shape)
