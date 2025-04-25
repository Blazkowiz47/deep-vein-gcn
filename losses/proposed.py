from logging import getLogger
import torch
from torch.nn import Linear, Module, Parameter
import torch.nn.functional as F


class Proposed(Module):
    def __init__(self, config, log, **kwargs):
        super(Proposed, self).__init__()
        self.log = log
        self.name = "Proposed Loss"
        self.num_classes = config["num_classes"]
        self.device = config["device"]
        self.centroids = Parameter(
            torch.randn((config["embedding_size"], self.num_classes)),
        )
        self.centroids.data.uniform_(-1, 1).renorm_(2, 1, 1)

        self.beta = torch.tensor(config["beta"]).to(self.device)

        self.fc = torch.nn.Linear(config["embedding_size"], self.num_classes)
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def margin_func(self, embds):
        return torch.norm(embds, p=2, dim=1)

    def forward(self, embds, labels, freeze_centroids=False):
        # xnorm = torch.norm(embds, p=2, dim=1).clamp(self.l_a, self.u_a)
        if len(embds.shape) != 2:
            embds = embds.view(1, embds.size(0))
        if len(labels.shape) != 2:
            labels = labels.view(1, labels.size(0))

        preds = self.fc(embds)
        loss1 = F.cross_entropy(preds, labels, reduction="mean")
        wnorm = F.normalize(self.centroids, p=2, dim=0)
        emb_norm = F.normalize(embds, p=2, dim=1)
        self.centroids.requires_grad = not freeze_centroids

        if freeze_centroids:
            with torch.no_grad():
                cos_theta = torch.matmul(emb_norm, wnorm)
                cos_theta = cos_theta.clamp(-1, 1)
                output = torch.acos(cos_theta)
                loss2 = F.cross_entropy(output, labels, reduction="mean")
        else:
            cos_theta = torch.matmul(emb_norm, wnorm)
            cos_theta = cos_theta.clamp(-1, 1)
            output = torch.acos(cos_theta)
            loss2 = F.cross_entropy(output, labels, reduction="mean")

        # self.log.info(f"Loss1: {loss1}, Loss2: {loss2}")
        if torch.isnan(loss1):
            self.log.error("Loss1 is NaN")
            exit(1)

        if torch.isnan(loss2):
            self.log.error("Loss2 is NaN")
            exit(1)
        return loss1, self.beta * loss2, preds


if __name__ == "__main__":
    config = {
        "num_classes": 301,
        "device": "cpu",
        "embedding_size": 512,
        "scale": 64,
        "margin": 0.5,
        "beta": 0.1,
    }
    log = getLogger("Proposed")
    log.setLevel("DEBUG")
    loss = Proposed(config, log)
    embds = torch.randn(10, 512)
    labels = torch.zeros(10, 301)
    for i in range(10):
        labels[i, i] = 1

    x1, x2 = loss(embds, labels)
    print(x1.shape, x2.shape)
