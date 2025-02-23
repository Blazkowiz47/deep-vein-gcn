from logging import getLogger
import torch
from torch.nn import Module, Parameter
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
        self.scale = torch.tensor(config["scale"]).to(self.device)
        self.margin = torch.tensor(config["margin"]).to(self.device)
        self.beta = torch.tensor(config["beta"]).to(self.device)
        self.l_margin = torch.tensor(0.15).to(self.device)

        self.fc = torch.nn.Linear(config["embedding_size"], self.num_classes)
        self.model_init()
        # self.l_a = config["l_a"]
        # self.u_a = config["u_a"]

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def margin_func(self, embds):
        return torch.norm(embds, p=2, dim=1)

    def forward(self, embds, labels, freeze_centroids=False):
        # xnorm = torch.norm(embds, p=2, dim=1).clamp(self.l_a, self.u_a)
        preds = self.fc(embds)

        self.centroids.requires_grad = not freeze_centroids

        wnorm = F.normalize(self.centroids, p=2, dim=0)
        emb_norm = F.normalize(embds, p=2, dim=1)

        margin = self.margin_func(embds)
        margin = torch.clamp(margin, self.l_margin, self.margin)
        margin = margin.unsqueeze(1)
        margin = margin.repeat(1, self.num_classes)

        cos_theta = torch.matmul(emb_norm, wnorm)
        cos_theta = cos_theta.clamp(-1, 1)
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))

        cos_m, sin_m = torch.cos(margin), torch.sin(margin)
        cos_theta_m = cos_theta * cos_m - sin_theta * sin_m

        one_hot = labels
        output = (one_hot * cos_theta_m) + ((1.0 - one_hot) * cos_theta)

        output = self.scale * output

        loss2 = F.cross_entropy(output, labels, reduction="mean")

        loss1 = F.cross_entropy(preds, labels, reduction="mean")
        # self.log.info(f"Loss1: {loss1}, Loss2: {loss2}")
        if torch.isnan(loss1):
            self.log.error("Loss1 is NaN")
            exit(1)

        if torch.isnan(loss2):
            self.log.error("Loss2 is NaN")
            exit(1)
        return loss1, self.beta * loss2


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
