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
        self.weight = Parameter(
            torch.Tensor(config["embedding_size"], self.num_classes)
        ).to(self.device)
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul(1e5)
        self.scale = torch.tensor(config["scale"]).to(self.device)
        self.margin = torch.tensor(config["margin"]).to(self.device)
        self.beta = torch.tensor(config["beta"]).to(self.device)
        self.l_margin = torch.tensor(0.15).to(self.device)
        # self.l_a = config["l_a"]
        # self.u_a = config["u_a"]

    def margin_func(self, embds):
        return torch.norm(embds, p=2, dim=1)

    def forward(self, embds, labels):
        # xnorm = torch.norm(embds, p=2, dim=1).clamp(self.l_a, self.u_a)
        wnorm = F.normalize(self.weight, p=2, dim=0)
        embdsn = F.normalize(embds, p=2, dim=1)
        margin = self.margin_func(embds)
        margin = torch.clamp(margin, self.l_margin, self.margin)
        margin = margin.unsqueeze(1)
        margin = margin.repeat(1, self.num_classes)

        cos_theta = torch.matmul(embdsn, wnorm)
        cos_theta = cos_theta.clamp(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))

        cos_m, sin_m = torch.cos(margin), torch.sin(margin)

        cos_theta_m = cos_theta * cos_m - sin_theta * sin_m
        one_hot = labels
        output = (one_hot * cos_theta_m) + ((1.0 - one_hot) * cos_theta)

        output = self.scale * output

        loss1 = F.cross_entropy(output, labels.argmax(dim=1))
        loss2 = (
            torch.where(
                one_hot > 0.0,
                torch.acos(cos_theta),
                torch.zeros_like(cos_theta),
            )
            .sum(dim=1)
            .mean()
        )

        return loss1 + self.beta * loss2
