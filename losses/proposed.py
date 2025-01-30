import torch
from torch.nn import Module, Parameter
import torch.nn.functional as F


class Proposed(Module):
    def __init__(self, config, log, **kwargs):
        super(Proposed, self).__init__()
        self.log = log
        self.name = "Proposed Loss"
        self.num_classes = config["num_classes"]
        self.weight = Parameter(
            torch.Tensor(config["embedding_size"], self.num_classes)
        ).to(config["device"])
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul(1e5)
        self.scale = torch.tensor(config["scale"]).to(config["device"])
        self.margin = torch.tensor(config["margin"]).to(config["device"])
        # self.l_a = config["l_a"]
        # self.u_a = config["u_a"]

    def margin_func(self, x, y):
        return torch.nn.functional.cross_entropy(x, y)

    def forward(self, embds, labels):
        # xnorm = torch.norm(embds, p=2, dim=1).clamp(self.l_a, self.u_a)
        wnorm = F.normalize(self.weight, p=2, dim=0)
        cos_theta = torch.matmul(embds, wnorm)
        cos_theta = cos_theta.clamp(-1, 1)
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
        cos_m, sin_m = torch.cos(self.margin), torch.sin(self.margin)
        cos_theta_m = cos_theta * cos_m - sin_theta * sin_m
        one_hot = labels
        output = (one_hot * cos_theta_m) + ((1.0 - one_hot) * cos_theta)
        output = self.scale * output
        loss = F.cross_entropy(output, labels.argmax(dim=1))

        return loss
