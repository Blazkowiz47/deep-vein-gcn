import math
from logging import getLogger

import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter


class AdaFace(Module):
    def __init__(self, config, log, **kwargs):
        super(AdaFace, self).__init__()
        self.log = log
        self.name = "AdaFace"
        self.num_classes = config["num_classes"]
        self.embedding_size = config["embedding_size"]
        self.margin = config.get("margin", 0.4)
        self.h = config.get("h", config.get("adaface_h", 0.333))
        self.scale = config.get("scale", 64.0)
        self.t_alpha = config.get("t_alpha", config.get("adaface_t_alpha", 0.01))
        self.eps = 1e-3

        self.kernel = Parameter(torch.empty(self.embedding_size, self.num_classes))
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

        self.register_buffer("batch_mean", torch.tensor(20.0))
        self.register_buffer("batch_std", torch.tensor(100.0))

    def forward(self, embds, labels, **kwargs):
        """Compute AdaFace logits with norm-based adaptive margins."""
        if labels.dim() == 2:
            labels = torch.argmax(labels, dim=1)

        norms = torch.norm(embds, p=2, dim=1, keepdim=True).clamp_min(self.eps)
        normalized_embds = embds / norms
        kernel_norm = F.normalize(self.kernel, p=2, dim=0)
        cosine = torch.mm(normalized_embds, kernel_norm)
        cosine = cosine.clamp(-1.0 + self.eps, 1.0 - self.eps)

        safe_norms = norms.detach().clamp(min=self.eps, max=100.0)
        with torch.no_grad():
            mean = safe_norms.mean()
            std = safe_norms.std(unbiased=False)
            self.batch_mean.mul_(1 - self.t_alpha).add_(mean * self.t_alpha)
            self.batch_std.mul_(1 - self.t_alpha).add_(std * self.t_alpha)

        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std + self.eps)
        margin_scaler = torch.clamp(margin_scaler * self.h, -1.0, 1.0)

        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()

        g_angular = -self.margin * margin_scaler
        theta = torch.acos(cosine)
        theta_m = torch.clamp(
            theta + (one_hot * g_angular),
            min=self.eps,
            max=math.pi - self.eps,
        )
        cosine = torch.cos(theta_m)

        g_add = self.margin + (self.margin * margin_scaler)
        logits = (cosine - (one_hot * g_add)) * self.scale
        loss = F.cross_entropy(logits, labels)

        return loss, torch.tensor(0.0, device=logits.device), logits


if __name__ == "__main__":
    config = {
        "num_classes": 301,
        "embedding_size": 512,
        "margin": 0.4,
        "scale": 64.0,
        "h": 0.333,
        "t_alpha": 0.01,
    }
    log = getLogger("AdaFace")
    log.setLevel("DEBUG")
    loss = AdaFace(config, log)
    embds = torch.randn(10, 512)
    labels = torch.zeros(10, 301)
    labels[torch.arange(10), torch.arange(10)] = 1

    x1, x2, preds = loss(embds, labels)
    print(x1.shape, x2.shape, preds.shape)
