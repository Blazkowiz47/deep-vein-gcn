import math
from logging import getLogger

import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter

from utils import compute_quality_components


class AdaFaceQ(Module):
    def __init__(self, config, log, **kwargs):
        super(AdaFaceQ, self).__init__()
        self.log = log
        self.name = "AdaFaceQ"
        self.num_classes = config["num_classes"]
        self.embedding_size = config["embedding_size"]
        self.margin = config.get("margin", 0.4)
        self.h = config.get("h", config.get("adaface_h", 0.333))
        self.scale = config.get("scale", 64.0)
        self.t_alpha = config.get("t_alpha", config.get("adaface_t_alpha", 0.01))
        self.eps = 1e-3
        self.quality_dark_threshold = config.get("quality_dark_threshold", 0.45)

        self.kernel = Parameter(torch.empty(self.embedding_size, self.num_classes))
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

        # Quality lives on a much smaller scale than feature norms, so the
        # running statistics need small initial values.
        self.register_buffer("batch_mean", torch.tensor(0.5))
        self.register_buffer("batch_std", torch.tensor(0.1))

    def compute_quality(self, image_batch, embds):
        if image_batch is None:
            return torch.norm(embds, p=2, dim=1, keepdim=True).detach()

        quality, _, _, _, _ = compute_quality_components(
            image_batch,
            dark_threshold=self.quality_dark_threshold,
        )
        return quality.unsqueeze(1).detach()

    def forward(self, embds, labels, image_batch=None, **kwargs):
        """Compute AdaFace-style adaptive-margin logits using vein quality.

        Math:
            q = vein_quality(image_batch)
            gamma = clip(h * (q - mu) / (sigma + eps), -1, 1)

            x_hat = normalize(embds)
            w_hat = normalize(kernel)
            cos_theta = x_hat @ w_hat

            theta_m = acos(cos_theta) - m * gamma
            cos_theta = cos(theta_m) - (m + m * gamma)
            logits = s * cos_theta

        Returns:
            A tuple of:
            - loss: Cross-entropy on adaptive-margin logits.
            - 0.0: Auxiliary loss placeholder for the training interface.
            - logits: Margin-adjusted class logits.
        """
        if labels.dim() == 2:
            labels = torch.argmax(labels, dim=1)

        norms = torch.norm(embds, p=2, dim=1, keepdim=True).clamp_min(self.eps)
        normalized_embds = embds / norms
        kernel_norm = F.normalize(self.kernel, p=2, dim=0)
        cosine = torch.mm(normalized_embds, kernel_norm)
        cosine = cosine.clamp(-1.0 + self.eps, 1.0 - self.eps)

        safe_quality = self.compute_quality(image_batch, embds).clamp(
            min=self.eps,
            max=100.0,
        )
        with torch.no_grad():
            mean = safe_quality.mean().detach()
            std = safe_quality.std(unbiased=False).detach()
            self.batch_mean.mul_(1 - self.t_alpha).add_(mean * self.t_alpha)
            self.batch_std.mul_(1 - self.t_alpha).add_(std * self.t_alpha)

        margin_scaler = (safe_quality - self.batch_mean) / (self.batch_std + self.eps)
        margin_scaler = torch.clamp(margin_scaler * self.h, -1.0, 1.0)

        m_arc = torch.zeros_like(cosine)
        m_arc.scatter_(1, labels.view(-1, 1), 1.0)
        g_angular = -self.margin * margin_scaler
        theta = torch.acos(cosine)
        theta_m = torch.clamp(
            theta + (m_arc * g_angular),
            min=self.eps,
            max=math.pi - self.eps,
        )
        cosine = torch.cos(theta_m)

        m_cos = torch.zeros_like(cosine)
        m_cos.scatter_(1, labels.view(-1, 1), 1.0)
        g_add = self.margin + (self.margin * margin_scaler)
        logits = (cosine - (m_cos * g_add)) * self.scale
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
    log = getLogger("AdaFaceQ")
    log.setLevel("DEBUG")
    loss = AdaFaceQ(config, log)
    embds = torch.randn(10, 512)
    labels = torch.zeros(10, 301)
    labels[torch.arange(10), torch.arange(10)] = 1
    image_batch = torch.rand(10, 3, 224, 224)

    x1, x2, preds = loss(embds, labels, image_batch=image_batch)
    print(x1.shape, x2.shape, preds.shape)
