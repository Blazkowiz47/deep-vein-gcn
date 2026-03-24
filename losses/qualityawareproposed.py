from logging import getLogger

import torch
import torch.nn.functional as F
from torch.nn import Linear, Module, Parameter
from utils import compute_quality_components


class QualityAwareProposed(Module):
    def __init__(self, config, log, **kwargs):
        super(QualityAwareProposed, self).__init__()
        self.log = log
        self.name = "QualityAwareProposed"
        self.num_classes = config["num_classes"]
        self.device = config["device"]
        self.centroids = Parameter(
            torch.randn((config["embedding_size"], self.num_classes)),
        )
        self.centroids.data.uniform_(-1, 1).renorm_(2, 1, 1)

        self.beta = torch.tensor(config["beta"]).to(self.device)
        self.quality_dark_threshold = config.get("quality_dark_threshold", 0.45)
        self.quality_min_weight = config.get("quality_min_weight", 0.5)
        self.quality_max_weight = config.get("quality_max_weight", 1.5)
        self.margin = config.get("margin", 0.1)
        self.soft_hinge_temperature = config.get("soft_hinge_temperature", 0.1)
        self.fc = Linear(config["embedding_size"], self.num_classes)
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def compute_quality_weights(self, image_batch: torch.Tensor) -> torch.Tensor:
        quality, _, _, _, _ = compute_quality_components(
            image_batch,
            dark_threshold=self.quality_dark_threshold,
        )
        quality = quality / (quality.mean().detach() + 1e-6)
        return quality.clamp(self.quality_min_weight, self.quality_max_weight)

    def weighted_mean(
        self, losses: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        return (losses * weights).sum() / (weights.sum() + 1e-6)

    def forward(
        self,
        embds,
        labels,
        freeze_centroids=False,
        image_batch=None,
        **kwargs,
    ):
        """Compute quality-weighted classification and centroid losses.

        Args:
            embds: Batch embeddings of shape [B, D].
            labels: One-hot labels of shape [B, C].
            freeze_centroids: If True, keep centroid weights fixed for loss2.
            image_batch: Input images used to derive per-sample quality weights.

        Math:
            quality = compute_quality_components(image_batch)
            weights = clamp(quality / mean(quality), min_w, max_w)

            targets = argmax(labels, dim=1)
            preds = fc(embds)
            loss1 = weighted_mean(CE(preds, targets, reduction="none"), weights)

            emb_norm = normalize(embds)
            cent_norm = normalize(centroids)
            cos_theta = emb_norm @ cent_norm

            target_sim = cos_theta[b, targets[b]]
            hard_neg = tau * logsumexp(non_target_sim / tau)
            attraction = 1 - target_sim
            repulsion = softplus(hard_neg - target_sim + margin)
            loss2 = weighted_mean(attraction + repulsion, weights)

            total = loss1 + beta * loss2

        Returns:
            A tuple of:
            - loss1: Quality-weighted cross-entropy on the classifier head.
            - beta * loss2: Quality-weighted smooth attraction/repulsion
              penalty against normalized class centroids.
            - preds: Class logits from the classifier head.
        """
        if len(embds.shape) != 2:
            embds = embds.view(1, embds.size(0))
        if len(labels.shape) != 2:
            labels = labels.view(1, labels.size(0))

        targets = labels.argmax(dim=1)
        sample_weights = torch.ones(embds.shape[0], device=embds.device)
        if image_batch is not None:
            sample_weights = self.compute_quality_weights(image_batch)

        preds = self.fc(embds)
        centroids = self.centroids.detach() if freeze_centroids else self.centroids
        loss1 = self.weighted_mean(
            F.cross_entropy(preds, targets, reduction="none"),
            sample_weights,
        )
        wnorm = F.normalize(centroids, p=2, dim=0)
        emb_norm = F.normalize(embds, p=2, dim=1)
        cos_theta = torch.matmul(emb_norm, wnorm).clamp(-1, 1)
        target_sim = cos_theta.gather(1, targets.unsqueeze(1)).squeeze(1)
        neg_sim = cos_theta.masked_fill(
            F.one_hot(targets, self.num_classes).bool(),
            float("-inf"),
        )
        hard_neg = self.soft_hinge_temperature * torch.logsumexp(
            neg_sim / self.soft_hinge_temperature,
            dim=1,
        )
        attraction = 1.0 - target_sim
        repulsion = F.softplus(hard_neg - target_sim + self.margin)
        output = attraction + repulsion

        loss2 = self.weighted_mean(
            output,
            sample_weights,
        )

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
        "beta": 0.3,
    }
    log = getLogger("QualityAwareProposed")
    log.setLevel("DEBUG")
    loss = QualityAwareProposed(config, log)
    embds = torch.randn(10, 512)
    labels = torch.zeros(10, 301)
    labels[torch.arange(10), torch.arange(10)] = 1
    image_batch = torch.rand(10, 3, 224, 224)

    x1, x2, preds = loss(embds, labels, image_batch=image_batch)
    print(x1.shape, x2.shape, preds.shape)
