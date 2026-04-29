from logging import Logger, StreamHandler, getLogger, DEBUG
from typing import Any, Dict
import torch
import torch.nn.functional as F
from torch.nn import Module, Linear, Identity
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights


class Resnet(Module):
    def __init__(self, config: Dict[str, Any], log: Logger, **kwargs):
        super(Resnet, self).__init__()
        self.name = "resnet"
        self.config = config
        self.log = log
        self.kwargs: Dict[str, Any] = kwargs
        self.log.debug("Initialised resnet model.")
        backbone = self.config.get("backbone", "resnet50")
        if backbone == "resnet18":
            self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
            self.backbone.fc = Identity()
        elif backbone == "resnet50":
            self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
            self.backbone.fc = Linear(512 * 4, 512)
        else:
            raise ValueError(f"Unsupported resnet backbone: {backbone}")

    def forward(self, x, **kwargs):
        return F.normalize(self.backbone(x), p=2, dim=1)


if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    logger = getLogger("LFEM")
    logger.setLevel(DEBUG)
    logger.addHandler(StreamHandler())
    logger.info("Testing Resnet")
    model = Resnet({"num_classes": 301}, logger)
    y = model(x, features=True)
    logger.info(f"Output shape: {y.shape}")
