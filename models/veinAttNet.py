from logging import DEBUG, Logger, StreamHandler, getLogger
from typing import Any, Dict

import torch
from torch.nn import (
    AvgPool2d,
    Conv2d,
    GroupNorm,
    LayerNorm,
    Linear,
    MaxPool2d,
    Module,
    MultiheadAttention,
    ReLU,
    Sequential,
)


class VeinAttNet(Module):
    def __init__(self, config: Dict[str, Any], log: Logger, **kwargs):
        super(VeinAttNet, self).__init__()
        self.name = "VeinAttNet"
        self.config = config
        self.log = log
        self.kwargs: Dict[str, Any] = kwargs
        self.log.debug("Initialised VeinAttNet model.")
        self.backbone = Sequential(
            Conv2d(3, 32, 7, 2, (7 - 1) // 2),
            GroupNorm(32, 32),
            ReLU(),
            MaxPool2d(3, 2),
            Conv2d(32, 32, 5, 2, (5 - 1) // 2),
            GroupNorm(32, 32),
            ReLU(),
            MaxPool2d(3, 2),
            Conv2d(32, 32, 3, 2, (3 - 1) // 2),
            GroupNorm(32, 32),
            ReLU(),
            MaxPool2d(3, 2),
            AvgPool2d(3),
        )
        self.queryenc = Linear(1, 64)
        self.keyenc = Linear(1, 64)
        self.valueenc = Linear(1, 64)
        self.attn = MultiheadAttention(
            64,
            4,
            batch_first=True,
        )
        self.enc = Linear(64, 1)
        self.head = LayerNorm(32)
        self.fc = Linear(32, config["num_classes"])

    def forward(self, x, **kwargs):
        x = self.backbone(x)
        x = x.view(x.size(0), x.size(1), -1)
        query = self.queryenc(x)
        key = self.keyenc(x)
        value = self.valueenc(x)
        out, _ = self.attn(query, key, value)
        out = self.enc(out)
        out = out.view(out.size(0), -1)
        out = self.head(out)
        out = self.fc(out)

        return out


if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    logger = getLogger("LFEM")
    logger.setLevel(DEBUG)
    logger.addHandler(StreamHandler())
    logger.info(f"Input shape: {x.shape}")
    model = VeinAttNet({"num_classes": 301}, logger)
    out = model(x)
    logger.info(f"VeinAttNet output:  {out.shape}")
