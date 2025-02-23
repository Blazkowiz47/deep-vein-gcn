"""
This file is re-implementation of `https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10887340`
"""

from logging import DEBUG, Logger, StreamHandler, getLogger
from typing import Any, Dict
import torch
from torch.nn import AvgPool2d, Conv2d, MaxPool2d, Module, Sequential, Sigmoid


class LFEM(Module):
    def __init__(self, config: Dict[str, Any], log: Logger, **kwargs):
        """
        Local Feaure Extraction Module.
        """
        super(LFEM, self).__init__()
        self.name = "LFEM"
        self.config = config
        self.log = log
        self.kwargs: Dict[str, Any] = kwargs
        self.log.debug("Initialised LFEM model.")

        self.branch1 = Sequential(
            Conv2d(config["indims"], config["indims"], 1, 1),
            Conv2d(
                config["indims"],
                config["indims"],
                3,
                1,
                padding=(3 - 1) // 2,
                groups=config["indims"],
            ),
        )
        self.branch2 = Sequential(
            Conv2d(config["indims"], config["indims"], 1, 1),
            Conv2d(
                config["indims"],
                config["indims"],
                5,
                1,
                padding=(5 - 1) // 2,
                groups=config["indims"],
            ),
        )
        self.branch3 = Sequential(
            Conv2d(config["indims"], config["indims"], 1, 1),
        )
        self.msfuser = Conv2d(config["indims"] * 3, config["indims"], 1, 1)
        self.avgpool = AvgPool2d(3, 1, (3 - 1) // 2)
        self.maxpool = MaxPool2d(3, 1, (3 - 1) // 2)
        self.poolfuser = Conv2d(config["indims"] * 4, config["indims"], 1, 1)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        x1, x2, x3 = self.branch1(x), self.branch2(x), self.branch3(x)
        x_b = torch.cat([x1, x2, x3], dim=1)
        msf = self.msfuser(x_b)
        msf_in = torch.cat([x, msf], dim=1)
        pooled = torch.cat([self.avgpool(msf_in), self.maxpool(msf_in)], dim=1)
        fusedpool = self.poolfuser(pooled)
        sigpool = self.sigmoid(fusedpool)
        assert sigpool.shape == x.shape, f"Shape mismatch: {sigpool.shape} != {x.shape}"
        assert (
            sigpool.shape == msf.shape
        ), f"Shape mismatch: {sigpool.shape} != {msf.shape}"
        return sigpool * x + sigpool * msf


class LGFIN(Module):
    def __init__(self, config: Dict[str, Any], log: Logger, **kwargs):
        super(LGFIN, self).__init__()
        self.name = "LGFIN"
        self.config = config
        self.log = log
        self.kwargs: Dict[str, Any] = kwargs
        self.log.debug("Initialised LGFIN model.")

    def forward(self, x):
        raise NotImplementedError()


if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    logger = getLogger("LFEM")
    logger.setLevel(DEBUG)
    logger.addHandler(StreamHandler())
    logger.info(f"Input shape: {x.shape}")
    model = LFEM({"indims": 3}, logger)
    out = model(x)
    logger.info(f"LFEM output:  {out.shape}")
