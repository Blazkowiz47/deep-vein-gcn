import math
from logging import Logger
from typing import Any, Dict, List

import torch
import torch.nn as nn
from torch.nn import AdaptiveAvgPool2d, BatchNorm2d, Conv2d, Module, Sequential
from utils.dscnet.S3_DSConv_pro import DSConv_pro
from utils.gcn_lib.torch_nn import act_layer
from utils.gcn_lib.torch_vertex import Grapher


class DSConv(Module):
    def __init__(
        self,
        indims: int,
        outdims: int,
        kernel: int = 3,
        stride: int = 1,
        bias: bool = True,
        device: str = "cpu",
        log: Logger = None,  # type: ignore
    ) -> None:
        super(DSConv, self).__init__()
        self.indims = indims
        self.outdims = outdims
        self.kernel = kernel
        self.stride = stride

        self.name = "DSConv"
        self.log = log

        self.conv = Conv2d(
            indims,
            outdims,
            kernel,
            stride=1,
            padding=math.ceil(kernel / 2) - 1,
            bias=bias,
        )
        self.xdsc = DSConv_pro(
            indims,
            outdims,
            kernel_size=kernel * kernel,
            morph=0,
            device=device,
        )
        self.ydsc = DSConv_pro(
            indims,
            outdims,
            kernel_size=kernel * kernel,
            morph=1,
            device=device,
        )

        self.enc = Conv2d(
            outdims * 3,
            outdims,
            kernel,
            stride=stride,
            padding=math.ceil(kernel / 2) - 1,
            bias=bias,
        )

    def __str__(self):
        return (
            f"{self.name}({self.indims}, {self.outdims}, {self.kernel}, {self.stride})"
        )

    def forward(self, inputs):
        c = self.conv(inputs)
        x = self.xdsc(inputs)
        y = self.ydsc(inputs)

        self.log.debug("DSConv")
        self.log.debug(f"Conv: {c.shape}, X: {x.shape}, Y: {y.shape}")
        return self.enc(torch.cat([c, x, y], dim=1))


class DeepVein(Module):
    def __init__(self, config: Dict[str, Any], log: Logger, **kwargs):
        super(DeepVein, self).__init__()
        self.name = "DeepVein"
        self.config = config
        self.log = log
        self.kwargs: Dict[str, Any] = kwargs
        self.log.debug("Initialised deepvein model.")
        layers: List[Module] = []
        width, height = config.get("width", 224), config.get("height", 224)
        layers.extend(
            [
                Conv2d(
                    config["indim_0"],
                    config["outdim_0"],
                    kernel_size=config["kernel_0"],
                    stride=config["stride_0"],
                    padding=math.ceil(config["kernel_0"] / 2) - 1,
                    bias=config.get("bias", True),
                ),
                BatchNorm2d(config["outdim_0"]),
                act_layer(config.get("act", "gelu")),
            ]
        )

        width, height = width // 2, height // 2
        for blockid, params in config.get("blocks", {}).items():
            layers.extend(
                [
                    DSConv(
                        params["indim"],
                        params["outdim"],
                        kernel=params["kernel"],
                        stride=params["stride"],
                        bias=config.get("bias", True),
                        device=config.get("device", "cpu"),
                        log=self.log,
                    ),
                    BatchNorm2d(params["outdim"]),
                    act_layer(config.get("act", "gelu")),
                    *[
                        Grapher(
                            params["outdim"],
                            kernel_size=min(
                                params.get("knn", 9), width * height
                            ),  # Ensures that the 'k' neighbours are not greater than
                            #     the number of pixels
                            act=config.get("act", "gelu"),
                            conv=config.get("conv", "mr"),
                            norm=config.get("norm", "batch"),
                            epsilon=config.get("epsilon", 0.2),
                            drop_path=config.get("drop_path", 0.0),
                            bias=config.get("bias", True),
                            r=params["reduce_ratio"],
                            n=height * width,
                        )
                        for _ in range(params["graphers"])
                    ],
                    Conv2d(
                        params["outdim"],
                        params["hidden"],
                        kernel_size=params["ffn_kernel"],
                        stride=params["ffn_stride"],
                        padding=math.ceil(params["ffn_kernel"] / 2) - 1,
                        bias=config.get("bias", True),
                    ),
                    Conv2d(
                        params["hidden"],
                        params["outdim"],
                        kernel_size=params["ffn_kernel"],
                        stride=params["ffn_stride"],
                        padding=math.ceil(params["ffn_kernel"] / 2) - 1,
                        bias=config.get("bias", True),
                    ),
                ]
            )
            self.log.debug(f"Built {blockid}")
            width, height = width // 2, height // 2
        layers.append(AdaptiveAvgPool2d((1, 1)))
        self.layers = Sequential(*layers)
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        self.log.debug(f"Input Shape: {x.shape}")
        for i, layer in enumerate(self.layers):
            x = layer(x)
            self.log.debug(f"Layer {layer} {i} Output Shape: {x.shape}")

        x = x.squeeze()
        return x
