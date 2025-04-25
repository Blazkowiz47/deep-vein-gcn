import math

import torch

from logging import Logger
from typing import Any, Dict, List
from torch.nn import Conv2d, GroupNorm, LayerNorm, Linear, Module, Parameter, Sequential
from torch.nn.functional import adaptive_avg_pool2d
from torch.nn.modules import BatchNorm2d
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
        self.log.debug(f"DSConv: {indims}, {outdims}, {kernel}, {stride}")
        self.log.debug(f"DSConv: {bias}, {device}")
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
            kernel_size=kernel,
            morph=0,
            device=device,
        )
        self.ydsc = DSConv_pro(
            indims,
            outdims,
            kernel_size=kernel,
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


class GrapherBlock(Module):
    def __init__(
        self,
        log: Logger,
        config: Dict[str, Any],
        block_config: Dict[str, Any],
        **kwargs: Dict[str, Any],
    ):
        super(GrapherBlock, self).__init__()
        """
        Needs height and width as kwargs
        """
        self.name = "Grapherblock"
        self.log = log
        self.grapherblock = Sequential(
            *self.build_grapher_ffn(config, block_config, **kwargs)
        )
        self.model_init()
        self.log.debug("Initialised GrapherBlock")

    def model_init(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def build_grapher_ffn(
        self, config: Dict[str, Any], block_config: Dict[str, Any], **kwargs
    ) -> List[Module]:
        blocks: List[Module] = []
        height = kwargs["height"]
        width = kwargs["width"]
        for _ in range(block_config["graphers"]):
            blocks.append(
                Sequential(
                    Grapher(
                        block_config["indim"],
                        min(config["kernel_size"], height * width),
                        config["dilation"],
                        config["conv"],
                        config["act"],
                        config["norm"],
                        config["bias"],
                        config["stochastic"],
                        config["epsilon"],
                        block_config["reduce_ratio"],
                        height * width,
                        relative_pos=True,
                    ),
                    Conv2d(
                        block_config["indim"],
                        block_config["hdim"],
                        config["ffn_kernel"],
                        config["ffn_stride"],
                    ),
                    BatchNorm2d(block_config["hdim"]),
                    act_layer(config["act"]),
                    Conv2d(
                        block_config["hdim"],
                        block_config["indim"],
                        config["ffn_kernel"],
                        config["ffn_stride"],
                    ),
                    BatchNorm2d(block_config["indim"]),
                )
            )
        if "shrinker_kernel" in block_config:
            blocks.append(
                Sequential(
                    Conv2d(
                        block_config["indim"],
                        block_config["outdim"],
                        block_config["shrinker_kernel"],
                        block_config["shrinker_stride"],
                    ),
                    BatchNorm2d(block_config["outdim"]),
                )
            )
        self.log.debug(f"GrapherBlock: {min(config['kernel_size'], height * width)}")
        return blocks

    def forward(self, x):
        return self.grapherblock(x)


class Dscgrapher(Module):
    def __init__(self, config: Dict[str, Any], log: Logger, **kwargs):
        super(Dscgrapher, self).__init__()
        self.name = "dscgrapher"
        self.config = config
        self.log = log
        self.kwargs: Dict[str, Any] = kwargs
        self.device = config["device"]

        self.stem = Sequential(*self.build_stem(config)).to(self.device)
        self.log.debug("Initialised stem")
        self.check_stem()

        height = config["height"]
        width = config["width"]
        for _ in range(config["stem"]["depth"]):
            height = height // 2
            width = width // 2

        self.pos_embed = Parameter(
            torch.zeros(  # pylint: disable=E1101
                1,
                config["stem"]["outdim"],
                height,
                width,
            )
        )
        self.backbone = Sequential(*self.build_backbone(config)).to(self.device)
        self.check_backbone()
        self.log.debug("Initialised dscgrapher model.")

        self.model_init()

        for param in self.parameters():
            param.requires_grad = True

    def model_init(self):
        for m in self.modules():
            if isinstance(m, (Conv2d, Linear)):
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, (BatchNorm2d, GroupNorm,LayerNorm)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def build_stem(self, config: Dict[str, Any]) -> List[Module]:
        stem_config = config["stem"]

        indim = stem_config["indim"]
        outdim = stem_config["outdim"]
        depth = stem_config["depth"]
        kernels = stem_config["kernels"]
        stride = stem_config["stride"]
        bias = stem_config["bias"]
        act = stem_config["act"]

        for _ in range(depth - 1):
            outdim = max(4, outdim // 2)

        stem = []
        for _, kernel in enumerate(kernels):
            stem.append(
                DSConv(
                    indim,
                    outdim,
                    kernel,
                    stride=stride,
                    bias=bias,
                    device=config["device"],
                    log=self.log,
                )
            )
            stem.append(BatchNorm2d(outdim))
            if _ + 1 != depth:
                stem.append(act_layer(act))
            indim = outdim
            outdim = outdim * 2

        return stem

    def build_backbone(self, config: Dict[str, Any]) -> List[Module]:
        height = config["height"]
        width = config["width"]
        for _ in range(config["stem"]["depth"]):
            height = height // 2
            width = width // 2

        backbone_config = config["backbone"]
        depth = backbone_config["depth"]

        backbone = []
        for blocknum in range(depth):
            block_config = backbone_config[f"block{blocknum}"]
            backbone.append(
                GrapherBlock(
                    self.log, backbone_config, block_config, height=height, width=width
                )
            )
            height = height // 2
            width = width // 2

        return backbone

    def build_head(self, config: Dict[str, Any]) -> List[Module]:
        head_config = config["head"]
        head = []
        head.append(
            Conv2d(
                head_config["indim"],
                head_config["outdim"],
                head_config["kernel"],
                head_config["stride"],
            )
        )
        head.append(BatchNorm2d(head_config["outdim"]))
        head.append(act_layer(head_config["act"]))
        head.append(
            Conv2d(
                head_config["outdim"],
                head_config["outdim"],
                head_config["kernel"],
                head_config["stride"],
            )
        )
        return head

    def check_stem(self):
        self.log.debug("Checking stem")
        x = torch.randn(2, 3, 224, 224).to(self.device)
        y = self.stem(x)
        self.log.debug(f"Stem output: {y.shape}")
        self.log.debug("Stem checked")

    def check_backbone(self):
        self.log.debug("Checking backbone")
        x = torch.randn(2, 3, 224, 224).to(self.device)
        x = self.stem(x)
        y = self.backbone(x)
        self.log.debug(f"Backbone output: {y.shape}")
        self.log.debug("Backbone checked")

    def forward(self, x, **kwargs):
        x = self.stem(x) + self.pos_embed
        x = self.backbone(x)
        x = adaptive_avg_pool2d(x, (1, 1))
        self.log.debug(f"Pooled shape: {x.shape}")
        x = x.squeeze()
        return x
