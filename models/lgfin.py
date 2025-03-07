"""
This file is re-implementation of `https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10887340`
"""

from logging import DEBUG, Logger, StreamHandler, getLogger
from typing import Any, Dict
import yaml

import torch
from torch.nn import (
    GELU,
    AdaptiveAvgPool2d,
    AvgPool2d,
    BatchNorm2d,
    Conv2d,
    LayerNorm,
    Linear,
    MaxPool2d,
    Module,
    Parameter,
    Sequential,
    Sigmoid,
)


class DepthConvBlock(Module):
    def __init__(self, config: Dict[str, Any], log: Logger, **kwargs):
        super(DepthConvBlock, self).__init__()
        self.name = "DepthConvBlock"
        self.config = config
        self.log = log
        self.kwargs: Dict[str, Any] = kwargs
        self.log.debug("Initialised DepthConvBlock model.")
        indims = config["indims"]
        outdims = config["outdims"]

        self.block = Sequential(
            Conv2d(
                indims,
                indims,
                3,
                1,
                padding=1,
                groups=indims,
            ),
            BatchNorm2d(indims),
            GELU(),
            Conv2d(
                indims,
                outdims,
                1,
                1,
            ),
            BatchNorm2d(outdims),
            GELU(),
            Conv2d(
                outdims,
                outdims,
                3,
                1,
                padding=1,
                groups=outdims,
            ),
            BatchNorm2d(outdims),
            GELU(),
            MaxPool2d(2, 2),
        )

    def forward(self, x):
        return self.block(x)


class FeatureInteractionModule(Module):
    def __init__(self, config: Dict[str, Any], log: Logger, **kwargs):
        super(FeatureInteractionModule, self).__init__()
        self.name = "FeatureInteractionModule"
        self.config = config
        self.log = log
        self.kwargs: Dict[str, Any] = kwargs
        self.log.debug("Initialised FeatureInteractionModule model.")
        self.f3 = Conv2d(config["indims"], config["indims"], 3, 1, padding=1)
        self.f5 = Conv2d(config["indims"], config["indims"], 5, 1, padding=2)
        self.alpha = Parameter(
            torch.tensor(config.get("alpha", 0.5)), requires_grad=True
        )
        self.avgpool = AdaptiveAvgPool2d(1)
        self.mlp = Linear(config["indims"], config["indims"])
        self.sigmoid = Sigmoid()

    def forward(self, local_feature, global_feature):
        assert (
            local_feature.shape == global_feature.shape
        ), f"Shape mismatch: {local_feature.shape} != {global_feature.shape}"

        f = local_feature + global_feature
        f3 = self.f3(f)
        f5 = self.f5(f)
        spatial_weight = f3 * f5
        spatial_feature = spatial_weight * f
        spatial_output = spatial_feature + self.alpha * f
        channel_weight = (
            self.sigmoid(self.mlp(self.avgpool(spatial_output).squeeze()))
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        interacted_local_feature = local_feature * channel_weight
        interacted_global_feature = global_feature * channel_weight
        return (
            local_feature + interacted_global_feature,
            global_feature + interacted_local_feature,
        )


class GFEM(Module):
    def __init__(self, config: Dict[str, Any], log: Logger, **kwargs):
        super(GFEM, self).__init__()
        self.name = "GFEM"
        self.log = log
        self.kwargs: Dict[str, Any] = kwargs
        self.channels = config["indims"]
        self.height = config["height"]
        self.width = config["width"]
        # Point-wise (1x1) convolutions for Q, K, V projections
        self.pointwise_q = Conv2d(
            self.channels, self.channels, kernel_size=1, bias=kwargs.get("bias", False)
        )
        self.pointwise_k = Conv2d(
            self.channels, self.channels, kernel_size=1, bias=kwargs.get("bias", False)
        )
        self.pointwise_v = Conv2d(
            self.channels, self.channels, kernel_size=1, bias=kwargs.get("bias", False)
        )

        # Depth-wise (3x3) convolutions for Q, K, V
        self.depthwise_q = Conv2d(
            self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
            groups=self.channels,
            bias=kwargs.get("bias", False),
        )
        self.depthwise_k = Conv2d(
            self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
            groups=self.channels,
            bias=kwargs.get("bias", False),
        )
        self.depthwise_v = Conv2d(
            self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
            groups=self.channels,
            bias=kwargs.get("bias", False),
        )
        self.projection = Conv2d(
            self.channels, self.channels, kernel_size=1, bias=kwargs.get("bias", False)
        )

        # Begin FFN
        self.layer_norm = LayerNorm([self.height, self.width])
        self.ffn1 = Sequential(
            Conv2d(self.channels, self.channels, 1, 1),
            Conv2d(self.channels, self.channels, 3, 1, padding=1, groups=self.channels),
        )
        self.ffn2 = Sequential(
            Conv2d(self.channels, self.channels, 1, 1),
            Conv2d(self.channels, self.channels, 3, 1, padding=1, groups=self.channels),
            GELU(),
        )
        self.ffn_projection = Conv2d(self.channels, self.channels, 1, 1)

    def forward(self, x):
        # MDTA
        Q = self.depthwise_q(self.pointwise_q(x))
        K = self.depthwise_k(self.pointwise_k(x))
        V = self.depthwise_v(self.pointwise_v(x))

        attention_map = Q * K
        feature_map = attention_map * V

        projection = self.projection(feature_map)

        assert (
            projection.shape == x.shape
        ), f"Shape mismatch: {projection.shape} != {x.shape}"
        # projection is output of MDTA
        # Begin FFN
        mdta_output = x + projection

        output = self.layer_norm(mdta_output)
        branch1 = self.ffn1(output)
        branch2 = self.ffn2(output)
        ffn_output = branch1 * branch2
        ffn_output = self.ffn_projection(ffn_output)
        assert (
            ffn_output.shape == mdta_output.shape
        ), f"Shape mismatch: {ffn_output.shape} != {mdta_output.shape}"

        return mdta_output + ffn_output


class SpatialFeatureFusion(Module):
    def __init__(self, config: Dict[str, Any], log: Logger, **kwargs):
        super(SpatialFeatureFusion, self).__init__()
        self.name = "SpatialFeatureFusion"
        self.config = config
        self.log = log
        self.kwargs: Dict[str, Any] = kwargs
        self.log.debug("Initialised SpatialFeatureFusion model.")

        self.avgpool = AvgPool2d(3, 1, (3 - 1) // 2)
        self.maxpool = MaxPool2d(3, 1, (3 - 1) // 2)
        self.poolfuser = Conv2d(config["indims"] * 4, config["indims"], 1, 1)
        self.sigmoid = Sigmoid()

    def forward(self, feature1, feature2):
        assert (
            feature1.shape == feature2.shape
        ), f"Shape mismatch: {feature1.shape} != {feature2.shape}"
        msf_in = torch.cat([feature1, feature2], dim=1)
        pooled = torch.cat([self.avgpool(msf_in), self.maxpool(msf_in)], dim=1)
        fusedpool = self.poolfuser(pooled)
        sigpool = self.sigmoid(fusedpool)
        assert (
            sigpool.shape == feature1.shape
        ), f"Shape mismatch: {sigpool.shape} != {feature1.shape}"
        assert (
            sigpool.shape == feature2.shape
        ), f"Shape mismatch: {sigpool.shape} != {feature2.shape}"
        return sigpool * feature1 + sigpool * feature2


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
        self.sff = SpatialFeatureFusion(config, log)

    def forward(self, x):
        x1, x2, x3 = self.branch1(x), self.branch2(x), self.branch3(x)
        x_b = torch.cat([x1, x2, x3], dim=1)
        msf = self.msfuser(x_b)
        sff = self.sff(x, msf)
        return sff


class Lgfin(Module):
    def __init__(self, config: Dict[str, Any], log: Logger, **kwargs):
        super(Lgfin, self).__init__()
        self.name = "LocalGlobalFeatureInteractionNetwork"
        self.config = config
        self.log = log
        self.kwargs: Dict[str, Any] = kwargs
        self.log.debug("Initialised LocalGlobalFeatureInteractionNetwork model.")

        self.dwconv0 = DepthConvBlock(config["dwconv0"], log)

        # Block 1
        block1 = config["block1"]
        self.lfem1 = LFEM(block1, log)
        self.gfem1 = GFEM(block1, log)
        self.dwconv1_l = DepthConvBlock(block1, log)
        self.dwconv1_g = DepthConvBlock(block1, log)

        self.fim1 = FeatureInteractionModule(block1, log)

        # Block 2
        block2 = config["block2"]
        self.lfem2 = LFEM(block2, log)
        self.gfem2 = GFEM(block2, log)
        self.dwconv2_l = DepthConvBlock(block2, log)
        self.dwconv2_g = DepthConvBlock(block2, log)
        self.fim2 = FeatureInteractionModule(block2, log)

        # Block 3
        block3 = config["block3"]
        self.lfem3 = LFEM(block3, log)
        self.gfem3 = GFEM(block3, log)
        self.dwconv3_l = DepthConvBlock(block3, log)
        self.dwconv3_g = DepthConvBlock(block3, log)

        self.fim3 = FeatureInteractionModule(block3, log)

        self.sff = SpatialFeatureFusion({"indims": config["outdims"]}, log)
        self.avg = AdaptiveAvgPool2d(1)
        self.fc1 = Linear(config["outdims"], config["hdims"])
        self.fc2 = Linear(config["hdims"], config["num_classes"])

    def forward(self, x, **kwargs):
        x = self.dwconv0(x)

        # Block 1
        x_l1 = self.lfem1(x)
        x_g1 = self.gfem1(x)
        x_l1_if, x_g1_if = self.fim1(x_l1, x_g1)
        x_l1 = self.dwconv1_l(x_l1 + x_l1_if)
        x_g1 = self.dwconv1_g(x_g1 + x_g1_if)

        # Block 2
        x_l2 = self.lfem2(x_l1)
        x_g2 = self.gfem2(x_g1)
        x_l2_if, x_g2_if = self.fim2(x_l2, x_g2)
        x_l2 = self.dwconv2_l(x_l2 + x_l2_if)
        x_g2 = self.dwconv2_g(x_g2 + x_g2_if)

        # Block 3
        x_l3 = self.lfem3(x_l2)
        x_g3 = self.gfem3(x_g2)
        x_l3_if, x_g3_if = self.fim3(x_l3, x_g3)
        x_l3 = self.dwconv3_l(x_l3 + x_l3_if)
        x_g3 = self.dwconv3_g(x_g3 + x_g3_if)

        x = self.sff(x_l3, x_g3)
        x = self.avg(x)
        x = x.squeeze()
        if kwargs.get("features", False):
            return x
        x = self.fc1(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    logger = getLogger("LFEM")
    logger.setLevel(DEBUG)
    logger.addHandler(StreamHandler())
    logger.info(f"LFEM input: {x.shape}")
    model = LFEM({"indims": 3}, logger)
    lfem_out = model(x)
    logger.info(f"LFEM output:  {lfem_out.shape}")

    logger.info(f"GFEM input: {x.shape}")
    model = GFEM({"indims": 3, "height": 224, "width": 224}, logger)
    gfem_out = model(x)
    logger.info(f"GFEM output:  {gfem_out.shape}")

    model = FeatureInteractionModule({"indims": 3, "alpha": 0.5}, logger)
    lfemint, gfemint = model(lfem_out, gfem_out)
    logger.info(f"LFEMINT output:  {lfemint.shape}")
    logger.info(f"GFEMINT output:  {gfemint.shape}")

    logger.info(f"SpatialFeatureFusion input: {x.shape}")
    model = SpatialFeatureFusion({"indims": 3}, logger)
    sff_out = model(x, x)
    logger.info(f"SpatialFeatureFusion output:  {sff_out.shape}")

    logger.info(f"DepthConvBlock input: {x.shape}")
    model = DepthConvBlock({"indims": 3, "outdims": 8}, logger)
    dcb_out = model(x)
    logger.info(f"DepthConvBlock output:  {dcb_out.shape}")

    logger.info(f"LocalGlobalFeatureInteractionNetwork input: {x.shape}")

    with open("./configs/lgim.yaml", "r") as f:
        config = yaml.safe_load(f)

    model = Lgfin(config, logger)
    lgfin_out = model(x)
    logger.info(f"LGFIN output:  {lgfin_out.shape}")
