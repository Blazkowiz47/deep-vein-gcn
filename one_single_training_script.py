import logging
import math
import os
import pprint as pp
import random
from abc import abstractmethod
from logging import Logger
from math import nan
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import einops
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from timm.models.layers import DropPath
from torch.nn import (
    GELU,
    AdaptiveAvgPool2d,
    BatchNorm2d,
    Conv2d,
    Dropout2d,
    GroupNorm,
    Hardswish,
    Identity,
    InstanceNorm2d,
    LayerNorm,
    LeakyReLU,
    Linear,
    Module,
    Parameter,
    PReLU,
    ReLU,
    Sequential,
    Tanh,
    functional,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import MulticlassAccuracy
from torchvision import set_image_backend
from torchvision import transforms as A
from tqdm import tqdm

# Incase you use wandb uncomment following line

set_image_backend("accimage")  # Faster image loading than PIL


def set_seeds(log: Logger, seed: int):
    """
    Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 2024.
    """
    log.debug(f"Setting seed to: {seed}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuddeterministic = True
    torch.backends.cudbenchmark = False
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_logger(logfile: str = "", level: str = "DEBUG") -> logging.Logger:
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    logger = logging.getLogger(logfile)
    logger.setLevel(level)

    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "{asctime} - {levelname} - {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M",
    )
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    if logfile:
        file_handler = logging.FileHandler(logfile, mode="a+", encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    return logger


# %%
###################################
# Load the dataset
###################################
image_extensions: List[str] = [".jpg", ".png", ".jpeg", ".bmp"]


class DatasetGenerator(Dataset):
    def __init__(self, data: List[Any], transform: Callable, **kwargs) -> None:
        self.data = data
        self.transform = transform
        self.kwargs = kwargs

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Tuple[Any]:
        datapoint = self.data[index]
        return self.transform(datapoint, **self.kwargs)


class Wrapper:
    def __init__(
        self,
        log: Logger,
        rdir: str = "",
        batch_size: int = 1,
        num_workers: int = 1,
        num_classes: Optional[int] = None,
    ) -> None:
        self.batch_size = batch_size
        self.log = log
        self.num_workers = num_workers
        self.num_classes = None
        self.rdir = rdir
        self.name = ""

    @abstractmethod
    def loop_splitset(self, ssplit: str) -> List[Any]:
        """
        Loops through the given directory.
        Practically, one should only change this function and get various splits.
        Returns: List of files to load along with its class label.
        """
        raise NotImplementedError("")

    def get_split(
        self,
        split: str,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        **kwargs: Any,
    ) -> DataLoader:
        """
        Generates the given split.
        """
        self.log.debug(f"Generating {split} split for {self.name} dataset.")
        batch_size = batch_size or self.batch_size
        data = self.loop_splitset(split)
        self.log.debug(f"Total files: {len(data)}")
        return DataLoader(
            DatasetGenerator(data, self.transform, **kwargs),
            num_workers=num_workers or self.num_workers,
            batch_size=batch_size or self.batch_size,
        )

    @abstractmethod
    def augment(self, image):
        """
        Augments the given image.
        """
        raise NotImplementedError()

    @abstractmethod
    def transform(self, datapoint: Iterable[Any]) -> Tuple:
        """
        Transforms the given datapoint.
        """
        raise NotImplementedError()


class LeaveoneoutWrapper(Wrapper):
    def __init__(
        self,
        config: Dict[str, Any],
        log: Logger,
        rdirs: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        This is a demo wrapper.
        Update the `loop_splitset` method for looping your dataset in a custom manner.
        By default it fetches data stored in following manner relative to `./data/leaveoneout':
            ROOT-DIR
                -> ClassId1
                    -> train
                        -> Image1.jpg
                    -> test
                        -> Image1.jpeg
                    -> validation
                        -> Image1.jpeg
                ...
        """

        self.name = "leaveoneout"
        self.log = log
        self.kwargs: Dict[str, Any] = kwargs
        self.height = config.get("height", 224)
        self.width = config.get("width", 224)
        # TODO: You can add the root dirs here directly:
        self.rdirs: List[str] = rdirs or [
            "./data/fvusm",
            "./data/fv300",
        ]

        self.batch_size = config["batch_size"]
        self.num_workers = config.get("num_workers", 4)
        self.total_data: Dict[str, List[str]] = {}
        self.num_classes = None
        self.mintrain_imgs = 90
        self.initialise_db()

        self.augmentations = A.Compose(
            [
                A.ToTensor(),
                A.RandomHorizontalFlip(),
                A.RandomVerticalFlip(),
                A.RandomAutocontrast(p=0.05),
                A.RandomRotation(45),
                A.Resize((self.height, self.width)),
            ]
        )

    def initialise_db(self) -> None:
        for rdir in self.rdirs:
            self._internal_loop(rdir, self.total_data)
        dataset_length = sum([len(v) for v in self.total_data.values()])
        self.num_classes = len(self.total_data)
        self.log.debug(f"Data-length for {self.name} split: {dataset_length}")
        self.log.debug(f"Number of classes for {self.name} split: {self.num_classes}")

        for cid in self.total_data:
            random.shuffle(self.total_data[cid])
            partition_index = int(self.partition_split * len(self.total_data[cid]))
            self.train_data[cid] = self.total_data[cid][:partition_index]
            self.test_data[cid] = self.total_data[cid][partition_index:]

    def _internal_loop(self, rdir: str, prev: Dict[str, List[str]]) -> None:
        rdir = Path(rdir)
        for cid in rdir.iterdir():
            if not cid.is_dir():
                continue
            cid_str = str(cid)
            if cid_str not in prev:
                prev[cid_str] = []
            for img in cid.rglob("*"):
                if img.suffix.lower() in image_extensions:
                    prev[cid_str].append(str(img))

    def loop_splitset(self, ssplit: str) -> List[Any]:
        data = self.total_data

        datalist: List[Any] = []
        for cid, class_name in enumerate(data.keys()):
            for img in data[class_name]:
                datalist.append((img, cid))

        return datalist

    def get_dataloader(
        self,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        **kwargs: Any,
    ) -> DataLoader:
        batch_size = batch_size or self.batch_size
        data = self.loop_splitset()
        self.log.debug("Data-length for : %d" % len(data))
        return DataLoader(
            DatasetGenerator(data, self.transform, **kwargs),
            num_workers=num_workers or self.num_workers,
            batch_size=batch_size or self.batch_size,
            shuffle=True,
            prefetch_factor=num_workers or self.num_workers,
        )

    def augment(self, image: Any) -> Any:
        return self.augmentations(image)

    def transform(self, datapoint: Iterable[Any], **kwargs) -> Tuple:
        imgfname, lbl = datapoint
        if self.num_classes is None:
            raise ValueError("Num classes not set.")
        # Initialise label
        img = Image.open(imgfname)
        label = torch.zeros(self.num_classes)
        label[lbl] = 1

        imgarray = np.array(img)
        # imgarray = (imgarray - imgarray.min()) / (imgarray.max() - imgarray.min() + 1e-6)
        imgarray = imgarray / 255.0
        imgarray = np.stack([imgarray, imgarray, imgarray], axis=2)

        imgarray = self.augment(imgarray)

        if kwargs.get("return_filename"):
            return imgarray.float(), label.float(), imgfname

        return imgarray.float(), label.float()


# %%


###################################
# Load the model
###################################


####################################
# DSC Conv for stem
####################################


class DSConv_pro(Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 9,
        extend_scope: float = 1.0,
        morph: int = 0,
        if_offset: bool = True,
        device: str | torch.device = "cuda",
    ):
        """
        A Dynamic Snake Convolution Implementation

        Based on:

            TODO

        Args:
            in_ch: number of input channels. Defaults to 1.
            out_ch: number of output channels. Defaults to 1.
            kernel_size: the size of kernel. Defaults to 9.
            extend_scope: the range to expand. Defaults to 1 for this method.
            morph: the morphology of the convolution kernel is mainly divided into two types along the x-axis (0) and the y-axis (1) (see the paper for details).
            if_offset: whether deformation is required,  if it is False, it is the standard convolution kernel. Defaults to True.

        """

        super().__init__()

        if morph not in (0, 1):
            raise ValueError("morph should be 0 or 1.")

        self.kernel_size = kernel_size
        self.extend_scope = extend_scope
        self.morph = morph
        self.if_offset = if_offset
        self.device = torch.device(device)
        self.to(device)

        # self.bn = BatchNorm2d(2 * kernel_size)
        self.gn_offset = GroupNorm(kernel_size, 2 * kernel_size)
        self.gn = GroupNorm(out_channels // 4, out_channels)
        self.relu = ReLU(inplace=True)
        self.tanh = Tanh()

        self.offset_conv = Conv2d(in_channels, 2 * kernel_size, 3, padding=1)

        self.dsc_conv_x = Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            stride=(kernel_size, 1),
            padding=0,
        )
        self.dsc_conv_y = Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, kernel_size),
            stride=(1, kernel_size),
            padding=0,
        )

    def forward(self, input: torch.Tensor):
        # Predict offset map between [-1, 1]
        offset = self.offset_conv(input)
        # offset = self.bn(offset)
        offset = self.gn_offset(offset)
        offset = self.tanh(offset)

        # Run deformative conv
        y_coordinate_map, x_coordinate_map = get_coordinate_map_2D(
            offset=offset,
            morph=self.morph,
            extend_scope=self.extend_scope,
            device=self.device,
        )
        deformed_feature = get_interpolated_feature(
            input,
            y_coordinate_map,
            x_coordinate_map,
        )

        if self.morph == 0:
            output = self.dsc_conv_x(deformed_feature)
        elif self.morph == 1:
            output = self.dsc_conv_y(deformed_feature)
        else:
            raise ValueError(f"Invalid morph value: {self.morph}")

        # Groupnorm & ReLU
        output = self.gn(output)
        output = self.relu(output)

        return output


def get_coordinate_map_2D(
    offset: torch.Tensor,
    morph: int,
    extend_scope: float = 1.0,
    device: str | torch.device = "cuda",
):
    """Computing 2D coordinate map of DSCNet based on: TODO

    Args:
        offset: offset predict by network with shape [B, 2*K, W, H]. Here K refers to kernel size.
        morph: the morphology of the convolution kernel is mainly divided into two types along the x-axis (0) and the y-axis (1) (see the paper for details).
        extend_scope: the range to expand. Defaults to 1 for this method.
        device: location of data. Defaults to 'cuda'.

    Return:
        y_coordinate_map: coordinate map along y-axis with shape [B, K_H * H, K_W * W]
        x_coordinate_map: coordinate map along x-axis with shape [B, K_H * H, K_W * W]
    """

    if morph not in (0, 1):
        raise ValueError("morph should be 0 or 1.")

    batch_size, _, width, height = offset.shape
    kernel_size = offset.shape[1] // 2
    center = kernel_size // 2
    device = torch.device(device)

    y_offset_, x_offset_ = torch.split(offset, kernel_size, dim=1)

    y_center_ = torch.arange(0, width, dtype=torch.float32, device=device)
    y_center_ = einops.repeat(y_center_, "w -> k w h", k=kernel_size, h=height)

    x_center_ = torch.arange(0, height, dtype=torch.float32, device=device)
    x_center_ = einops.repeat(x_center_, "h -> k w h", k=kernel_size, w=width)

    if morph == 0:
        """
        Initialize the kernel and flatten the kernel
            y: only need 0
            x: -num_points//2 ~ num_points//2 (Determined by the kernel size)
        """
        y_spread_ = torch.zeros([kernel_size], device=device)
        x_spread_ = torch.linspace(-center, center, kernel_size, device=device)

        y_grid_ = einops.repeat(y_spread_, "k -> k w h", w=width, h=height)
        x_grid_ = einops.repeat(x_spread_, "k -> k w h", w=width, h=height)

        y_new_ = y_center_ + y_grid_
        x_new_ = x_center_ + x_grid_

        y_new_ = einops.repeat(y_new_, "k w h -> b k w h", b=batch_size)
        x_new_ = einops.repeat(x_new_, "k w h -> b k w h", b=batch_size)

        y_offset_ = einops.rearrange(y_offset_, "b k w h -> k b w h")
        y_offset_new_ = y_offset_.detach().clone()

        # The center position remains unchanged and the rest of the positions begin to swing
        # This part is quite simple. The main idea is that "offset is an iterative process"

        y_offset_new_[center] = 0

        for index in range(1, center + 1):
            y_offset_new_[center + index] = (
                y_offset_new_[center + index - 1] + y_offset_[center + index]
            )
            y_offset_new_[center - index] = (
                y_offset_new_[center - index + 1] + y_offset_[center - index]
            )

        y_offset_new_ = einops.rearrange(y_offset_new_, "k b w h -> b k w h")

        y_new_ = y_new_.add(y_offset_new_.mul(extend_scope))

        y_coordinate_map = einops.rearrange(y_new_, "b k w h -> b (w k) h")
        x_coordinate_map = einops.rearrange(x_new_, "b k w h -> b (w k) h")

    elif morph == 1:
        """
        Initialize the kernel and flatten the kernel
            y: -num_points//2 ~ num_points//2 (Determined by the kernel size)
            x: only need 0
        """
        y_spread_ = torch.linspace(-center, center, kernel_size, device=device)
        x_spread_ = torch.zeros([kernel_size], device=device)

        y_grid_ = einops.repeat(y_spread_, "k -> k w h", w=width, h=height)
        x_grid_ = einops.repeat(x_spread_, "k -> k w h", w=width, h=height)

        y_new_ = y_center_ + y_grid_
        x_new_ = x_center_ + x_grid_

        y_new_ = einops.repeat(y_new_, "k w h -> b k w h", b=batch_size)
        x_new_ = einops.repeat(x_new_, "k w h -> b k w h", b=batch_size)

        x_offset_ = einops.rearrange(x_offset_, "b k w h -> k b w h")
        x_offset_new_ = x_offset_.detach().clone()

        # The center position remains unchanged and the rest of the positions begin to swing
        # This part is quite simple. The main idea is that "offset is an iterative process"

        x_offset_new_[center] = 0

        for index in range(1, center + 1):
            x_offset_new_[center + index] = (
                x_offset_new_[center + index - 1] + x_offset_[center + index]
            )
            x_offset_new_[center - index] = (
                x_offset_new_[center - index + 1] + x_offset_[center - index]
            )

        x_offset_new_ = einops.rearrange(x_offset_new_, "k b w h -> b k w h")

        x_new_ = x_new_.add(x_offset_new_.mul(extend_scope))

        y_coordinate_map = einops.rearrange(y_new_, "b k w h -> b w (h k)")
        x_coordinate_map = einops.rearrange(x_new_, "b k w h -> b w (h k)")

    return y_coordinate_map, x_coordinate_map


def get_interpolated_feature(
    input_feature: torch.Tensor,
    y_coordinate_map: torch.Tensor,
    x_coordinate_map: torch.Tensor,
    interpolate_mode: str = "bilinear",
):
    """From coordinate map interpolate feature of DSCNet based on: TODO

    Args:
        input_feature: feature that to be interpolated with shape [B, C, H, W]
        y_coordinate_map: coordinate map along y-axis with shape [B, K_H * H, K_W * W]
        x_coordinate_map: coordinate map along x-axis with shape [B, K_H * H, K_W * W]
        interpolate_mode: the arg 'mode' of functional.grid_sample, can be 'bilinear' or 'bicubic' . Defaults to 'bilinear'.

    Return:
        interpolated_feature: interpolated feature with shape [B, C, K_H * H, K_W * W]
    """

    if interpolate_mode not in ("bilinear", "bicubic"):
        raise ValueError("interpolate_mode should be 'bilinear' or 'bicubic'.")

    y_max = input_feature.shape[-2] - 1
    x_max = input_feature.shape[-1] - 1

    y_coordinate_map_ = _coordinate_map_scaling(y_coordinate_map, origin=[0, y_max])
    x_coordinate_map_ = _coordinate_map_scaling(x_coordinate_map, origin=[0, x_max])

    y_coordinate_map_ = torch.unsqueeze(y_coordinate_map_, dim=-1)
    x_coordinate_map_ = torch.unsqueeze(x_coordinate_map_, dim=-1)

    # Note here grid with shape [B, H, W, 2]
    # Where [:, :, :, 2] refers to [x ,y]
    grid = torch.cat([x_coordinate_map_, y_coordinate_map_], dim=-1)

    interpolated_feature = functional.grid_sample(
        input=input_feature,
        grid=grid,
        mode=interpolate_mode,
        padding_mode="zeros",
        align_corners=True,
    )

    return interpolated_feature


def _coordinate_map_scaling(
    coordinate_map: torch.Tensor,
    origin: list,
    target: list = [-1, 1],
):
    """Map the value of coordinate_map from origin=[min, max] to target=[a,b] for DSCNet based on: TODO

    Args:
        coordinate_map: the coordinate map to be scaled
        origin: original value range of coordinate map, e.g. [coordinate_map.min(), coordinate_map.max()]
        target: target value range of coordinate map,Defaults to [-1, 1]

    Return:
        coordinate_map_scaled: the coordinate map after scaling
    """
    min, max = origin
    a, b = target

    coordinate_map_scaled = torch.clamp(coordinate_map, min, max)

    scale_factor = (b - a) / (max - min)
    coordinate_map_scaled = a + scale_factor * (coordinate_map_scaled - min)

    return coordinate_map_scaled


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
        res = self.enc(torch.cat([c, x, y], dim=1))
        self.log.debug(f"Res: {res.shape}")
        return res


def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer

    act = act.lower()
    if act == "relu":
        layer = ReLU(inplace)
    elif act == "leakyrelu":
        layer = LeakyReLU(neg_slope, inplace)
    elif act == "prelu":
        layer = PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == "gelu":
        layer = GELU()
    elif act == "hswish":
        layer = Hardswish(inplace)
    else:
        raise NotImplementedError("activation layer [%s] is not found" % act)
    return layer


####################################
# Grapher Block
####################################
def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_relative_pos_embed(embed_dim, grid_size):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, grid_size*grid_size]
    """
    pos_embed = get_2d_sincos_pos_embed(embed_dim, grid_size)
    relative_pos = 2 * np.matmul(pos_embed, pos_embed.transpose()) / pos_embed.shape[1]
    return relative_pos


def batched_index_select(x, idx):
    r"""fetches neighbors features from a given neighbor idx

    Args:
        x (Tensor): input feature Tensor
                :math:`\mathbf{X} \in \mathbb{R}^{B \times C \times N \times 1}`.
        idx (Tensor): edge_idx
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times l}`.
    Returns:
        Tensor: output neighbors features
            :math:`\mathbf{X} \in \mathbb{R}^{B \times C \times N \times k}`.
    """
    batch_size, num_dims, num_vertices_reduced = x.shape[:3]
    _, num_vertices, k = idx.shape
    idx_base = (
        torch.arange(0, batch_size, device=idx.device).view(-1, 1, 1)
        * num_vertices_reduced
    )
    idx = idx + idx_base
    idx = idx.contiguous().view(-1)

    x = x.transpose(2, 1)
    feature = x.contiguous().view(batch_size * num_vertices_reduced, -1)[idx, :]
    feature = (
        feature.view(batch_size, num_vertices, k, num_dims)
        .permute(0, 3, 1, 2)
        .contiguous()
    )
    return feature


def norm_layer(norm, nc):
    # normalization layer 2d
    norm = norm.lower()
    if norm == "batch":
        layer = BatchNorm2d(nc, affine=True)
    elif norm == "instance":
        layer = InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm)
    return layer


class MLP(Sequential):
    def __init__(self, channels, act="relu", norm=None, bias=True):
        m = []
        for i in range(1, len(channels)):
            m.append(Linear(channels[i - 1], channels[i], bias))
            if act is not None and act.lower() != "none":
                m.append(act_layer(act))
            if norm is not None and norm.lower() != "none":
                m.append(norm_layer(norm, channels[-1]))
        super(MLP, self).__init__(*m)


class BasicConv(Sequential):
    def __init__(self, channels, act="relu", norm=None, bias=True, drop=0.0):
        m = []
        for i in range(1, len(channels)):
            m.append(Conv2d(channels[i - 1], channels[i], 1, bias=bias, groups=4))
            if norm is not None and norm.lower() != "none":
                m.append(norm_layer(norm, channels[-1]))
            if act is not None and act.lower() != "none":
                m.append(act_layer(act))
            if drop > 0:
                m.append(Dropout2d(drop))

        super(BasicConv, self).__init__(*m)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, BatchNorm2d) or isinstance(m, InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class MRConv2d(Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type
    """

    def __init__(self, in_channels, out_channels, act="relu", norm=None, bias=True):
        super(MRConv2d, self).__init__()
        self.nn = BasicConv([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(x_j - x_i, -1, keepdim=True)
        b, c, n, _ = x.shape
        x = torch.cat([x.unsqueeze(2), x_j.unsqueeze(2)], dim=2).reshape(b, 2 * c, n, _)
        return self.nn(x)


class EdgeConv2d(Module):
    """
    Edge convolution layer (with activation, batch normalization) for dense data type
    """

    def __init__(self, in_channels, out_channels, act="relu", norm=None, bias=True):
        super(EdgeConv2d, self).__init__()
        self.nn = BasicConv([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        max_value, _ = torch.max(
            self.nn(torch.cat([x_i, x_j - x_i], dim=1)), -1, keepdim=True
        )
        return max_value


class GraphSAGE(Module):
    """
    GraphSAGE Graph Convolution (Paper: https://arxiv.org/abs/1706.02216) for dense data type
    """

    def __init__(self, in_channels, out_channels, act="relu", norm=None, bias=True):
        super(GraphSAGE, self).__init__()
        self.nn1 = BasicConv([in_channels, in_channels], act, norm, bias)
        self.nn2 = BasicConv([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(self.nn1(x_j), -1, keepdim=True)
        return self.nn2(torch.cat([x, x_j], dim=1))


class GINConv2d(Module):
    """
    GIN Graph Convolution (Paper: https://arxiv.org/abs/1810.00826) for dense data type
    """

    def __init__(self, in_channels, out_channels, act="relu", norm=None, bias=True):
        super(GINConv2d, self).__init__()
        self.nn = BasicConv([in_channels, out_channels], act, norm, bias)
        eps_init = 0.0
        self.eps = Parameter(torch.Tensor([eps_init]))

    def forward(self, x, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j = torch.sum(x_j, -1, keepdim=True)
        return self.nn((1 + self.eps) * x + x_j)


class GraphConv2d(Module):
    """
    Static graph convolution layer
    """

    def __init__(
        self, in_channels, out_channels, conv="edge", act="relu", norm=None, bias=True
    ):
        super(GraphConv2d, self).__init__()
        if conv == "edge":
            self.gconv = EdgeConv2d(in_channels, out_channels, act, norm, bias)
        elif conv == "mr":
            self.gconv = MRConv2d(in_channels, out_channels, act, norm, bias)
        elif conv == "sage":
            self.gconv = GraphSAGE(in_channels, out_channels, act, norm, bias)
        elif conv == "gin":
            self.gconv = GINConv2d(in_channels, out_channels, act, norm, bias)
        else:
            raise NotImplementedError("conv:{} is not supported".format(conv))

    def forward(self, x, edge_index, y=None):
        return self.gconv(x, edge_index, y)


class DenseDilated(nn.Module):
    """
    Find dilated neighbor from neighbor list

    edge_index: (2, batch_size, num_points, k)
    """

    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DenseDilated, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k

    def forward(self, edge_index):
        if self.stochastic:
            if torch.rand(1) < self.epsilon and self.training:
                num = self.k * self.dilation
                randnum = torch.randperm(num)[: self.k]
                edge_index = edge_index[:, :, :, randnum]
            else:
                edge_index = edge_index[:, :, :, :: self.dilation]
        else:
            edge_index = edge_index[:, :, :, :: self.dilation]

        return edge_index


def pairwise_distance(x):
    """
    Compute pairwise distance of a point cloud.
    Args:
        x: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    with torch.no_grad():
        x_inner = -2 * torch.matmul(x, x.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        return x_square + x_inner + x_square.transpose(2, 1)


def part_pairwise_distance(x, start_idx=0, end_idx=1):
    """
    Compute pairwise distance of a point cloud.
    Args:
        x: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    with torch.no_grad():
        x_part = x[:, start_idx:end_idx]
        x_square_part = torch.sum(torch.mul(x_part, x_part), dim=-1, keepdim=True)
        x_inner = -2 * torch.matmul(x_part, x.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        return x_square_part + x_inner + x_square.transpose(2, 1)


def xy_pairwise_distance(x, y):
    """
    Compute pairwise distance of a point cloud.
    Args:
        x: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    with torch.no_grad():
        xy_inner = -2 * torch.matmul(x, y.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        y_square = torch.sum(torch.mul(y, y), dim=-1, keepdim=True)
        return x_square + xy_inner + y_square.transpose(2, 1)


def dense_knn_matrix(x, k=16, relative_pos=None):
    """Get KNN based on the pairwise distance.
    Args:
        x: (batch_size, num_dims, num_points, 1)
        k: int
    Returns:
        nearest neighbors: (batch_size, num_points, k) (batch_size, num_points, k)
    """
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)
        batch_size, n_points, n_dims = x.shape
        # memory efficient implementation
        n_part = 10000
        if n_points > n_part:
            nn_idx_list = []
            groups = math.ceil(n_points / n_part)
            for i in range(groups):
                start_idx = n_part * i
                end_idx = min(n_points, n_part * (i + 1))
                dist = part_pairwise_distance(x.detach(), start_idx, end_idx)
                if relative_pos is not None:
                    dist += relative_pos[:, start_idx:end_idx]
                _, nn_idx_part = torch.topk(-dist, k=k)
                nn_idx_list += [nn_idx_part]
            nn_idx = torch.cat(nn_idx_list, dim=1)
        else:
            dist = pairwise_distance(x.detach())
            if relative_pos is not None:
                dist += relative_pos
            _, nn_idx = torch.topk(-dist, k=k)  # b, n, k
        ######
        center_idx = (
            torch.arange(0, n_points, device=x.device)
            .repeat(batch_size, k, 1)
            .transpose(2, 1)
        )
    return torch.stack((nn_idx, center_idx), dim=0)


def xy_dense_knn_matrix(x, y, k=16, relative_pos=None):
    """Get KNN based on the pairwise distance.
    Args:
        x: (batch_size, num_dims, num_points, 1)
        k: int
    Returns:
        nearest neighbors: (batch_size, num_points, k) (batch_size, num_points, k)
    """
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)
        y = y.transpose(2, 1).squeeze(-1)
        batch_size, n_points, n_dims = x.shape
        dist = xy_pairwise_distance(x.detach(), y.detach())
        if relative_pos is not None:
            dist += relative_pos
        _, nn_idx = torch.topk(-dist, k=k)
        center_idx = (
            torch.arange(0, n_points, device=x.device)
            .repeat(batch_size, k, 1)
            .transpose(2, 1)
        )
    return torch.stack((nn_idx, center_idx), dim=0)


class DenseDilatedKnnGraph(nn.Module):
    """
    Find the neighbors' indices based on dilated knn
    """

    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DenseDilatedKnnGraph, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k
        self._dilated = DenseDilated(k, dilation, stochastic, epsilon)

    def forward(self, x, y=None, relative_pos=None):
        if y is not None:
            x = functional.normalize(x, p=2.0, dim=1)
            y = functional.normalize(y, p=2.0, dim=1)
            edge_index = xy_dense_knn_matrix(x, y, self.k * self.dilation, relative_pos)
        else:
            x = functional.normalize(x, p=2.0, dim=1)
            edge_index = dense_knn_matrix(x, self.k * self.dilation, relative_pos)
        return self._dilated(edge_index)


class DyGraphConv2d(GraphConv2d):
    """
    Dynamic graph convolution layer
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=9,
        dilation=1,
        conv="edge",
        act="relu",
        norm=None,
        bias=True,
        stochastic=False,
        epsilon=0.0,
        r=1,
    ):
        super(DyGraphConv2d, self).__init__(
            in_channels, out_channels, conv, act, norm, bias
        )
        self.k = kernel_size
        self.d = dilation
        self.r = r
        self.dilated_knn_graph = DenseDilatedKnnGraph(
            kernel_size, int(dilation), stochastic, epsilon
        )

    def forward(self, x, relative_pos=None):
        """
        Forward pass.
        """
        B, C, H, W = x.shape
        y = None
        if self.r > 1:
            y = functional.avg_pool2d(x, self.r, self.r)
            y = y.reshape(B, C, -1, 1).contiguous()
        x = x.reshape(B, C, -1, 1).contiguous()
        edge_index = self.dilated_knn_graph(x, y, relative_pos)
        x = super(DyGraphConv2d, self).forward(x, edge_index, y)
        return x.reshape(B, -1, H, W).contiguous()


class Grapher(Module):
    """
    Grapher module with graph convolution and fc layers
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 9,
        dilation: int = 1,
        conv: str = "edge",
        act: str = "relu",
        norm: str = None,
        bias: bool = True,
        stochastic: bool = False,
        epsilon: float = 0.0,
        r: int = 1,
        n: int = 32,
        drop_path: float = 0.0,
        relative_pos: bool = False,
    ):
        super(Grapher, self).__init__()
        self.channels = in_channels
        self.n = n
        self.r = r
        self.fc1 = Sequential(
            Conv2d(
                in_channels,
                in_channels,
                1,
                stride=1,
                padding=0,
            ),
            BatchNorm2d(in_channels),
        )
        self.knn = kernel_size
        self.graph_conv = DyGraphConv2d(
            in_channels,
            in_channels,
            kernel_size,
            dilation,
            conv,
            act,
            norm,
            bias,
            stochastic,
            epsilon,
            r,
        )
        self.fc2 = Sequential(
            Conv2d(
                in_channels,
                in_channels,
                1,
                stride=1,
                padding=0,
            ),
            BatchNorm2d(in_channels),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()
        self.relative_pos = None
        if relative_pos:
            relative_pos_tensor = (
                torch.from_numpy(
                    np.float32(get_2d_relative_pos_embed(in_channels, int(n**0.5)))
                )
                .unsqueeze(0)
                .unsqueeze(1)
            )
            relative_pos_tensor = functional.interpolate(
                relative_pos_tensor,
                size=(n, n // (r * r)),
                mode="bicubic",
                align_corners=False,
            )
            self.relative_pos = Parameter(
                -relative_pos_tensor.squeeze(1), requires_grad=False
            )

    def __str__(self):
        return f"Grapher (in_channels={self.channels}, n={self.n}, r={self.r}, knn={self.knn})"

    def _get_relative_pos(self, relative_pos, H, W):
        if relative_pos is None or H * W == self.n:
            return relative_pos
        else:
            N = H * W
            N_reduced = N // (self.r * self.r)
            return functional.interpolate(
                relative_pos.unsqueeze(0), size=(N, N_reduced), mode="bicubic"
            ).squeeze(0)

    def forward(self, x):
        """
        Forward pass.
        """
        _tmp = x
        x = self.fc1(x)
        B, C, H, W = x.shape
        relative_pos = self._get_relative_pos(self.relative_pos, H, W)
        x = self.graph_conv(x, relative_pos)
        x = self.fc2(x)
        x = self.drop_path(x) + _tmp
        return x


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
                torch.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, (torch.BatchNorm2d, torch.GroupNorm)):
                torch.init.constant_(m.weight, 1)
                torch.init.constant_(m.bias, 0)

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
        res = self.grapherblock(x)
        self.log.debug(f"Grapher block output: {res.shape}")
        return res


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
                torch.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, (BatchNorm2d, GroupNorm, LayerNorm)):
                torch.init.constant_(m.weight, 1)
                torch.init.constant_(m.bias, 0)

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

        backbone.append(AdaptiveAvgPool2d((1, 1)))
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
        self.log.debug(f"Pooled shape: {x.shape}")
        x = x.squeeze(2).squeeze(2)
        return x


# %%


###################################
# Load the loss function
###################################
class Proposed(Module):
    def __init__(self, config, log, **kwargs):
        super(Proposed, self).__init__()
        self.log = log
        self.name = "Proposed Loss"
        self.num_classes = config["num_classes"]
        self.device = config["device"]
        self.centroids = Parameter(
            torch.randn((config["embedding_size"], self.num_classes)),
        )
        self.centroids.data.uniform_(-1, 1).renorm_(2, 1, 1)

        self.beta = torch.tensor(config["beta"]).to(self.device)

        self.fc = torch.nn.Linear(config["embedding_size"], self.num_classes)
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def margin_func(self, embds):
        return torch.norm(embds, p=2, dim=1)

    def forward(self, embds, labels, freeze_centroids=False):
        # xnorm = torch.norm(embds, p=2, dim=1).clamp(self.l_a, self.u_a)
        if len(embds.shape) != 2:
            embds = embds.view(1, embds.size(0))
        if len(labels.shape) != 2:
            labels = labels.view(1, labels.size(0))

        preds = self.fc(embds)
        loss1 = functional.cross_entropy(preds, labels, reduction="mean")
        wnorm = functional.normalize(self.centroids, p=2, dim=0)
        emb_norm = functional.normalize(embds, p=2, dim=1)
        self.centroids.requires_grad = not freeze_centroids

        if freeze_centroids:
            with torch.no_grad():
                cos_theta = torch.matmul(emb_norm, wnorm)
                cos_theta = cos_theta.clamp(-1, 1)
                output = torch.pi - torch.acos(cos_theta)
                loss2 = functional.cross_entropy(output, labels, reduction="mean")
        else:
            cos_theta = torch.matmul(emb_norm, wnorm)
            cos_theta = cos_theta.clamp(-1, 1)
            output = torch.pi - torch.acos(cos_theta)
            loss2 = functional.cross_entropy(output, labels, reduction="mean")

        # self.log.info(f"Loss1: {loss1}, Loss2: {loss2}")
        if torch.isnan(loss1):
            self.log.error("Loss1 is NaN")
            exit(1)

        if torch.isnan(loss2):
            self.log.error("Loss2 is NaN")
            exit(1)
        return loss1, self.beta * loss2, preds


###################################
# Training loop
###################################


def driver(config):
    logger_level = "DEBUG"
    log = get_logger("one_single_training_script", logger_level)

    set_seeds(log, config["seed"])
    # Add the data dirs
    train_wrapper = LeaveoneoutWrapper(
        config, log, rdirs=["./data/train_db1", "./data/train_db2"]
    )
    train_ds = train_wrapper.get_dataloader()

    test_db = LeaveoneoutWrapper(
        config, log, rdirs=["./data/train_db1"]
    ).get_dataloader()

    if config["num_classes"] != train_wrapper.num_classes:
        config["num_classes"] = train_wrapper.num_classes

    model = Dscgrapher(config, log).to(config["device"])
    log.info("Model:")
    log.info(str(model))

    criterion = Proposed(config, log).to(config["device"])
    params = [p for p in model.parameters() if p.requires_grad]
    params.extend([p for p in criterion.parameters() if p.requires_grad])

    optimizer = AdamW(
        params,
        lr=config["lr"],
        weight_decay=0.05,
    )
    metric = MulticlassAccuracy(num_classes=config["num_classes"]).cuda()
    device = config["device"]
    best_acc_val = 0
    loss_is_nan = False
    early_stop = 10
    validation_acc_didnt_increase = early_stop
    epochs = config["epochs"]
    ckptdir = "./run/checkpoints"
    os.makedirs(ckptdir, exist_ok=True)
    validate_after_epochs = config["validate_after_epochs"]
    for epoch in range(epochs):
        model.train()
        criterion.train()
        train_losses = []
        step1_train_losses = []
        step2_train_losses = []
        wandblog = {}
        pbar = tqdm(train_ds, desc=f"Epoch {epoch + 1}")
        i = 0
        for image, label in pbar:
            optimizer.zero_grad()
            image, label = image.to(device), label.to(device)
            preds = model(image)
            loss1, loss2, preds = criterion(
                preds, label, freeze_centroids=epoch > config["freeze_centroids"]
            )
            metric.update(preds.softmax(dim=1), label.argmax(dim=1))
            step_loss = loss1 + loss2
            step_loss.backward()
            optimizer.step()

            train_losses.append(step_loss.detach().cpu().item())
            step1_train_losses.append(loss1.detach().cpu().item())
            step2_train_losses.append(loss2.detach().cpu().item())
            pbar.set_postfix({"loss": np.mean(train_losses)})
            pbar.update(1)
            if train_losses[-1] == nan:
                loss_is_nan = True
                break

            i += 1
            if i == 10:
                continue
        pbar.close()
        if loss_is_nan:
            break

        log.info(f"Average train step loss: {np.mean(train_losses)}")
        wandblog = {
            "train_loss": np.mean(train_losses),
            "train_Step1_loss": np.mean(step1_train_losses),
            "train_acc": metric.compute().detach().cpu().item(),
            "train_Step2_loss": np.mean(step2_train_losses),
        }

        torch.save(
            model.state_dict(),
            os.path.join(ckptdir, f"epoch_{epoch}.pt"),
        )
        if not epoch % validate_after_epochs:
            validation_losses = []
            step1_losses = []
            step2_losses = []
            model.eval()
            criterion.eval()
            pbar = tqdm(test_db, desc="Validation")
            for image, label in pbar:
                image, label = image.to(device), label.to(device)
                preds = model(image)
                loss1, loss2, preds = criterion(preds, label)
                step_loss = loss1 + loss2
                validation_losses.append(step_loss.detach().cpu().item())
                metric.update(preds.softmax(dim=1), label.argmax(dim=1))
                step1_losses.append(loss1.detach().cpu().item())
                step2_losses.append(loss2.detach().cpu().item())
                pbar.set_postfix({"loss": np.mean(validation_losses)})
                if validation_losses[-1] == nan:
                    loss_is_nan = True
                    break

            pbar.close()
            if loss_is_nan:
                break
            validation_loss = np.mean(validation_losses)
            log.info(f"Average validation step loss: {validation_loss}")
            validation_acc = metric.compute().detach().cpu().item()
            wandblog = {
                **wandblog,
                "validation_loss": np.mean(validation_losses),
                "val_Step1_loss": np.mean(step1_losses),
                "val_Step2_loss": np.mean(step2_losses),
                "val_acc": validation_acc,
            }
            loss1total = np.mean(step1_losses)
            if validation_acc > best_acc_val:
                best_acc_val = validation_acc
                torch.save(
                    model.state_dict(),
                    os.path.join(ckptdir, "best_model.pt"),
                )
                validation_acc_didnt_increase = early_stop
            else:
                validation_acc_didnt_increase -= 1
                if validation_acc_didnt_increase == 0:
                    log.info(
                        f"Validation loss didn't decrease for {early_stop} epochs. Stopping training."
                    )
                    break
        # scheduler.step() # disable for arcvein
        # if not epoch % 30:
        #     for group in optimizer.param_groups:
        #         group["lr"] /= 10
        log.info(f"Wandblog: {pp.pformat(wandblog)}")


if __name__ == "__main__":
    config = {
        "batch_size": 128,
        "epochs": 150,
        "device": "cuda",
        "width": 224,
        "height": 224,
        "lr": 0.001,
        "validate_after_epochs": 1,
        "seed": 2025,
        "num_workers": 8,
        # dataset seed
        "stat_seed": 0,
        "leaveoutds": "polyu",
        # Hyper parameters for loss funtion
        "loss": "proposed",
        "num_classes": 301,  # For training on FV-300
        "freeze_centroids": 30,
        "embedding_size": 256,
        "beta": 0.3,
        # Hyper parameters for model
        "model": "dscgrapher",
        "conv": "mr",
        "stem": {
            "depth": 3,
            "indim": 3,
            "outdim": 64,
            "kernels": [9, 7, 3],
            "stride": 2,
            "bias": True,
            "act": "gelu",
        },
        "backbone": {
            "act": "gelu",
            "conv": "mr",
            "norm": "batch",
            "epsilon": 0.2,
            "drop_path": 0.0,
            "bias": True,
            "kernel_size": 9,
            "dilation": 1,
            "stochastic": False,
            "ffn_kernel": 1,
            "ffn_stride": 1,
            "depth": 2,
            "block0": {
                "indim": 64,
                "hdim": 256,
                "outdim": 128,
                "reduce_ratio": 1,
                "graphers": 4,
                "shrinker_kernel": 3,
                "shrinker_stride": 2,
                "neighbour_number": 18,
            },
            "block1": {
                "indim": 128,
                "hdim": 512,
                "outdim": 256,
                "reduce_ratio": 1,
                "graphers": 6,
                "shrinker_kernel": 3,
                "shrinker_stride": 2,
                "neighbour_number": 9,
            },
        },
    }
    driver(config)
