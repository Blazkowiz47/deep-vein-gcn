import os
import random
from logging import Logger
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import set_image_backend
from torchvision import transforms as A

from utils import DatasetGenerator, Wrapper, image_extensions

set_image_backend("accimage")  # Faster image loading than PIL


def build_kernel(kernel_values: list[list[float]], dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(kernel_values, dtype=dtype).view(1, 1, 3, 3)


def apply_input_mode(grayscale: torch.Tensor, mode: str | None) -> torch.Tensor:
    grayscale = grayscale.unsqueeze(0).unsqueeze(0)
    dtype = grayscale.dtype
    sobel_x = build_kernel(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype,
    )
    sobel_y = build_kernel(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        dtype,
    )
    diag_r_x = build_kernel(
        [[0.0, -1.0, -2.0], [1.0, 0.0, -1.0], [2.0, 1.0, 0.0]],
        dtype,
    )
    diag_r_y = build_kernel(
        [[-2.0, -1.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 1.0, 2.0]],
        dtype,
    )
    diag_l_x = build_kernel(
        [[0.0, 1.0, 2.0], [-1.0, 0.0, 1.0], [-2.0, -1.0, 0.0]],
        dtype,
    )
    diag_l_y = build_kernel(
        [[2.0, 1.0, 0.0], [1.0, 0.0, -1.0], [0.0, -1.0, -2.0]],
        dtype,
    )

    grad_x = F.conv2d(grayscale, sobel_x, padding=1)
    grad_y = F.conv2d(grayscale, sobel_y, padding=1)
    diag_r_resp_x = F.conv2d(grayscale, diag_r_x, padding=1)
    diag_r_resp_y = F.conv2d(grayscale, diag_r_y, padding=1)
    diag_l_resp_x = F.conv2d(grayscale, diag_l_x, padding=1)
    diag_l_resp_y = F.conv2d(grayscale, diag_l_y, padding=1)

    if mode is None:
        stacked = torch.cat((grayscale, grayscale, grayscale), dim=1)
    elif mode == "xy":
        stacked = torch.cat((grayscale, grad_x, grad_y), dim=1)
    elif mode == "dr":
        stacked = torch.cat((grayscale, diag_r_resp_x, diag_r_resp_y), dim=1)
    elif mode == "dl":
        stacked = torch.cat((grayscale, diag_l_resp_x, diag_l_resp_y), dim=1)
    else:
        raise ValueError(f"Unsupported input mode: {mode}")

    return stacked.squeeze(0)


class LeaveoneoutWrapper(Wrapper):
    def __init__(
        self,
        config: Dict[str, Any],
        log: Logger,
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
        self.stat_seed = config.get("stat_seed", 0)
        self.partition_split = kwargs.get("partition_split", 0.8)
        self.leaveoutds = config.get("leaveoutds", "vera")
        self.height = config.get("height", 224)
        self.width = config.get("width", 224)
        self.mode = config.get("mode")
        self.rdirs: List[str] = []

        for dataset in ["fvusm", "fv300", "mmcbnu", "polyu", "vera"]:
            if dataset != self.leaveoutds and not dataset.startswith("leaveoutds_"):
                self.rdirs.append(os.path.join("./data", dataset, str(self.stat_seed)))

        self.batch_size = config["batch_size"]
        self.num_workers = config.get("num_workers", 4)
        self.total_data: Dict[str, List[Any]] = {}
        self.train_data: Dict[str, List[Any]] = {}
        self.test_data: Dict[str, List[Any]] = {}
        self.num_classes = None
        self.mintrain_imgs = 90
        # self.initialise_db() # Prefering old for now
        self.initialise_db_old()

        self.train_augmentations = A.Compose(
            [
                A.RandomHorizontalFlip(),
                A.RandomVerticalFlip(),
                A.RandomAutocontrast(p=0.05),
                A.RandomRotation(45),
            ]
        )

    def initialise_db_old(self) -> None:
        for rdir in self.rdirs:
            self._internal_loop(rdir, "test", self.test_data)
            self._internal_loop(rdir, "train", self.train_data)

        self.num_classes = len(self.train_data)
        self.log.info(f"Number of classes: {self.num_classes}")

    def initialise_db(self) -> None:
        for ssplit in ["train", "test"]:
            self._internal_loop(ssplit, self.total_data)
        dataset_length = sum([len(v) for v in self.total_data.values()])
        self.num_classes = len(self.total_data)
        self.log.debug(f"Data-length for {self.name} split: {dataset_length}")
        self.log.debug(f"Number of classes for {self.name} split: {self.num_classes}")

        for cid in self.total_data:
            random.shuffle(self.total_data[cid])
            partition_index = int(self.partition_split * len(self.total_data[cid]))
            self.train_data[cid] = self.total_data[cid][:partition_index]
            self.test_data[cid] = self.total_data[cid][partition_index:]

    def _internal_loop(
        self, rdir: str, ssplit: str, prev: Dict[str, List[Any]]
    ) -> None:
        rdir = os.path.join(rdir, ssplit)
        for cid in os.listdir(rdir):
            ds = rdir.split("/")[-3]
            if cid not in prev:
                prev[ds + "_" + cid] = []
            cdir = os.path.join(rdir, cid)
            imgs = 0
            for img in os.listdir(cdir):
                if "." + img.split(".")[-1].lower() in image_extensions:
                    img_path = os.path.join(cdir, img)
                    with Image.open(img_path) as image:
                        prev[ds + "_" + cid].append((np.array(image), img_path))
                    imgs += 1

            if ssplit == "train" and self.mintrain_imgs > imgs:
                self.mintrain_imgs = imgs

    def loop_splitset(self, ssplit: str) -> List[Any]:
        if ssplit == "train":
            data = self.train_data
        else:
            data = self.test_data

        datalist: List[Any] = []
        for cid, class_name in enumerate(data.keys()):
            for img in data[class_name]:
                datalist.append((img, cid))

        return datalist

    def get_split(
        self,
        split: str,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        **kwargs: Any,
    ) -> DataLoader:
        batch_size = batch_size or self.batch_size
        self.log.debug("Looping through %s split." % split)
        data = self.loop_splitset(split)
        self.log.debug("Data-length for %s split: %d" % (split, len(data)))
        return DataLoader(
            DatasetGenerator(
                data,
                self.train_transform if split == "train" else self.test_transform,
                **kwargs,
            ),
            num_workers=num_workers or self.num_workers,
            batch_size=batch_size or self.batch_size,
            shuffle=True,
            prefetch_factor=num_workers or self.num_workers,
        )

    def train_augment(self, image: Any) -> Any:
        return self.train_augmentations(image)

    def test_augment(self, image: Any) -> Any:
        return image

    def test_transform(self, datapoint: Iterable[Any], **kwargs) -> Tuple:
        img_data, lbl = datapoint
        if self.num_classes is None:
            raise ValueError("Num classes not set.")
        # Initialise label
        img, imgfname = img_data
        label = torch.zeros(self.num_classes)
        label[lbl] = 1

        imgarray = np.array(img)
        grayscale = torch.tensor(imgarray / 255.0, dtype=torch.float32)
        imgarray = apply_input_mode(grayscale, self.mode)
        imgarray = self.test_augment(imgarray)
        imgarray = F.interpolate(
            imgarray.unsqueeze(0),
            size=(self.height, self.width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        if kwargs.get("return_filename"):
            return imgarray.float(), label.float(), imgfname

        return imgarray.float(), label.float()

    def train_transform(self, datapoint: Iterable[Any], **kwargs) -> Tuple:
        img_data, lbl = datapoint
        if self.num_classes is None:
            raise ValueError("Num classes not set.")
        # Initialise label
        img, imgfname = img_data
        label = torch.zeros(self.num_classes)
        label[lbl] = 1

        imgarray = np.array(img)
        grayscale = torch.tensor(imgarray / 255.0, dtype=torch.float32)
        imgarray = apply_input_mode(grayscale, self.mode)
        imgarray = self.train_augment(imgarray)
        imgarray = F.interpolate(
            imgarray.unsqueeze(0),
            size=(self.height, self.width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        if kwargs.get("return_filename"):
            return imgarray.float(), label.float(), imgfname

        return imgarray.float(), label.float()
