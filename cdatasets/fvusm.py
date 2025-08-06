import os
import random
from logging import Logger
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torchvision import transforms as A
from PIL import Image
from torch.utils.data import DataLoader
from utils import DatasetGenerator, Wrapper, image_extensions


class FvusmWrapper(Wrapper):
    def __init__(
        self,
        config: Dict[str, Any],
        log: Logger,
        **kwargs,
    ):
        self.name = "fvusm"
        self.log = log
        self.kwargs: Dict[str, Any] = kwargs
        self.stat_seed = kwargs.get("stat_seed", 0)
        self.partition_split = kwargs.get("partition_split", 0)
        self.height = config.get("height", 224)
        self.width = config.get("width", 224)
        self.rdir = f"./data/fvusm/{self.stat_seed}"

        self.batch_size = config["batch_size"]
        self.num_workers = config["num_workers"]
        self.total_data: Dict[str, List[str]] = {}
        self.train_data: Dict[str, List[str]] = {}
        self.test_data: Dict[str, List[str]] = {}
        self.num_classes = None
        self.initialise_db()

        self.augmentations = A.Compose(
            [
                A.ToTensor(),
                A.Resize((self.height, self.width)),
            ]
        )

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

    def _internal_loop(self, ssplit: str, prev: Dict[str, List[str]]) -> None:
        rdir = os.path.join(self.rdir, ssplit)
        for cid in os.listdir(rdir):
            if cid not in prev:
                prev[cid] = []
            cdir = os.path.join(rdir, cid)

            for img in os.listdir(cdir):
                if "." + img.split(".")[-1].lower() in image_extensions:
                    prev[cid].append(os.path.join(cdir, img))

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
    ) -> DataLoader:
        batch_size = batch_size or self.batch_size
        self.log.debug("Looping through %s split." % split)
        data = self.loop_splitset(split)
        self.log.debug("Data-length for %s split: %d" % (split, len(data)))
        return DataLoader(
            DatasetGenerator(data, self.transform),
            num_workers=num_workers or self.num_workers,
            batch_size=batch_size or self.batch_size,
            shuffle=True,
            prefetch_factor=2,
        )

    def augment(self, image: Any) -> Any:
        return self.augmentations(image)

    def transform(self, datapoint: Iterable[Any]) -> Tuple:
        img, lbl = datapoint
        if self.num_classes is None:
            raise ValueError("Num classes not set.")
        # Initialise label
        label = torch.zeros(self.num_classes)
        label[lbl] = 1

        # Initialise image
        imgarray = np.array(img)
        imgarray = (imgarray - imgarray.min()) / (imgarray.max() - imgarray.min())
        imgarray = np.stack([imgarray, imgarray, imgarray], axis=2)

        imgarray = self.augment(imgarray)

        return imgarray.float(), label.float()
