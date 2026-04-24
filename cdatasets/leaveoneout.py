import os
import random
from logging import Logger
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import set_image_backend
from torchvision import transforms as A

from utils import DatasetGenerator, Wrapper, image_extensions

set_image_backend("accimage")  # Faster image loading than PIL


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
        self.rdirs: List[str] = []

        for dataset in ["fvusm", "fv300", "mmcbnu", "polyu", "vera"]:
            if dataset != self.leaveoutds and not dataset.startswith("leaveoutds_"):
                self.rdirs.append(os.path.join("./data", dataset, str(self.stat_seed)))

        self.batch_size = config["batch_size"]
        self.num_workers = config.get("num_workers", 4)
        self.total_data: Dict[str, List[str]] = {}
        self.train_data: Dict[str, List[str]] = {}
        self.test_data: Dict[str, List[str]] = {}
        self.num_classes = None
        self.mintrain_imgs = 90
        # self.initialise_db() # Prefering old for now
        self.initialise_db_old()

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
        self, rdir: str, ssplit: str, prev: Dict[str, List[str]]
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
                    prev[ds + "_" + cid].append(os.path.join(cdir, img))
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
            DatasetGenerator(data, self.transform, **kwargs),
            num_workers=num_workers or self.num_workers,
            batch_size=batch_size or self.batch_size,
            shuffle=kwargs.get("shuffle", True),
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
