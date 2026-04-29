from datetime import datetime
from logging import Logger
import os
import random
from abc import abstractmethod
from typing import Any, Callable, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset


image_extensions: List[str] = [".jpg", ".png", ".jpeg", ".bmp"]
video_extensions: List[str] = [".mov", ".mp4"]


def set_seeds(log: Logger, seed: int):
    """
    Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 2024.
    """
    log.debug(f"Setting seed to: {seed}")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_run_name(model: str, dataset: str, seed=None) -> str:
    if seed is not None:
        return f"{model}_{dataset}_{seed}"
    return f"{model}_{dataset}_{datetime.now().strftime('%d_%m_%y_%H_%M_2.2%S').replace('.', '_')}"


def initialise_dirs(model_name: str):
    """
    Initialises all the required directories.
    """
    os.makedirs(rf"tmp/{model_name}/checkpoints", exist_ok=True)


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
