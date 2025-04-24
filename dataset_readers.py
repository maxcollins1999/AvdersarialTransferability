### Preamble ##########################################################################################################

"""
Pytorch dataset classes for image datasets used in experiments.
"""

#######################################################################################################################

### Imports ###########################################################################################################

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from concurrent.futures import ThreadPoolExecutor
from torchvision.utils import save_image
from datetime import datetime, timedelta
import os
import pandas as pd
import timm
import pathlib
import json
import pickle
import shutil
import logging
from tqdm import tqdm
import argparse

from typing import Union, Iterable, Optional, Tuple, List, Any, Callable, Dict


### ImageNet Classes ##################################################################################################

with open("imagenet_classes.json", "r") as fstrm:
    IMAGENET_CLASSES = json.load(fstrm)
IMAGENET_CLASSES["id2label"] = {
    int(k): v for k, v in IMAGENET_CLASSES["id2label"].items()
}
IMAGENET_CLASSES["id2synset"] = {
    int(k): v for k, v in IMAGENET_CLASSES["id2synset"].items()
}

#######################################################################################################################

VALID_IMAGE_EXT = [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif"]


class ImageDataset(Dataset):
    """
    Dataset class for reading any images in a directory or subdirectory.
    """

    def __init__(
        self,
        image_dir: Union[str, pathlib.PosixPath, list[str], list[pathlib.PosixPath]],
        transforms: torch.nn.Module = None,
    ):
        """
        :param image_dir: str, pathlib.PosixPath or list of strings or pathlib.PosixPath objects.
            Directory or directories containing the images
        :param transforms: torch.nn.Module
            Transforms to apply to the images.
        """

        if not isinstance(image_dir, list):
            image_dir = [image_dir]

        for i, dir in enumerate(image_dir):
            if not isinstance(dir, pathlib.PosixPath):
                image_dir[i] = pathlib.Path(dir)

        self.image_dir = image_dir
        self.transforms = transforms

        self.image_paths = []

        for dir in self.image_dir:
            for dirpath, _, files in os.walk(dir):
                for f in files:
                    if any([f.lower().endswith(ext) for ext in VALID_IMAGE_EXT]):
                        self.image_paths.append(os.path.join(dirpath, f))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: Union[int, List[int]]) -> torch.Tensor:
        img_path = self.image_paths[idx]
        img = read_image(img_path, mode=ImageReadMode.RGB)
        if self.transforms is not None:
            img = self.transforms(img)

        return img


class NatAdvDiffImageDataset(Dataset):
    """
    Dataset class for reading the outputs of the adversarial diffusion experiments. Returns `[image, diffusion label,
    adversarial label]`. Note that if the experiment does not have an adversarial target, then the adversarial label
    will be a `torch.nan`.
    """

    def __init__(
        self,
        image_dir: Union[str, pathlib.PosixPath],
        transforms: torch.nn.Module = None,
    ):
        """
        :param image_dir: str or pathlib.PosixPath
            Directory containing the images
        :param transforms: torch.nn.Module
            Transforms to apply to the images.
        """

        if not isinstance(image_dir, pathlib.PosixPath):
            image_dir = pathlib.Path(image_dir)

        self.image_dir = image_dir
        self.transforms = transforms
        self.exp_id = []

        self.image_files = [
            f
            for f in os.listdir(self.image_dir)
            if any([f.lower().endswith(ext) for ext in VALID_IMAGE_EXT])
        ]
        self.image_files.sort()

        # Verify that the file names are as expected and get experiment id
        for f in self.image_files:
            if not f.startswith("image)") and not any(
                [f.lower().endswith(ext) for ext in VALID_IMAGE_EXT]
            ):
                raise FileNotFoundError(
                    f"Expected image with name image_<ID> and valid image extension, got {f}"
                )
            self.exp_id.append(int(f.split("_")[1].split(".")[0]))

        # Get metadata
        metadata_path = image_dir / "metadata.txt"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Did not find metadata file at {metadata_path}")
        metadata = pd.read_csv(metadata_path, delimiter="\t")

        exp_index = [ele - 1 for ele in self.exp_id]

        # Get diffusion target
        if "Target Diffusion Class" in metadata.columns:
            self.diffusion_target = torch.tensor(
                metadata["Target Diffusion Class"], dtype=torch.int
            )[exp_index]
        else:
            raise KeyError(
                "The `metadata.txt` file must have a 'Target Diffusion Class' column."
            )

        # Get adversarial target
        if "Target Adversarial Class" in metadata.columns:
            self.adversarial_target = torch.tensor(
                metadata["Target Adversarial Class"], dtype=torch.int
            )[exp_index]
        else:
            self.adversarial_target = None

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(
        self, idx: Union[int, List[int]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_name = self.image_files[idx]
        img = read_image(self.image_dir / img_name, mode=ImageReadMode.RGB)
        if self.transforms is not None:
            img = self.transforms(img)

        diff_target = self.diffusion_target[idx]

        if self.adversarial_target is not None:
            adv_target = self.adversarial_target[idx]
        else:
            adv_target = torch.nan

        return img, diff_target, adv_target


class ImageNetDataset(Dataset):
    """
    Dataset class for reading the unpacked ImageNet or ImageNet-A datasets. Returns [image, label]. Note that if the
    'test' ImageNet dataset is selected then `label` will be `torch.nan`.
    """

    def __init__(
        self,
        image_dir: Union[str, pathlib.PosixPath],
        imagenet_type: str,
        transforms: torch.nn.Module = None,
    ):
        """
        :param image_dir: str or pathlib.PosixPath
            Directory containing the images.
        :param imagenet_type: str
            One of `train`, `val`, `test`, or `imagenet-a`. Denotes the type of of the imagenet dataset
        :param transforms: torch.nn.Module
            Transforms to apply to the images.
        """

        if not isinstance(image_dir, pathlib.PosixPath):
            image_dir = pathlib.Path(image_dir)

        parent_dir = image_dir
        if imagenet_type.lower() == "train":
            self.image_dir = image_dir / "ILSVRC/Data/CLS-LOC/train"
        elif imagenet_type.lower() == "imagenet-a":
            self.image_dir = image_dir
        elif imagenet_type.lower() == "val":
            self.image_dir = image_dir / "ILSVRC/Data/CLS-LOC/val"
        elif imagenet_type.lower() == "test":
            self.image_dir = image_dir / "ILSVRC/Data/CLS-LOC/test"
        else:
            raise ValueError(
                "Unsupported ImageNet imagenet_type, expected one of `train`, `val`, `test`, or ",
                f"`imagenet-a`, got{imagenet_type}",
            )

        self.transforms = transforms
        self.image_paths = []
        self.image_classes = []

        if imagenet_type.lower() in ["train", "imagenet-a"]:
            iter_ims = os.walk(self.image_dir)
            next(iter_ims)  # Skipping the root directory
            for dirpath, _, files in iter_ims:
                synset = dirpath.split(os.sep)[-1]
                image_class = IMAGENET_CLASSES["synset2id"][synset]
                for f in files:
                    if any([f.lower().endswith(ext) for ext in VALID_IMAGE_EXT]):
                        self.image_paths.append(os.path.join(dirpath, f))
                        self.image_classes.append(image_class)
            self.image_classes = torch.tensor(self.image_classes, dtype=torch.int)
        elif imagenet_type.lower() == "val":
            image_files = [
                f
                for f in os.listdir(self.image_dir)
                if any([f.lower().endswith(ext) for ext in VALID_IMAGE_EXT])
            ]
            # Get image-class mappings
            class_mapping = {}
            with open(parent_dir / "LOC_val_solution.csv", "r") as fstrm:
                lines = fstrm.readlines()
                for line in lines[1:]:
                    tmp = line.split(",")
                    class_mapping[tmp[0]] = IMAGENET_CLASSES["synset2id"][
                        tmp[1].split(" ")[0]
                    ]

            # Get image classes
            for f in image_files:
                self.image_paths.append(self.image_dir / f)
                self.image_classes.append(class_mapping[f.split(".")[0]])
            self.image_classes = torch.tensor(self.image_classes, dtype=torch.int)
        elif imagenet_type.lower() == "test":
            self.image_paths = [
                self.image_dir / f
                for f in os.listdir(self.image_dir)
                if any([f.lower().endswith(ext) for ext in VALID_IMAGE_EXT])
            ]
            self.image_classes = None

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(
        self, idx: Union[int, List[int]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        img = read_image(img_path, mode=ImageReadMode.RGB)
        if self.transforms is not None:
            img = self.transforms(img)

        if self.image_classes is not None:
            img_class = self.image_classes[idx]
        else:
            img_class = torch.nan

        return img, img_class


#######################################################################################################################
