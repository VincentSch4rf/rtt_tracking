#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 26.03.22 10:33
# @Author  : Vincent Scharf
# @File    : datasets.py

# Reference Link: https://pytorch.org/vision/master/_modules/torchvision/datasets/kitti.html
import csv
import json
import os
from typing import Dict, Tuple, Any, Optional, Callable, List

import torch
import yaml
from PIL import Image
from torch import Tensor
from torchvision.datasets import VisionDataset


class AtWork(VisionDataset):
    """RoboCup @ Work KITTI style dataset.

    Args:
        root (string): Root directory where images are downloaded to.
            Expects the following folder structure if download=False:

            .. code::

                <root>
                    └── AtWork
                        └─ atwork_augmented_and_real_robocup_2019
                            ├── training
                            |   ├── image_2
                            |   └── label_2
                            └── testing
                                └── image_2
        train (bool, optional): Use ``train`` split if true, else ``test`` split.
            Defaults to ``train``.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample
            and its target as entry and returns a transformed version.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    data_url = ""
    resources = [
        "data_object_image_2.zip",
        "data_object_label_2.zip",
    ]
    image_dir_name = "image_2"
    labels_dir_name = "label_2"

    def __init__(
            self,
            root: str,
            augmented: bool = True,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
            download: bool = False,
    ):
        super().__init__(
            root,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
        )
        self.images = []
        self.targets = []
        self.root = root
        self.train = train
        self._augmented = augmented
        self._location = "training" if self.train else "testing"
        self._labels = ["axis", "bearing", "bearing_box_ax01", "bearing_box_ax16", "container_box_blue",
                        "container_box_red", "distance_tube", "em_01", "em_02", "f20_20_B", "f20_20_G", "m20",
                        "m20_100", "m30", "motor", "r20", "s40_40_B", "s40_40_G"]
        self._label_lookup = {self._labels[i]: i for i in range(len(self._labels))}

        if download:
            self.download()
        if not self._check_exists():
            raise RuntimeError("Dataset not found. You may use download=True to download it.")

        image_dir = os.path.join(self._raw_folder, self._location, self.image_dir_name)
        if self.train:
            labels_dir = os.path.join(self._raw_folder, self._location, self.labels_dir_name)
        for img_file in os.listdir(image_dir):
            self.images.append(os.path.join(image_dir, img_file))
            if self.train:
                self.targets.append(os.path.join(labels_dir, f"{img_file.rsplit('.', 1)[0]}.txt"))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Get item at a given index.

        Args:
            index (int): Index
        Returns:
            tuple: (image, target), where
            target is a list of dictionaries with the following keys:

            - type: str
            - truncated: float
            - occluded: int
            - alpha: float
            - bbox: float[4]
            - dimensions: float[3]
            - locations: float[3]
            - rotation_y: float

        """
        image = Image.open(self.images[index])
        target = self._parse_target(index) if self.train else None
        if self.transforms:
            image, target = self.transforms(image, target)
        return image, target

    def _calc_areas(self, coords) -> Tensor:
        areas = []
        for coord in coords:
            areas.append(torch.mul(torch.abs(torch.sub(coord[0], coord[2])), torch.abs(torch.sub(coord[1], coord[3]))))
        return torch.as_tensor(areas)

    def _parse_target(self, index: int) -> Dict:
        target = {"image_id": index}
        with open(self.targets[index]) as inp:
            content = csv.reader(inp, delimiter=" ")
            for i, line in enumerate(content):
                if i == 0:
                    target["labels"] = [self._label_lookup[line[0]]]
                    target["boxes"] = [[float(x) for x in line[4:8]]]
                else:
                    target["labels"].append(self._label_lookup[line[0]])
                    target["boxes"].append([float(x) for x in line[4:8]])
            try:
                target["labels"] = torch.tensor(target["labels"], dtype=torch.int64)
                target["boxes"] = torch.tensor(target["boxes"], dtype=torch.float)
                target["area"] = self._calc_areas(target["boxes"])
            except KeyError:
                # Empty tensors in case of no detections present in frame
                target["labels"] = torch.zeros(0, dtype=torch.int64)
                target["boxes"] = torch.zeros(0, 4)
                target["area"] = torch.zeros(0)
            target["scores"] = torch.ones(target["labels"].size(0))
            target["image_id"] = torch.tensor(target["image_id"], dtype=torch.int64)
            target["iscrowd"] = torch.zeros(target["labels"].size(0), dtype=torch.uint8)

        return target

    def __len__(self) -> int:
        return len(self.images)

    @property
    def _raw_folder(self) -> str:
        # "atwork_augmented_and_real_robocup_2019"
        return os.path.join(self.root, self.__class__.__name__, "augmented" if self._augmented else "real")

    def _check_exists(self) -> bool:
        """Check if the data directory exists."""
        folders = [self.image_dir_name]
        if self.train:
            folders.append(self.labels_dir_name)
        return all(os.path.isdir(os.path.join(self._raw_folder, self._location, fname)) for fname in folders)

    def download(self) -> None:
        """Download the KITTI data if it doesn't exist already."""

        if self._check_exists():
            return

        os.makedirs(self._raw_folder, exist_ok=True)

        # download files
        # for fname in self.resources:
        #     download_and_extract_archive(
        #         url=f"{self.data_url}{fname}",
        #         download_root=self._raw_folder,
        #         filename=fname,
        #     )


class RttDataset(VisionDataset):
    """RoboCup @ Work KITTI style dataset.

        Args:
            root (string): Root directory where images are downloaded to.
                Expects the following folder structure if download=False:

                .. code::

                    <root>
                        └── RttDataset
                            ├── training
                            |   ├── image_2
                            |   └── label_2
                            └── testing
                                └── image_2
            train (bool, optional): Use ``train`` split if true, else ``test`` split.
                Defaults to ``train``.
            transform (callable, optional): A function/transform that takes in a PIL image
                and returns a transformed version. E.g, ``transforms.ToTensor``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
            transforms (callable, optional): A function/transform that takes input sample
                and its target as entry and returns a transformed version.
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.

        """

    data_url = ""
    resources = [
        "data_object_image_2.zip",
        "data_object_label_2.zip",
    ]
    image_dir_name = "images"
    labels_dir_name = "labels"

    def __init__(
            self,
            root: str,
            split: str = "train",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
            download: bool = False,
    ):
        super().__init__(
            root,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
        )
        self.images = []
        self.targets = []
        self.root = root
        self.split = split
        # self.include_labels = True if self.split in ["train", "valid"] else False
        self.__parse_metadata(os.path.join(self._raw_folder, "data.yaml"))

        if download:
            self.download()
        if not self._check_exists():
            raise RuntimeError("Dataset not found. You may use download=True to download it.")

        image_dir = os.path.join(self._raw_folder, self.split, self.image_dir_name)
        labels_dir = os.path.join(self._raw_folder, self.split, self.labels_dir_name)
        for img_file in os.listdir(image_dir):
            self.images.append(os.path.join(image_dir, img_file))
            self.targets.append(os.path.join(labels_dir, f"{img_file.rsplit('.', 1)[0]}.json"))

    def __parse_metadata(self, _file):
        with open(_file, "r") as f:
            metadata = yaml.safe_load(f)
        self.nc = metadata["nc"]
        self.labels = {metadata["names"][i]: i for i in range(len(metadata["names"]))}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Get item at a given index.

        Args:
            index (int): Index
        Returns:
            tuple: (image, target), where
            target is a list of dictionaries with the following keys:

            - type: str
            - truncated: float
            - occluded: int
            - alpha: float
            - bbox: float[4]
            - dimensions: float[3]
            - locations: float[3]
            - rotation_y: float

        """
        image = Image.open(self.images[index])
        target = self._parse_target(index)
        if self.transforms:
            image, target = self.transforms(image, target)
        return image, target

    def _parse_target(self, index: int) -> Dict:
        target = {
            "labels": [],
            "boxes": [],
            "track_ids": []
        }
        with open(self.targets[index]) as inp:
            content = json.load(inp)

        for shape in content["shapes"]:
            label = shape["label"].split(" ")
            target["labels"].append(self.labels[label[0]])
            try:
                target["track_ids"].append(int(label[1]))
            except IndexError:
                pass
            target["boxes"].append([value for coord in shape["points"] for value in coord])
            target["frame"] = content["imagePath"]
            target["label_file"] = self.targets[index]

        target["labels"] = torch.tensor(target["labels"], dtype=torch.int)
        target["boxes"] = torch.tensor(target["boxes"], dtype=torch.float)
        target["track_ids"] = torch.tensor(target["track_ids"], dtype=torch.int)
        # except KeyError:
        #     # Empty tensors in case of no detections present in frame
        #     target["labels"] = torch.zeros(0, dtype=torch.int)
        #     target["boxes"] = torch.zeros(0, 4, dtype=torch.float)
        #     target["track_ids"] = torch.zeros(0, dtype=torch.int)

        return target

    def __len__(self) -> int:
        return len(self.images)

    @property
    def _raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__)

    def _check_exists(self) -> bool:
        """Check if the data directory exists."""
        folders = [self.image_dir_name]
        folders.append(self.labels_dir_name)
        return all(os.path.isdir(os.path.join(self._raw_folder, self.split, fname)) for fname in folders)

    def download(self) -> None:
        if self._check_exists():
            return
        os.makedirs(self._raw_folder, exist_ok=True)


def box_xyxy_to_cxcywh(boxes: torch.Tensor, size: Tuple[int, int]):
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2 / size[0]
    cy = (y1 + y2) / 2 / size[1]
    w = abs(x2 - x1) / size[0]
    h = abs(y2 - y1) / size[1]

    boxes = torch.stack((cx, cy, w, h), dim=-1)

    return boxes


if __name__ == "__main__":
    for split in ["train", "test", "valid"]:
        ds = RttDataset("../../data/", split=split)
        for frame in ds:
            image, target = frame
            with open(os.path.join(f"../../data/RttDataset_yolo/{split}/labels", target["frame"].split('.')[0] + ".txt"), "w") as f:
                writer = csv.writer(f, delimiter=' ')
                for i in range(len(target["labels"])):
                    row = [target["labels"][i].tolist()]
                    row.extend(box_xyxy_to_cxcywh(target["boxes"][i], (640, 480)).tolist())
                    writer.writerow(row)
