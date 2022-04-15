#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 16.03.22 20:15
# @Author  : Vincent Scharf
# @File    : rtt_yolo.py
from typing import List, Optional, Dict, Tuple

import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch import nn, Tensor
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, ToTensor
from yolort.models import yolov5s
from yolort.trainer import DefaultTask
from pytorch_lightning.trainer import Trainer
from pytorch_lightning import loggers as pl_loggers

from datasets import AtWork, RttDataset


# torch.multiprocessing.set_sharing_strategy('file_system')


def collate_fn(batch):
    return tuple(zip(*batch))


class YoloAtWork(DefaultTask):

    def __init__(self, arch, version, lr, root: str = 'data', **kwargs):
        super().__init__(arch, version, lr, num_classes=18, **kwargs)

        transform = Compose([
            ToTensor(),
        ])

        self.train_dataset = RttDataset(root, split="train", transform=transform)
        print(f"Train Size: {len(self.train_dataset)}")
        self.val_dataset = RttDataset(root, split="valid", transform=transform)
        print(f"Valid Size: {len(self.val_dataset)}")
        self.test_dataset = RttDataset(root, split="test", transform=transform)
        print(f"Test Size: {len(self.test_dataset)}")
        self.batch_size = 16

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True,
                          num_workers=8, persistent_workers=True, collate_fn=collate_fn)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True,
                          num_workers=8, persistent_workers=True, collate_fn=collate_fn)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True,
                          num_workers=8, persistent_workers=True, collate_fn=collate_fn)


def main():
    arch = "yolov5s"
    version = "r6.0"
    lr = 0.01
    task = YoloAtWork(arch=arch, version=version, lr=lr, root="../../data", pretrained=False, annotation_path="../../data/RttDataset/test/labels")
    logger = pl_loggers.TensorBoardLogger(save_dir=f"lightning_logs/rtt",
                                          name=f"{arch}_{version}_{lr}")
    checkpoint_callback = ModelCheckpoint(
        monitor="avg_val_iou",
        dirpath=f"./out/models/",
        filename=f"{arch}_{version}",
        save_top_k=1,
        mode="max",
    )

    callbacks = [checkpoint_callback]
    trainer = Trainer(logger=logger, max_epochs=300, log_every_n_steps=1, gpus=-1,
                      check_val_every_n_epoch=1, enable_progress_bar=True, enable_model_summary=False,
                      callbacks=callbacks)  # resume_from_checkpoint="./out/models/yolov5s_r6.0-v1.ckpt")
    # tuner = Tuner(trainer)
    # tuner.scale_batch_size(task)
    trainer.fit(task)
    trainer.test(task)


if __name__ == "__main__":
    main()
