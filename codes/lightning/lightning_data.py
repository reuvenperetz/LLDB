from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch.utils.data as data
import torch.distributed as dist

from data import create_dataset


class LLDBDataModule(pl.LightningDataModule):
    def __init__(self, opt: Dict[str, Any]):
        super().__init__()
        self.opt = opt
        self.train_opt = opt["datasets"]["train"]
        self.val_opt = opt["datasets"].get("val")
        self.test_opt = opt["datasets"].get("test")

        self._train_set = None
        self._val_set = None
        self._test_set = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self._train_set = create_dataset(self.train_opt)
            max_train_images = self.train_opt.get("max_train_images")
            if max_train_images is not None:
                self._train_set = data.Subset(
                    self._train_set, list(range(min(len(self._train_set), max_train_images)))
                )
            if self.val_opt is not None:
                self._val_set = create_dataset(self.val_opt)
        if stage in (None, "test"):
            if self.test_opt is not None:
                self._test_set = create_dataset(self.test_opt)

    def train_dataloader(self):
        batch_size = self.train_opt["batch_size"]
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            if batch_size % world_size != 0:
                raise ValueError(
                    f"batch_size ({batch_size}) must be divisible by world_size ({world_size})"
                )
            batch_size = batch_size // world_size
        num_workers = self.train_opt.get("n_workers", 4)
        return data.DataLoader(
            self._train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=False,
        )

    def val_dataloader(self):
        if self._val_set is None:
            return None
        num_workers = self.val_opt.get("n_workers", 0)
        return data.DataLoader(
            self._val_set,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        if self._test_set is None:
            return None
        num_workers = self.test_opt.get("n_workers", 0)
        return data.DataLoader(
            self._test_set,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
