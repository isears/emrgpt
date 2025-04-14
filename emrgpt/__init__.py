print("*** EMR GPT ***")

import lightning as L
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Optional
import os
import torch
from torch.nn import functional as F
from typing import Literal


class BasicDM(L.LightningDataModule):

    def __init__(
        self,
        ds: Dataset,
        batch_size: int = 32,
        workers: Optional[int] = None,
        train_valid_splits: tuple[float, float] = (0.9, 0.1),
    ):
        super().__init__()

        self.batch_size = batch_size
        self.train_valid_splits = train_valid_splits
        self.core_ds = ds

        if workers:
            self.cores_available = workers
        else:
            self.cores_available = len(os.sched_getaffinity(0))

        print(f"[*] Initializing DM with {self.cores_available} workers")

    def setup(self, stage: str):
        self.train_ds, self.valid_ds = random_split(
            self.core_ds,
            self.train_valid_splits,
            generator=torch.Generator().manual_seed(42),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, num_workers=self.cores_available, batch_size=self.batch_size
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_ds, num_workers=self.cores_available, batch_size=self.batch_size
        )
