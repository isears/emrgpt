from torch.utils.data import Dataset, DataLoader, Subset
import torch
import lightning as L
from emrgpt import BasicDM


class ShakespeareDS(Dataset):
    def __init__(self, path: str, block_size: int):
        super().__init__()
        torch.manual_seed(1337)

        self.path = path
        self.block_size = block_size

        with open(self.path, "r", encoding="utf-8") as f:
            self.text = f.read()

        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)

        self.encoding_map = {ch: i for i, ch in enumerate(self.chars)}
        self.decoding_map = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text: str) -> torch.Tensor:
        encoded = [self.encoding_map[c] for c in text]
        return torch.tensor(encoded, dtype=torch.long)

    def decode(self, data: torch.Tensor) -> str:
        decoded = [self.decoding_map[i] for i in data.tolist()]
        return "".join(decoded)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.text[index : index + self.block_size]
        y = self.text[index + 1 : index + self.block_size + 1]

        return self.encode(x), self.encode(y)


class ShakespeareDM(BasicDM):
    def __init__(self, path: str, batch_size: int, block_size: int):
        super().__init__(ShakespeareDS(path, block_size), batch_size)

    def setup(self, stage: str):
        n = int(0.9 * len(self.core_ds))
        self.train_ds = Subset(self.core_ds, indices=range(0, n))
        self.valid_ds = Subset(
            self.core_ds, indices=range(n, len(self.core_ds) - self.core_ds.block_size)
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            num_workers=self.cores_available,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_ds,
            num_workers=self.cores_available,
            batch_size=self.batch_size,
            shuffle=True,
        )
