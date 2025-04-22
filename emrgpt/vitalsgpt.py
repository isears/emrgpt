"""
Very minimal example of EMR GPT operating only on vitals
"""

import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from emrgpt.model import EventBasedEmrGPT
from torch.nn import functional as F
from torchinfo import summary

BLOCK_SIZE = 256
MAX_EPOCHS = 10
EVAL_INTERVAL = 10
VALUE_LOSS_MULTIPLIER = 0.0001
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
EVAL_STEPS = 100
DEVICE = "cuda"
N_HEAD = 6
N_LAYER = 6
N_EMBD = 384
DROPOUT = 0.2


class VitalsDS(Dataset):
    def __init__(self, data: pd.DataFrame):
        super().__init__()
        self.df = data

        # TODO: ultimately this should be subject_ids, not stay_ids, but using stay_ids for simplicity
        # self.unique_subjects = {
        #     sid: list(data[data["subject_id"] == sid]["stay_id"].unique())
        #     for sid in data["subject_id"].unique()
        # }

        self.unique_stays = self.df["stay_id"].drop_duplicates().values

        # Tokenize
        self.tokenization_map = {
            s: i for i, s in enumerate(vitals_df["event_type"].unique())
        }
        self.detokenization_map = {v: k for k, v in self.tokenization_map.items()}
        vitals_df["event_id"] = vitals_df["event_type"].apply(
            lambda e: self.tokenization_map[e]
        )

    def encode_event(self, s: str):
        return self.tokenization_map[s]

    def decode_event(self, i: int):
        return self.detokenization_map[i]

    def __len__(self):
        return len(self.unique_stays)

    def __getitem__(self, index: int):
        selected_stay = self.unique_stays[index]

        ret = self.df[self.df["stay_id"] == selected_stay][
            ["tidx", "event_id", "event_value"]
        ]

        # NOTE: returning block size + 1 so that both an x and a y can be created from the batch
        if len(ret) > BLOCK_SIZE + 1:
            start_idx = torch.randint(0, (len(ret) - (BLOCK_SIZE + 1)), (1,)).item()

            ret = ret[start_idx : start_idx + BLOCK_SIZE + 1]
        else:
            pad_len = (BLOCK_SIZE + 1) - len(ret)
            pad = pd.DataFrame(
                data={
                    "tidx": [ret["tidx"].max()] * pad_len,
                    "event_id": [0] * pad_len,
                    "event_value": [0.0] * pad_len,
                }
            )
            ret = pd.concat([ret, pad])

        tidx = torch.tensor(ret["tidx"].values, dtype=torch.long)
        id = torch.tensor(ret["event_id"].values, dtype=torch.long)
        val = torch.tensor(ret["event_value"].values, dtype=torch.float)

        assert len(tidx) == len(id) == len(val) == BLOCK_SIZE + 1

        return (
            (tidx[:-1], id[:-1], val[:-1]),
            (tidx[1:], id[1:], val[1:]),
        )


def calculate_losses(model, x, y):
    x_t, x_id, x_v = x
    y_t, y_id, y_v = y

    x_t, x_id, x_v = x_t.to(DEVICE), x_id.to(DEVICE), x_v.to(DEVICE)
    y_t, y_id, y_v = y_t.to(DEVICE), y_id.to(DEVICE), y_v.to(DEVICE)

    value_estimates = model(x_t, x_id, x_v)

    y_id = y_id.view(y_id.shape[0] * y_id.shape[1])
    y_v = y_v.view(y_v.shape[0] * y_v.shape[1])
    # Compute loss
    # event_loss = F.cross_entropy(event_logits, y_id)
    # Select only value estimates corresponding to the ground-truth event that happened
    relevant_value_estimates = torch.gather(
        value_estimates, axis=1, index=y_id.unsqueeze(1)
    ).squeeze(1)
    value_loss = F.mse_loss(relevant_value_estimates, y_v)
    # combined_loss = event_loss + value_loss

    return value_loss


if __name__ == "__main__":
    torch.manual_seed(42)

    # Data setup
    vitals_df = pd.read_parquet("cache/vitals.parquet")
    ds = VitalsDS(vitals_df)
    train_ds, val_ds = random_split(ds, lengths=[0.9, 0.1])
    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        num_workers=4,
        pin_memory=True,
        pin_memory_device=DEVICE,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        num_workers=4,
        pin_memory=True,
        pin_memory_device=DEVICE,
    )

    # Instantiate model
    model = EventBasedEmrGPT(
        vocab_size=len(ds.tokenization_map),
        n_embd=N_EMBD,
        block_size=BLOCK_SIZE,
        max_len=vitals_df["tidx"].max(),
        n_head=N_HEAD,
        n_layer=N_LAYER,
        dropout=DROPOUT,
    ).to(DEVICE)

    summary(
        model,
        input_data={
            "tidx": torch.zeros(
                (BATCH_SIZE, BLOCK_SIZE), dtype=torch.long, device=DEVICE
            ),
            "event_id": torch.zeros(
                (BATCH_SIZE, BLOCK_SIZE), dtype=torch.long, device=DEVICE
            ),
            "value": torch.zeros(
                (BATCH_SIZE, BLOCK_SIZE), dtype=torch.float, device=DEVICE
            ),
        },
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    for epoch in range(MAX_EPOCHS):
        model.eval()
        # val_event_id_losses = list()
        val_event_value_losses = list()
        for batch in val_dl:
            x_val, y_val = batch
            event_value_loss = calculate_losses(model, x_val, y_val)
            val_event_value_losses.append(event_value_loss.item())

        print(
            f"Epoch {epoch} validation loss: {sum(val_event_value_losses) / len(val_event_value_losses)}"
        )
        # print(f"\tEvent ID: {sum(val_event_id_losses) / len(val_event_id_losses)}")
        # print(
        #     f"\tEvent Value: {sum(val_event_value_losses) / len(val_event_value_losses)}"
        # )

        model.train()

        for batch in train_dl:

            x, y = batch
            loss = calculate_losses(model, x, y)
            loss.backward()
            optimizer.step()

    print("Done")
