from torch.utils.data import Dataset, DataLoader, Subset, random_split
import psycopg2
import psycopg2.extras
import torch
import pandas as pd
import atexit
from emrgpt.model import TimelineBasedEmrGPT
from torchinfo import summary
import torch.nn.functional as F
import warnings
from tqdm import tqdm

# stfu pandas
warnings.simplefilter(action="ignore", category=FutureWarning)

BLOCK_SIZE = 24
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
DL_WORKERS = 0


class VitalsDS(Dataset):

    def __init__(self):
        super().__init__()

        # Temporary conn to get metadata (e.g. len)
        # NOTE: assuming database info in environ
        # i.e. PGHOST, PGUSER, PGPASSWORD, PGDATABASE
        c = psycopg2.connect("")
        cursor = c.cursor()
        cursor.execute(
            """
            --sql
            SELECT stay_id FROM mimiciv_derived.icustay_detail 
            WHERE (icu_outtime - icu_intime) > INTERVAL '%s hours';
            """,
            (BLOCK_SIZE,),
        )
        res = cursor.fetchall()
        self.stay_ids = [i[0] for i in res]

        c.close()

        self.conn = None

    def _lazy_get_conn(self):
        self.conn = psycopg2.connect("")
        atexit.register(self._teardown)

    def _teardown(self):
        print("Dataset teardown called")
        self.conn.close()

    def __len__(self):
        return len(self.stay_ids)

    def __getitem__(self, index):
        this_sid = self.stay_ids[index]

        if self.conn is None:
            self._lazy_get_conn()

        cursor = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cursor.execute(
            """
            --sql
            SELECT time_bucket('1 hour', charttime) AS tidx,
            avg(heart_rate) AS heart_rate,
            avg(sbp) AS sbp,
            avg(dbp) AS dbp,
            avg(resp_rate) as resp_rate,
            avg(temperature) as temperature,
            avg(spo2) as spo2,
            avg(glucose) as glucose

            FROM mimiciv_derived.vitalsign WHERE stay_id = %s GROUP BY tidx ORDER BY tidx ASC;
            """,
            (this_sid,),
        )

        res = cursor.fetchall()

        df = pd.DataFrame(res)
        df["temperature"] = df["temperature"].astype(float)
        df.drop(columns=["tidx"], inplace=True)

        if len(df) < BLOCK_SIZE + 1:
            pad_len = (BLOCK_SIZE + 1) - len(df)
            pad = pd.DataFrame(data={cname: [pd.NA] * pad_len for cname in df.columns})
            df = pd.concat([df, pad])
        elif len(df) == BLOCK_SIZE + 1:
            pass

        else:
            start_idx = torch.randint(0, (len(df) - (BLOCK_SIZE + 1)), (1,)).item()
            df = df[start_idx : start_idx + BLOCK_SIZE + 1]

        # TODO: is there any utility to also passing an x_nanmask?
        X = df[0:BLOCK_SIZE]
        y = df[1 : BLOCK_SIZE + 1]
        y_nanmask = torch.tensor(y.isna().values, dtype=torch.bool)

        y = y.fillna(0.0)
        X = X.fillna(0.0)

        return (
            torch.tensor(X.values, dtype=torch.float),
            torch.tensor(y.values, dtype=torch.float),
            y_nanmask,
        )


def calculate_losses(m, x, y, y_nanmasks):
    x, y, y_nanmasks = x.to(DEVICE), y.to(DEVICE), y_nanmasks.to(DEVICE)

    B, T, C = y.shape
    y = y.view(B * T, C)
    y_nanmasks = y_nanmasks.view(B * T, C)

    preds = m(x)
    preds_masked = preds * ~y_nanmasks

    loss = F.mse_loss(preds_masked, y)

    return loss


if __name__ == "__main__":
    torch.manual_seed(42)

    ds = VitalsDS()

    test_subset = Subset(ds, range(0, 1000))
    train_ds, val_ds = random_split(test_subset, lengths=[0.9, 0.1])
    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        num_workers=DL_WORKERS,
    )
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=DL_WORKERS)

    model = TimelineBasedEmrGPT(
        n_event_types=7,
        d_model=N_EMBD,
        block_size=BLOCK_SIZE,
        max_len=BLOCK_SIZE,
        n_head=N_HEAD,
        n_layer=N_LAYER,
        dropout=DROPOUT,
    ).to(DEVICE)

    summary(
        model,
        input_data=torch.zeros(
            (BATCH_SIZE, BLOCK_SIZE, 7), dtype=torch.float, device=DEVICE
        ),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(MAX_EPOCHS):
        model.eval()
        val_losses = list()

        print(f"Validation epoch {epoch}")

        for batch in tqdm(val_dl):
            x_val, y_val, y_nanmasks = batch
            val_losses.append(calculate_losses(model, x_val, y_val, y_nanmasks).item())

        avg_val_loss = sum(val_losses) / len(val_losses)
        print(f"Epoch {epoch} validation loss: {avg_val_loss}")

        model.train()

        print(f"Training epoch {epoch}")
        for batch in tqdm(train_dl):
            x, y, y_nanmasks = batch
            loss = calculate_losses(model, x, y, y_nanmasks)
            loss.backward()
            optimizer.step()

    print("Done")
