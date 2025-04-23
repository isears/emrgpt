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
from psycopg2.extensions import AsIs

# stfu pandas
warnings.simplefilter(action="ignore", category=FutureWarning)

BLOCK_SIZE = 24
MAX_EPOCHS = 10
LEARNING_RATE = 1e-5
BATCH_SIZE = 32
DEVICE = "cuda"
N_HEAD = 10
N_LAYER = 10
N_EMBD = 32
DROPOUT = 0.2
DL_WORKERS = 6
VAL_CHECK_INTERVAL = 100


class TimelineDS(Dataset):

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
            SELECT stay_id FROM mimiciv_derived.icustay_times 
            WHERE (outtime_hr - intime_hr) > INTERVAL '%s hours';
            """,
            (BLOCK_SIZE,),
        )
        res = cursor.fetchall()
        self.stay_ids = [i[0] for i in res]

        # Also going to go ahead and calculate min / max for normalization
        # NOTE: this should technically be done per train / val split
        cursor.execute(
            """
            --sql
            SELECT * FROM information_schema.columns
            WHERE table_schema = 'mimiciv_local'
            AND table_name = 'timebuckets';
            """
        )

        res = cursor.fetchall()
        self.features = [row[3] for row in res if row[3] not in ["stay_id", "tidx"]]
        self.feature_stats = dict()

        for f in self.features:
            cursor.execute(
                """
                --sql
                SELECT
                    MIN(%s),
                    MAX(%s),
                    AVG(%s),
                    STDDEV(%s)
                FROM mimiciv_local.timebuckets;
                """,
                (AsIs(f),) * 4,
            )
            res = cursor.fetchall()

            self.feature_stats[f] = {
                "min": res[0][0],
                "max": res[0][1],
                "avg": res[0][2],
                "stddev": res[0][3],
            }

        c.close()

        self.conn = None

    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        assert data.ndim == 2
        assert data.shape[1] == len(self.features)

        max_t = torch.tensor(
            [v["max"] for k, v in self.feature_stats.items()],
            dtype=torch.float,
            device=data.device,
        )
        min_t = torch.tensor(
            [v["min"] for k, v in self.feature_stats.items()],
            dtype=torch.float,
            device=data.device,
        )

        return (data - min_t) / (max_t - min_t)

    def denormalize(self, data: torch.Tensor) -> torch.Tensor:
        max_t = torch.tensor(
            [v["max"] for k, v in self.feature_stats.items()],
            dtype=torch.float,
            device=data.device,
        )
        min_t = torch.tensor(
            [v["min"] for k, v in self.feature_stats.items()],
            dtype=torch.float,
            device=data.device,
        )

        return (data * (max_t - min_t)) + min_t

    def standardize(self, data: torch.Tensor) -> torch.Tensor:
        assert data.ndim == 2
        assert data.shape[1] == len(self.features)

        mu = torch.tensor(
            [v["avg"] for k, v in self.feature_stats.items()],
            dtype=torch.float,
            device=data.device,
        )
        sigma = torch.tensor(
            [v["stddev"] for k, v in self.feature_stats.items()],
            dtype=torch.float,
            device=data.device,
        )

        return (data - mu) / sigma

    def destandardize(self, data: torch.Tensor) -> torch.Tensor:
        assert data.ndim == 2
        assert data.shape[1] == len(self.features)

        mu = torch.tensor(
            [v["avg"] for k, v in self.feature_stats.items()],
            dtype=torch.float,
            device=data.device,
        )
        sigma = torch.tensor(
            [v["stddev"] for k, v in self.feature_stats.items()],
            dtype=torch.float,
            device=data.device,
        )

        return (data * sigma) + mu

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
            SELECT * FROM mimiciv_local.timebuckets WHERE stay_id = %s ORDER BY tidx ASC;
            """,
            (this_sid,),
        )

        res = cursor.fetchall()

        df = pd.DataFrame(res)
        df.drop(columns=["tidx", "stay_id"], inplace=True)

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
            self.normalize(torch.tensor(X.values, dtype=torch.float)),
            self.normalize(torch.tensor(y.values, dtype=torch.float)),
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

    ds = TimelineDS()

    # test_subset = Subset(ds, range(0, 1000))
    train_ds, val_ds = random_split(ds, lengths=[0.9, 0.1])
    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        num_workers=DL_WORKERS,
    )
    val_dl = DataLoader(val_ds, batch_size=512, num_workers=DL_WORKERS)

    model = TimelineBasedEmrGPT(
        n_event_types=13,
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
            (BATCH_SIZE, BLOCK_SIZE, 13), dtype=torch.float, device=DEVICE
        ),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    best_val_loss = float("inf")

    for epoch in range(MAX_EPOCHS):
        print(f"--> Training epoch {epoch}")
        for batchnum, batch in enumerate(train_dl):
            if batchnum % VAL_CHECK_INTERVAL == 0:
                model.eval()
                val_losses = list()

                for batch in val_dl:
                    x_val, y_val, y_nanmasks = batch
                    val_losses.append(
                        calculate_losses(model, x_val, y_val, y_nanmasks).item()
                    )

                avg_val_loss = sum(val_losses) / len(val_losses)

                if avg_val_loss < best_val_loss:
                    # print(f"{avg_val_loss} < {best_val_loss}, saving checkpoint")
                    best_val_loss = avg_val_loss
                    torch.save(
                        model.state_dict(),
                        f"cache/savedmodels/{model.__class__.__name__}.pt",
                    )

                    print(f"Step {batchnum} validation loss: {avg_val_loss} (*)")
                else:
                    print(f"Step {batchnum} validation loss: {avg_val_loss}")

                model.train()

            x, y, y_nanmasks = batch
            loss = calculate_losses(model, x, y, y_nanmasks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
            optimizer.step()

    print("Done")
