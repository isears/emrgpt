from torch.utils.data import Dataset, DataLoader, Subset
from psycopg2.extensions import AsIs
import pandas as pd
import atexit
import psycopg2
import psycopg2.extras
import torch


class TimelineDS(Dataset):

    def __init__(self, block_size: int):
        super().__init__()

        self.block_size = block_size

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
            (block_size,),
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

        if len(df) < self.block_size + 1:
            pad_len = (self.block_size + 1) - len(df)
            pad = pd.DataFrame(data={cname: [pd.NA] * pad_len for cname in df.columns})
            df = pd.concat([df, pad])
        elif len(df) == self.block_size + 1:
            pass

        else:
            start_idx = torch.randint(0, (len(df) - (self.block_size + 1)), (1,)).item()
            df = df[start_idx : start_idx + self.block_size + 1]

        # TODO: is there any utility to also passing an x_nanmask?
        X = df[0 : self.block_size]
        y = df[1 : self.block_size + 1]
        y_nanmask = torch.tensor(y.isna().values, dtype=torch.bool)

        y = y.fillna(0.0)
        X = X.fillna(0.0)

        return (
            self.normalize(torch.tensor(X.values, dtype=torch.float)),
            self.normalize(torch.tensor(y.values, dtype=torch.float)),
            y_nanmask,
        )
