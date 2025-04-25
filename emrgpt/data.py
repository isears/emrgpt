from torch.utils.data import Dataset, DataLoader, Subset
from psycopg2.extensions import AsIs
import pandas as pd
import atexit
import psycopg2
import psycopg2.extras
import torch
import datetime


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

    @staticmethod
    def _get_stay_id(
        stay_id: int, conn: psycopg2.extensions.connection
    ) -> pd.DataFrame:
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cursor.execute(
            """
            --sql
            SELECT * FROM mimiciv_local.timebuckets WHERE stay_id = %s ORDER BY tidx ASC;
            """,
            (stay_id,),
        )

        res = cursor.fetchall()

        return pd.DataFrame(res)

    @staticmethod
    def _pad(df_in: pd.DataFrame, desired_len: int) -> pd.DataFrame:
        pad_len = (desired_len) - len(df_in)
        pad = pd.DataFrame(data={cname: [pd.NA] * pad_len for cname in df_in.columns})
        return pd.concat([pad, df_in])

    def __len__(self):
        return len(self.stay_ids)

    def __getitem__(self, index):
        if self.conn is None:
            self._lazy_get_conn()

        this_sid = self.stay_ids[index]

        df = self._get_stay_id(this_sid, self.conn)
        df.drop(columns=["tidx", "stay_id"], inplace=True)

        if len(df) < self.block_size + 1:
            df = self._pad(df, self.block_size + 1)
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


class ReintubationDS(Dataset):

    def __init__(
        self, tlds: TimelineDS, stay_ids: list[int], prediction_window: int = 24
    ):
        super().__init__()
        self.tlds = tlds
        self.prediction_window = prediction_window

        # Inclusion criteria:
        # - Never had tracheostomy
        # - Extubated > 12 hours prior to end of ICU stay (to exclude extubation immediately prior to death)
        c = psycopg2.connect("")
        cursor = c.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cursor.execute(
            """
            --sql
            WITH non_tracheostomy AS (
                SELECT
                    stay_id
                FROM mimiciv_derived.ventilation
                GROUP BY stay_id
                HAVING BOOL_OR(ventilation_status = 'Tracheostomy') = false
            ),
            first_extubation AS (
                SELECT
                    stay_id,
                    min(endtime) AS extubation_time
                FROM mimiciv_derived.ventilation WHERE ventilation_status = 'InvasiveVent'
                AND stay_id = ANY(%s)
                GROUP BY stay_id
            )

            SELECT non_tracheostomy.stay_id, extubation_time FROM non_tracheostomy
                INNER JOIN first_extubation ON non_tracheostomy.stay_id = first_extubation.stay_id
                LEFT JOIN mimiciv_derived.icustay_detail ON non_tracheostomy.stay_id = mimiciv_derived.icustay_detail.stay_id
            WHERE (icu_outtime - extubation_time) > (INTERVAL '12 hours')
            ;
            """,
            (stay_ids,),
        )
        self.extubations_df = pd.DataFrame(data=cursor.fetchall())

        c.close()
        self.conn = None

    def _lazy_get_conn(self):
        self.conn = psycopg2.connect("")
        atexit.register(self._teardown)

    def _teardown(self):
        print("Dataset teardown called")
        self.conn.close()

    def __len__(self) -> int:
        return len(self.extubations_df)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        if self.conn is None:
            self._lazy_get_conn()

        stay_id, extubation_time = self.extubations_df.iloc[index]
        stay_id = int(stay_id)

        df = self.tlds._get_stay_id(stay_id, self.conn)

        # Per exclusion criteria should not have any tracheostomy patients
        assert df["vent_trach"].sum() == 0

        pre_extubation_df = df[df["tidx"] < extubation_time]
        post_extubation_df = df[
            (df["tidx"] > extubation_time)
            & (
                df["tidx"]
                < (extubation_time + datetime.timedelta(hours=self.prediction_window))
            )
        ]

        # TODO: is it better to set extubation back one hour or add a row with no data except the extubation event?
        pre_extubation_df.at[len(pre_extubation_df) - 1, "vent_invasive"] = 0.0

        # Cut down to just the last 24 hrs
        pre_extubation_df = pre_extubation_df[-self.tlds.block_size :]

        pre_extubation_df.drop(columns=["tidx", "stay_id"], inplace=True)

        if len(pre_extubation_df) < self.tlds.block_size:
            pre_extubation_df = self.tlds._pad(pre_extubation_df, self.tlds.block_size)

        pre_extubation_df = pre_extubation_df.fillna(0.0)

        return self.tlds.normalize(
            torch.tensor(pre_extubation_df.values, dtype=torch.float)
        ), torch.tensor(post_extubation_df["vent_invasive"].max(), dtype=torch.float)
