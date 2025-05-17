from torch.utils.data import Dataset
import psycopg2
import psycopg2.extras
import atexit
import torch
import numpy as np


class PostgresUtil:
    def __init__(self):
        super().__init__()
        self.conn = None
        self.conn_initialized = False

        c = psycopg2.connect("")
        cursor = c.cursor()

        # Get vocab
        cursor.execute(
            """
            --sql
            SELECT token_id, token FROM mimiciv_local.d_tokens;
            """
        )

        res = cursor.fetchall()
        # nop event is defined as token 0
        # TODO: could include this in d_items table
        self.id2token_map = {**{i[0]: i[1] for i in res}, **{0: "nop"}}
        self.token2id_map = {**{i[1]: i[0] for i in res}, **{"nop": 0}}
        self.vocab_size = len(self.id2token_map)
        # Precompute so can be used later
        self._hourtokens = torch.tensor(
            [v for k, v in self.token2id_map.items() if k.startswith("hour.")],
            dtype=torch.long,
        )
        assert len(self._hourtokens) == 24

        # Get memory vector size
        cursor.execute(
            """
            --sql
            SELECT count(*) FROM information_schema.columns WHERE 
            table_name = 'staticfeats' AND table_schema = 'mimiciv_local' 
            AND column_name != 'stay_id';
            """
        )

        res = cursor.fetchall()
        # TODO: will need to add more complex logic once have more drugs
        # For now manually +1 for icu_los
        self.memory_size = res[0][0] + 1

        c.close()

    def _lazy_init(self):
        if not self.conn_initialized:
            self.conn = psycopg2.connect("")
            atexit.register(self._teardown)
            self.conn_initialized = True

    def _teardown(self):
        print("Dataset teardown called")
        if self.conn is not None:
            self.conn.close()

    def _build_memory_vector(
        self, stay_id: int, X: torch.Tensor, history: torch.Tensor
    ):
        self._lazy_init()

        cursor = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cursor.execute(
            """
            --sql
            SELECT * FROM mimiciv_local.staticfeats
            WHERE stay_id = %s;
            """,
            (stay_id,),
        )

        res = cursor.fetchall()
        assert len(res) == 1, "Should only be one entry per stay_id in staticfeats"

        static_feats = {k: v for k, v in res[0].items() if k != "stay_id"}
        for k, v in static_feats.items():
            if v is None:
                static_feats[k] = 0.0

        # Do some manual normalization
        # TODO: could do a better job of this
        # weight is strongly left skewed, so doing a log normalization
        static_feats["age"] = static_feats["age"] / 120
        static_feats["gender"] = 1.0 if static_feats["gender"] == "F" else 0.0
        static_feats["height"] = static_feats["height"] / 200
        static_feats["weight"] = np.log(static_feats["weight"] + 1) / np.log(635)

        los_hours = 0.0
        if history is not None:

            # Count # of hour events that have transpired in history
            los_hours = (history.unsqueeze(0) == self._hourtokens.unsqueeze(1)).sum()
            # Also log-normalizing los-icu
            los_hours = np.log(los_hours + 1) / np.log(5434)

        memory = torch.tensor(
            list(static_feats.values()) + [los_hours], dtype=torch.float
        )

        assert len(memory) == self.memory_size
        return memory

    def _get_token_stream(self, stay_id: int):
        self._lazy_init()
        cursor = self.conn.cursor()
        cursor.execute(
            """
            --sql
            SELECT token_id
            FROM mimiciv_local.tokenevents
            WHERE stay_id = %s;
            """,
            (stay_id,),
        )

        res = cursor.fetchall()
        token_stream = torch.tensor(res, dtype=torch.long).flatten()
        return token_stream


class TokenStreamDS(Dataset):

    def __init__(self, block_size: int, testset: bool = False):
        super().__init__()
        self.postgresUtil = PostgresUtil()

        self.block_size = block_size

        c = psycopg2.connect("")
        cursor = c.cursor()

        # Get stay ids
        cursor.execute(
            """
            --sql
            SELECT stay_id FROM mimiciv_local.splits
            WHERE testset = %s;
            """,
            ("true" if testset else "false",),
        )

        res = cursor.fetchall()
        self.stay_ids = [i[0] for i in res]

        c.close()

        print("Initiated dataset with:")
        print(f"\tICU stays: {len(self.stay_ids)}")
        print(f"\tVocab size: {self.postgresUtil.vocab_size}")
        print(f"\tBlock size: {self.block_size}")

    def __len__(self):
        return len(self.stay_ids)

    def __getitem__(self, index):
        stay_id = self.stay_ids[index]
        token_stream = self.postgresUtil._get_token_stream(stay_id)

        truncation_idx = torch.randint(1, len(token_stream) - 1, (1,)).item()
        start_idx = max(0, truncation_idx - self.block_size)
        X = token_stream[start_idx:truncation_idx]
        if start_idx > 0:
            history = token_stream[0:start_idx]
        else:
            history = None
        y = token_stream[start_idx : truncation_idx + 1]

        if len(X) < self.block_size:
            X = torch.nn.functional.pad(X, (self.block_size - len(X), 0))

        if len(y) < self.block_size + 1:
            y = torch.nn.functional.pad(y, ((self.block_size + 1) - len(y), 0))

        memory = self.postgresUtil._build_memory_vector(stay_id, X, history)

        assert len(X) == self.block_size
        assert len(y) == self.block_size + 1
        assert len(memory) == self.postgresUtil.memory_size

        return X, memory, y


if __name__ == "__main__":
    ds = TokenStreamDS(block_size=256)

    for idx in range(0, len(ds)):
        out = ds[idx]
