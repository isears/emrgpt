from torch.utils.data import Dataset
import psycopg2
import psycopg2.extras
import atexit
import torch
import numpy as np


class TokenStreamDS(Dataset):

    def __init__(self, block_size: int):
        super().__init__()

        self.block_size = block_size

        c = psycopg2.connect("")
        cursor = c.cursor()

        # Get stay ids
        cursor.execute(
            """
            --sql
            SELECT stay_id FROM mimiciv_local.splits
            WHERE testset = false;
            """
        )

        res = cursor.fetchall()
        self.stay_ids = [i[0] for i in res]

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

        c.close()
        self.conn = None  # will lazy init later
        self.conn_initialized = False

        print("Initiated dataset with:")
        print(f"\tICU stays: {len(self.stay_ids)}")
        print(f"\tVocab size: {self.vocab_size}")
        print(f"\tBlock size: {self.block_size}")

    def _lazy_init(self):
        if not self.conn_initialized:
            self.conn = psycopg2.connect("")
            atexit.register(self._teardown)
            self.conn_initialized = True

    def _teardown(self):
        print("Dataset teardown called")
        if self.conn is not None:
            self.conn.close()

    def __len__(self):
        return len(self.stay_ids)

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
        static_feats["weight"] = np.log(static_feats["weight"]) / np.log(635)

        # TODO: for now just looking for norepinephrine outside the context window
        # Will have to be more thorough about this in the future when have more meds
        last_dose = 0.0
        if history is not None:
            indices = (
                history == self.token2id_map["norepinephrine_equivalent_dose.rate"]
            ).nonzero(as_tuple=True)[0]

            # Get the last one
            if indices.numel() > 0:
                last_index = indices[-1].item()

                if last_index != len(history) - 1:
                    last_dose_token = self.id2token_map[history[last_index + 1].item()]
                else:
                    last_dose_token = self.id2token_map[X[0]]

                assert last_dose_token.startswith("magnitude.")
                last_dose = int(last_dose_token.split(".")[-1]) / 10
                assert last_dose >= 0 and last_dose <= 1

        return torch.tensor(
            list(static_feats.values()) + [last_dose], dtype=torch.float
        )

    def __getitem__(self, index):
        self._lazy_init()

        stay_id = self.stay_ids[index]

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

        truncation_idx = torch.randint(1, len(token_stream) - 1, (1,)).item()
        start_idx = max(0, truncation_idx - self.block_size)
        X = token_stream[start_idx:truncation_idx]
        if start_idx > 0:
            history = token_stream[0:start_idx]
        else:
            history = None
        start_idx = max(0, (truncation_idx + 1) - self.block_size)
        y = token_stream[start_idx : truncation_idx + 1]

        if len(X) < self.block_size:
            X = torch.nn.functional.pad(X, (self.block_size - len(X), 0))

        if len(y) < self.block_size:
            y = torch.nn.functional.pad(y, (self.block_size - len(y), 0))

        memory = self._build_memory_vector(stay_id, X, history)

        return X, memory, y


if __name__ == "__main__":
    ds = TokenStreamDS(block_size=256)

    for idx in range(0, len(ds)):
        out = ds[idx]
