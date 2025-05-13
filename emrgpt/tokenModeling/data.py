from torch.utils.data import Dataset
import psycopg2
import atexit
import torch


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

        print("Initiated dataset with:")
        print(f"\tICU stays: {len(self.stay_ids)}")
        print(f"\tVocab size: {self.vocab_size}")
        print(f"\tBlock size: {self.block_size}")

    def _lazy_get_conn(self):
        self.conn = psycopg2.connect("")
        atexit.register(self._teardown)

    def _teardown(self):
        print("Dataset teardown called")
        if self.conn is not None:
            self.conn.close()

    def __len__(self):
        return len(self.stay_ids)

    def __getitem__(self, index):
        stay_id = self.stay_ids[index]

        if self.conn is None:
            self._lazy_get_conn()

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
        start_idx = max(0, (truncation_idx + 1) - self.block_size)
        y = token_stream[start_idx : truncation_idx + 1]

        if len(X) < self.block_size:
            X = torch.nn.functional.pad(X, (self.block_size - len(X), 0))

        if len(y) < self.block_size:
            y = torch.nn.functional.pad(y, (self.block_size - len(y), 0))

        return X, y


if __name__ == "__main__":
    ds = TokenStreamDS(block_size=256)

    out = ds[5406]
