"""
Evaluate how good the GPT is at predicting acute overnight events

Targets:
- Need for new intubation
- Need for new pressors

Inclusion criteria:
- In test set
- Stay > 24 hrs.
- For pressors eval: no pressors prior
- For ventilation eval: no ventilation prior

TODO: will probably need o2 delivery data before new ventilation becomes possible
"""

from torch.utils.data import Dataset
from emrgpt.data import PostgresUtil
import psycopg2
import torch


# TODO
class OvernightPressors(Dataset):

    def __init__(self, block_size: int):
        super().__init__()

        self.pgutil = PostgresUtil()
        self.block_size = block_size

        c = psycopg2.connect("")
        cursor = c.cursor()
        cursor.execute(
            """
            --sql
            SELECT ;
            """
        )

        c.close()


class OvernightBlood(Dataset):
    def __init__(self, block_size: int):
        super().__init__()
        self.pgutil = PostgresUtil()
        self.block_size = block_size

        c = psycopg2.connect("")
        cursor = c.cursor()
        # TODO: examples are shifts not stays
        cursor.execute(
            """
            --sql
            SELECT stay_id, shift_starttime, blood_given
            FROM mimiciv_local.overnight_blood;
            """
        )

        res = cursor.fetchall()
        self.examples = [(i[0], i[1], i[2]) for i in res]

        c.close()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        stay_id, shift_starttime, blood_given = self.examples[index]
        token_stream, mem = self.pgutil._get_tokens_mem(
            stay_id, self.block_size, shift_starttime
        )

        return token_stream, mem, torch.tensor(blood_given)


if __name__ == "__main__":
    ds = OvernightBlood(block_size=512)

    for x, mem, y in ds:
        print(".")
