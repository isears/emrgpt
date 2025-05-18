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
