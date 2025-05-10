"""
Datasets used to validate the clinical utility of GPT predictions
"""

from torch.utils.data import Dataset
import psycopg2
import psycopg2.extras
from emrgpt.baggedEventModeling.data import EventSequenceDS, EventSequence
import atexit
from abc import ABC, abstractmethod
import pandas as pd
import torch


class ValidationDS(ABC, Dataset):
    # NOTE: Block size hard-coded b/c changing it requires updating the postgres table
    block_size = 24

    def __init__(self):
        super().__init__()

        c = psycopg2.connect("")
        cursor = c.cursor()

        cursor.execute(
            """
            --sql
            SELECT count(*) FROM mimiciv_local.item_encoding;
            """
        )
        res = cursor.fetchall()
        self.vocab_size = res[0][0] + 1
        c.close()
        self.conn = None

    def _lazy_get_conn(self):
        self.conn = psycopg2.connect("")
        atexit.register(self._teardown)

    def _teardown(self):
        print("Dataset teardown called")
        if self.conn is not None:
            self.conn.close()


class SepsisDS(ValidationDS): ...


class HyperkalemiaDS(ValidationDS): ...


class AcuteKidneyInjuryDS(ValidationDS): ...


class NewPressorsDS(ValidationDS): ...


class InUnitMortalityDS(ValidationDS): ...


class NewVentilationDS(ValidationDS):

    def __init__(self):
        super().__init__()

        c = psycopg2.connect("")
        cursor = c.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        cursor.execute(
            """
            --sql
            SELECT stay_id, base_tidx, vent_initiation_36h FROM mimiciv_local.val_ventilation;
            """,
        )

        res = cursor.fetchall()
        self.examples = pd.DataFrame(data=res)

        c.close()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index: int):
        if self.conn is None:
            self._lazy_get_conn()

        stay_id, base_tidx, label = self.examples.iloc[index]

        cursor = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cursor.execute(
            """
            --sql
            SELECT encodings, values, offsets
            FROM mimiciv_local.sequences 
            WHERE stay_id = %s;
            """,
            (int(stay_id),),
        )

        res = cursor.fetchall()
        assert len(res) == 1, "Should be only one result per stay_id"

        encodings = torch.tensor(res[0]["encodings"], dtype=torch.long)
        values = torch.tensor(res[0]["values"], dtype=torch.float)
        offsets = torch.tensor(res[0]["offsets"], dtype=torch.long)

        assert (
            len(offsets) >= self.block_size + 1
        ), "Shouldn't include ICU stays smaller than block_size (padding not implemented)"

        offsets = torch.cat([offsets, torch.tensor([len(encodings)])])

        offsets_block = (
            offsets[base_tidx : base_tidx + self.block_size] - offsets[base_tidx]
        )
        encodings_block = encodings[
            offsets[base_tidx] : offsets[base_tidx + self.block_size + 1]
        ]
        values_block = values[
            offsets[base_tidx] : offsets[base_tidx + self.block_size + 1]
        ]

        return EventSequence(
            offsets=offsets_block,
            encodings=encodings_block,
            values=values_block,
            vocab_size=self.vocab_size,
            block_size=self.block_size,
        ), torch.tensor(int(label))

    @staticmethod
    def collate_fn(batch: list[tuple]):
        x = EventSequence.collate([batch_x for batch_x, _ in batch])
        y = torch.stack([batch_y for _, batch_y in batch], dim=0)

        return x, y


if __name__ == "__main__":
    ds = NewVentilationDS()

    print(len(ds))

    print(ds[38])
