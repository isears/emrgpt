import psycopg2.extras
from torch.utils.data import Dataset, DataLoader
import torch
import psycopg2
import atexit
import pandas as pd


class EventSequenceDS(Dataset):

    def __init__(self, block_size: int):
        super().__init__()
        self.block_size = block_size

        # Inclusion criteria: any stays shorter than block_size
        c = psycopg2.connect("")
        cursor = c.cursor()
        # TODO: this takes forever, precompute?
        # NOTE: seems like result gets cached so maybe ok?
        cursor.execute(
            """
            --sql
            SELECT stay_id FROM mimiciv_local.sequences 
            WHERE array_length(offsets, 1) >= %s;
            """,
            (block_size + 1,),
        )

        res = cursor.fetchall()
        self.stay_ids = [i[0] for i in res]

        self.conn = None

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
        if self.conn is None:
            self._lazy_get_conn()

        this_sid = self.stay_ids[index]

        cursor = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cursor.execute(
            """
            --sql
            SELECT encodings, values, offsets
            FROM mimiciv_local.sequences 
            WHERE stay_id = %s;
            """,
            (this_sid,),
        )

        res = cursor.fetchall()
        assert len(res) == 1, "Should be only one result per stay_id"

        encodings = torch.tensor(res[0]["encodings"], dtype=torch.long)
        values = torch.tensor(res[0]["values"], dtype=torch.float)
        offsets = torch.tensor(res[0]["offsets"], dtype=torch.long)

        assert (
            len(offsets) >= self.block_size + 1
        ), "Shouldn't include ICU stays smaller than block_size (padding not implemented)"

        excess_length = len(offsets) - (self.block_size + 1)
        # Randint: generates random int between low_bound (inclusive) and high_bound (exclusive)
        # start_idx will at most be excess_length
        start_idx = torch.randint(0, excess_length + 1, (1,)).item()
        end_idx = start_idx + self.block_size

        # The last offset is technically the end of the sequence
        # appending this value to end of offsets helps handle edge cases
        # NOTE: offsets will actually be block_size + 1 to handle this edge case
        offsets = torch.cat((offsets, torch.tensor([len(encodings)])))
        offsets_x = offsets[start_idx : end_idx + 1]
        offsets_y = offsets[start_idx + 1 : end_idx + 2]

        encodings_x = encodings[offsets_x[0] : offsets_x[-1]]
        values_x = values[offsets_x[0] : offsets_x[-1]]

        encodings_y = encodings[offsets_y[0] : offsets_y[-1]]
        values_y = values[offsets_y[0] : offsets_y[-1]]

        return offsets_x, encodings_x, values_x, offsets_y, encodings_y, values_y

    @staticmethod
    def collate_fn(batch: list[tuple]):
        offsets_x_aug, offsets_y_aug = list(), list()

        additive_offset = 0
        for ox, _, _, oy, _, _ in batch:
            offsets_x_aug.append(ox + additive_offset)
            offsets_y_aug.append(oy + additive_offset)

        offsets_x_collated = torch.cat(offsets_x_aug)
        offsets_y_collated = torch.cat(offsets_y_aug)

        encodings_x_collated = torch.cat([b[1] for b in batch])
        values_x_collated = torch.cat([b[2] for b in batch])
        encodings_y_collated = torch.cat([b[4] for b in batch])
        values_y_collated = torch.cat([b[5] for b in batch])

        return (
            offsets_x_collated,
            encodings_x_collated,
            values_x_collated,
            offsets_y_collated,
            encodings_y_collated,
            values_y_collated,
        )


if __name__ == "__main__":
    ds = EventSequenceDS(block_size=24)

    dl = DataLoader(ds, batch_size=2, num_workers=0, collate_fn=ds.collate_fn)

    for batch in dl:
        print(batch)
        break
