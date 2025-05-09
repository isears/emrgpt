import psycopg2.extras
from torch.utils.data import Dataset, DataLoader
import torch
import psycopg2
import atexit
from dataclasses import dataclass


@dataclass
class EventSequence:
    offsets: torch.Tensor
    encodings: torch.Tensor
    values: torch.Tensor
    vocab_size: int
    block_size: int

    def to_ohe(self) -> torch.Tensor:
        # TODO: verify
        # NOTE: also assuming offsets don't include len(encodings) as last element
        timesteps = (
            torch.bucketize(
                torch.arange(len(self.encodings), device=self.encodings.device),
                # torch.cat([self.offsets[1:], torch.tensor([len(self.encodings)])]),
                self.offsets,
                right=True,
            )
            - 1
        )
        one_hot = torch.zeros(
            self.vocab_size,
            self.block_size,
            dtype=torch.float,
            device=self.encodings.device,
        )
        one_hot.index_put_(
            (self.encodings, timesteps),
            torch.ones_like(
                self.encodings, dtype=torch.float, device=self.encodings.device
            ),
            accumulate=True,
        )

        return one_hot.T

    @classmethod
    def from_ohe(cls, encoding_ohe: torch.Tensor) -> "EventSequence":
        # TODO: batched encodings (ndim = 3)
        assert encoding_ohe.ndim == 2
        block_size, vocab_size = encoding_ohe.shape

        timesteps, encodings = encoding_ohe.nonzero(as_tuple=True)

        # Only necessary if it's possible to have more than one event of a given type per timestep
        # counts = encoding_ohe[timesteps, encodings].int()
        # encodings = event_ids.repeat_interleave(counts)
        # timesteps = timesteps.repeat_interleave(counts)

        # timesteps, sort_idx = torch.sort(timesteps)
        # encodings = encodings[sort_idx]

        counts_per_timestep = torch.bincount(timesteps, minlength=block_size)
        offsets = torch.cat(
            [
                torch.tensor([0], device=encoding_ohe.device),
                counts_per_timestep.cumsum(0)[0:-1],
            ]
        )

        return EventSequence(
            offsets, encodings, torch.zeros_like(encodings), vocab_size, block_size
        )

    @classmethod
    def collate(cls, batch: list["EventSequence"]) -> "EventSequence":
        # TODO unfinished
        # NOTE: so far writing this assuming offsets don't include last element (i.e. len(encodings))
        assert (
            len(set([es.vocab_size for es in batch])) == 1
        ), "All elements in batch should have same vocab"
        assert (
            len(set([es.block_size for es in batch])) == 1
        ), "All elements in batch should have same block size"

        offsets_aug = list()

        additive_offset = 0
        for es in batch:
            offsets_aug.append(es.offsets + additive_offset)
            additive_offset += len(es.encodings)

        offsets_collated = torch.cat(offsets_aug)

        encodings_collated = torch.cat([es.encodings for es in batch])
        values_collated = torch.cat([es.values for es in batch])

        return EventSequence(
            offsets_collated,
            encodings_collated,
            values_collated,
            batch[0].vocab_size,
            batch[0].block_size,
        )

    def set_device(self, device: str):
        self.offsets = self.offsets.to(device)
        self.encodings = self.encodings.to(device)
        self.values = self.values.to(device)


class EventSequenceDS(Dataset):

    def __init__(self, block_size: int, stay_ids: list[int] = None):
        super().__init__()
        self.block_size = block_size

        # Inclusion criteria: any stays shorter than block_size
        c = psycopg2.connect("")
        cursor = c.cursor()

        if stay_ids is None:
            cursor.execute(
                """
                --sql
                SELECT stay_id FROM mimiciv_local.sequences 
                WHERE array_length(offsets, 1) >= %s 
                AND testset = false;
                """,
                (block_size + 1,),
            )

            res = cursor.fetchall()
            self.stay_ids = [i[0] for i in res]
        else:
            self.stay_ids = stay_ids

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

    def __len__(self):
        return len(self.stay_ids)

    def __getitem__(self, index) -> tuple[EventSequence, torch.Tensor]:
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
        # TODO: not sure if the edge case where a timestep has 0 events is handled well here, it may be a silent failure
        offsets = torch.cat((offsets, torch.tensor([len(encodings)])))
        offsets_x = offsets[start_idx : end_idx + 1]
        offsets_y = offsets[start_idx + 1 : end_idx + 2]

        encodings_x = encodings[offsets_x[0] : offsets_x[-1]]
        values_x = values[offsets_x[0] : offsets_x[-1]]
        offsets_x = offsets_x - offsets_x[0]

        encodings_y = encodings[offsets_y[0] : offsets_y[-1]]
        values_y = values[offsets_y[0] : offsets_y[-1]]
        offsets_y = offsets_y - offsets_y[0]

        es_x = EventSequence(
            offsets_x[:-1], encodings_x, values_x, self.vocab_size, self.block_size
        )
        es_y = EventSequence(
            offsets_y[:-1], encodings_y, values_y, self.vocab_size, self.block_size
        )

        return es_x, es_y.to_ohe()

    @staticmethod
    def collate_fn(batch: list[tuple]):
        x = EventSequence.collate([batch_x for batch_x, _ in batch])
        y = torch.stack([batch_y for _, batch_y in batch], dim=0)

        return x, y


if __name__ == "__main__":
    ds = EventSequenceDS(block_size=24)

    dl = DataLoader(ds, batch_size=2, num_workers=0, collate_fn=ds.collate_fn)

    for batch in dl:
        print(batch)
        break
