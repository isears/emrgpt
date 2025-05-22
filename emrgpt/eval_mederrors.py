"""
Evaluates the ability of the GPT to predict outcome of medical error

Errors:
- High-dose insulin iso normo- or hypo- glycemia
- Supplemental potassium iso hyperkalemia
- Warfarin overdose

TODO: to improve performance on this type of out-of-distribution inference test,
may have to manually build out some examples

E.g.:
- Take a random ICU stay
- Insert insulin overdose sequence randomly
- Insert sequence of response tokens:
    - Fingerstick blood glucose mag.0
    - Chem 7 w/hypoK, hypoglycemia, etc.
    - Ultimately ending in mort token
- Should base the response sequence on case reports
- Should add slight random variations to each example
"""

from torch.utils.data import Dataset, DataLoader
from emrgpt.data import PostgresUtil
import psycopg2
from emrgpt.trainer import *
import pandas as pd


class InsulinOverdoseDS(Dataset):

    def __init__(self, block_size: int):
        super().__init__()

        self.pgutil = PostgresUtil()

        self.block_size = block_size

        c = psycopg2.connect("")
        cursor = c.cursor()

        # Get stay ids
        # TODO: better inclusion criteria, or maybe do post-inference
        # icustay > 2 days (so that we can cut at 24 hrs)
        cursor.execute(
            """
            --sql
            SELECT 
                s.stay_id, 
                s.testset, 
                id.los_icu 
            FROM mimiciv_local.splits s
            LEFT JOIN mimiciv_derived.icustay_detail id 
            ON s.stay_id = id.stay_id
            WHERE testset = %s AND id.los_icu > 2;
            """,
            ("true",),
        )

        res = cursor.fetchall()
        self.stay_ids = [i[0] for i in res]

        c.close()

    def __len__(self):
        return len(self.stay_ids)

    def __getitem__(self, index):
        # TODO: does not currently support batching (variable length / non-tensor outputs)
        stay_id = self.stay_ids[index]
        token_stream = self.pgutil._get_token_stream(stay_id)

        # TODO: for loop probably not most efficient, may optimize later
        hour_count = 0
        most_recent_glucose = float("nan")
        # TODO: this will break if we move from deciles to percentiles
        insulin_history = {
            "Insulin - 70/30": [0] * 10,
            "Insulin - Glargine": [0] * 10,
            "Insulin - Humalog": [0] * 10,
            "Insulin - Humalog 75/25": [0] * 10,
            "Insulin - NPH": [0] * 10,
            "Insulin - Novolog": [0] * 10,
            "Insulin - Regular": [0] * 10,
            "Insulin - U500": [0] * 10,
        }
        for idx in range(0, len(token_stream)):
            curr_token = self.pgutil.id2token_map[token_stream[idx].item()]

            if curr_token.startswith("hour."):
                hour_count += 1

                if hour_count == 24:
                    hour_truncation_idx = idx
                    break

            elif curr_token == "chemistry.glucose" and idx < len(token_stream):
                glucose_magnitude_token = self.pgutil.id2token_map[
                    token_stream[idx + 1].item()
                ]
                assert glucose_magnitude_token.startswith("magnitude.")
                most_recent_glucose = float(glucose_magnitude_token.split(".")[-1])

            elif curr_token in insulin_history.keys() and idx < len(token_stream) - 1:
                dose_token = self.pgutil.id2token_map[token_stream[idx + 2].item()]
                assert dose_token.startswith("magnitude.")
                dose_magnitude = int(dose_token.split(".")[-1])
                insulin_history[curr_token][dose_magnitude] += 1

        first_24h = token_stream[0 : hour_truncation_idx + 1]

        if len(first_24h) > self.block_size:
            start_idx = len(first_24h) - self.block_size
            X = first_24h[start_idx:]
            history = first_24h[0:start_idx]
        elif len(first_24h) < self.block_size:
            # If decide to support batching later on, will need to actually implement a pad here
            # So that all are equal length
            X = first_24h
            history = None
        else:
            X = first_24h
            history = None

        memory = self.pgutil._build_memory_vector(stay_id, X, history)

        return X, memory, most_recent_glucose, pd.DataFrame(data=insulin_history)


def append_maintain_blocksize(
    original: torch.Tensor, appendee: torch.Tensor, block_size: int
):
    assert original.ndim == appendee.ndim == 1, "[-] Only supports 1d vectors"
    if appendee.device != original.device:
        appendee = appendee.to(original.device)
    concat = torch.cat([original, appendee], dim=0)
    if len(concat) > block_size:
        concat = concat[len(concat) - block_size :]

    assert len(concat) <= block_size
    return concat


def simulate_overdose(
    model: TokenStreamGPT,
    stream_tokens: torch.Tensor,
    memory: torch.Tensor,
    pgutil: PostgresUtil,
):
    lasttoken = pgutil.id2token_map[stream_tokens[-1].item()]
    assert lasttoken.startswith("hour.")
    lasthour = int(lasttoken.split(".")[-1])
    nexthour = (lasthour + 1) % 24

    overdose_sequence = torch.tensor(
        [
            pgutil.token2id_map["Insulin - Regular"],
            pgutil.token2id_map["units"],
            pgutil.token2id_map["magnitude.9"],
            pgutil.token2id_map[f"hour.{nexthour}"],
        ]
    )

    chem7_tokens = [
        "bicarbonate",
        "bun",
        "calcium",
        "chloride",
        "creatinine",
        "glucose",
        "potassium",
        "sodium",
    ]

    with torch.no_grad():

        # next_hr_tokens = model.generate(
        #     seed=append_maintain_blocksize(
        #         stream_tokens, overdose_sequence, block_size=BLOCK_SIZE
        #     ).unsqueeze(0),
        #     memory=memory.unsqueeze(0),
        #     lookahead_hrs=1,
        #     hourtokens=pgutil._hourtokens,
        # ).squeeze(dim=0)

        # TODO: check for glucose intervention (e.g. D50)

        # TODO: if weird behavior, may have to update memory

        # future_stream = append_maintain_blocksize(
        #     stream_tokens, next_hr_tokens, block_size=BLOCK_SIZE
        # )

        future_stream = append_maintain_blocksize(
            stream_tokens, overdose_sequence, block_size=BLOCK_SIZE
        )

        # Generate a chem7
        # TODO: could also save potassium here
        glucose_magnitude = float("nan")
        for token in chem7_tokens:
            future_stream = append_maintain_blocksize(
                future_stream,
                torch.tensor([pgutil.token2id_map[f"chemistry.{token}"]]),
                block_size=BLOCK_SIZE,
            )

            next_token = model.generate_next(
                future_stream.unsqueeze(0), mem.unsqueeze(0).to(DEVICE)
            ).squeeze(dim=0)

            magnitude_token = pgutil.id2token_map[next_token.detach().cpu().item()]
            if not magnitude_token.startswith("magnitude."):
                print(f"[-] WARN expected magnitude token but got {magnitude_token}")
            else:

                future_stream = append_maintain_blocksize(
                    future_stream, next_token, block_size=BLOCK_SIZE
                )

                if token == "glucose":
                    glucose_magnitude = float(magnitude_token.split(".")[-1])

        return glucose_magnitude


if __name__ == "__main__":

    model = TokenStreamGPT.load("cache/TokenStreamGPT.ckpt")

    ds = InsulinOverdoseDS(block_size=model.conf.block_size)

    model.eval()
    model = model.to("cuda")

    pre_insulin_glucose = list()
    post_insulin_glucose = list()
    insulin_histories = list()

    for X, mem, last_glucose, insulin_history in tqdm(ds):
        pre_insulin_glucose.append(last_glucose)
        post_insulin_glucose.append(
            simulate_overdose(model, X.to(DEVICE), mem.to(DEVICE), ds.pgutil)
        )
        insulin_histories.append(insulin_history)

    pre_insulin_glucose = torch.tensor(pre_insulin_glucose)
    post_insulin_glucose = torch.tensor(post_insulin_glucose)
    avg_pre = pre_insulin_glucose.nanmean()
    avg_post = post_insulin_glucose.nanmean()
    avg_delta = (post_insulin_glucose - pre_insulin_glucose).nanmean()

    print(f"Avg pre-insulin glucose: {avg_pre}")
    print(f"Avg post-insulin glucose: {avg_post}")
    print(f"Avg deltaG: {avg_delta}")
