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
from emrgpt.model import TokenStreamGPT
from emrgpt.trainer import *
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm


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

    model_path = "./cache/TokenStreamGPT.pt"

    ds = OvernightBlood(block_size=512)

    model = TokenStreamGPT(
        vocab_size=ds.pgutil.vocab_size,
        memory_size=ds.pgutil.memory_size,
        n_embd=N_EMBD,
        dropout=DROPOUT,
        block_size=BLOCK_SIZE,
        n_layer=N_LAYER,
        n_head=N_HEAD,
    )

    model.load_state_dict(torch.load(model_path))
    model.eval()
    model = model.to("cuda")

    y_true = list()
    y_pred = list()

    for X, mem, y in tqdm(ds):
        with torch.no_grad():
            _, probs = model.generate_nonbatch(
                X.to("cuda"),
                mem.to("cuda"),
                lookahead_hrs=12,
                hourtokens=ds.pgutil._hourtokens,
            )

        y_true.append(y.item())
        y_pred.append(
            probs[ds.pgutil.token2id_map["Packed Red Blood Cells"]]
            .detach()
            .cpu()
            .item()
        )

    print(f"AUROC: {roc_auc_score(y_true, y_pred)}")
    print(f"AUPRC: {average_precision_score(y_true, y_pred)}")
