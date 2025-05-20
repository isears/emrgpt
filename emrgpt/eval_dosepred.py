"""
Similar to mederrors eval, try to determine when wrong med is being given

Instead of predicting physiology, predict expected dose
given a token sequence that terminates with the medication token
"""

import torch
from torch.utils.data import Dataset, DataLoader
from emrgpt.data import PostgresUtil
import psycopg2
from emrgpt.trainer import *
from emrgpt.model import TokenStreamGPT
from sklearn.metrics import roc_auc_score, average_precision_score
from tabulate import tabulate


class MedDoseDS(Dataset):

    def __init__(self, block_size: int, med: str):
        super().__init__()
        self.block_size = block_size
        self.pgutil = PostgresUtil()
        self.med_token = self.pgutil.token2id_map[med]

        c = psycopg2.connect("")
        cursor = c.cursor()

        cursor.execute(
            """
            --sql
            SELECT tokenevents.stay_id, charttime 
            FROM mimiciv_local.tokenevents
            LEFT JOIN mimiciv_local.splits 
            ON splits.stay_id = tokenevents.stay_id
            WHERE token_id = %s AND splits.testset = true;
            """,
            (self.med_token,),
        )

        self.examples = [(i[0], i[1]) for i in cursor.fetchall()]

        c.close()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        stay_id, charttime = self.examples[index]

        token_stream, memory = self.pgutil._get_tokens_mem(
            stay_id, self.block_size, limit=charttime
        )

        med_indices = (token_stream == self.med_token).nonzero(as_tuple=True)[0]
        assert len(med_indices) > 0
        last_med_index = med_indices[-1].item()

        should_be_unit_token = token_stream[last_med_index + 1].item()

        should_be_magnitude_token = self.pgutil.id2token_map[
            token_stream[last_med_index + 2].item()
        ]

        assert should_be_magnitude_token.startswith("magnitude.")
        magnitude = int(should_be_magnitude_token.split(".")[-1])

        y = torch.zeros((10,), dtype=int)
        y[magnitude] = 1

        token_stream = token_stream[:last_med_index]
        token_stream = torch.cat(
            (token_stream, torch.tensor([self.med_token, should_be_unit_token]))
        )

        # Need to re-pad because we may have change the size of the tokens stream
        # Updating memory not necessary though
        if len(token_stream) > self.block_size:
            start_idx = len(token_stream) - self.block_size
            token_block = token_stream[start_idx:]
        elif len(token_stream) < self.block_size:

            token_block = torch.nn.functional.pad(
                token_stream, (self.block_size - len(token_stream), 0)
            )
        else:
            token_block = token_stream

        return token_block, memory, y


if __name__ == "__main__":
    model_path = "./cache/TokenStreamGPT.pt"
    results = dict()

    for med in [
        "Insulin - Regular",
        "Coumadin (Warfarin)",
        "Argatroban",
        "KCL (Bolus)",
        "Digoxin (Lanoxin)",
        "Fentanyl",
    ]:
        print(f"Evaluating: {med}")
        ds = MedDoseDS(BLOCK_SIZE, med=med)
        dl = DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            num_workers=DL_WORKERS,
        )

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

        y_preds = list()
        y_actual = list()

        magnitude_tokens = [
            ds.pgutil.token2id_map[f"magnitude.{i}"] for i in range(0, 10)
        ]

        for X, mem, y in tqdm(dl):
            with torch.no_grad():
                logits = model(X.to("cuda"), mem.to("cuda"))[:, -1, :]
                probs = F.softmax(logits, dim=-1)

                magnitude_probs = probs[:, magnitude_tokens].detach().cpu()
                assert magnitude_probs.shape == y.shape

                y_preds.append(magnitude_probs)
                y_actual.append(y)

        y_preds = torch.cat(y_preds, dim=0)
        y_actual = torch.cat(y_actual, dim=0)

        auroc = roc_auc_score(y_actual, y_preds)
        auprc = average_precision_score(y_actual, y_preds)

        results[f"{med} overall AUROC"] = auroc
        results[f"{med} overall AUPRC"] = auprc
        results[f"{med} 90%-ile AUROC"] = roc_auc_score(y_actual[:, 9], y_preds[:, 9])
        results[f"{med} 90%-ile AUPRC"] = average_precision_score(
            y_actual[:, 9], y_preds[:, 9]
        )

    print(tabulate([(k, v) for k, v in results.items()]))
