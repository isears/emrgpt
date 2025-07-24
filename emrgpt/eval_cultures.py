"""
Evaluates how good the GPT is at guessing blood culture results

Targets:
- Blood culture +/-
- Blood culture GN
- Blood culture GP
- Blood culture Fungal

Inclusion criteria:
- Test set
"""

from emrgpt.model import TokenStreamGPT
from emrgpt.trainer import *
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
from tabulate import tabulate

from torch.utils.data import Dataset
import psycopg2
import torch
from emrgptdata.mimic import PostgresUtil
from emrgpt.eval_labs import LabValueDS


if __name__ == "__main__":

    model = TokenStreamGPT.load("cache/archivedmodels/TokenStreamGPT07232025.ckpt")
    result_tokens = ["Growth", "No Growth"]
    ds = LabValueDS(
        model.conf.block_size,
        target_token="bcresults.result",
        expect_tokens=result_tokens,
    )
    dl = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=DL_WORKERS)

    model.eval()
    model = model.to("cuda")

    y_preds = list()
    y_actual = list()

    growth_token_id = ds.pgutil.token2id_map["Growth"]

    for X, mem, y in tqdm(dl):
        with torch.no_grad():
            logits = model(X.to("cuda"), mem.to("cuda"))[:, -1, :]
            probs = F.softmax(logits, dim=-1)

        y_actual.append(y[:, 0])
        y_preds.append(
            probs[:, growth_token_id].detach().cpu(),
        )

    y_actual = torch.cat(y_actual, dim=0)
    y_preds = torch.cat(y_preds, dim=0)

    auroc = roc_auc_score(y_actual, y_preds)
    auprc = average_precision_score(y_actual, y_preds)

    print(f"AUROC: {auroc}")
    print(f"AUPRC: {auprc}")
