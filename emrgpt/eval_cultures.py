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

    model = TokenStreamGPT.load("cache/archivedmodels/TokenStreamGPT.ckpt")
    result_tokens = [
        "Gram Positive",
        "Gram Negative",
        "Fungal",
        "Growth Uncategorized",
        "No Growth",
    ]
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

    gn_token_id = ds.pgutil.token2id_map["Gram Positive"]
    gp_token_id = ds.pgutil.token2id_map["Gram Negative"]
    fungus_token_id = ds.pgutil.token2id_map["Fungal"]
    uncategorized_token_id = ds.pgutil.token2id_map["Growth Uncategorized"]
    nogrowth_token_id = ds.pgutil.token2id_map["No Growth"]

    for X, mem, y in tqdm(dl):
        with torch.no_grad():
            logits = model(X.to("cuda"), mem.to("cuda"))[:, -1, :]
            probs = F.softmax(logits, dim=-1)

        y_actual.append(y)
        y_preds.append(
            torch.stack(
                (
                    probs[:, gn_token_id],
                    probs[:, gp_token_id],
                    probs[:, fungus_token_id],
                    probs[:, uncategorized_token_id],
                    probs[:, nogrowth_token_id],
                ),
                dim=1,
            )
            .detach()
            .cpu(),
        )

    y_actual = torch.cat(y_actual, dim=0)
    y_preds = torch.cat(y_preds, dim=0)

    # nogrowth_auroc = roc_auc_score(y_actual[:, -1], y_preds[:, -1])
    # nogrowth_auprc = average_precision_score(y_actual[:, -1], y_preds[:, -1])

    # overall_auroc = roc_auc_score(y_actual, y_preds)
    # overall_auprc = average_precision_score(y_actual, y_preds)

    # print(f"Growth v. No Growth AUROC: {nogrowth_auroc}")
    # print(f"Growth v. No Growth AUPRC: {nogrowth_auprc}")
    # print(f"Overall AUROC: {overall_auroc}")
    # print(f"Overall AUPRC: {overall_auprc}")

    print("OVR results by class:")
    for idx, result_token in enumerate(result_tokens):
        auroc = roc_auc_score(y_actual[:, idx], y_preds[:, idx])
        auprc = average_precision_score(y_actual[:, idx], y_preds[:, idx])

        print(f"{result_token} AUROC: {auroc}")
        print(f"{result_token} AUPRC: {auprc}")

    # y_actual_growth = 1 - y_actual[:, -1]
    # y_preds_growth = y_preds[:, 0:-1].sum(dim=-1)
    # growth_auroc = roc_auc_score(y_actual_growth, y_preds_growth)
    # growth_auprc = average_precision_score(y_actual_growth, y_preds_growth)

    # print(f"Growth v. No Growth AUROC: {growth_auroc}")
    # print(f"Growth v. No Growth AUPRC: {growth_auprc}")
