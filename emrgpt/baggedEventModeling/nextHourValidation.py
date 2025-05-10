from dataclasses import dataclass
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from emrgpt.baggedEventModeling.data import EventSequenceDS
from emrgpt.model import EventBasedEmrGPT
from torch.utils.data import DataLoader
from tqdm import tqdm
from emrgpt.util import *
import pandas as pd

import warnings

warnings.filterwarnings("ignore", module="sklearn")

if __name__ == "__main__":
    model_path = "./cache/EventBasedEmrGPT.pt"

    model = EventBasedEmrGPT(
        vocab_size=5018, n_embd=64, dropout=0.2, block_size=24, n_layer=10, n_head=6
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model = model.to("cuda")

    ds = EventSequenceDS(block_size=24, test=True)
    dl = DataLoader(dataset=ds, batch_size=32, num_workers=5, collate_fn=ds.collate_fn)

    y_hat = list()
    y_true = list()

    print("Generating preds...")
    for x, y in tqdm(dl):
        x.set_device("cuda")

        _, probabilities = model.generate(seed=x, lookahead=1)
        probabilities = probabilities.detach().cpu().squeeze(1)

        y_true.append(y[:, -1, :].bool())
        y_hat.append(probabilities)

        # for eoi in EVENTS_OF_INTEREST:
        #     eoi.add_batched_preds(
        #         y[:, -1, eoi.encoding_id].bool(), probabilities[:, eoi.encoding_id]
        #     )

    # for eoi in EVENTS_OF_INTEREST:
    #     eoi.print_metrics()

    y_true = torch.cat(y_true)
    y_hat = torch.cat(y_hat)

    scoring_data = []

    print("Scoring...")
    for encoding_id, token_name in tqdm(get_encoding_map().items()):
        scoring_data.append(
            {
                "Name": token_name,
                "AUROC": roc_auc_score(y_true[:, encoding_id], y_hat[:, encoding_id]),
                "AUPRC": average_precision_score(
                    y_true[:, encoding_id], y_hat[:, encoding_id]
                ),
            }
        )

    df = pd.DataFrame(data=scoring_data)

    print(f"Avg AUROC: {df['AUROC'].mean()}")
    print(f"Avg AUPRC: {df['AUPRC'].mean()}")
    print(df.nlargest(n=25, columns="AUPRC"))
