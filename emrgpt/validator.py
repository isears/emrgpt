import sys
from emrgpt.model import EventBasedEmrGPT
from emrgpt.validationData import NewVentilationDS
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader

VENT_ENCODINGS = [368, 369, 370, 371]

if __name__ == "__main__":
    model_path = "./cache/EventBasedEmrGPT.pt"

    model = EventBasedEmrGPT(
        vocab_size=5018, n_embd=64, dropout=0.2, block_size=24, n_layer=10, n_head=6
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model = model.to("cuda")

    ds = NewVentilationDS()
    dl = DataLoader(
        dataset=ds, batch_size=128, num_workers=10, collate_fn=ds.collate_fn
    )

    thresholds = torch.load("cache/thresholds.pt").to("cuda")

    preds = list()
    y_true = list()

    for x, y in tqdm(dl):
        x.set_device("cuda")
        out, probabilities = model.generate(seed=x, lookahead=12, thresholds=thresholds)
        window_probability = torch.amax(probabilities[:, :, VENT_ENCODINGS], dim=(1, 2))

        preds.append(window_probability.detach().cpu())
        y_true.append(y)

    preds = torch.cat(preds)
    y_true = torch.cat(y_true)

    print(preds.shape)
    print(y_true.shape)
    print("Done")

    print(f"AUROC: {roc_auc_score(y_true, preds)}")
    print(f"AUPRC: {average_precision_score(y_true, preds)}")
