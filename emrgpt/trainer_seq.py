from torch.utils.data import Dataset, DataLoader, Subset, random_split
import torch
from emrgpt.model import EventBasedEmrGPT
from torchinfo import summary
import torch.nn.functional as F
import warnings
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from emrgpt.sequenceData import EventSequenceDS, EventSequence

# stfu pandas
warnings.simplefilter(action="ignore", category=FutureWarning)

BLOCK_SIZE = 24
MAX_EPOCHS = 50
LEARNING_RATE = 1e-4
BATCH_SIZE = 128
DEVICE = "cuda"
N_HEAD = 6
N_LAYER = 10
N_EMBD = 64
DROPOUT = 0.2
DL_WORKERS = 12
VAL_CHECK_INTERVAL = 200


def calculate_losses(
    m: EventBasedEmrGPT, batch: tuple[EventSequence, torch.Tensor], loss_module
):
    es_x, y = batch
    y = y.to(DEVICE)
    es_x.set_device(DEVICE)

    preds = m(es_x.offsets, es_x.encodings, es_x.values)
    B, T, C = y.shape
    loss = loss_module(preds, y.view(B * T, C))

    return loss


def calculate_utility_reintubation(
    m: EventBasedEmrGPT, reintubation_dl: DataLoader
) -> float:
    m.eval()
    ventilation_idx = reintubation_dl.dataset.tlds.features.index("vent_invasive")
    y_trues = list()
    y_preds = list()

    for batchnum, batch in enumerate(reintubation_dl):
        X, y = batch
        synthetic_data = m.generate(
            max_new_steps=reintubation_dl.dataset.prediction_window, seed=X.to("cuda")
        )

        preds = synthetic_data[:, :, ventilation_idx].sum(dim=1)

        y_trues.append(y)
        y_preds.append(preds.detach().cpu())

    m.train()
    return roc_auc_score(torch.cat(y_trues), torch.cat(y_preds))


if __name__ == "__main__":
    torch.manual_seed(42)

    ds = EventSequenceDS(BLOCK_SIZE)

    train_ds, val_ds = random_split(ds, lengths=[0.9, 0.1])

    # validation_stay_ids = [val_ds.dataset.stay_ids[i] for i in val_ds.indices]
    # reintubation_validation_ds = ReintubationDS(tlds=ds, stay_ids=validation_stay_ids)

    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        num_workers=DL_WORKERS,
        collate_fn=ds.collate_fn,
    )
    val_dl = DataLoader(
        val_ds, batch_size=512, num_workers=DL_WORKERS, collate_fn=ds.collate_fn
    )
    # reintubation_dl = DataLoader(
    #     reintubation_validation_ds, batch_size=32, num_workers=DL_WORKERS
    # )

    model = EventBasedEmrGPT(
        vocab_size=ds.vocab_size,
        n_embd=N_EMBD,
        block_size=BLOCK_SIZE,
        n_head=N_HEAD,
        n_layer=N_LAYER,
        dropout=DROPOUT,
    ).to(DEVICE)

    x, _ = next(iter(train_dl))
    x.set_device(DEVICE)
    summary(
        model,
        input_data=(x.offsets, x.encodings, x.values),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_module = torch.nn.BCEWithLogitsLoss()
    best_val_loss = float("inf")

    for epoch in range(MAX_EPOCHS):
        print(f"--> Training epoch {epoch}")
        # reintubation_utility = calculate_utility_reintubation(model, reintubation_dl)
        # print(f"\tReintubation score: {reintubation_utility}")

        for batchnum, batch in enumerate(train_dl):
            if batchnum % VAL_CHECK_INTERVAL == 0:
                model.eval()
                val_losses = list()

                for batch in val_dl:
                    val_losses.append(
                        calculate_losses(model, batch, loss_module).item()
                    )

                avg_val_loss = sum(val_losses) / len(val_losses)

                if avg_val_loss < best_val_loss:
                    # print(f"{avg_val_loss} < {best_val_loss}, saving checkpoint")
                    best_val_loss = avg_val_loss
                    torch.save(
                        model.state_dict(),
                        f"cache/savedmodels/{model.__class__.__name__}.pt",
                    )

                    print(f"Step {batchnum:04d} validation loss: {avg_val_loss} (*)")
                else:
                    print(f"Step {batchnum:04d} validation loss: {avg_val_loss}")

                model.train()

            loss = calculate_losses(model, batch, loss_module)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
            optimizer.step()

    print("Done")
