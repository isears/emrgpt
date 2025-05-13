from torch.utils.data import Dataset, DataLoader, Subset, random_split
import torch
from torchinfo import summary
import torch.nn.functional as F
import warnings
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from emrgpt.tokenModeling.model import TokenStreamGPT
from emrgpt.tokenModeling.data import TokenStreamDS


BLOCK_SIZE = 256
MAX_EPOCHS = 50
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
DEVICE = "cuda"
N_HEAD = 6
N_LAYER = 6
N_EMBD = 384
DROPOUT = 0.2
DL_WORKERS = 12
VAL_CHECK_INTERVAL = 200


def calculate_losses(m: TokenStreamGPT, batch: torch.Tensor):
    X, y = batch
    logits = m(X.to(DEVICE))

    B, T, C = logits.shape

    logits = logits.view(B * T, C)
    y = y.view(B * T)

    return F.cross_entropy(logits, y.to(DEVICE))


if __name__ == "__main__":
    torch.manual_seed(42)

    ds = TokenStreamDS(BLOCK_SIZE)

    train_ds, val_ds = random_split(ds, lengths=[0.9, 0.1])

    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        num_workers=DL_WORKERS,
    )
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=DL_WORKERS)

    model = TokenStreamGPT(
        vocab_size=ds.vocab_size,
        n_embd=N_EMBD,
        block_size=BLOCK_SIZE,
        n_head=N_HEAD,
        n_layer=N_LAYER,
        dropout=DROPOUT,
    ).to(DEVICE)

    summary(
        model,
        input_data=torch.zeros(
            (BATCH_SIZE, BLOCK_SIZE), dtype=torch.long, device=DEVICE
        ),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
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
                    val_losses.append(calculate_losses(model, batch).item())

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

            loss = calculate_losses(model, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
            optimizer.step()

    print("Done")
