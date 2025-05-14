from emrgpt.tokenModeling.model import TokenStreamGPT
from emrgpt.tokenModeling.trainer import *
from emrgpt.tokenModeling.data import TokenStreamDS


if __name__ == "__main__":
    model_path = "./cache/TokenStreamGPT.pt"
    ds = TokenStreamDS(BLOCK_SIZE, testset=True)
    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        num_workers=DL_WORKERS,
    )

    model = TokenStreamGPT(
        vocab_size=ds.vocab_size,
        memory_size=ds.memory_size,
        n_embd=N_EMBD,
        dropout=DROPOUT,
        block_size=BLOCK_SIZE,
        n_layer=N_LAYER,
        n_head=N_HEAD,
    )

    model.load_state_dict(torch.load(model_path))
    model.eval()
    model = model.to("cuda")

    X, mem, y = next(iter(dl))

    preds = model.generate(X.to(DEVICE), mem.to(DEVICE), 12, ds._hourtokens)

    print("Done")
