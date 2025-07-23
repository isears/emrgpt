from emrgpt.model import TokenStreamGPT
import torch
import random
from emrgptdata.mimic import TokenStreamDS


if __name__ == "__main__":
    model = TokenStreamGPT.load("cache/TokenStreamGPT.ckpt")

    model.eval()
    model = model.to("cuda")

    ds = TokenStreamDS(block_size=model.conf.block_size, testset=True)
    rand_idx = random.choice(range(0, len(ds)))
    X, mem, _ = ds[rand_idx]
    seed = X[X != 0][0:12].to("cuda")

    for x in seed:
        print(ds.postgresUtil.id2token_map[x.item()])

    next_token = ""

    while True:
        seq = list()
        while True:

            next_token = input('Next Action ("" to quit): ')

            if next_token != "":
                if next_token in ds.postgresUtil.token2id_map.keys():
                    seq.append(ds.postgresUtil.token2id_map[next_token])
                else:
                    print(f"{next_token} not found in dict, try again")

            else:
                break

        seq_t = torch.tensor(seq, dtype=torch.long, device="cuda")

        with torch.no_grad():
            if len(seq_t) > 0:
                seed = torch.cat([seed, seq_t])

            generated_token = model.generate_next(
                seed.unsqueeze(0), mem.unsqueeze(0).to("cuda")
            ).squeeze()

            print(ds.postgresUtil.id2token_map[generated_token.item()])

            seed = torch.cat([seed, generated_token.unsqueeze(0)])
