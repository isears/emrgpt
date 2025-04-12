import sys
from emrgpt import BaseGptLM
from emrgpt.data import ShakespeareDS


def usage():
    print(f"{sys.argv[0]} path_to_ckpt.ckpt path_to_shakespeare.txt")
    quit()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        usage()

    ds = ShakespeareDS(path=sys.argv[2], block_size=1)
    lm = BaseGptLM.load_from_checkpoint(sys.argv[1])
    generated_data = lm.model.generate(device=lm.device)

    print(ds.decode(generated_data))
