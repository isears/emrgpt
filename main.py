from lightning.pytorch.cli import LightningCLI
from emrgpt import BasicDM, BaseGptLM
import torch

# https://discuss.pytorch.org/t/runtimeerror-received-0-items-of-ancdata/4999/3
# NOTE: set ulimit -n to 65535 but can try below if that doesn't work
# torch.multiprocessing.set_sharing_strategy("file_system")


def cli_main():
    cli = LightningCLI(
        BaseGptLM,
        BasicDM,
        subclass_mode_data=True,
        subclass_mode_model=True,
        save_config_callback=None,
        parser_kwargs={"default_config_files": ["configs/default_trainer.yaml"]},
        # run=False,
    )


if __name__ == "__main__":
    cli_main()
