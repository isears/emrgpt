from dataclasses import dataclass
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from emrgpt.sequenceData import EventSequenceDS
from emrgpt.model import EventBasedEmrGPT
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class EventOfInterest:
    name: str
    encoding_id: int

    def __post_init__(self):
        self.y_true = list()
        self.y_hat = list()

    def add_batched_preds(self, y_true: torch.Tensor, y_hat: torch.Tensor):
        assert y_true.ndim == 1
        assert y_hat.ndim == 1
        assert len(y_true) == len(y_hat)

        self.y_true.append(y_true)
        self.y_hat.append(y_hat)

    def print_metrics(self):
        y_true = torch.cat(self.y_true)
        y_hat = torch.cat(self.y_hat)

        auroc = roc_auc_score(y_true, y_hat)
        auprc = average_precision_score(y_true, y_hat)

        print(f"{self.name} ({self.encoding_id})")
        print(f"\tAUROC: {auroc}")
        print(f"\tAUPRC: {auprc}")


# Common, clinically relevant events
EVENTS_OF_INTEREST = [
    EventOfInterest("A-fib", 114),
    EventOfInterest("HR Alarm - High", 106),
    EventOfInterest("HR Alarm - Low", 107),
    EventOfInterest("Non-invasive BP Alarm - High", 182),
    EventOfInterest("Non-invasive BP Alarm - Low", 183),
    EventOfInterest("Arterial BP Alarm - High", 136),
    EventOfInterest("Arterial BP Alarm - Low", 135),
    EventOfInterest("Pulse Ox Alarm - High", 106),
    EventOfInterest("Pulse Ox Alarm - Low", 107),
]

if __name__ == "__main__":
    model_path = "./cache/EventBasedEmrGPT.pt"

    model = EventBasedEmrGPT(
        vocab_size=5018, n_embd=64, dropout=0.2, block_size=24, n_layer=10, n_head=6
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model = model.to("cuda")

    ds = EventSequenceDS(block_size=24)
    dl = DataLoader(dataset=ds, batch_size=32, num_workers=5, collate_fn=ds.collate_fn)

    y_hat = list()
    y_true = list()

    for idx, (x, y) in tqdm(enumerate(dl)):
        x.set_device("cuda")

        _, probabilities = model.generate(seed=x, lookahead=1)
        probabilities = probabilities.detach().cpu().squeeze(1)

        for eoi in EVENTS_OF_INTEREST:
            eoi.add_batched_preds(
                y[:, -1, eoi.encoding_id].bool(), probabilities[:, eoi.encoding_id]
            )

    for eoi in EVENTS_OF_INTEREST:
        eoi.print_metrics()
