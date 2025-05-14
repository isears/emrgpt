from emrgpt.tokenModeling.model import TokenStreamGPT
from emrgpt.tokenModeling.trainer import *
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
from tabulate import tabulate

from torch.utils.data import Dataset
import psycopg2
import torch
from emrgpt.tokenModeling.data import PostgresUtil


class LabValueDS(Dataset):

    def __init__(self, block_size: int, target_token: int):
        super().__init__()
        self.pgutil = PostgresUtil()
        self.block_size = block_size
        self.target_token = target_token
        self.target_token_id = self.pgutil.token2id_map[target_token]

        c = psycopg2.connect("")
        cursor = c.cursor()

        cursor.execute(
            """
            --sql
            SELECT
                stay_id,
                count('any') AS count
            FROM (
                SELECT
                    t.stay_id,
                    t.token_id
                FROM mimiciv_local.splits s
                LEFT JOIN mimiciv_local.tokenevents t ON t.stay_id = s.stay_id
                WHERE testset = true
            ) testset_tokens
            WHERE token_id = %s GROUP BY stay_id;
            """,
            (self.target_token_id,),
        )

        res = cursor.fetchall()
        self.examples = [(sid, i) for sid, c in res for i in range(0, c)]
        # Filter out any stays where the event never occurred
        self.examples = [(sid, i) for sid, i in self.examples if i > 0]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        stay_id, event_idx = self.examples[index]

        token_stream = self.pgutil._get_token_stream(stay_id)
        all_event_indices = (token_stream == self.target_token_id).nonzero(
            as_tuple=True
        )[0]
        truncation_idx = all_event_indices[event_idx].item() + 1
        start_idx = max(0, truncation_idx - self.block_size)
        X = token_stream[start_idx:truncation_idx]
        if start_idx > 0:
            history = token_stream[0:start_idx]
        else:
            history = None

        if len(X) < self.block_size:
            X = torch.nn.functional.pad(X, (self.block_size - len(X), 0))

        memory = self.pgutil._build_memory_vector(stay_id, X, history)

        # y with two classes per example: 10th percentile and 90th percentile
        val_token = self.pgutil.id2token_map[token_stream[truncation_idx].item()]
        assert val_token.startswith("magnitude.")
        actual_val = int(val_token.split(".")[-1])
        # TODO: will have to redifine this if we move from deciles to percentiles
        low_val = actual_val <= 0
        high_val = actual_val >= 9
        y = torch.tensor([low_val, high_val], dtype=float)

        return X, memory, y


if __name__ == "__main__":
    model_path = "./cache/TokenStreamGPT.pt"

    results = {}

    for target_token in [
        "chemistry.potassium",
        "chemistry.glucose",
        "chemistry.creatinine",
        "cardiac_marker.troponin_t",
        "complete_blood_count.hemoglobin",
        "complete_blood_count.hematocrit",
    ]:
        print(f"Evaluating: {target_token}")
        ds = LabValueDS(BLOCK_SIZE, target_token=target_token)
        dl = DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            num_workers=DL_WORKERS,
        )

        model = TokenStreamGPT(
            vocab_size=ds.pgutil.vocab_size,
            memory_size=ds.pgutil.memory_size,
            n_embd=N_EMBD,
            dropout=DROPOUT,
            block_size=BLOCK_SIZE,
            n_layer=N_LAYER,
            n_head=N_HEAD,
        )

        model.load_state_dict(torch.load(model_path))
        model.eval()
        model = model.to("cuda")

        y_preds = list()
        y_actual = list()

        low_val_token_id = ds.pgutil.token2id_map["magnitude.0"]
        # TODO: technically this should include 10
        high_val_token_id = ds.pgutil.token2id_map["magnitude.9"]

        for X, mem, y in tqdm(dl):
            with torch.no_grad():
                logits = model(X.to("cuda"), mem.to("cuda"))[:, -1, :]
                probs = F.softmax(logits, dim=-1)

            y_actual.append(y)
            y_preds.append(
                torch.stack(
                    (probs[:, low_val_token_id], probs[:, high_val_token_id]), dim=1
                )
                .detach()
                .cpu()
            )

        y_actual = torch.cat(y_actual, dim=0)
        y_preds = torch.cat(y_preds, dim=0)

        results[f"{target_token} low val AUROC"] = roc_auc_score(
            y_actual[:, 0], y_preds[:, 0]
        )
        results[f"{target_token} low val AUPRC"] = average_precision_score(
            y_actual[:, 0], y_preds[:, 0]
        )
        results[f"{target_token} high val AUROC"] = roc_auc_score(
            y_actual[:, 1], y_preds[:, 1]
        )
        results[f"{target_token} high val AUPRC"] = average_precision_score(
            y_actual[:, 1], y_preds[:, 1]
        )

    print(tabulate([(k, v) for k, v in results.items()]))
