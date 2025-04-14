import torch
import lightning as L
from typing import Literal
from torch.nn import functional as F


class BaseGptLM(L.LightningModule):

    def __init__(self, lr: float, model: torch.nn.Module):
        super().__init__()
        torch.set_float32_matmul_precision("high")

        self.loss = F.cross_entropy
        self.model = model
        self.lr = lr

        self.save_hyperparameters(ignore=["model"])

    def _run_model(self, batch):
        x, y = batch
        B, T = x.shape
        y_hat = self.model(x)
        return y.view(B * T), y_hat

    def _do_step(self, batch, stage: Literal["train", "val", "test"]):
        expected_output, actual_output = self._run_model(batch)
        loss = self.loss(actual_output, expected_output)

        self.log(
            f"{stage}_loss",
            loss,
            on_step=True,
            prog_bar=True,
            on_epoch=True,
            sync_dist=True,
        )

        return loss

    def training_step(self, batch):
        return self._do_step(batch, stage="training")

    def validation_step(self, batch):
        return self._do_step(batch, stage="val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer


class EventBasedGptLM(BaseGptLM):

    def __init__(self, lr, model):
        super().__init__(lr, model)

        self.value_loss_multiplier = 1

    def _do_step(self, batch, stage: Literal["train", "val", "test"]):
        tidx, event_id, value = batch

        # Do right-shift of y timeline, truncate x timeline
        x_tidx, x_event_id, x_value = tidx[:, :-1], event_id[:, :-1], value[:, :-1]
        y_tidx, y_event_id, y_value = tidx[:, 1:], event_id[:, 1:], value[:, 1:]

        pred_event_logits, pred_event_values = self.model(x_tidx, x_event_id, x_value)
        # Compute loss for estimated next event type
        event_loss = F.cross_entropy(pred_event_logits, y_event_id)

        # Compute loss value of event
        value_loss = F.mse_loss(pred_event_values[y_event_id], y_value)

        loss = event_loss + (self.value_loss_multiplier * value_loss)

        self.log(
            f"{stage}_loss",
            loss,
            on_step=True,
            prog_bar=True,
            on_epoch=True,
            sync_dist=True,
        )

        return loss
