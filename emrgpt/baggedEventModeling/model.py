from torch import nn
from emrgpt.model import *
from emrgpt.baggedEventModeling.data import EventSequence


class BaggedEventEmrGPT(nn.Module):
    """
    This model is closer in spirit to NLM

    - Input is a stream of event ids representing chartevents (later maybe inputevents, etc.)
    - Bag embedding aggregates all the events that happened every hour
    - Output is block_size x vocab_size
    - B/c of aggregation over every timestep, problem is multilabel clf rather than multiclass clf
    - No softmax for generation, instead need to define a cutoff value
    - All outputs that exceed cutoff value are predicted event_ids for next timestep

    # TODO: how define cutoff value?
    # TODO: some chartevents have associated values, how can those be incorporated?
    """

    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        dropout: float,
        block_size: int,
        n_layer: int,
        n_head: int,
    ):
        super().__init__()

        self.block_size = block_size
        self.n_embd = n_embd
        self.vocab_size = vocab_size

        # TODO: mode=sum / mode=max ?
        self.embedding_table = nn.EmbeddingBag(vocab_size, n_embd, mode="mean")
        self.positional_encoding = FixedPositionalEncoding(n_embd, block_size, dropout)
        self.blocks = nn.Sequential(
            *[
                AkDecoderBlock(n_embd, n_head, block_size, dropout)
                for _ in range(n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(
        self, offsets: torch.Tensor, encodings: torch.Tensor, values: torch.Tensor
    ):
        assert offsets.ndim == encodings.ndim == values.ndim
        assert encodings.size() == values.size()
        assert len(offsets) % self.block_size == 0, "Should not have a partial block"

        batch_size = len(offsets) // self.block_size

        out = self.embedding_table(encodings, offsets=offsets)  # B * T, C
        out = out.view(batch_size, self.block_size, self.n_embd)

        # PE wants [seq_len, batch_size, d_model] shapes
        out = self.positional_encoding(out.permute(1, 0, 2)).permute(1, 0, 2)

        out = self.blocks(out)
        out = self.ln_f(out)
        out = self.lm_head(out)

        return out.view(batch_size * self.block_size, self.vocab_size)

    def generate(
        self, seed: EventSequence, lookahead: int = 12, thresholds: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert seed.block_size == self.block_size
        assert seed.vocab_size == self.vocab_size

        if thresholds is not None:
            assert len(thresholds) == self.vocab_size

        batch_size = len(seed.offsets) // seed.block_size
        collected_probs = list()

        with torch.no_grad():
            for step_idx in range(0, lookahead):
                next_step_logits = self(seed.offsets, seed.encodings, seed.values).view(
                    batch_size, seed.block_size, seed.vocab_size
                )[:, -1, :]

                next_step_probs = F.sigmoid(next_step_logits)
                collected_probs.append(next_step_probs)

                if thresholds is not None:
                    next_step_preds = next_step_probs > thresholds
                else:
                    next_step_preds = next_step_probs > 0.01

                sliding_window = torch.cat(
                    [seed.to_ohe()[:, 1:, :], next_step_preds.unsqueeze(1)], dim=1
                )

                seed = EventSequence.from_ohe(sliding_window)

        return seed, torch.stack(collected_probs, dim=1)
