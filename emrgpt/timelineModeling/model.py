from emrgpt.model import *
from torch import nn


class TimelineBasedEmrGPT(nn.Module):
    """
    This model is closer to the original Time Series Transformer
    - Input is a B x T x n_event_types matrix
    - Embedding layer is a linear layer n_event_types -> d_model
    - Loss is ideally computed only on available data
        (i.e. don't punish model for not knowing t+1 CBC if it was never taken)

    # TODO: embedding layer can be massive in terms of number of parameters.
    Is there a more efficient alternative to linear here?
    """

    def __init__(
        self,
        n_event_types: int,
        d_model: int,
        block_size: int,
        max_len: int,
        n_head: int,
        n_layer: int,
        dropout: float,
    ):
        super().__init__()

        self.block_size = block_size
        self.d_model = d_model
        self.n_event_types = n_event_types
        self.proj = nn.Linear(n_event_types, d_model)
        self.positional_encoding = FixedPositionalEncoding(d_model, max_len, dropout)

        self.blocks = nn.Sequential(
            *[
                AkDecoderBlock(d_model, n_head, block_size, dropout)
                for _ in range(n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, n_event_types)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # batch_dim, seq_len, feat_dim
        assert x.ndim == 3
        B, T, C = x.shape

        x = self.proj(x) * math.sqrt(self.d_model)

        # PE wants [seq_len, batch_size, d_model] shapes
        x = self.positional_encoding(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.blocks(x)
        x = self.ln_f(x)

        preds = self.lm_head(x).view(B * T, C)

        return F.sigmoid(preds)

    def generate(
        self, max_new_steps: int = 12, device: str = "cuda", seed: torch.Tensor = None
    ) -> torch.Tensor:
        if seed is None:
            generated_data = torch.zeros(
                (1, 1, self.n_event_types),
                dtype=torch.float,
                device=device,
            )
        else:
            assert seed.ndim == 3
            generated_data = seed

        B, T, C = generated_data.shape
        for _ in range(max_new_steps):
            pred = self(generated_data).view(B, T, C)[:, -1, :]
            generated_data = torch.cat(
                (generated_data[:, 1:, :], pred.unsqueeze(1)), dim=1
            )

        return generated_data
