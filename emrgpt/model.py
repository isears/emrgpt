"""
Inspired by Andrej Karpathy's nanoGPT
https://github.com/karpathy/ng-video-lecture
"""

import torch.nn as nn
import torch
from torch.nn import functional as F
import math


class AkSelfAttentionHead(nn.Module):
    def __init__(self, n_embd: int, head_size: int, block_size: int, dropout: float):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v
        return out


class AkMultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        n_embd: int,
        head_size: int,
        block_size: int,
        dropout: float,
    ):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                AkSelfAttentionHead(n_embd, head_size, block_size, dropout)
                for _ in range(num_heads)
            ]
        )
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):

    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class AkDecoderBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        head_size = n_embd // n_head
        self.self_attention = AkMultiHeadAttention(
            num_heads=n_head,
            n_embd=n_embd,
            head_size=head_size,
            block_size=block_size,
            dropout=dropout,
        )
        self.feedforward = FeedForward(n_embd=n_embd, dropout=dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.self_attention(self.ln1(x))
        x = x + self.feedforward(self.ln2(x))
        return x


# From https://github.com/pytorch/examples/blob/master/word_language_model/model.py
class FixedPositionalEncoding(torch.nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model, max_len, dropout=0.1, scale_factor=1.0):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer(
            "pe", pe
        )  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[: x.size(0), :]  # type: ignore
        return self.dropout(x)


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

        return preds

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
