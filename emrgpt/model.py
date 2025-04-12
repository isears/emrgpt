"""
Inspired by Andrej Karpathy's nanoGPT
https://github.com/karpathy/ng-video-lecture
"""

import torch.nn as nn
import torch
from torch.nn import functional as F


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


class AkGPT(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        block_size: int,
        n_head: int,
        n_layer: int,
        dropout: float,
    ):
        super().__init__()

        self.block_size = block_size

        # TODO
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # TODO
        self.position_embedding_table = nn.Embedding(self.block_size, n_embd)

        self.blocks = nn.Sequential(
            *[
                AkDecoderBlock(n_embd, n_head, self.block_size, dropout)
                for _ in range(n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, x: torch.Tensor):
        B, T = x.shape
        tok_emb = self.token_embedding_table(x)
        # TODO
        pos_emb = self.position_embedding_table(torch.arange(T, device=x.device))

        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)
        B, T, C = logits.shape
        logits = logits.view(B * T, C)

        return logits

    def generate(self, max_new_tokens: int = 500, device: str = "cuda"):
        generated_data = torch.zeros((1, 1), dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            # crop to blocksize before sending to model
            logits = self(generated_data[:, -self.block_size :])

            # Just get the last timestep (the prediction)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)

            next_data = torch.multinomial(probs, num_samples=1)
            generated_data = torch.cat((generated_data, next_data), dim=1)

        return generated_data
