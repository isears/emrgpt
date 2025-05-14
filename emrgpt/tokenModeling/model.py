from emrgpt.model import *


class TokenStreamGPT(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        memory_size: int,
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

        self.embedding_table = nn.Embedding(vocab_size, n_embd)
        self.memory_projection = nn.Linear(memory_size, n_embd)
        self.positional_encoding = FixedPositionalEncoding(
            n_embd, block_size + 1, dropout
        )
        self.blocks = nn.Sequential(
            *[
                AkDecoderBlock(n_embd, n_head, block_size + 1, dropout)
                for _ in range(n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        x_embd = self.embedding_table(x)  # (B,T,C)
        mem_proj = self.memory_projection(memory)
        x_mem = torch.cat((mem_proj.unsqueeze(1), x_embd), dim=1)
        out = self.positional_encoding(x_mem.permute(1, 0, 2)).permute(1, 0, 2)
        out = self.blocks(out)
        out = self.ln_f(out)
        logits = self.lm_head(out)

        return logits

    def generate(self, seed: torch.Tensor) -> torch.Tensor: ...
