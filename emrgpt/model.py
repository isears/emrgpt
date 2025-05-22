from emrgpt.ak_transformer import *
from dataclasses import dataclass


@dataclass
class TokenStreamGptConfig:
    vocab_size: int
    memory_size: int
    n_embd: int
    dropout: float
    block_size: int
    n_layer: int
    n_head: int


class TokenStreamGPT(nn.Module):

    def __init__(self, conf: TokenStreamGptConfig):
        super().__init__()

        self.conf = conf

        self.embedding_table = nn.Embedding(self.conf.vocab_size, self.conf.n_embd)
        self.memory_projection = nn.Linear(self.conf.memory_size, self.conf.n_embd)
        self.positional_encoding = FixedPositionalEncoding(
            self.conf.n_embd, self.conf.block_size + 1, self.conf.dropout
        )
        self.blocks = nn.Sequential(
            *[
                AkDecoderBlock(
                    self.conf.n_embd,
                    self.conf.n_head,
                    self.conf.block_size + 1,
                    self.conf.dropout,
                )
                for _ in range(self.conf.n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(self.conf.n_embd)
        self.lm_head = nn.Linear(self.conf.n_embd, self.conf.vocab_size)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        x_embd = self.embedding_table(x)  # (B,T,C)
        mem_proj = self.memory_projection(memory)
        x_mem = torch.cat((mem_proj.unsqueeze(1), x_embd), dim=1)
        out = self.positional_encoding(x_mem.permute(1, 0, 2)).permute(1, 0, 2)
        out = self.blocks(out)
        out = self.ln_f(out)
        logits = self.lm_head(out)

        return logits

    def generate(
        self,
        seed: torch.Tensor,
        memory: torch.Tensor,
        lookahead_hrs: int,
        hourtokens: torch.Tensor,
    ) -> torch.Tensor:
        assert seed.ndim == 2
        B, T = seed.shape
        # assert T == self.block_size

        generated_tokens = torch.tensor([], dtype=torch.long, device=seed.device)
        hour_counts = torch.zeros((B,), dtype=torch.int, device=seed.device)
        hourtokens = hourtokens.to(seed.device)

        while not (hour_counts >= lookahead_hrs).all():
            generated_count = (
                generated_tokens.shape[1] if generated_tokens.ndim == 2 else 0
            )

            if generated_count < self.conf.block_size:
                x = torch.cat(
                    [
                        seed[:, generated_count:],
                        generated_tokens,
                    ],
                    dim=1,
                )
            else:
                x = generated_tokens[:, (generated_count - self.conf.block_size) :]

            assert x.ndim == 2
            assert x.shape[0] == B
            assert x.shape[1] == T

            logits = self(x, memory)[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            hour_counts += (next_token == hourtokens.unsqueeze(0)).sum(dim=1)
            # zero-out any event that happen after lookahead hours
            # may help downstream tasks recognize when sequence has ended
            # NOTE: this potentially wastes a lot of computation for large batches
            next_token = next_token.squeeze() * (hour_counts <= lookahead_hrs)
            generated_tokens = torch.cat(
                (generated_tokens, next_token.unsqueeze(1)), dim=1
            )

        return generated_tokens

    def generate_next(
        self, seed: torch.Tensor, memory: torch.Tensor, return_probs: bool = False
    ) -> torch.Tensor:
        assert seed.ndim == 2
        B, T = seed.shape

        logits = self(seed, memory)[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        if return_probs:
            return next_token, probs
        else:
            return next_token

    # TODO: I suspect this occassionally hangs when the model falls into a
    # trap where it generates a sequence with no hour tokens
    # Should specify a max_tokens_per_hour thing and manually insert them
    # to ensure no hangs
    def generate_nonbatch(
        self,
        seed: torch.Tensor,
        memory: torch.Tensor,
        lookahead_hrs: int,
        hourtokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert seed.ndim == 1

        generated_tokens = torch.tensor([], dtype=torch.long, device=seed.device)
        hour_count = 0
        generated_count = 0
        hourtokens = hourtokens.to(seed.device)
        probs_accumulated = torch.zeros((self.conf.vocab_size,), device=seed.device)

        while not (hour_count >= lookahead_hrs):
            # TODO: need some way to update memory

            if generated_count < self.conf.block_size:
                x = torch.cat(
                    [
                        seed[generated_count:],
                        generated_tokens,
                    ]
                )
            else:
                x = generated_tokens[(generated_count - self.conf.block_size) :]

            logits = self(x.unsqueeze(0), memory.unsqueeze(0))[0, -1, :]
            probs = F.softmax(logits, dim=0)
            probs_accumulated += probs
            next_token = torch.multinomial(probs, num_samples=1)

            hour_count += (next_token == hourtokens).sum()
            generated_tokens = torch.cat((generated_tokens, next_token))
            generated_count += 1

        return generated_tokens, probs_accumulated / generated_count

    def save(self, path: str, training_metadata: dict = None):
        save_data = {
            "state_dict": self.state_dict(),
            "config": self.conf,
            "meta": training_metadata,
        }

        torch.save(save_data, path)

    @classmethod
    def load(cls, path: str) -> "TokenStreamGPT":
        save_data = torch.load(path, weights_only=False)

        model = cls(conf=save_data["config"])
        model.load_state_dict(save_data["state_dict"])

        return model
