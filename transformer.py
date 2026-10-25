"""A small, readable implementation of the original Transformer.

This file intentionally uses ordinary PyTorch tensor operations instead of
``torch.nn.Transformer``.  Reading the code from ``Transformer.forward`` down
to ``MultiHeadAttention.forward`` is a useful way to follow the data flow.

The model is an encoder-decoder Transformer, suitable for sequence-to-
sequence tasks such as translation.  The command-line demo at the bottom
trains it on a tiny copy task (the target sequence should equal the source).

Tensor conventions used throughout this file:

    tokens:  [batch, sequence_length]
    hidden:  [batch, sequence_length, d_model]
    heads:   [batch, num_heads, sequence_length, head_dim]
    logits:  [batch, sequence_length, vocabulary_size]
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class TransformerConfig:
    """Hyperparameters kept in one place so experiments are easy to change."""

    src_vocab_size: int
    tgt_vocab_size: int
    d_model: int = 128
    num_heads: int = 4
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    d_ff: int = 512
    dropout: float = 0.1
    max_seq_len: int = 128

    def __post_init__(self) -> None:
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")


def subsequent_mask(size: int, device: torch.device | None = None) -> Tensor:
    """Return a boolean causal mask: True means attention is allowed.

    A decoder position i may read positions 0..i, but must not peek at future
    target tokens.  ``torch.tril`` creates exactly that lower-triangular rule.
    Shape is [1, 1, size, size], which broadcasts over batch and heads.
    """

    return torch.tril(torch.ones(size, size, dtype=torch.bool, device=device))[
        None, None
    ]


def padding_mask(tokens: Tensor, pad_id: int) -> Tensor:
    """Return [batch, 1, 1, sequence] mask where non-padding is True."""

    return (tokens != pad_id)[:, None, None, :]


class TokenAndPositionEmbedding(nn.Module):
    """Token embeddings plus fixed sinusoidal position information.

    Self-attention alone is permutation-invariant: without positions, the
    model sees a bag of tokens.  The sine/cosine table gives every position a
    distinct, smoothly varying signal without adding trainable parameters.
    """

    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int, dropout: float):
        super().__init__()
        self.token = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)

        positions = torch.arange(max_seq_len, dtype=torch.float32)[:, None]
        frequencies = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / d_model)
        )
        table = torch.zeros(max_seq_len, d_model)
        table[:, 0::2] = torch.sin(positions * frequencies)
        table[:, 1::2] = torch.cos(positions * frequencies)
        self.register_buffer("position", table[None], persistent=False)

    def forward(self, tokens: Tensor) -> Tensor:
        length = tokens.size(1)
        if length > self.position.size(1):
            raise ValueError("sequence is longer than max_seq_len")
        # sqrt(d_model) keeps token and position magnitudes comparable.
        x = self.token(tokens) * math.sqrt(self.token.embedding_dim)
        return self.dropout(x + self.position[:, :length])


class MultiHeadAttention(nn.Module):
    """Scaled dot-product attention split into several independent heads."""

    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: Tensor) -> Tensor:
        batch, length, _ = x.shape
        # [B, T, H*D] -> [B, H, T, D]
        return x.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))

        # Scores are pairwise token affinities: [B, H, query_len, key_len].
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        if mask is not None:
            # Masked logits must be -infinity before softmax, so their
            # probability becomes zero. Bool masks use True = keep.
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = self.dropout(F.softmax(scores, dim=-1))
        attended = weights @ v  # [B, H, query_len, head_dim]

        # Join heads back: [B, H, T, D] -> [B, T, H*D].
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(query.size(0), query.size(1), -1)
        return self.out_proj(attended)


class FeedForward(nn.Module):
    """The position-wise MLP applied independently to every token."""

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class EncoderLayer(nn.Module):
    """One encoder block: self-attention followed by a feed-forward network."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.self_attn = MultiHeadAttention(config.d_model, config.num_heads, config.dropout)
        self.feed_forward = FeedForward(config.d_model, config.d_ff, config.dropout)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor, src_mask: Tensor | None) -> Tensor:
        # Pre-norm makes optimization more stable for deep Transformers.
        x = x + self.dropout(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), src_mask))
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x


class DecoderLayer(nn.Module):
    """One decoder block with masked self-attention and cross-attention."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.self_attn = MultiHeadAttention(config.d_model, config.num_heads, config.dropout)
        self.cross_attn = MultiHeadAttention(config.d_model, config.num_heads, config.dropout)
        self.feed_forward = FeedForward(config.d_model, config.d_ff, config.dropout)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.norm3 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None,
        memory_mask: Tensor | None,
    ) -> Tensor:
        normalized = self.norm1(x)
        x = x + self.dropout(self.self_attn(normalized, normalized, normalized, tgt_mask))
        normalized = self.norm2(x)
        x = x + self.dropout(self.cross_attn(normalized, memory, memory, memory_mask))
        x = x + self.dropout(self.feed_forward(self.norm3(x)))
        return x


class Transformer(nn.Module):
    """Complete encoder-decoder Transformer.

    During training, ``tgt_input`` is the target shifted right (starts with
    <bos>) and ``tgt_output`` is the same sequence shifted left (ends with
    <eos>). This is teacher forcing: every position predicts its known next
    token in parallel.
    """

    def __init__(self, config: TransformerConfig, pad_id: int = 0):
        super().__init__()
        self.config = config
        self.pad_id = pad_id
        self.src_embedding = TokenAndPositionEmbedding(
            config.src_vocab_size, config.d_model, config.max_seq_len, config.dropout
        )
        self.tgt_embedding = TokenAndPositionEmbedding(
            config.tgt_vocab_size, config.d_model, config.max_seq_len, config.dropout
        )
        self.encoder = nn.ModuleList(EncoderLayer(config) for _ in range(config.num_encoder_layers))
        self.decoder = nn.ModuleList(DecoderLayer(config) for _ in range(config.num_decoder_layers))
        self.encoder_norm = nn.LayerNorm(config.d_model)
        self.decoder_norm = nn.LayerNorm(config.d_model)
        self.output = nn.Linear(config.d_model, config.tgt_vocab_size)

    def encode(self, src: Tensor, src_mask: Tensor | None = None) -> Tensor:
        x = self.src_embedding(src)
        for layer in self.encoder:
            x = layer(x, src_mask)
        return self.encoder_norm(x)

    def decode(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
    ) -> Tensor:
        x = self.tgt_embedding(tgt)
        for layer in self.decoder:
            x = layer(x, memory, tgt_mask, memory_mask)
        return self.decoder_norm(x)

    def forward(self, src: Tensor, tgt_input: Tensor) -> Tensor:
        src_mask = padding_mask(src, self.pad_id)
        # Combine the causal rule with target padding. Broadcasting produces
        # [batch, 1, target_len, target_len].
        tgt_mask = subsequent_mask(tgt_input.size(1), tgt_input.device)
        tgt_mask = tgt_mask & padding_mask(tgt_input, self.pad_id)
        memory = self.encode(src, src_mask)
        hidden = self.decode(tgt_input, memory, tgt_mask, src_mask)
        return self.output(hidden)


def greedy_decode(
    model: Transformer,
    src: Tensor,
    bos_id: int,
    eos_id: int,
    max_length: int,
) -> Tensor:
    """Autoregressively generate one token at a time for inference."""

    model.eval()
    with torch.no_grad():
        memory = model.encode(src, padding_mask(src, model.pad_id))
        generated = torch.full((src.size(0), 1), bos_id, dtype=torch.long, device=src.device)
        for _ in range(max_length - 1):
            hidden = model.decode(
                generated,
                memory,
                subsequent_mask(generated.size(1), src.device),
                padding_mask(src, model.pad_id),
            )
            next_token = model.output(hidden[:, -1:]).argmax(dim=-1)
            generated = torch.cat([generated, next_token], dim=1)
            if bool((next_token == eos_id).all()):
                break
    return generated


def copy_task_demo(steps: int = 200, device: str = "cpu") -> None:
    """Train briefly on a copy task so the implementation can be exercised."""

    torch.manual_seed(0)
    device_obj = torch.device(device)
    # 0=padding, 1=<bos>, 2=<eos>, 3..11=payload tokens.
    config = TransformerConfig(12, 12, d_model=64, num_heads=4, d_ff=128, max_seq_len=16, dropout=0.0)
    model = Transformer(config, pad_id=0).to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for step in range(1, steps + 1):
        payload = torch.randint(3, 12, (32, 5), device=device_obj)
        src = torch.cat([payload, torch.full((32, 1), 2, device=device_obj)], dim=1)
        target = torch.cat([torch.full((32, 1), 1, device=device_obj), payload, torch.full((32, 1), 2, device=device_obj)], dim=1)
        optimizer.zero_grad(set_to_none=True)
        logits = model(src, target[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, 12), target[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        if step == 1 or step % max(1, steps // 5) == 0:
            print(f"step {step:>4}/{steps}  loss={loss.item():.4f}")

    sample = torch.tensor([[3, 7, 5, 9, 4, 2]], device=device_obj)
    print("source:   ", sample.tolist()[0])
    print("generated:", greedy_decode(model, sample, bos_id=1, eos_id=2, max_length=7).tolist()[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the educational Transformer on a copy task")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    copy_task_demo(args.steps, args.device)
