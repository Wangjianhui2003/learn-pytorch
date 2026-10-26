"""MHA vs MQA vs GQA — the same attention, with one number changed.

Run it:  python GQA-demo.py

--------------------------------------------------------------------------
The one-sentence version
--------------------------------------------------------------------------
MHA, MQA and GQA are *exactly the same attention algorithm*.  The only
difference is **how many Key/Value heads you have** compared to Query heads:

    MHA  (Multi-Head Attention)      : #KV heads == #Q heads   (1 : 1)
    GQA  (Grouped-Query Attention)   : #KV heads <  #Q heads   (a few Q heads share one KV)
    MQA  (Multi-Query Attention)     : #KV heads == 1          (every Q head shares the SAME KV)

So the whole file is one class (`GroupedQueryAttention`) plus two tiny
subclasses that just pick different numbers.  If you understand the numbers
below, you understand all three.

--------------------------------------------------------------------------
The high-school analogy
--------------------------------------------------------------------------
Attention in one picture:

    "Every word asks a question (Q), and every other word hands out
     a name tag (K) and some content (V)."

* Q = Query  : "what am I looking for?"
* K = Key    : "what do I contain?"        (used to decide WHO to listen to)
* V = Value  : "here is my actual info"    (the stuff you copy once you chose)

Score = Q · K  ->  softmax  ->  weighted average of the V's.

A "head" is like a group of students reading the same sentence with
different highlighters: one head tracks grammar, another tracks who-did-what.
`num_heads` is how many highlighters you own.

Now the interesting part.  Each head needs its own K and V too.  In MHA that
means 32 heads -> 32 sets of K/V.  During generation we must remember every
past K and V (the "KV cache"), and that gets huge:

    KV cache size ∝ num_kv_heads × head_dim × sequence_length × batch

GQA's trick: let several Q heads **share** one K/V set.  Fewer KV heads =
smaller cache = faster long-context generation, while keeping many Q heads
(so the model still gets lots of different "viewpoints").

    MHA:  Q0 Q1 Q2 Q3  ... Q7        KV0 KV1 KV2 KV3 ... KV7   (8 caches)
    GQA:  Q0 Q1 | Q2 Q3 | ...        KV0     KV1               (2 caches, 4:1)
    MQA:  Q0 Q1 Q2 Q3 ... Q7         KV0                       (1 cache)

Tensor conventions in this file:

    x:            [batch, seq_len, d_model]      ordinary hidden states
    q:            [batch, num_heads,    seq_len, head_dim]
    k, v (cached):[batch, num_kv_heads, seq_len, head_dim]   <-- smaller!
    scores:       [batch, num_heads, seq_len, seq_len]       one score per pair

--------------------------------------------------------------------------
How the KV cache in this file works
--------------------------------------------------------------------------
A transformer is autoregressive: to pick token 100 it must "look at" tokens
1..99.  Recomputing their K and V on every step would be wasted work, because
a token's K and V never change once computed.  So we keep them: that saved
tensor is the KV cache.

Two phases, one code path (`forward` with an optional `kv_cache` argument):

  PREFILL  kv_cache=None, x = whole prompt        -> fills the cache
  DECODE   kv_cache=(k, v), x = one new token     -> appends, then reads it

Shape trace (batch=1, num_heads=8, num_kv_heads=2, head_dim=64):

  prefill 4 tokens:  k = [1, 2, 4, 64]   scores = [1, 8, 4, 4]
  decode token 5:    k = [1, 2, 1, 64] --cat--> [1, 2, 5, 64]
                     --_repeat_kv--> [1, 8, 5, 64] --scores--> [1, 8, 1, 5]

Only K and V are cached, never Q: a query is used once, by its own token, and
then thrown away.  K/V are the ones every *future* token needs to read.

Two things this file does on purpose and two it skips:

  * The cache holds the SMALL un-broadcast tensor (num_kv_heads, not
    num_heads).  `_repeat_kv` runs after the concat, on a temporary.
  * The causal mask is shifted by `k_len - q_len` (line ~213).  With a cache
    the query is the newest token, so it must see ALL cached keys; a plain
    lower triangle would hide all but the oldest one.
  * Skipped: position bookkeeping (real models must tell the model *where*
    the new token sits, e.g. RoPE applied to the cached K) and preallocated
    buffers (torch.cat copies the whole cache every step, O(T) per token;
    vLLM-style "paged" caches write into fixed slots instead).
  * Skipped: eviction / sliding windows / quantization of the cache.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class AttentionConfig:
    """All the knobs. Note that only `num_kv_heads` decides MHA/MQA/GQA."""

    d_model: int = 512
    num_heads: int = 8  # how many Query heads ("how many highlighters")
    num_kv_heads: int = 8  # how many Key/Value heads ("how many shared caches")
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        # Each group of Q heads needs a whole KV head, so the split must be even.
        if self.num_heads % self.num_kv_heads:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be a multiple of "
                f"num_kv_heads ({self.num_kv_heads})"
            )

    @property
    def head_dim(self) -> int:
        # The size of one head's little vector. 512 / 8 = 64 numbers per head.
        return self.d_model // self.num_heads

    @property
    def group_size(self) -> int:
        # How many Q heads share one KV head. MHA=1, MQA=num_heads, GQA=in between.
        return self.num_heads // self.num_kv_heads


class GroupedQueryAttention(nn.Module):
    """Causal self-attention where Q heads outnumber KV heads.

    Set num_kv_heads == num_heads -> this *is* MHA.
    Set num_kv_heads == 1         -> this *is* MQA.
    Anything in between          -> this is GQA.
    """

    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.group_size = config.group_size
        self.dropout = config.dropout

        # --- the only structural difference between MHA / GQA / MQA -----------
        # Q must produce one vector per Query head, so its output size is
        # num_heads * head_dim (= d_model, unchanged).
        self.q_proj = nn.Linear(
            config.d_model, config.num_heads * config.head_dim, bias=False
        )
        # K and V only need num_kv_heads * head_dim outputs. With GQA/MQA these
        # two matrices are physically smaller, which is where the savings live.
        self.k_proj = nn.Linear(
            config.d_model, config.num_kv_heads * config.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.d_model, config.num_kv_heads * config.head_dim, bias=False
        )

        # Back to d_model at the end, exactly like normal attention.
        self.out_proj = nn.Linear(
            config.num_heads * config.head_dim, config.d_model, bias=False
        )

    # ---------------------------------------------------------------- helpers
    def _split_heads(self, x: Tensor, num_heads: int) -> Tensor:
        """[B, T, H*D] -> [B, H, T, D].

        We chop the last dimension into H chunks of size D and move H next to
        the batch, because heads are processed independently (like separate
        small attention problems running in parallel).
        """
        batch, length, _ = x.shape
        return x.view(batch, length, num_heads, self.head_dim).transpose(1, 2)

    def _repeat_kv(self, x: Tensor) -> Tensor:
        """Copy each KV head `group_size` times so it can pair with its Q heads.

        Example with 4 Q heads and 2 KV heads (group_size = 2):

            kv heads : [k0, k1]
            after    : [k0, k0, k1, k1]
                         ^Q0 ^Q1 ^Q2 ^Q3

        So Q0 and Q1 both look at k0/v0, and Q2 and Q3 both look at k1/v1.
        This is the entire GQA mechanism. Nothing else changes.

        Note: this makes a real copy of the tensors for the matrix multiply.
        Production kernels skip the copy and just *read* the same KV memory
        several times. The math is identical; only the speed differs.
        """
        if self.group_size == 1:  # MHA: nothing to do
            return x
        return x.repeat_interleave(self.group_size, dim=1)

    # ---------------------------------------------------------------- forward
    def forward(
        self, x: Tensor, kv_cache: tuple[Tensor, Tensor] | None = None
    ) -> Tensor:
        """x: [batch, seq_len, d_model] -> same shape.

        `kv_cache` is optional: pass the previously stored K/V to generate one
        token at a time (that's what a chat model does).
        """
        batch, length, _ = x.shape

        # 1) Project the input into Q, K, V and cut them into heads.
        q = self._split_heads(self.q_proj(x), self.num_heads)  # [B, H,   T, D]
        k = self._split_heads(self.k_proj(x), self.num_kv_heads)  # [B, Hkv, T, D]
        v = self._split_heads(self.v_proj(x), self.num_kv_heads)  # [B, Hkv, T, D]

        # 2) Incremental decoding: stick the new K/V onto the old ones.
        #    This cache only stores num_kv_heads vectors per token, not
        #    num_heads -> that is exactly the memory win of GQA/MQA.
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        # 3) Fan the shared KV heads back out to one-per-Q-head ("broadcast").
        k = self._repeat_kv(k)  # [B, H, T_all, D]
        v = self._repeat_kv(v)

        # 4) Classic scaled dot-product attention.
        #    q @ k^T gives one "how much do I like you" score for every
        #    (query token, key token) pair. Divide by sqrt(D) so the numbers
        #    don't blow up as D grows (keeps softmax out of the flat regions).
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)

        # 5) Causal mask: a token may look at itself and the past, never the
        #    future (otherwise the model could cheat while being trained).
        #    Lower triangle = allowed (True), upper triangle = blocked.
        #
        #    The `diagonal` shift matters for incremental decoding. With a KV
        #    cache, q_len can be 1 while k  _len is large: that single query is
        #    the NEWEST token, so it may see everything, i.e. the mask must be
        #    all True. diagonal = k_len - q_len lines the two sequences up at
        #    their right edges:
        #        q_len=5, k_len=5 -> shift 0 -> normal triangle
        #        q_len=1, k_len=5 -> shift 4 -> [[1, 1, 1, 1, 1]]
        q_len, k_len = scores.shape[-2], scores.shape[-1]
        allowed = torch.ones(q_len, k_len, dtype=torch.bool, device=x.device)
        allowed = torch.tril(allowed, diagonal=k_len - q_len)
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)

        # 6) Softmax turns scores into percentages that sum to 1 per query.
        weights = F.softmax(scores, dim=-1)
        weights = F.dropout(weights, p=self.dropout, training=self.training)

        # 7) Weighted average of the V's: each query collects a blend of the
        #    content it decided was relevant.
        out = weights @ v  # [B, H, q_len, D]

        # 8) Glue the heads back together and mix them with out_proj.
        out = out.transpose(1, 2).reshape(batch, q_len, -1)  # [B, q_len, H*D]
        return self.out_proj(out)

    # ---------------------------------------------- cache bookkeeping (demo)
    def kv_cache_bytes_per_token(self, bytes_per_number: int = 2) -> int:
        """Memory needed to remember one token's K and V in this layer.

        K and V each store num_kv_heads * head_dim numbers.
        (fp16 / bf16 = 2 bytes per number, which is what LLMs usually use.)
        """
        return 2 * self.num_kv_heads * self.head_dim * bytes_per_number


# ---------------------------------------------------------------------------
# The three famous variants, each one line away from the others
# ---------------------------------------------------------------------------
class MultiHeadAttention(GroupedQueryAttention):
    """MHA — the original Transformer (2017) and GPT-3 style.

    Every Query head owns a private Key/Value head.
    Best quality, biggest KV cache.
    """

    def __init__(self, d_model: int = 512, num_heads: int = 8, dropout: float = 0.0):
        super().__init__(
            AttentionConfig(d_model, num_heads, num_kv_heads=num_heads, dropout=dropout)
        )


class MultiQueryAttention(GroupedQueryAttention):
    """MQA — used by PaLM, Falcon, StarCoder.

    ALL Query heads share ONE Key/Value head.
    Tiny KV cache and very fast generation, but a small quality drop and it can
    be less stable to train, because 8 heads all compete for the same 1 cache.
    """

    def __init__(self, d_model: int = 512, num_heads: int = 8, dropout: float = 0.0):
        super().__init__(
            AttentionConfig(d_model, num_heads, num_kv_heads=1, dropout=dropout)
        )


class GroupedQueryAttentionLayer(GroupedQueryAttention):
    """GQA — the practical middle ground (Llama 2/3, Mistral, Qwen, Gemma).

    Real-world head counts:
        Llama 3 8B    : 32 Q heads,  8 KV heads  (group of 4)
        Llama 2 70B   : 64 Q heads,  8 KV heads  (group of 8)
        Mistral 7B    : 32 Q heads,  8 KV heads  (group of 4)

    Nearly MHA quality, nearly MQA memory savings. That is why everyone uses it.
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        num_kv_heads: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__(AttentionConfig(d_model, num_heads, num_kv_heads, dropout))


# ---------------------------------------------------------------------------
# Demo / sanity checks
# ---------------------------------------------------------------------------
def describe(name: str, layer: GroupedQueryAttention, d_model: int) -> None:
    params_q = layer.q_proj.weight.numel()
    params_kv = layer.k_proj.weight.numel() + layer.v_proj.weight.numel()
    per_token = layer.kv_cache_bytes_per_token()
    print(
        f"{name:<6} Q heads={layer.num_heads:>3}  KV heads={layer.num_kv_heads:>3}  "
        f"group={layer.group_size:>2}  "
        f"q/kv params={params_q:>7,}/{params_kv:>7,}  "
        f"KV cache={per_token:>5,} B/token/layer"
    )


# This function checks that attention gives the same results when it processes a sequence all at once
# as when it processes it one token at a time, reusing cached K/V.
def kv_cache_matches_full_forward(layer: GroupedQueryAttention, x: Tensor) -> float:
    """Generate token-by-token with the cache and compare to one full pass.

    If these two agree (to tiny float error) then the GQA math and the
    head-sharing are implemented correctly.
    """
    with torch.no_grad():
        expected = layer(x)  # everything at once, causal mask does the work
        cache: tuple[Tensor, Tensor] | None = None
        steps = []
        for t in range(x.size(1)):
            new_x = x[:, t : t + 1]  # get exactly one token
            steps.append(layer(new_x, kv_cache=cache))
            # Remember only the SMALL kv tensors (num_kv_heads, not num_heads).
            k = layer._split_heads(layer.k_proj(new_x), layer.num_kv_heads)
            v = layer._split_heads(layer.v_proj(new_x), layer.num_kv_heads)
            cache = (
                (k, v)
                if cache is None
                else (
                    torch.cat([cache[0], k], dim=2),
                    torch.cat([cache[1], v], dim=2),
                )
            )
    return (expected - torch.cat(steps, dim=1)).abs().max().item()


def main() -> None:
    torch.manual_seed(0)
    d_model, num_heads, seq_len, batch = 512, 8, 16, 2

    print("=" * 96)
    print(
        f"d_model={d_model}, num_heads={num_heads} (head_dim={d_model // num_heads}), "
        f"seq_len={seq_len}, batch={batch}"
    )
    print("=" * 96)

    mha = MultiHeadAttention(d_model, num_heads)
    gqa = GroupedQueryAttentionLayer(d_model, num_heads, num_kv_heads=2)
    mqa = MultiQueryAttention(d_model, num_heads)

    for name, layer in [("MHA", mha), ("GQA", gqa), ("MQA", mqa)]:
        describe(name, layer, d_model)

    print()
    print("Same input for all three, just to prove the shapes line up:")
    x = torch.randn(batch, seq_len, d_model)
    for name, layer in [("MHA", mha), ("GQA", gqa), ("MQA", mqa)]:
        print(f"  {name}: {tuple(x.shape)} -> {tuple(layer(x).shape)}")

    print()
    print("KV-cache sanity check (token-by-token vs one full pass, max abs diff):")
    for name, layer in [("MHA", mha), ("GQA", gqa), ("MQA", mqa)]:
        print(f"  {name}: {kv_cache_matches_full_forward(layer, x):.2e}")

    print()
    print("Head -> KV sharing map for GQA (num_heads=8, num_kv_heads=2):")
    mapping = [i // gqa.group_size for i in range(mha.num_heads)]
    print("  Q head    :", " ".join(f"Q{i}" for i in range(mha.num_heads)))
    print("  uses KV   :", " ".join(f"k{j}" for j in mapping))

    # One more equivalence check: GQA with num_kv_heads == num_heads behaves
    # like MHA (here we compare the *shape* of the intermediate K tensor).
    same = GroupedQueryAttentionLayer(d_model, num_heads, num_kv_heads=num_heads)
    print(
        f"\nGQA with num_kv_heads=num_heads == MHA ? "
        f"{same.k_proj.out_features == mha.k_proj.out_features}"
    )

    # KV cache for a realistic model: 32 layers, 8192-token context, batch 1.
    print("\nKV cache for a 32-layer model, 8192-token context, batch=1 (bf16):")
    for name, layer in [("MHA", mha), ("GQA", gqa), ("MQA", mqa)]:
        total = layer.kv_cache_bytes_per_token() * 8192 * 32
        print(f"  {name}: {total / 1e6:>7.1f} MB")
    print(
        "\n(That is the whole point: GQA gets MHA-ish quality with a fraction "
        "of the memory, so long conversations stay cheap.)"
    )


if __name__ == "__main__":
    main()
