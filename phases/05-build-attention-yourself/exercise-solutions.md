# Phase 5 — Exercise Solutions

*Try building each piece yourself first (see the main [README](README.md)). These are the full, tested implementations to check your work against — not necessarily the only correct way to write them.*

---

## 1. Single-head attention

```python
import torch
import torch.nn.functional as F

def single_head_attention(Q, K, V):
    # Q, K, V: shape (seq_len, dim)
    d_k = Q.shape[-1]
    scores = Q @ K.T / (d_k ** 0.5)      # (seq_len, seq_len)
    weights = F.softmax(scores, dim=-1)  # (seq_len, seq_len), each row sums to 1
    output = weights @ V                  # (seq_len, dim)
    return output

# Test
Q = torch.randn(4, 8)
K = torch.randn(4, 8)
V = torch.randn(4, 8)
out = single_head_attention(Q, K, V)
print(out.shape)   # torch.Size([4, 8])

# Sanity check: every row of the attention weights should sum to ~1.0
d_k = Q.shape[-1]
weights = F.softmax(Q @ K.T / (d_k ** 0.5), dim=-1)
print(weights.sum(dim=-1))   # tensor([1.0000, 1.0000, 1.0000, 1.0000])
```

**Common mistake:** forgetting the `/ (d_k ** 0.5)` scaling. The code will still *run* without it — shapes are unaffected — which is exactly why this bug is dangerous. It only shows up as unstable training later, not as an error now. Always double check it's there.

---

## 2. Multi-head attention (full class)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, dim=64, n_heads=8):
        super().__init__()
        assert dim % n_heads == 0, "dim must divide evenly into heads"
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, T, D = x.shape

        Q = self.q_proj(x)   # (B, T, D)
        K = self.k_proj(x)   # (B, T, D)
        V = self.v_proj(x)   # (B, T, D)

        # (B, T, D) -> (B, n_heads, T, head_dim)
        Q = Q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / (self.head_dim ** 0.5)   # (B, n_heads, T, T)
        weights = F.softmax(scores, dim=-1)
        out = weights @ V                                             # (B, n_heads, T, head_dim)

        # merge heads: (B, n_heads, T, head_dim) -> (B, T, D)
        out = out.transpose(1, 2).contiguous().view(B, T, D)

        return self.out_proj(out)

# Test
mha = MultiHeadAttention(dim=64, n_heads=8)
x = torch.randn(2, 10, 64)
out = mha(x)
print(out.shape)   # torch.Size([2, 10, 64])
```

**Common mistake:** calling `.view()` directly after `.transpose()` without `.contiguous()` first. `.transpose()` only changes how PyTorch *reads* the underlying memory, without physically rearranging it — `.view()` then gets confused because it expects memory laid out in a specific order. `.contiguous()` forces an actual physical rearrangement first, so `.view()` has something valid to work with. If you skip it, PyTorch raises a clear `RuntimeError` naming exactly this — read that error message if you see it, it's telling you precisely this.

**Shape trace, step by step (for the checkpoint question):**---

## 3. Adding causal masking

```python
def forward(self, x, causal=False):
    B, T, D = x.shape
    Q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
    K = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
    V = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

    scores = Q @ K.transpose(-2, -1) / (self.head_dim ** 0.5)   # (B, n_heads, T, T)

    if causal:
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))

    weights = F.softmax(scores, dim=-1)
    out = weights @ V
    out = out.transpose(1, 2).contiguous().view(B, T, D)
    return self.out_proj(out)
```

**Verifying it worked (the checkpoint's most important check):**

```python
mha = MultiHeadAttention(dim=64, n_heads=8)
x = torch.randn(1, 5, 64)

# Monkeypatch forward for this test, or copy the method above into your class first
out = mha.forward(x, causal=True)

# Manually inspect the weights for head 0, batch 0
B, T, D = x.shape
Q = mha.q_proj(x).view(B, T, mha.n_heads, mha.head_dim).transpose(1, 2)
K = mha.k_proj(x).view(B, T, mha.n_heads, mha.head_dim).transpose(1, 2)
scores = Q @ K.transpose(-2, -1) / (mha.head_dim ** 0.5)
mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
scores = scores.masked_fill(mask, float('-inf'))
weights = F.softmax(scores, dim=-1)

print(weights[0, 0])
# Expect a LOWER TRIANGULAR pattern -- every value above the diagonal should be exactly 0.0
# tensor([[1.00, 0.00, 0.00, 0.00, 0.00],
#         [0.43, 0.57, 0.00, 0.00, 0.00],
#         [0.21, 0.35, 0.44, 0.00, 0.00],
#         [0.18, 0.20, 0.31, 0.31, 0.00],
#         [0.15, 0.18, 0.22, 0.20, 0.25]])
```

If you see any non-zero value above the diagonal, the masking isn't applied correctly — double check `diagonal=1` (not `diagonal=0`, which would also block a token from attending to itself).

---

## Checkpoint answers (Phase 5 README questions)

1. **Attention formula:** `Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V`
2. **What `.view(...).transpose(1,2)` does:** first regroups the last dimension into `(heads, head_dim)` without moving any data, then physically reorders the axes so heads become the second dimension instead of the last — needed so each head's attention can be computed independently and in parallel.
3. **Why negative infinity, not zero:** softmax is applied *after* masking. `softmax` of `0` is still a small positive number, meaning the model would still weakly attend to "the future." `softmax` of `-infinity` becomes exactly `0` after the exponential step (Phase 2, Step 3) — the only way to guarantee zero attention weight.
4. **Without the `/sqrt(head_dim)` scaling:** raw dot products grow large as `head_dim` increases, pushing softmax's inputs to extremes — producing near one-hot attention distributions (almost all weight on a single token) very early in training, which makes gradients unstable and training difficult (echoing the vanishing/exploding gradient problem from Phase 3).
