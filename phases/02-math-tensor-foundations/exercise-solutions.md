# Phase 2 — Exercise Solutions

*Try each exercise yourself first (see the main [README](README.md), Step 5). These solutions are here so you can check your work, not to be read first.*

---

### 1. Split `x` of shape `(B=2, T=5, D=64)` into 8 heads of dimension 8

```python
import torch

x = torch.randn(2, 5, 64)   # (B=2, T=5, D=64)

# Step 1: split the last dimension (64) into (heads=8, head_dim=8)
x = x.view(2, 5, 8, 8)      # (B, T, H, d)

# Step 2: move H before T, since attention wants heads as their own leading dimension
x = x.permute(0, 2, 1, 3)   # (B, H, T, d)

print(x.shape)   # torch.Size([2, 8, 5, 8])
```

**Why `.permute` and not `.view` for the second step:** `view` can only reorganize memory that's already contiguous in the order you want. Swapping dimensions (not just regrouping them) requires `.permute` (or `.transpose`), which changes the *order* dimensions are read in, not just how they're grouped.

---

### 2. Implement softmax from scratch, verify it matches `torch.softmax`

```python
import torch

def my_softmax(x, dim=-1):
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)

x = torch.tensor([2.5, 1.2, 0.3])

mine = my_softmax(x, dim=0)
builtin = torch.softmax(x, dim=0)

print(mine)      # tensor([0.7054, 0.2119, 0.0827])
print(builtin)   # tensor([0.7054, 0.2119, 0.0827])
print(torch.allclose(mine, builtin))   # True
```

**A subtlety worth knowing (not required to pass the exercise, but good to understand):** real implementations first subtract the max value from `x` before exponentiating (`x - x.max()`), purely to avoid overflow when values get large. Mathematically it produces the identical result, since that shift cancels out in the division — but it's more numerically stable. Try adding it yourself and confirm `torch.allclose` still passes.

---

### 3. Output shape of `Q @ K^T` for `Q: (2, 8, 10, 64)`, `K: (2, 8, 10, 64)`

```python
import torch

Q = torch.randn(2, 8, 10, 64)   # (batch, heads, seq_len, head_dim)
K = torch.randn(2, 8, 10, 64)

scores = Q @ K.transpose(-2, -1)   # transpose the LAST TWO dims of K only
print(scores.shape)   # torch.Size([2, 8, 10, 10])
```

**Why `.transpose(-2, -1)` and not `.T`:** `.T` on a 4D tensor reverses *all* dimensions, not just the last two — that would wreck the batch and head dimensions. `.transpose(-2, -1)` swaps only the last two axes (`seq_len` and `head_dim`), leaving `batch` and `heads` untouched — exactly what you want for batched matrix multiplication (Phase 2, Step 2).

---

### 4. Token IDs `(B, T)` → embeddings `(B, T, D)` using `nn.Embedding`

```python
import torch
import torch.nn as nn

def token_ids_to_embeddings(token_ids, vocab_size, embed_dim):
    embedding_table = nn.Embedding(vocab_size, embed_dim)
    return embedding_table(token_ids)

token_ids = torch.tensor([[5, 12, 890], [3, 3, 40]])   # (B=2, T=3)
vectors = token_ids_to_embeddings(token_ids, vocab_size=1000, embed_dim=16)

print(vectors.shape)   # torch.Size([2, 3, 16])
```

**Why this works with no extra reshaping:** `nn.Embedding` is built to accept any shape of integer tensor and simply adds one new dimension (`embed_dim`) at the end — `(B, T)` in, `(B, T, D)` out, automatically. This is the exact mechanism from Phase 4, Step 2.

---

## Checkpoint self-check

If your own attempts matched (or were logically equivalent to) these, Phase 2 is genuinely solid. If any of them surprised you, re-read [Phase 2's README](README.md) at the relevant step before moving on to Phase 3.
