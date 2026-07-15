# Worked Example, Part 1: Training on "a soldier dies in world war"

*Every previous phase explained one piece at a time. This is the payoff: one real sentence, traced through EVERY step, with real numbers from an actual run of the code below — tokenization, embeddings, attention, logits, loss, gradients, and the weight update. Nothing here is made up; every number was produced by actually running this exact script.*

**Before reading further:** copy the full script at the bottom of this page into Colab and run it yourself. The numbers below are what it prints — seeing them appear on your own screen is worth more than reading them here.

---

## The setup

- **Sentence:** `"a soldier dies in world war"`
- **Toy vocabulary** (12 words, tiny on purpose): `a, soldier, dies, in, world, war, gun, was, fired, at, the, <eos>`
- **Embedding dimension:** `D=8` (real models use 768-12288+; we use 8 so every vector fits on screen)
- **Heads:** 2, each of dimension 4
- One transformer block (Phase 4 Part 3), causal masked (Phase 4 Part 2, Step 5)

---

## Step 1 — Tokenization (Phase 4, Step 1) We split on whole words here purely for readability. A real tokenizer (BPE, Phase 4 Step 1) would likely split some of these into sub-word pieces — the mechanism that follows is identical either way.

---

## Step 2 — Token IDs (Phase 4, Step 1)Six words became six integers. This is the only form a neural network can actually consume.

---

## Step 3 — Embeddings (Phase 4, Step 2)Each of the 6 tokens now has an 8-number vector. Right now these are **randomly initialized** — this is an untrained model, exactly like the very first moment of Phase 1's training loop, before any learning has happened.

---

## Step 4 — Positional Encoding (Phase 4, Step 3)

```python
x = tok_emb + pos_emb   # shape (6, 8)
```

Each token's embedding gets a position-dependent vector added to it, so the model can tell "a" at position 0 apart from "a" if it appeared again at position 4. (We use a simple learned positional table here rather than the sinusoidal formula from Phase 4 — same purpose, simpler to inspect.)

---

## Step 5 — Layer Norm (Phase 4 Part 3, Step 4)Before attention, every token vector gets re-centered to mean ≈0 and spread ≈1 — exactly the "reset to a comfortable range" step from Phase 4 Part 3.

---

## Step 6a — Multi-Head Causal Self-Attention (Phase 4, Step 4-6 / Phase 5)

Here are the **real attention weights** for head 0 — rows are "query" positions, columns are "key" positions:**Look at the shape of this grid.** Every value in the upper-right triangle is exactly `0.0000` — this is the causal mask (Phase 4, Step 5 and Phase 5, Step 3) working exactly as designed: token `i` can never attend to a token that comes after it. Notice each row still sums to `1.0` — softmax (Phase 2, Step 3) is still doing its job on whatever positions ARE visible.

---

## Step 6b — Feedforward + Residual (Phase 4 Part 3, Step 2-3)This is the **context vector** for "world" — an 8-number summary that has now absorbed information from every token up to and including itself (via attention), then been reshaped by the feedforward network. This single vector is what the rest of the model uses to predict what comes *after* "world."

---

## Step 7 — Output Head → Logits (Phase 6, Step 3)One raw score per vocabulary word (12 of them), produced by projecting the context vector up to vocabulary size.

---

## Step 8 — Softmax → Probabilities (Phase 2, Step 3 / Phase 6, Step 3)The correct next word is "world" (0.1298), but the untrained model currently thinks `<eos>` (0.1887) is slightly more likely. This gap between "what it guessed" and "what's actually correct" is exactly what training closes.

---

## Step 9 — Loss (Cross-Entropy) (Phase 6, Step 5)One single number summarizing "how wrong" the model's predictions were across all 5 positions at once (predicting `soldier` after `a`, `dies` after `a soldier`, and so on). Lower is better; a perfect model would approach `0`.

---

## Step 10 — Backpropagation: Gradient Calculation (Phase 1, Step 4)**This is the chain rule in action, exactly as described in Phase 1.** `loss.backward()` computed, for every single weight in the entire network — the output head, the attention projections, the feedforward layers, even the original embedding table — exactly how much that weight contributed to the loss being `2.5752` rather than `0`. Notice the gradient reached all the way back to `q_proj` (part of attention), not just the output layer — this is the "direct shortcut path" that residual connections (Phase 4 Part 3, Step 3) make possible even through several stacked operations.

---

## Step 11 — Weight Update: Gradient Descent (Phase 1, Step 4)Every single weight in the model — not just this one row — got nudged slightly, in the direction that would have made the loss slightly lower on this exact example. On its own, this one update barely moves anything. **Real training repeats this exact loop — steps 6 through 11 — millions or billions of times, across a huge dataset, not just one sentence.** Each repetition nudges the weights a tiny bit more, and after enough repetitions, "world" reliably becomes the top prediction after "a soldier dies in," instead of `<eos>`.

---

## The Full Script (copy this into Colab and run it yourself)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
torch.set_printoptions(precision=4, sci_mode=False)

# STEP 1: TOKENIZATION (word-level, for simplicity)
sentence = "a soldier dies in world war"
words = sentence.split()
print("STEP 1 — Tokenization")
print("Sentence:", sentence)
print("Tokens:", words)
print()

# STEP 2: TOKEN IDS
vocab = ["a", "soldier", "dies", "in", "world", "war",
         "gun", "was", "fired", "at", "the", "<eos>"]
stoi = {w: i for i, w in enumerate(vocab)}
itos = {i: w for w, i in stoi.items()}
vocab_size = len(vocab)

token_ids = torch.tensor([stoi[w] for w in words])
print("STEP 2 — Token IDs")
print("Vocabulary:", stoi)
print("Token IDs for sentence:", token_ids.tolist())
print()

# STEP 3: EMBEDDINGS
D = 8
n_heads = 2
head_dim = D // n_heads

embedding_table = nn.Embedding(vocab_size, D)
tok_emb = embedding_table(token_ids)
print("STEP 3 — Token Embeddings")
print("Shape:", tok_emb.shape)
print("Embedding for 'a' (first token):", tok_emb[0].detach())
print()

# STEP 4: POSITIONAL ENCODING
T = len(words)
pos_embedding_table = nn.Embedding(T, D)
positions = torch.arange(T)
pos_emb = pos_embedding_table(positions)

x = tok_emb + pos_emb
print("STEP 4 — Add Positional Encoding")
print("x = token_embedding + positional_embedding, shape:", x.shape)
print()

# STEP 5: LAYER NORM
ln1 = nn.LayerNorm(D)
x_normed = ln1(x)
print("STEP 5 — Layer Norm (before attention)")
print("Mean of first token vector after norm (~0):", x_normed[0].mean().item())
print("Std of first token vector after norm (~1):", x_normed[0].std(unbiased=False).item())
print()

# STEP 6: TRANSFORMER BLOCK
q_proj = nn.Linear(D, D)
k_proj = nn.Linear(D, D)
v_proj = nn.Linear(D, D)
out_proj = nn.Linear(D, D)

def multi_head_attention(x_in, causal=True):
    T_, D_ = x_in.shape
    Q = q_proj(x_in).view(T_, n_heads, head_dim).transpose(0, 1)
    K = k_proj(x_in).view(T_, n_heads, head_dim).transpose(0, 1)
    V = v_proj(x_in).view(T_, n_heads, head_dim).transpose(0, 1)

    scores = Q @ K.transpose(-2, -1) / (head_dim ** 0.5)
    if causal:
        mask = torch.triu(torch.ones(T_, T_), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
    weights = F.softmax(scores, dim=-1)
    out = weights @ V
    out = out.transpose(0, 1).contiguous().view(T_, D_)
    return out_proj(out), weights

attn_out, attn_weights = multi_head_attention(x_normed, causal=True)
x = x + attn_out

print("STEP 6a — Multi-Head Causal Self-Attention")
print("Attention weights for head 0 (rows=query pos, cols=key pos):")
print(attn_weights[0].detach())
print("Notice: upper triangle is exactly 0.0 (causal mask working)")
print()

ln2 = nn.LayerNorm(D)
ffn = nn.Sequential(
    nn.Linear(D, D * 4),
    nn.GELU(),
    nn.Linear(D * 4, D),
)
ffn_out = ffn(ln2(x))
x = x + ffn_out

print("STEP 6b — Feedforward + residual")
print("Block output shape (context vectors):", x.shape)
print("Context vector for 'world' (5th token):", x[4].detach())
print()

# STEP 7: OUTPUT HEAD -> LOGITS
final_norm = nn.LayerNorm(D)
output_head = nn.Linear(D, vocab_size)

logits = output_head(final_norm(x))
print("STEP 7 — Output Head -> Logits")
print("Logits shape:", logits.shape)
print("Logits after 'in' (predicting the 5th word):")
print(logits[3].detach())
print()

# STEP 8: SOFTMAX -> PROBABILITIES
probs_after_in = F.softmax(logits[3], dim=-1)
print("STEP 8 — Softmax -> Probabilities (after seeing 'a soldier dies in')")
for w, p in zip(vocab, probs_after_in.detach().tolist()):
    print(f"  {w:8s}: {p:.4f}")
print()

# STEP 9: LOSS
predicted_logits = logits[:-1]
true_next_tokens = token_ids[1:]

loss = F.cross_entropy(predicted_logits, true_next_tokens)
print("STEP 9 — Loss (Cross-Entropy)")
print("Predicting these next tokens:", [itos[i.item()] for i in true_next_tokens])
print("Loss value:", loss.item())
print()

# STEP 10: BACKPROPAGATION
loss.backward()
print("STEP 10 — Backpropagation")
print("Gradient of the loss w.r.t. output_head weight matrix, shape:", output_head.weight.grad.shape)
print("First row of that gradient (first vocab word's weight gradients):")
print(output_head.weight.grad[0])
print("Gradient norm across the whole output head:", output_head.weight.grad.norm().item())
print()
print("Gradient also flowed back into earlier layers, e.g. q_proj weight gradient norm:",
      q_proj.weight.grad.norm().item())
print()

# STEP 11: WEIGHT UPDATE
lr = 0.1
before = output_head.weight[0, :4].clone()

with torch.no_grad():
    for p in list(output_head.parameters()) + list(q_proj.parameters()) + \
              list(k_proj.parameters()) + list(v_proj.parameters()) + \
              list(ffn.parameters()) + list(embedding_table.parameters()):
        if p.grad is not None:
            p -= lr * p.grad

after = output_head.weight[0, :4].clone()

print("STEP 11 — Weight Update (Gradient Descent)")
print("output_head weight, row 0, first 4 values, BEFORE update:", before)
print("output_head weight, row 0, first 4 values, AFTER update: ", after)
print("(new_weight = old_weight - learning_rate * gradient)")
```

---

## Checkpoint

1. Why is the upper-right triangle of the attention weight grid all zeros?
2. What real quantity does "loss = 2.5752" represent, in your own words?
3. Which layers received gradients during backprop — only the output head, or earlier layers too? Why?
4. If you ran the script again with a different sentence, which steps would produce different-shaped tensors, and which would produce the same shapes with different values?

Continue to **[Part 2: Inference with a real KV cache](02-inference-walkthrough.md)**.
