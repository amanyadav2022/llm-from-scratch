# Worked Example — One Sentence, Every Step

*Phases 1-11 explain concepts one at a time. This is the payoff: a single real example traced through the entire pipeline, with real numbers from real code — not illustrations.*

1. **[Part 1 — Training](01-training-walkthrough.md)** — `"a soldier dies in world war"` traced through tokenization, embeddings, attention, logits, loss, backpropagation, and a real weight update.
2. **[Part 2 — Inference](02-inference-walkthrough.md)** — the prompt `"a gun was fired"` generated token by token, using a real, working KV cache you can watch grow.

**Best read after finishing Phase 7** (KV cache) at minimum, and ideally after Phase 9 — this worked example assumes you already know what each piece is *for*; it exists to show you what actually happens numerically, not to introduce the concepts for the first time.

Both parts include the full, runnable script — copy it into Colab and run it yourself rather than just reading the numbers.
