# Neural network based sorter
<img width="3780" height="1929" alt="image" src="https://github.com/user-attachments/assets/e0f7e2fe-5d7e-4217-992d-db92af3907ef" />

# What is it?

A neural network that sorts 16 numbers by learning a **permutation matrix**, built in Rust with the [Burn](https://burn.dev) deep learning framework.

The key idea: instead of having the network output sorted values directly (which allows it to hallucinate numbers that weren't in the input), we constrain the architecture so that the output is *structurally* a reordering of the input. The network only learns *which input goes where* — the actual values pass through untouched.

## What are we trying to do?

We have 16 numbers coming in, and we want to output those same 16 numbers in ascending order. The constraint: the output must contain *exactly* the same values as the input — no approximations, no fabricated numbers.

So really, all we need to figure out is **which input goes to which output position**. That's a permutation. If we can learn the right permutation, we just apply it and we're done.

## What is a permutation, concretely?

Say the input is `[0.7, 0.2, 0.9, 0.1]` (4 elements for simplicity). The sorted output should be `[0.1, 0.2, 0.7, 0.9]`. The permutation that does this is:

```
output[0] = input[3]   (0.1)
output[1] = input[1]   (0.2)
output[2] = input[0]   (0.7)
output[3] = input[2]   (0.9)
```

We can represent this as a matrix. Each row is an output position, each column is an input position. Put a 1 where we want to "pick" from:

```
         input[0]  input[1]  input[2]  input[3]
out[0]:     0         0         0         1       ← picks input[3]
out[1]:     0         1         0         0       ← picks input[1]
out[2]:     1         0         0         0       ← picks input[0]
out[3]:     0         0         1         0       ← picks input[2]
```

This is a **permutation matrix**. Each row has exactly one 1 (each output picks one input), each column has exactly one 1 (each input is used exactly once). Multiplying this matrix by the input vector gives the permuted output — that's just linear algebra: `output = P · input`.

## The problem: we can't learn a permutation matrix directly

A permutation matrix is *discrete* — every entry is either 0 or 1. Neural networks learn by gradient descent, which needs smooth, continuous values. If the matrix entries are 0s and 1s, there's no gradient — "slightly pick input[3]" isn't a meaningful operation.

So we need a continuous relaxation — a "soft" version of a permutation matrix that we can gradually sharpen.

## Doubly stochastic matrices: the soft version

What if instead of hard 0/1 entries, we allow any non-negative values, with two constraints: every row sums to 1, and every column sums to 1?

```
         input[0]  input[1]  input[2]  input[3]
out[0]:   0.05      0.05      0.05      0.85      ← mostly picks input[3]
out[1]:   0.05      0.80      0.10      0.05      ← mostly picks input[1]
out[2]:   0.80      0.05      0.10      0.05      ← mostly picks input[0]
out[3]:   0.10      0.10      0.75      0.05      ← mostly picks input[2]
```

This is a **doubly stochastic matrix**. Every row sums to 1 (it's a probability distribution over inputs), every column sums to 1 (every input is "used up" exactly once, spread across outputs).

When you multiply this by the input vector, each output is a *weighted average* of the inputs, with most weight on one element. As the matrix gets sharper (0.85 → 0.99 → 1.0), it approaches a true permutation.

There's a beautiful result (Birkhoff's theorem) that says: the set of all doubly stochastic matrices is exactly the convex hull of all permutation matrices. Every doubly stochastic matrix is a blend of permutations, and the extreme points (corners) of this set are the hard permutation matrices. So as we "sharpen" a doubly stochastic matrix, we're moving toward a corner — toward a true permutation.

## How do we make a doubly stochastic matrix?

The network outputs 16×16 = 256 raw numbers (the "score matrix"). These can be anything — positive, negative, whatever. We need to turn them into a doubly stochastic matrix.

**Step 1: Make everything positive.** We divide by a temperature parameter τ and exponentiate: `exp(score / τ)`. Now every entry is positive. The temperature controls sharpness — small τ makes large scores dominate exponentially.

**Step 2: The Sinkhorn algorithm.** This is the key piece, and it's almost trivially simple:

```
repeat 20 times:
    divide each row by its row sum      → now rows sum to 1
    divide each column by its column sum → now columns sum to 1
```

That's it. Just alternate between "make rows sum to 1" and "make columns sum to 1." Each step breaks the other constraint slightly, but the alternation converges to a matrix where *both* hold simultaneously. After 20 iterations you have a doubly stochastic matrix.

Why does this work? Normalizing rows is like running softmax on each row — it turns it into a probability distribution. Normalizing columns does the same vertically. The alternation is a fixed-point iteration that converges to the unique doubly stochastic matrix in the "scaling family" of the input matrix. The proof is from Sinkhorn and Knopp (1967), but the algorithm itself is just two lines in a loop.

## The full pipeline

Putting it all together:

```
input: [0.7, 0.2, 0.9, 0.1]     ← 4 raw values
         │
         ▼
    MLP network (learnable weights)
         │
         ▼
raw scores: 4×4 matrix            ← 16 arbitrary numbers
         │
         ▼
    exp(scores / τ)                ← make positive, control sharpness
         │
         ▼
    Sinkhorn (20 iterations)       ← make doubly stochastic
         │
         ▼
P: 4×4 soft permutation matrix    ← rows & cols sum to 1
         │
         ▼
output = P · input                 ← weighted selection of inputs
```

The MLP's job is to look at all input values and produce a score matrix where `score[i][j]` is high when input `j` should go to output position `i`. The MLP learns this from data — it sees millions of random inputs paired with their sorted versions, and adjusts its weights so the scores produce the right permutation.

In the actual implementation, the input is 16 elements, so the score matrix is 16×16 = 256 values, produced by an MLP with 4 hidden layers of width 128.

## Training: the soft forward pass

During training, we compute `output = P_soft · input` where `P_soft` is the doubly stochastic matrix (not a hard permutation). This output is differentiable — gradients flow through the matrix multiplication, through the Sinkhorn iterations, through the exp, back into the MLP weights. We compute MSE between this soft output and the sorted target, and backpropagate.

The soft output is a blurry version of the right answer — each output position is a weighted average that's close to the correct value but not exact. That's fine for training; the MSE gradient tells the MLP "make `score[2][0]` bigger" (push input[0] more toward output position 2), which nudges the weights in the right direction.

We also use a **pairwise ordering loss** that operates directly on the soft permutation matrix. For each input element, we compute its "expected output position" — a weighted average of all row indices, weighted by the column of P corresponding to that input. Then for every pair of inputs where `input[j] < input[k]`, we penalize if `expected_pos[j] > expected_pos[k]` (wrong order) with a hinge loss. This gives the network direct gradient signal for pairwise ordering, which MSE alone can miss for close-valued elements.

## Evaluation: the hard forward pass

At eval time we want exact values, not blurry averages. So we take the same soft P matrix and extract a hard permutation from it: for each output row, which input column has the highest weight? That's our pick.

We do this greedily — process the most confident row first (the one with the highest peak value), assign its best input, mark that input as used, move to the next row. This guarantees every input is used exactly once.

The output is now `input[perm[i]]` — actual input values selected by index. The set is guaranteed to match the input perfectly, by construction. The only question is whether the ordering is right.

## The temperature parameter τ

This controls how "peaked" the soft permutation is. Think of it as a focus knob.

At **high τ** (say 0.5): `exp(score / 0.5)` compresses differences. A score of 2.0 vs 1.0 gives `exp(4)` vs `exp(2)` ≈ 55 vs 7 — the winner gets about 88% of the weight. Gradients flow to many elements, good for early learning, but the soft output is blurry.

At **low τ** (say 0.01): `exp(score / 0.01)` amplifies differences enormously. A score of 2.0 vs 1.0 gives `exp(200)` vs `exp(100)` — the winner gets essentially 100%. The matrix is nearly one-hot, but gradients only flow to the winning element, which makes learning harder.

We use **τ = 0.1** as a fixed training temperature — sharp enough that each output position has a clear "favorite" input, soft enough that gradients flow to 2-3 candidates for error correction. At eval time, the hard argmax extraction means the exact tau value doesn't matter much — we just need the scores to rank correctly.

## What the network actually learns

The MLP sees all 16 inputs as a flat vector and produces 256 scores. Internally, it's learning something like: "input[3] is small, input[7] is large, input[3] is smaller than input[11]..." — building up a ranking of all elements, then converting that ranking into score assignments.

The hard part isn't the Sinkhorn machinery — that's just fixed math. The hard part is the MLP learning to produce good scores for all possible inputs. With 16 elements, there are 120 pairwise comparisons to get right, and the MLP has to encode all of them through shared hidden layers.

## Why the output can't hallucinate values

This is the architectural guarantee. The final output is computed as `P · input` (soft, during training) or `input[perm[i]]` (hard, during eval). In both cases, the output is derived from the input values themselves — the network never directly produces output numbers. It only produces the *routing* (the permutation), and the routing is applied to the actual input.

Even with a completely untrained, random network at epoch 0, the output values come from the input. They might be in the wrong order, but they're never fabricated. This is what a naive MLP (Level 0) can't do — a raw MLP outputs 16 floats with no structural connection to the input values, so it can and does hallucinate.

## Project structure

```
sort_net_level1/
├── Cargo.toml              # burn 0.20 + wgpu backend
├── src/
│   └── main.rs             # model, training loop, metrics, checkpointing
└── checkpoints/            # auto-created during training
    ├── sort_sinkhorn.json  # latest checkpoint (for --resume)
    └── sort_sinkhorn_epoch_*.json
```

## Running

```bash
# Fresh training run (runs forever, Ctrl+C to stop)
cargo run --release

# Resume from checkpoint
cargo run --release -- --resume
```

The training loop logs every 10 epochs and prints detailed sample tests every 200 epochs. Checkpoints are saved every 200 epochs in JSON format (human-readable weights).

## Visualizer

The included `sinkhorn_visualizer.html` is a standalone HTML file that runs the full inference pipeline in JavaScript. Open it in a browser, paste the contents of a checkpoint JSON file, and interactively explore:

- Adjust the 16 input sliders or use presets (random, ascending, descending, wiggle, near-duplicates)
- See the MLP activations, raw score matrix, Sinkhorn permutation matrix, hard permutation arrows, and output vs ground truth
- All inference runs client-side with no server

## Architecture summary

| Component | Shape | Description |
|---|---|---|
| Input | `[B, 16]` | 16 float values to sort |
| MLP | `16 → 128 → 128 → 128 → 128 → 256` | Produces raw score matrix |
| Reshape | `[B, 256] → [B, 16, 16]` | Reinterpret as score matrix |
| Exp + Scale | `exp(scores / τ)` | Make positive, control sharpness |
| Sinkhorn | 20 iterations of row/col normalization | Doubly stochastic matrix |
| Hard Perm | Greedy argmax extraction | True permutation |
| Output | `input[perm[i]]` | Exactly the input values, reordered |

**Parameters**: ~101K (mostly in the MLP). The Sinkhorn operator and hard permutation extraction have zero learnable parameters.
