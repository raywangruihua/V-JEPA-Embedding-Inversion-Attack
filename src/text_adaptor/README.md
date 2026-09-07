# text_adaptor

Project V-JEPA embeddings into a frozen LLM's input space and read the clip back
out **as language**, instead of as pixels.

The premise is the measured negative from the earlier directories. The supervised
adaptor reached R² ≈ 0.30 overall but only **0.03 in the detail band**, and
`--match-variance` at 0.6 and 1.0 did not produce shapes — which proves the
high-frequency content is *absent from the prediction* rather than under-scaled.
The diffusion decoder sharpened objects and still recovered no action. V-JEPA
discards appearance by design and encodes semantics extremely well; decoding to
pixels fights the model, decoding to language uses it.

So the claim this directory is built to test is:

> **V-JEPA embeddings leak semantics, not appearance.**

The negative latent-decoding results supply the appearance half. What is measured
here is the semantics half, and — more usefully — *which* semantics, as a
per-attribute profile rather than one score.

## The command

Tuned defaults for a 12 GB card, so there are no performance flags to remember.
Substitute the two paths and go:

```bash
# 1. sanity, ~10s, downloads nothing
uv run text_adaptor/run.py selftest

# 2. the floor. no training, no LLM, no GPU
uv run text_adaptor/run.py baseline \
    --vjepa-dir  /path/to/vjepa \
    --captions   /path/to/something-something-v2-train.json \
    --jepa-grid  8,14,14 \
    --out        runs/nn_baseline

# 3. the run
uv run text_adaptor/run.py train \
    --vjepa-dir  /path/to/vjepa \
    --captions   /path/to/something-something-v2-train.json \
    --jepa-grid  8,14,14 \
    --proj       perceiver \
    --split      runs/nn_baseline/split.json \
    --out        runs/cap_perceiver

# 4. the full control table
uv run text_adaptor/run.py evaluate \
    --ckpt       runs/cap_perceiver/best.pt \
    --vjepa-dir  /path/to/vjepa \
    --captions   /path/to/something-something-v2-train.json
```

`--split runs/nn_baseline/split.json` in step 3 is worth keeping: it makes the
learned projector and the nearest-neighbour floor the same held-out clips, so the
comparison is exact rather than approximate.

Then the other two rungs of the ladder, reusing that split — and **`--pool-grid
4,4,4`**, which is not the default and is not optional. See below.

```bash
uv run text_adaptor/run.py train ... --proj pool --pool-grid 4,4,4 --split runs/cap_perceiver/split.json --out runs/cap_pool
uv run text_adaptor/run.py train ... --proj mean --split runs/cap_perceiver/split.json --out runs/cap_mean
```

**What the defaults already do for you**, measured on an RTX 3060 with 5000 clips:

- **gradient checkpointing off** — it costs a third of the FLOPs and a 1.5B model
  at this sequence length fits 12 GB without it. If a step OOMs anyway, the run
  turns it on by itself, says so, and continues. `--checkpointing` to start that
  way.
- **`--workers 4`** on Linux (0 on Windows, which cannot fork).
- **`--batch-size 4`, `--eval-batch-size 2`, `--retrieval-chunk 16`** — sized so
  the 152k-row vocab projection does not blow the card.
- **`--eval-clips 32`** — retrieval is O(N^2), so this is the expensive knob.
  The number for the writeup comes from `run.py evaluate` at the end, which
  takes its own larger `--retrieval-clips`.
- **greedy decoding with `repetition_penalty=1.0`**, overriding whatever the
  checkpoint ships.

Reach for `run.py pack` only if the timing line says you are waiting on disk. On
a local SSD you will not be.

## Running it

The directory is self-contained. Copy it to the Linux box and run:

```bash
uv run text_adaptor/run.py selftest
```

`run.py` carries a PEP 723 inline dependency block, so `uv` builds an isolated
environment on the first call and caches it after. No venv, no `requirements.txt`,
no install step. This is the same `uv run --no-project --with torch ...`
arrangement already used for verification on the Windows box, moved into the file
so the invocation stops having to carry it.

Nothing here imports from `standard_adaptors/` or `diffusion_adaptor/`. The
loading and standardization code is vendored, for the same reason it was vendored
into `diffusion_adaptor/` — a `sys.path` insert into a sibling directory broke
once already when `src/` was renamed, and it would have broken again when
`adaptor/` became `standard_adaptors/`.

### The order to run things in

**1. Self-test.** Seconds, CPU, downloads nothing.

```bash
uv run text_adaptor/run.py selftest
```

It checks the things that fail *quietly*: projector shapes, that the perceiver
actually distinguishes frame order, that the loss mask lands on caption positions
and nowhere else, and that all three caption file formats parse. Add
`--llm hf-internal-testing/tiny-random-LlamaForCausalLM` (a few MB) to also verify
that gradients reach the projector through a frozen, gradient-checkpointed stack —
that one is worth running on the Linux box before booking a long run, because when
it fails the loss curve still falls and nothing is learning.

**2. Nearest-neighbour baseline.** One pass, no training, no LLM, no GPU.

```bash
uv run text_adaptor/run.py baseline \
    --vjepa-dir /path/to/vjepa --captions ssv2-labels.json --jepa-grid 8,14,14
```

For each held-out embedding, find its nearest captioned neighbour by cosine and
copy the caption. **Get this number before training anything.** It answers the
question the expensive route assumes the answer to — is caption-relevant
information in there at all — and it is the floor a learned projector has to
clear. A projector that does not beat it has learned to write SSv2 sentences, not
to read embeddings.

**3. Train the ladder.** Three runs, sharing one split.

```bash
uv run text_adaptor/run.py train \
    --vjepa-dir /path/to/vjepa --captions ssv2-labels.json \
    --jepa-grid 8,14,14 --proj perceiver --out runs/cap_perceiver

uv run text_adaptor/run.py train ... --proj pool --pool-grid 4,4,4 --split runs/cap_perceiver/split.json --out runs/cap_pool
uv run text_adaptor/run.py train ... --proj mean --split runs/cap_perceiver/split.json --out runs/cap_mean
```

**4. Score with the full control table.**

```bash
uv run text_adaptor/run.py evaluate \
    --ckpt runs/cap_perceiver/best.pt \
    --vjepa-dir /path/to/vjepa --captions ssv2-labels.json
```

Writes `eval.json` and `captions.jsonl` beside the checkpoint.

## The ladder

Same logic as `standard_adaptors/adaptor.py`'s `linear → mlp → conv`: run all
three or a win is uninterpretable.

| `--proj` | what it does | what it isolates |
|---|---|---|
| `mean` | pools every patch into one vector, expands to K tokens | the floor: what a **gist vector** is worth |
| `pool` | `adaptive_avg_pool3d` onto a small grid, positionwise MLP | adds spatial and temporal **structure** |
| `perceiver` | K learned queries cross-attend into all 1568 patches | adds learned **selection** |

`mean → pool` isolates the value of structure; `pool → perceiver` isolates the
value of choosing what to keep. If `mean` matches `perceiver`, the honest headline
is that V-JEPA leaks a gist vector and nothing finer.

### Match the token budget, or the ladder measures nothing

The rungs have to differ in *one* thing. At stock defaults they do not: `mean` and
`perceiver` emit `--n-tokens 64` soft tokens, while `pool`'s count comes from
`--pool-grid`, whose default `2,4,4` is **32**. Run it that way and
`pool → perceiver` changes the architecture *and* doubles the budget — exactly the
two-things-at-once trap the ladder exists to avoid.

So pass **`--pool-grid 4,4,4`** (= 64) and leave `--n-tokens` alone. Of the
factorizations of 64, that one keeps the spatial aspect square and gives **four
temporal bins instead of two**; on a dataset where motion is the entire signal,
the default's two halves could sink `pool` for a reason that has nothing to do
with structure. `2,4,8` also reaches 64 but breaks the aspect.

If you would rather match downward, `--proj perceiver --n-tokens 32` against the
stock `2,4,4` is the other consistent choice, and is cheaper — but only if you
have not already trained the perceiver at 64.

The three rungs are **not** parameter-matched, and that is fine. `mean`'s output
layer alone is `--proj-hidden × K × d_llm` ≈ 200M, against ~53M for the perceiver
at width 1024 / depth 4 and ~10M for `pool`'s positionwise MLP. The asymmetry runs
in the useful direction: the floor is the *largest* model, so `mean` matching
`perceiver` cannot be waved away as the floor being starved. `--proj-hidden 4096`
will armor `pool` against the same objection if a reviewer asks, but a positionwise
projection is the least likely place for capacity to bind.

## Reading the output

`val` is the next-token loss and is nearly uninformative alone — a model that
ignored its input entirely would still drive it down by learning SSv2's idiom.
Don't tune against it.

**`top1`** is caption-to-clip retrieval: score `log p(caption_i | embedding_j)`
over candidate clips and check whether the true one wins. This is the metric
fluency cannot game — a caption produced from the prior scores identically against
every clip, so it lands at chance by construction. When it disagrees with the
generative numbers, it is right.

**`action`** vs **`object`** is the per-attribute leakage profile, and it is the
result the project has been working toward. Action accuracy snaps the generated
caption onto the nearest of SSv2's 174 templates (chance 0.6%); object recall
checks whether the clip's placeholder words ("book", "mug") survived. The
prediction from everything measured so far is action yes, object no.

**`shufAct`** is the control, and it is the column to read first.

## The controls, and why they are not optional

A frozen LLM is a fluency machine. Point one at noise and it writes a confident,
well-formed, plausibly-SSv2 caption, because that is what the projector's
unconditional prior plus a competent decoder produce. So the question is never
"are the captions good" but **"are they better than the same model given
nothing"**.

| control | conditioning | catches |
|---|---|---|
| `none` | the real embedding | — |
| `random` | N(0,1) in standardized space | a projector emitting a fixed prior |
| `shuffle` | another clip's real embedding | the same, but strictly in-distribution |
| `zero` | zeros | the obvious null |

`shuffle` is the strict one: it cannot be beaten by a model that has merely
learned what V-JEPA embeddings look like in general. **`none` minus `shuffle` is
the leakage.** If that gap is small, the captions are the prior talking and
nothing was recovered, regardless of how well they read.

BLEU and CIDEr are deliberately absent — they reward exactly the fluency that is
the failure mode here.

### Decoding is set explicitly, not inherited

Checkpoints ship a `generation_config.json` and it is not neutral. Qwen2.5-Instruct
sets `repetition_penalty=1.05`, `do_sample=True`, `temperature=0.7`, `top_p=0.8`,
`top_k=20` — tuned for chat, not for measurement. Inheriting that would mean the
decoding strategy silently differs for every model swapped in, while the code
claims to be doing greedy decoding.

So `CaptionModel.generation_config` builds the config explicitly, defaults
`--repetition-penalty` to **1.0 (off)**, and records the whole thing in
`config.json` and `eval.json`. A repetition penalty makes argmax decoding
not-argmax, and degenerate repetitive output is one of the signals separating a
caption read off an embedding from one confabulated by the prior — suppressing it
would blur the comparison the controls exist to make.

Passing a `GenerationConfig` does **not** replace the model's.
`_prepare_generation_config` deep-copies yours and then fills every attribute
still set to `None` from `model.generation_config`, and in transformers v5 a bare
`GenerationConfig()` leaves `temperature`, `top_p` and `top_k` at `None`. Leaving
them unset therefore hands control straight back to the checkpoint. They are
pinned to `1.0 / 1.0 / 50` under greedy decoding — the values `validate()` treats
as inactive, and inert anyway because the top-k and top-p warpers are only built
when `do_sample=True`. **Do not "tidy" these to `0` or `None`:** `0` trips the
validator, `None` re-opens the merge. Verified against a model carrying Qwen's
exact shipped values.

## Speed

At 5000 clips on a 12 GB card, both of the things that can dominate an epoch cost
about the same, so **read the timing line before tuning anything**:

```
      time: train 412s | val 9s | generate 31s | retrieval 15s   (4500 clips, 4 workers)
```

Also in `history.json` as `train_s` / `val_s` / `generate_s` / `retrieval_s`.

**The I/O side.** One clip is `1024 × 8 × 14 × 14` fp32 = 6.42 MB, so 5000 clips
is **32 GB read per epoch**, spread over 5000 separate `torch.load` calls.
`run.py pack` folds them into one fp16 memmap: 16 GB, one file handle, contiguous
rows. On a box with ≥32 GB RAM that fits the OS page cache, so epoch 2 onward
reads from memory. fp16 round-trip error is ~4e-4 relative, and retrieval numbers
are unchanged.

**The compute side.** Qwen2.5-1.5B at 122 tokens is ~3.07 GFLOP/token (15% of
that is the 152k-row `lm_head` alone), so a batch-4 step is ~4.5 TFLOP with
checkpointing. On an RTX 3060 that is 0.4–0.9 s/step depending on achieved
throughput, or **460–1000 s/epoch at 4500 clips**. If `nvidia-smi` shows the GPU
busy, an epoch in that range is the hardware, not a bug.

Levers, largest first:

| lever | effect | cost |
|---|---|---|
| checkpointing off | **−33% FLOPs** — already the default | more activation memory |
| `--n-tokens 32` | −26% sequence length (64 soft tokens → 32) | smaller projector budget |
| smaller LLM (Qwen2.5-0.5B) | ~3× less compute | weaker decoder |
| `--max-caption-len 32` | shorter sequences | truncates long captions |
| `--eval-every 4`, `--eval-clips 32` | eval cost only | coarser curves |

`--workers` defaults to 4 on Linux, 0 on Windows. **0 leaves the GPU waiting on
disk** — that default exists only because Windows cannot fork.

Note that `--workers > 0` requires the dataset to survive pickling: Windows uses
spawn, and from **Python 3.14 the POSIX default changed to forkserver**, so code
that relied on inheriting closures through a fork breaks on an interpreter bump
alone. Both loaders are therefore classes, not closures, and `selftest` asserts
they pickle.

## Memory

Frozen does not mean cheap. The error signal still travels back through every
layer of the LLM to reach the projector, so this costs close to full training
memory even though only ~20–50M parameters move.

Gradient checkpointing is nonetheless **off** by default — it buys that memory
back for a third of the throughput, and measurement on a 3060 showed a 1.5B model
using only 4.2 GB of 12 without it. The run enables it automatically if a step
OOMs, so the fast path is the default and the safe path is the fallback.

### What actually dominates: the vocab projection

Counter-intuitively it is **not** the transformer activations. The sequence here
is short — 1 BOS + 64 soft + ~9 prompt + ~48 caption ≈ **122 tokens** — but
Qwen2.5's vocabulary is **151,936**, so the logits tensor is
`batch × 122 × 151936`:

| batch | bf16 logits | with an fp32 copy and a log_softmax |
|---|---|---|
| 2 | 0.07 GB | 0.37 GB |
| 4 | 0.15 GB | 0.74 GB |
| 8 | 0.30 GB | **1.48 GB** |

and the backward pass needs the same again. This is why `--batch-size` is the
first knob to turn, and why the defaults are deliberately low (train 4, eval 2).

`sequence_logprob` computes `logit[y] − logsumexp(logits)` by gathering
immediately and running the logsumexp in fp32 slices along the sequence, so only
one full-size tensor is ever live. The obvious spelling — `.float()`, then
`log_softmax`, then `gather` — allocates two more, and reshaping for
`cross_entropy` allocates another because the `[:, :-1]` slice is
non-contiguous. Verified numerically identical to the naive version.

Validation embeddings are held on the **CPU** and streamed to the GPU in chunks.
At the real shapes one clip is `1024 × 8 × 14 × 14 × 4 B` = 6.4 MB, so
`--eval-clips 128` resident would be 0.8 GB sitting idle beside the LLM.

### If you OOM anyway, in this order

1. `--checkpointing` — though the run does this for you on the first OOM.
2. `--batch-size 2 --accum 4` — same effective batch, quarter the logits peak.
3. `--eval-batch-size 1 --retrieval-chunk 4`.
4. `--max-caption-len 32` and `--n-tokens 32` — both shorten the sequence, and
   the second halves the soft-token budget, which is the part of the sequence you
   control.
5. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — worth trying early if the
   failure is a *small* allocation with a *large* total, which means
   fragmentation rather than genuine exhaustion.
6. `--load-4bit` (needs `bitsandbytes`) — drops the weights from ~3 GB to ~1 GB,
   but does nothing about the logits, which is the actual bottleneck here.
7. A model with a smaller vocabulary. Llama-3.2-1B's 128k vocab is ~16% cheaper
   per token than Qwen's 152k for exactly this reason.

- **Qwen2.5-1.5B in bf16**: ~3 GB weights plus activations. Fits 12 GB comfortably.
  This is the tested default.
- **gemma-3-4b-it**: ~8 GB. Loads through the same call — its SigLIP tower is dead
  weight here and is dropped when it can be found. Note it is *natively multimodal*,
  which makes it a better bluffer, so the `random` and `shuffle` controls matter more,
  not less.
- **Anything ≥ 7B**: use `--load-4bit` (needs `bitsandbytes`, add `--with bitsandbytes`
  to the `uv run` call). Be aware the quantization noise sits between the projector
  and the score, which muddies an already subtle measurement.

A bigger LLM produces better captions and a *worse* measurement — more of the
output comes from the model's prior and less from the embedding. Get the number
with a small model; use a big one for the demo.

## Data

Embeddings are read exactly as elsewhere in the project: one file per video in a
flat directory, keyed by filename stem, `(N, D)` per clip folded back onto
`--jepa-grid`.

Captions come from one JSON file in any of three shapes:

- **SSv2 label file** — `something-something-v2-train.json` as shipped. Richest
  form: `template` and `placeholders` are what make the action/object split
  possible with no extra annotation work.
- **flat mapping** — `{"16290": "a caption"}`, which is what a VLM re-captioning
  pass will produce.
- **JSONL** — one object per line.

If the stems carry a prefix the labels don't (`embedding_16290` vs `16290`), pass
`--caption-key-prefix embedding_`.

**Pairing is by key and nothing downstream can detect a violation.** A systematic
off-by-one would train on mismatched pairs and look exactly like a clean negative
result — which is the failure this project is most exposed to, since it would also
explain the earlier latent-decoding negatives. Confirm a few by hand.

## Known open items

- **Near-duplicate leakage across the split** is still unchecked and it bites
  here specifically. Splits are by key, keys do not know their source video, and
  if the encoding pass wrote several crops per video then near-duplicates straddle
  the train/val boundary and inflate both retrieval and the NN baseline. Worth
  settling before any number goes in the writeup.
- **`--jepa-grid 8,14,14`** has still never been confirmed by `locality_score`.
  It fails loudly if it does not divide N, so it is probably right — but the
  perceiver's position embeddings assume it, and a wrong factorization would
  scatter spatial neighbours to arbitrary positions.
- **Templates only cover SSv2 idiom.** Trained on SSv2 labels alone the model can
  only ever speak SSv2. The re-captioning step (Qwen2-VL / InternVL; the LTX repo
  ships `scripts/caption_videos.py`) is what turns this from one number into a
  real per-attribute profile covering objects, colour, background and camera.
