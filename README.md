# V-JEPA Embedding Inversion Attack

I target the V-JEPA (ViT-L variant) model, and try to reconstruct videos by mapping V-JEPA latents to LTX-2 VAE latent space, then decoding those videos. The use of generative AI is very heavy in this project in order to test ideas, generate code and write reports to speed up work.

Refer to the [summary](SUMMARY.md) for a more detailed report.

## Setup

### 1. Clone with submodules

The two model repos are pinned as submodules under `third_party/`, so the exact
encoder/decoder code that produced the embeddings comes with the checkout:

```bash
git clone --recursive https://github.com/raywangruihua/V-JEPA-Embedding-Inversion-Attack.git
cd V-JEPA-Embedding-Inversion-Attack
```

Already cloned without `--recursive`:

```bash
git submodule update --init --recursive
```

| Submodule | Fork | Used for |
| --------- | ---- | -------- |
| `third_party/jepa` | [raywangruihua/jepa](https://github.com/raywangruihua/jepa) | V-JEPA encoding, attentive-probe classification |
| `third_party/LTX-2` | [raywangruihua/LTX-2](https://github.com/raywangruihua/LTX-2) | LTX-2 VAE encoding and decoding |

### 2. Three environments, not one

The submodules are pinned *sources*, not a merged environment. V-JEPA needs
`decord`, LTX-2 needs `torchcodec`, and they do not coexist. Keep three:

```bash
# this repo -- the adaptors, alignment and evaluation
uv venv && uv pip install torch numpy transformers pandas scipy scikit-learn plotly umap-learn

# V-JEPA               (see third_party/jepa/README.md)
# LTX-2                (see third_party/LTX-2/README.md)
```

`src/text_adaptor/run.py` is the exception: it carries a PEP 723 inline
dependency block, so `uv run` builds its own environment and needs no install
step at all.

### 3. Models and data

- **V-JEPA ViT-L/16** checkpoint plus the SSv2 attentive probe, for the encoder
  and the classifier baseline.
- **LTX-2** VAE weights, for the reconstruction model.
- **Something-Something V2** videos, plus the shipped
  `something-something-v2-train.json` label file — its `template` and
  `placeholders` fields are what make the action/object split possible.

## Layout

```
.
├── data/                        SSv2 split generation
│   ├── generate_split.py        build the train/validation dataloader CSVs
│   ├── class_to_index.json
│   └── video_to_class.json
├── src/
│   ├── otalign/                 Gromov-Wasserstein comparison and alignment
│   │   ├── comparison.py        Monte Carlo GW distance between the two spaces
│   │   ├── otalign.py           the GW aligner
│   │   └── vjepa_to_ltx2_align.py
│   ├── standard_adaptors/       supervised V-JEPA -> LTX-2 latent adaptors
│   │   ├── adaptor.py           the heads: linear, mlp, conv, plus the metrics
│   │   ├── latents.py           paired latent loading and standardization
│   │   └── train_adaptor.py
│   ├── diffusion_adaptor/       conditional DiT over LTX-2 latents
│   │   ├── model.py             the denoiser
│   │   ├── flow.py              rectified flow: objective and sampler
│   │   ├── data.py
│   │   ├── train.py
│   │   └── sample.py            latents only; the decode happens in LTX-2
│   ├── text_adaptor/            V-JEPA -> frozen LLM soft tokens
│   │   ├── README.md            the full walkthrough for this stage
│   │   ├── run.py               single entry point (PEP 723, use `uv run`)
│   │   ├── projector.py         the heads: mean, pool, perceiver
│   │   ├── model.py             projector + frozen LLM, loss and scoring
│   │   ├── baseline.py          nearest-neighbour floor
│   │   ├── train.py
│   │   ├── evaluate.py          the control table
│   │   ├── data.py
│   │   └── pack.py              fold per-clip files into one fp16 memmap
│   └── utils.py                 normalisation and UMAP plots
├── outputs/                     gitignored
│   ├── embeddings/
│   │   ├── jepa/                one .pt per clip, keyed by filename stem
│   │   └── ltx-2/
│   └── runs/                    checkpoints, config.json, history.json, eval.json
├── third_party/                 pinned submodules
│   ├── jepa/                    V-JEPA encoding and attentive-probe classification
│   └── LTX-2/                   LTX-2 VAE encoding and decoding
├── README.md
└── SUMMARY.md                   thesis, results, findings
```

## Running the pipeline

### 0. Build the split

Edit the paths at the top of [data/generate_split.py](data/generate_split.py),
then:

```bash
python data/generate_split.py
```

Writes `train_dataloader.csv` / `validation_dataloader.csv` in the format the
V-JEPA scripts expect. Videos shorter than `MIN_FRAMES` (33) are dropped.

### 1. Encode

Both encoders live in the submodules and run under *their own* environments.
**Their input and output paths are constants at the top of each file, not CLI
flags — edit them before running.**

```bash
# V-JEPA environment
python third_party/jepa/vjepa_encode.py      # -> one .pt per clip

# LTX-2 environment
python third_party/LTX-2/ltx2_encode.py      # -> one .pt per clip, 33 frames @ 256x256
```

The rest of this repo reads embeddings as **one file per video in a flat
directory, keyed by filename stem**. [comparison.py:26-27](src/otalign/comparison.py#L26-L27)
and the alignment script default to `outputs/embeddings/jepa` and
`outputs/embeddings/ltx-2`, so point the encoders there or pass the directories
explicitly downstream.

> [!NOTE]
> Pairing is by key, and **nothing downstream can detect a mismatch.** A
> systematic off-by-one would train on mismatched pairs and look exactly like a
> clean negative result. Confirm a few by hand.

### 2. Compare the two latent spaces

```bash
python src/otalign/comparison.py          # Monte Carlo Gromov-Wasserstein
python src/otalign/vjepa_to_ltx2_align.py # GW alignment -> outputs/embeddings/aligned
```

Both take their paths *and* their solver settings from module constants, not
flags. The one to know is the entropic regularization (`ENTREG` in
`comparison.py`, `entreg=5e-4` at
[vjepa_to_ltx2_align.py:41](src/otalign/vjepa_to_ltx2_align.py#L41)): the GW
gradient scales like `1/n`, so the usable value tracks the batch size. Too small
for the batch underflows the kernel and returns a degenerate plan — **on a small
run the safe direction is larger.**

### 3. Supervised adaptors

The ladder is `linear -> mlp -> conv`; run all three or a win is
uninterpretable.

```bash
python src/standard_adaptors/train_adaptor.py \
    --vjepa-dir outputs/embeddings/jepa \
    --ltx-dir   outputs/embeddings/ltx-2 \
    --jepa-grid 8,14,14 \
    --kind      mlp \
    --out       outputs/runs/adaptor_mlp
```

Use `--workers 0` on Windows. `--split` reuses an existing `split.json` so runs
share held-out clips.

### 4. Diffusion adaptor

```bash
python src/diffusion_adaptor/train.py \
    --vjepa-dir outputs/embeddings/jepa \
    --ltx-dir   outputs/embeddings/ltx-2 \
    --jepa-grid 8,14,14 \
    --out       outputs/runs/diffusion

python src/diffusion_adaptor/sample.py \
    --ckpt        outputs/runs/diffusion/best.pt \
    --vjepa-dir   outputs/embeddings/jepa \
    --out-dir     outputs/runs/diffusion/samples \
    --num-samples 8 --cfg 1.5
```

`sample.py` writes one subdirectory per source video, so hand the decoder
`samples/<index>/` rather than `samples/`. With `--num-samples > 1` the
candidates are `{index}/{key}_s{i}.pt` precisely so they can be reranked.

### 5. Decode to video

```bash
# LTX-2 environment -- edit INPUT_DIR / OUTPUT_DIR at the top first
python third_party/LTX-2/ltx2_decode.py
```

### 6. Text reconstruction

This is the part that worked. It has its own walkthrough in
**[src/text_adaptor/README.md](src/text_adaptor/README.md)** — the ladder, the
controls, and the memory/speed notes. The short version:

```bash
uv run src/text_adaptor/run.py selftest      # seconds, CPU, downloads nothing

uv run src/text_adaptor/run.py baseline \
    --vjepa-dir outputs/embeddings/jepa \
    --captions  something-something-v2-train.json \
    --jepa-grid 8,14,14 --out outputs/runs/nn_baseline

uv run src/text_adaptor/run.py train \
    --vjepa-dir outputs/embeddings/jepa \
    --captions  something-something-v2-train.json \
    --jepa-grid 8,14,14 --proj perceiver \
    --split     outputs/runs/nn_baseline/split.json \
    --out       outputs/runs/cap_perceiver

uv run src/text_adaptor/run.py evaluate \
    --ckpt      outputs/runs/cap_perceiver/best.pt \
    --vjepa-dir outputs/embeddings/jepa \
    --captions  something-something-v2-train.json
```

Get the nearest-neighbour baseline **before** training anything — a projector
that does not clear it has learned to write SSv2 sentences, not to read
embeddings. And read `shufAct` before any headline number: a frozen LLM writes a
confident SSv2 caption from pure noise, so `none` minus `shuffle` is the leakage.

## Gotchas

- **The encoder/decoder scripts hardcode absolute Linux paths** as module
  constants. Every one of them needs editing before it will run anywhere else.
- **`--jepa-grid 8,14,14`** is assumed throughout and has never been confirmed by
  `locality_score`. It fails loudly if it does not divide N, so it is probably
  right, but a wrong factorization would scatter spatial neighbours.
- **Submodule updates are a two-step commit.** Change something in
  `third_party/`, commit and push it *there* first, then commit the moved pointer
  here.
