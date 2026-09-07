"""Projector + frozen LLM: sequence assembly, the loss, sampling, and scoring.

The whole model is one small trainable head in front of a language model whose
weights never move. What that buys is not only the 12 GB budget -- it is the
*claim*. A frozen LLM never sees a gradient from the captions, so it cannot
memorize the caption distribution, and any dataset-specific information in the
output had to arrive through the K soft tokens. Leakage measured here is
therefore attributable to the embedding.

That argument is not airtight, and the gap is worth naming precisely: the
*projector* can still learn an unconditional prior ("SSv2 captions usually begin
with Pushing") and emit it whatever the input. That is exactly what the controls
in ``evaluate.py`` are for, and it is why they are not optional.

The sequence
------------
::

    [BOS]  [soft x K]  [prompt tokens]  [caption tokens]  [EOS]
     -100    -100        -100            <- loss lives here ->

Three things about this are easy to get wrong and expensive to notice:

1. **Soft tokens bypass the embedding table.** There is no integer id for "the
   third soft token", so the sequence is assembled at the embedding level and
   handed to the LLM as ``inputs_embeds``. Passing ``input_ids`` alongside would
   be a silent contradiction.
2. **The loss is masked to caption positions.** Soft-token positions have no
   correct next token, and training on the fixed prompt teaches the projector to
   reproduce boilerplate. Neither crashes; both waste the budget.
3. **Frozen does not mean cheap.** The error signal still has to travel back
   through every layer of the LLM to reach the projector, so this costs close to
   full training memory even though only ~30M parameters move.

   Gradient checkpointing is nonetheless **off** by default: it buys that memory
   back at the price of a recompute forward pass, which is a third of the FLOPs,
   and a 1.5B model at this sequence length fits a 12 GB card without it.
   ``train.py`` turns it on by itself if a step OOMs. When it is on,
   ``use_reentrant=False`` is not optional -- the reentrant implementation looks
   for parameters requiring grad, finds none in a fully frozen stack, and
   silently declines to build a graph, so the projector would receive no
   gradient while the loss curve carried on looking healthy.

The prompt is deliberately plain text rather than a chat template. Chat templates
differ per model and inject tokens whose positions the masking above would have
to track; the measurement does not benefit, and the failure would be quiet.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE_INDEX = -100
CONTROLS = ("none", "random", "shuffle", "zero")

# Sequence positions upcast to fp32 at once inside sequence_logprob. Trades a
# short Python loop for a bounded fp32 working set; see the comment there.
LOGSUMEXP_CHUNK = 32


# ---------------------------------------------------------------------------
# Loading the frozen language model
# ---------------------------------------------------------------------------


def text_hidden_size(config) -> int:
    """The *text* stack's width, which is what a soft token has to match.

    Multimodal checkpoints (gemma-3-4b and friends) nest the language model's
    config under ``text_config``, and their top-level ``hidden_size`` -- when it
    exists at all -- may describe the vision tower instead. Reading the wrong one
    produces a projector whose output width is wrong by a factor that only shows
    up as a shape error deep in the first forward pass.
    """
    getter = getattr(config, "get_text_config", None)
    if callable(getter):
        try:
            return int(getter().hidden_size)
        except Exception:
            pass
    inner = getattr(config, "text_config", None)
    if inner is not None and hasattr(inner, "hidden_size"):
        return int(inner.hidden_size)
    return int(config.hidden_size)


def load_llm(
    name: str,
    dtype: str = "bfloat16",
    load_4bit: bool = False,
    device: Optional[torch.device] = None,
    gradient_checkpointing: bool = False,
    drop_vision_tower: bool = True,
):
    """Load a causal LM frozen and ready to be conditioned on ``inputs_embeds``.

    Returns ``(llm, tokenizer, hidden_size)``.

    Qwen2.5-1.5B-Instruct is the tested path. Multimodal checkpoints such as
    ``google/gemma-3-4b-it`` load through the same call because
    ``get_input_embeddings``, ``forward(inputs_embeds=...)`` and
    ``generate(inputs_embeds=...)`` are common interface -- their SigLIP tower is
    dead weight here, since the projector is replacing its job, so it is dropped
    when it can be found.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = getattr(torch, dtype)
    kwargs: Dict[str, object] = {"dtype": torch_dtype}

    if load_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise SystemExit(
                "--load-4bit needs bitsandbytes. Re-run with "
                "'uv run --with bitsandbytes src/text_adaptor/run.py ...'"
            ) from exc
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(name)
    llm = AutoModelForCausalLM.from_pretrained(name, **kwargs)

    if drop_vision_tower:
        # Saves ~0.8 GB on gemma-3-4b. Guarded because the attribute name is not
        # stable across architectures and losing the saving is harmless.
        for attr in ("vision_tower", "vision_model", "visual"):
            holder = llm.model if hasattr(llm, "model") else llm
            if hasattr(holder, attr):
                try:
                    setattr(holder, attr, None)
                except Exception:
                    pass

    llm.requires_grad_(False)
    llm.eval()
    llm.config.use_cache = not gradient_checkpointing
    if gradient_checkpointing:
        llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    if not load_4bit and device is not None:
        llm.to(device)

    return llm, tokenizer, text_hidden_size(llm.config)


# ---------------------------------------------------------------------------
# Sequence assembly -- pure tensor logic, deliberately testable without an LLM
# ---------------------------------------------------------------------------


def assemble_batch(
    soft: torch.Tensor,
    pre_ids: torch.Tensor,
    mid_ids: torch.Tensor,
    caption_ids: Sequence[torch.Tensor],
    embed,
    pad_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Splice soft tokens into a padded batch of captions.

    Parameters
    ----------
    soft:
        ``(B, K, d)``, already in the LLM's dtype and on its device.
    pre_ids:
        ``(P0,)`` ids placed *before* the soft tokens -- BOS, if the tokenizer
        has one. May be empty.
    mid_ids:
        ``(P1,)`` prompt ids placed *between* the soft tokens and the caption.
        May be empty.
    caption_ids:
        ``B`` variable-length id tensors, each already terminated with EOS.
    embed:
        ``ids -> embeddings``; ``llm.get_input_embeddings()`` in production, a
        plain ``nn.Embedding`` in the self-test.
    pad_id:
        Filler for the ragged tail. Masked out of both attention and loss, so
        its value never reaches a gradient -- but it must be a *valid* index or
        the embedding lookup raises.

    Returns ``(inputs_embeds, attention_mask, labels)``.

    Right-padding is correct here because every padded position is masked out of
    the loss. It would *not* be correct for generation, which is why
    :meth:`CaptionModel.generate` never pads: its prompt is
    ``P0 + K + P1`` tokens for every row in the batch.
    """
    b, k, _ = soft.shape
    device = soft.device
    lengths = [int(c.numel()) for c in caption_ids]
    t_max = max(lengths) if lengths else 0

    padded = torch.full((b, t_max), pad_id, dtype=torch.long, device=device)
    cap_mask = torch.zeros((b, t_max), dtype=torch.bool, device=device)
    for i, ids in enumerate(caption_ids):
        n = lengths[i]
        padded[i, :n] = ids.to(device)
        cap_mask[i, :n] = True

    parts, masks, labels = [], [], []

    def add(chunk_emb, chunk_mask, chunk_labels):
        parts.append(chunk_emb)
        masks.append(chunk_mask)
        labels.append(chunk_labels)

    if pre_ids.numel():
        emb = embed(pre_ids.to(device))[None].expand(b, -1, -1)
        n = pre_ids.numel()
        add(emb,
            torch.ones((b, n), dtype=torch.long, device=device),
            torch.full((b, n), IGNORE_INDEX, dtype=torch.long, device=device))

    add(soft,
        torch.ones((b, k), dtype=torch.long, device=device),
        torch.full((b, k), IGNORE_INDEX, dtype=torch.long, device=device))

    if mid_ids.numel():
        emb = embed(mid_ids.to(device))[None].expand(b, -1, -1)
        n = mid_ids.numel()
        add(emb,
            torch.ones((b, n), dtype=torch.long, device=device),
            torch.full((b, n), IGNORE_INDEX, dtype=torch.long, device=device))

    if t_max:
        add(embed(padded).to(soft.dtype),
            cap_mask.long(),
            torch.where(cap_mask, padded, torch.full_like(padded, IGNORE_INDEX)))

    return (
        torch.cat(parts, dim=1),
        torch.cat(masks, dim=1),
        torch.cat(labels, dim=1),
    )


def apply_control(x: torch.Tensor, control: str, generator=None) -> torch.Tensor:
    """Corrupt the conditioning in a specified way, for the fluency controls.

    ``x`` is a standardized V-JEPA grid ``(B, D, F, H, W)``.

    * ``random`` -- draw from N(0, 1). Because the input is standardized this is
      a *plausible-looking* embedding, not an obviously broken one. If captions
      stay fluent and SSv2-shaped under this, what is being read is the prior.
    * ``shuffle`` -- roll the batch so every clip gets another clip's embedding.
      Strictly in-distribution, and therefore the harder control of the two: it
      cannot be passed by a model that merely learned "real embeddings look like
      this".
    * ``zero`` -- the weakest, kept only because it is the obvious thing to try
      and it is useful to see it fail differently from the other two.
    """
    if control in ("none", None):
        return x
    if control == "random":
        return torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
    if control == "shuffle":
        if x.shape[0] < 2:
            raise ValueError("the 'shuffle' control needs a batch of at least 2")
        return torch.roll(x, shifts=1, dims=0)
    if control == "zero":
        return torch.zeros_like(x)
    raise ValueError(f"unknown control: {control!r} (expected one of {CONTROLS})")


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


class CaptionModel(nn.Module):
    """Trainable projector in front of a frozen causal LM.

    The projector is held in float32 while the LLM runs in bf16. Small heads
    train more stably in full precision, and the only place the two meet is one
    cast on the soft tokens, which the backward pass handles without ceremony.
    """

    def __init__(self, projector: nn.Module, llm, tokenizer, prompt: str = "") -> None:
        super().__init__()
        self.projector = projector
        self.llm = llm
        self.tokenizer = tokenizer
        self.prompt = prompt

        bos = tokenizer.bos_token_id
        self.register_buffer(
            "pre_ids",
            torch.tensor([bos] if bos is not None else [], dtype=torch.long),
            persistent=False,
        )
        mid = tokenizer(prompt, add_special_tokens=False)["input_ids"] if prompt else []
        self.register_buffer("mid_ids", torch.tensor(mid, dtype=torch.long), persistent=False)

        # eos is what tells generate() to stop and what the loss teaches the model
        # to emit. A tokenizer without one would train a model that never
        # terminates, so fail here rather than at sampling time.
        if tokenizer.eos_token_id is None:
            raise SystemExit(f"tokenizer for {tokenizer.name_or_path} has no eos_token_id")
        self.eos_id = int(tokenizer.eos_token_id)
        self.pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else self.eos_id)

    # -- helpers -----------------------------------------------------------

    @property
    def llm_dtype(self) -> torch.dtype:
        return next(self.llm.parameters()).dtype

    @property
    def prefix_len(self) -> int:
        """How many non-caption positions sit in front of every caption."""
        return int(self.pre_ids.numel()) + self.projector.n_tokens + int(self.mid_ids.numel())

    def encode_caption(self, text: str, max_len: int = 64) -> torch.Tensor:
        """Caption text -> ids terminated with EOS, truncated to ``max_len``."""
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"][: max_len - 1]
        return torch.tensor(ids + [self.eos_id], dtype=torch.long)

    def soft_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, D, F, H, W)`` -> ``(B, K, d_llm)`` in the LLM's dtype."""
        return self.projector(x.float()).to(self.llm_dtype)

    # -- objectives --------------------------------------------------------

    def forward(self, x: torch.Tensor, caption_ids: Sequence[torch.Tensor]) -> torch.Tensor:
        """Next-token loss on the caption positions. ``x`` is standardized V-JEPA."""
        soft = self.soft_tokens(x)
        embeds, mask, labels = assemble_batch(
            soft, self.pre_ids, self.mid_ids, caption_ids,
            self.llm.get_input_embeddings(), self.pad_id,
        )
        out = self.llm(inputs_embeds=embeds, attention_mask=mask, labels=labels)
        return out.loss

    @torch.no_grad()
    def sequence_logprob(
        self, x: torch.Tensor, caption_ids: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        """``log p(caption | embedding)`` summed over caption tokens -> ``(B,)``.

        This is what caption-to-clip retrieval ranks on. Note that the length
        normalization usually needed for such scores is unnecessary in that use:
        retrieval holds the *caption* fixed and varies the embedding, so every
        score in a comparison covers exactly the same tokens.
        """
        soft = self.soft_tokens(x)
        embeds, mask, labels = assemble_batch(
            soft, self.pre_ids, self.mid_ids, caption_ids,
            self.llm.get_input_embeddings(), self.pad_id,
        )
        logits = self.llm(inputs_embeds=embeds, attention_mask=mask).logits

        # Shift by hand: position i predicts token i+1. HF does this internally
        # when given `labels`, but it also reduces to a scalar, and retrieval
        # needs the per-sequence total.
        shift_labels = labels[:, 1:]
        keep = shift_labels != IGNORE_INDEX
        safe = shift_labels.masked_fill(~keep, 0)

        # log p(y) = logit[y] - logsumexp(logits), computed so that no tensor of
        # B x L x vocab is ever allocated a second time.
        #
        # The obvious spelling -- .float() then torch.log_softmax then .gather --
        # allocates two *additional* full-size tensors, and with Qwen's 152k
        # vocab against a 122-token sequence that is ~0.6 GB each at batch 8.
        # Three live copies of the same thing is what put a 12 GB card over the
        # edge. Reshaping for cross_entropy is no better: the [:, :-1] slice is
        # non-contiguous, so .reshape would copy it too.
        #
        # gather reduces to (B, L) immediately, and logsumexp is a reduction --
        # but it needs fp32 to be trustworthy across 48 summed tokens, so it runs
        # in slices along the sequence. Each slice upcasts B x 32 x vocab, which
        # is tens of MB rather than hundreds.
        shifted = logits[:, :-1]
        picked = shifted.gather(-1, safe.unsqueeze(-1)).squeeze(-1).float()

        lse = torch.empty(safe.shape, dtype=torch.float32, device=shifted.device)
        for start in range(0, shifted.shape[1], LOGSUMEXP_CHUNK):
            stop = start + LOGSUMEXP_CHUNK
            lse[:, start:stop] = torch.logsumexp(shifted[:, start:stop].float(), dim=-1)

        per_token = picked - lse
        del logits, shifted, picked, lse
        return (per_token * keep).sum(dim=1)

    def generation_config(
        self,
        max_new_tokens: int = 32,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        num_return_sequences: int = 1,
    ):
        """An **explicit** decoding config, deliberately not the model's own.

        Checkpoints ship a ``generation_config.json`` and it is not neutral:
        Qwen2.5-Instruct sets ``repetition_penalty=1.05``, ``do_sample=True``,
        ``temperature=0.7``, ``top_p=0.8``, ``top_k=20``. Inheriting that would
        mean the decoding strategy is whatever the checkpoint's authors chose for
        chat, silently, unrecorded, and different for every model someone swaps
        in -- while this file claims to be doing greedy decoding.

        That matters more here than in ordinary use. A repetition penalty makes
        argmax decoding not-argmax, and degenerate output is one of the signals
        distinguishing a caption read off an embedding from one confabulated by
        the prior. Suppressing the symptom would blur the very comparison the
        controls exist to make.

        So every field is set explicitly, the defaults are neutral
        (``repetition_penalty=1.0`` is *off*), and the whole thing is recorded in
        ``config.json``.

        **Every field means every field, including the ones greedy decoding
        ignores.** Passing a ``GenerationConfig`` does not replace the model's:
        ``_prepare_generation_config`` deep-copies it and then fills in every
        attribute still set to ``None`` from ``model.generation_config``. In
        transformers v5 a bare ``GenerationConfig()`` leaves ``temperature``,
        ``top_p`` and ``top_k`` at ``None``, so leaving them unset hands control
        straight back to the checkpoint -- which is the thing this method exists
        to prevent.

        Under greedy decoding they are therefore pinned to the values the
        validator treats as inactive (``1.0``, ``1.0``, ``50``). Those are also
        functionally inert: the top-k and top-p logits warpers are only
        constructed when ``do_sample=True``, so nothing about the output changes.
        """
        from transformers import GenerationConfig

        # 1.0 / 1.0 / 50 are the "unset" sentinels validate() checks against, and
        # they are inert under greedy decoding. Do not replace them with 0 or
        # None: 0 trips the validator, None re-opens the merge.
        sampling = (
            {"temperature": temperature, "top_p": top_p, "top_k": 0}
            if do_sample
            else {"temperature": 1.0, "top_p": 1.0, "top_k": 50}
        )
        return GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.pad_id,
            eos_token_id=self.eos_id,
            **sampling,
        )

    @torch.no_grad()
    def generate(
        self,
        x: torch.Tensor,
        max_new_tokens: int = 32,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        num_return_sequences: int = 1,
    ) -> List[str]:
        """Sample captions. Greedy by default, because the measurement wants the
        model's single best answer rather than a fluent one.

        No padding is involved: the prompt is ``prefix_len`` positions wide for
        every row, so the left-padding requirement that normally applies to
        batched causal generation does not arise.
        """
        soft = self.soft_tokens(x)
        b = soft.shape[0]
        device = soft.device
        embed = self.llm.get_input_embeddings()

        parts = []
        if self.pre_ids.numel():
            parts.append(embed(self.pre_ids.to(device))[None].expand(b, -1, -1))
        parts.append(soft)
        if self.mid_ids.numel():
            parts.append(embed(self.mid_ids.to(device))[None].expand(b, -1, -1))
        embeds = torch.cat(parts, dim=1)
        mask = torch.ones(embeds.shape[:2], dtype=torch.long, device=device)

        was_cached = self.llm.config.use_cache
        self.llm.config.use_cache = True
        try:
            out = self.llm.generate(
                inputs_embeds=embeds,
                attention_mask=mask,
                generation_config=self.generation_config(
                    max_new_tokens=max_new_tokens, do_sample=do_sample,
                    temperature=temperature, top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    num_return_sequences=num_return_sequences,
                ),
            )
        finally:
            self.llm.config.use_cache = was_cached

        # With inputs_embeds and no input_ids, generate() returns only the new
        # tokens, so there is no prompt prefix to strip.
        return [t.strip() for t in self.tokenizer.batch_decode(out, skip_special_tokens=True)]

    # -- checkpointing -----------------------------------------------------

    def trainable_parameters(self):
        return [p for p in self.projector.parameters() if p.requires_grad]

    def projector_state(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu() for k, v in self.projector.state_dict().items()}


def load_checkpoint(path, device, gradient_checkpointing: bool = False):
    """Rebuild a :class:`CaptionModel` from a training checkpoint.

    Only the projector was saved -- a few tens of MB rather than a few GB -- so
    the LLM is re-fetched by the name recorded in ``config``. That keeps runs
    cheap to keep around, at the cost of needing the same checkpoint available
    when a run is re-scored.
    """
    from projector import build_projector  # local: avoids a cycle at import time

    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload["config"]

    llm, tokenizer, d_llm = load_llm(
        config["llm"], dtype=config.get("dtype", "bfloat16"),
        load_4bit=config.get("load_4bit", False), device=device,
        gradient_checkpointing=gradient_checkpointing,
    )
    if d_llm != config["d_llm"]:
        raise SystemExit(
            f"{config['llm']} now reports hidden size {d_llm} but the checkpoint "
            f"was trained against {config['d_llm']}. Different revision?"
        )

    projector = build_projector(
        config["proj"], in_dim=config["in_dim"], out_dim=d_llm,
        grid=tuple(config["grid"]), n_tokens=config["n_tokens"],
        pool_grid=tuple(config["pool_grid"]), width=config["proj_width"],
        depth=config["proj_depth"], heads=config["proj_heads"],
        hidden=config["proj_hidden"],
    )
    projector.load_state_dict(payload["projector"])
    projector.to(device).eval()

    model = CaptionModel(projector, llm, tokenizer, prompt=config.get("prompt", ""))
    return model, config
