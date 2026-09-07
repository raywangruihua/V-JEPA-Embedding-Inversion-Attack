"""The denoiser: a conditional DiT over LTX-2 VAE latents, conditioned on V-JEPA tokens.

This is the model that replaces ``standard_adaptors/adaptor.py``, and the reason for the
replacement is worth stating plainly, because it is the whole point of the
exercise.

The adaptor was a *regressor*: one V-JEPA embedding in, one LTX latent out,
trained under MSE. But V-JEPA is trained to predict representations, not pixels,
so it deliberately discards most pixel-level detail -- which makes the map from
embedding to video genuinely one-to-many. Asked for a one-to-many map under MSE,
a regressor returns the conditional *mean* over every video consistent with the
embedding, and the mean of many plausible videos is grey mush. That is what the
``--match-variance`` hack in ``standard_adaptors/export_latents.py`` was fighting, and why
it could never win: the averaging destroys the information before the rescale
ever sees it.

A diffusion model *samples* from ``p(latent | embedding)`` instead of averaging
it, so it returns one sharp member of that set rather than the blurred centroid
of all of them. Nothing else about the setup changes.

What conditions on what
-----------------------
The V-JEPA embedding is **conditioning**, never a diffusion target. Noise is
added to and removed from the LTX latent; the V-JEPA tokens enter only through
cross-attention. This matters because it means the two latent spaces need no
geometric correspondence whatsoever -- the LTX VAE is being used as a *codec*
that happens to have a working decoder, not as an alignment target. The negative
Gromov-Wasserstein and adaptor results were evidence against alignment, and
alignment is no longer part of the plan.

Architecture
------------
Standard DiT with adaLN-Zero, plus a cross-attention stream:

* the noisy LTX latent ``(B, C, F, H, W)`` is flattened to one token per grid
  position -- no further patchifying, since the VAE has already compressed hard
  enough that a 16-frame clip is only ~150 positions;
* V-JEPA ``(B, D, F', H', W')`` is projected to the model width and used as
  cross-attention memory, with its own factorised 3-D position embedding;
* the timestep and a mean-pooled summary of the condition drive adaLN
  modulation, giving the model a cheap global channel alongside the spatial one.

Every block's modulation projection is zero-initialised, so at step zero each
block is the identity and the residual stream is clean. This is the single most
reliable trick for making a DiT train stably from scratch on a small dataset.

Classifier-free guidance
------------------------
``forward`` takes a ``drop`` mask that swaps the condition for a learned null
token. Train with ``--cfg-dropout 0.1`` and you can trade fidelity against
sharpness at sampling time without retraining. It costs one parameter vector and
is painful to add retroactively, so it is here from the start.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10_000) -> torch.Tensor:
    """Sinusoidal features for a continuous ``t`` in ``[0, 1]``.

    Scaled by 1000 before the sinusoids, the usual convention for continuous-time
    models: it puts a unit-interval timestep on roughly the same footing as the
    discrete 0-1000 index the original schedules used, so the standard frequency
    range still resolves nearby timesteps.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float().unsqueeze(1) * 1000.0 * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """adaLN: rescale and offset a normed activation from a conditioning vector."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class GridPosEmbed(nn.Module):
    """Factorised learned position embedding over an ``(F, H, W)`` grid.

    Three small tables summed, rather than one table of ``F*H*W`` rows. With only
    a few thousand training clips a full table is mostly unconstrained
    parameters; the factorised form shares statistics across every position in a
    row or column and is much harder to overfit.
    """

    def __init__(self, grid: Sequence[int], width: int) -> None:
        super().__init__()
        f, h, w = (int(v) for v in grid)
        self.grid = (f, h, w)
        self.pf = nn.Parameter(torch.randn(f, width) * 0.02)
        self.ph = nn.Parameter(torch.randn(h, width) * 0.02)
        self.pw = nn.Parameter(torch.randn(w, width) * 0.02)

    @property
    def n_tokens(self) -> int:
        f, h, w = self.grid
        return f * h * w

    def forward(self) -> torch.Tensor:
        e = self.pf[:, None, None, :] + self.ph[None, :, None, :] + self.pw[None, None, :, :]
        return e.reshape(1, self.n_tokens, -1)


class Attention(nn.Module):
    """Multi-head attention. Self-attention when ``mem`` is None, cross otherwise."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        if width % heads:
            raise ValueError(f"width {width} not divisible by heads {heads}")
        self.heads = heads
        self.to_q = nn.Linear(width, width)
        self.to_kv = nn.Linear(width, 2 * width)
        self.proj = nn.Linear(width, width)

    def forward(self, x: torch.Tensor, mem: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, n, w = x.shape
        src = x if mem is None else mem
        m = src.shape[1]
        q = self.to_q(x).reshape(b, n, self.heads, -1).transpose(1, 2)
        kv = self.to_kv(src).reshape(b, m, 2, self.heads, -1).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(q, kv[0], kv[1])
        return self.proj(out.transpose(1, 2).reshape(b, n, w))


class DiTBlock(nn.Module):
    """Self-attention, cross-attention to the condition, MLP -- each adaLN-Zero gated."""

    def __init__(self, width: int, heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(width, heads)
        self.cross = Attention(width, heads)
        hidden = int(width * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(width, hidden), nn.GELU(approximate="tanh"), nn.Linear(hidden, width)
        )
        # Nine parameters: shift/scale/gate for each of the three sublayers.
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(width, 9 * width))

    def forward(self, x: torch.Tensor, c: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        s1, g1, a1, s2, g2, a2, s3, g3, a3 = self.ada(c).chunk(9, dim=-1)
        x = x + a1.unsqueeze(1) * self.attn(modulate(self.norm1(x), s1, g1))
        x = x + a2.unsqueeze(1) * self.cross(modulate(self.norm2(x), s2, g2), mem)
        x = x + a3.unsqueeze(1) * self.mlp(modulate(self.norm3(x), s3, g3))
        return x


class FinalLayer(nn.Module):
    """Project back to latent channels.

    Zero-initialised, so the model starts by predicting a velocity of exactly
    zero -- a stable place to begin, and it stops the first few steps from
    injecting large random gradients into the condition encoder.
    """

    def __init__(self, width: int, out_ch: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.proj = nn.Linear(width, out_ch)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.ada(c).chunk(2, dim=-1)
        return self.proj(modulate(self.norm(x), shift, scale))


class LatentDiT(nn.Module):
    """Predict the flow velocity for a noised LTX latent, given V-JEPA tokens.

    Parameters
    ----------
    in_ch, latent_grid
        Channels and ``(F, H, W)`` of the LTX-2 VAE latent -- the diffusion target.
    cond_dim, cond_grid
        Channels and ``(F, H, W)`` of the V-JEPA side. ``cond_grid`` is the grid
        recovered by the locality check, e.g. ``(8, 14, 14)`` for a 16-frame
        224px clip through a ``2x16x16`` tubelet.
    cond_pool
        Optionally average-pool the V-JEPA grid before projecting. A ViT-L clip is
        1568 tokens of cross-attention memory; pooling to ``(8, 7, 7)`` quarters
        that for what is usually a small quality loss, and is the first thing to
        reach for when VRAM is tight.
    """

    def __init__(
        self,
        in_ch: int,
        latent_grid: Sequence[int],
        cond_dim: int,
        cond_grid: Sequence[int],
        width: int = 768,
        depth: int = 12,
        heads: int = 12,
        mlp_ratio: float = 4.0,
        cond_pool: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.width = width
        self.latent_grid = tuple(int(v) for v in latent_grid)
        self.cond_grid = tuple(int(v) for v in cond_grid)
        self.cond_pool = tuple(int(v) for v in cond_pool) if cond_pool else None
        eff_cond_grid = self.cond_pool or self.cond_grid

        self.x_embed = nn.Linear(in_ch, width)
        self.x_pos = GridPosEmbed(self.latent_grid, width)

        self.c_embed = nn.Linear(cond_dim, width)
        self.c_pos = GridPosEmbed(eff_cond_grid, width)
        # One learned vector, broadcast over the memory length. A full-length null
        # would be ~1.2M parameters trained on 10% of batches; this is `width`
        # parameters and works as well in practice.
        self.null_token = nn.Parameter(torch.randn(1, 1, width) * 0.02)

        self.t_embed = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.c_pool_proj = nn.Linear(width, width)

        self.blocks = nn.ModuleList([DiTBlock(width, heads, mlp_ratio) for _ in range(depth)])
        self.final = FinalLayer(width, in_ch)

        self.apply(self._init_linear)
        # _init_linear would have clobbered the adaLN-Zero inits, so apply those
        # last. Order matters here: getting it backwards silently costs you the
        # identity-at-init property and the run just trains worse.
        for block in self.blocks:
            nn.init.zeros_(block.ada[1].weight)
            nn.init.zeros_(block.ada[1].bias)
        for layer in (self.final.ada[1], self.final.proj):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    @staticmethod
    def _init_linear(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def encode_cond(self, cond: torch.Tensor, drop: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``(B, D, F, H, W)`` V-JEPA grid -> ``(B, N, width)`` cross-attention memory."""
        if self.cond_pool is not None:
            cond = F.adaptive_avg_pool3d(cond, self.cond_pool)
        b, d = cond.shape[0], cond.shape[1]
        tokens = cond.reshape(b, d, -1).transpose(1, 2)  # (B, N, D)
        mem = self.c_embed(tokens) + self.c_pos()
        if drop is not None and bool(drop.any()):
            null = self.null_token.expand(b, mem.shape[1], -1).to(mem.dtype)
            mem = torch.where(drop.view(-1, 1, 1), null, mem)
        return mem

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        drop: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``x``: noised latent ``(B, C, F, H, W)``. ``t``: ``(B,)`` in ``[0, 1]``.

        Returns the predicted velocity, same shape as ``x``.
        """
        b, c = x.shape[0], x.shape[1]
        grid = x.shape[2:]

        h = x.reshape(b, c, -1).transpose(1, 2)  # (B, T, C)
        h = self.x_embed(h) + self.x_pos()

        mem = self.encode_cond(cond, drop)
        vec = self.t_embed(timestep_embedding(t, self.width).to(h.dtype))
        vec = vec + self.c_pool_proj(mem.mean(dim=1))

        for block in self.blocks:
            h = block(h, vec, mem)
        h = self.final(h, vec)

        return h.transpose(1, 2).reshape(b, c, *grid)
