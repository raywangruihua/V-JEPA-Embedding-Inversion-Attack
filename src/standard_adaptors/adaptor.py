"""The adaptor heads: ``linear``, ``mlp``, ``conv``, plus the metrics to judge them.

Each head maps a standardized V-JEPA grid ``(D, Fv, Hv, Wv)`` onto a standardized
LTX-2 latent grid ``(C, Fl, Hl, Wl)``. The two grids differ, so every head begins
by resampling the input onto the target grid and then transforms channels.

The ladder is the point
-----------------------
Run all three. ``linear -> mlp`` isolates the value of nonlinearity;
``mlp -> conv`` isolates the value of a spatial receptive field. Skipping ``mlp``
makes a ``conv`` win uninterpretable, because you cannot tell which of the two
changes bought it.

``linear`` and ``mlp`` are **positionwise**: the output at one grid position is a
function of the input at that position alone. ``conv`` is the first head that can
see its neighbours. That difference is the whole experiment -- colour is
low-frequency and positionwise-recoverable, shape is high-frequency and needs
spatial context.

The resample is not innocent
----------------------------
``--resample interp`` uses ``F.interpolate(..., "trilinear")``, which has no
anti-aliasing and *samples* rather than averages. On (8,14,14) -> (2,7,7) the
outputs land at input coordinates 1.5 and 5.5, so **frames 0, 3, 4 and 7
contribute nothing at all**. This was confirmed empirically by perturbing one
input frame at a time.

``--resample pool`` (``adaptive_avg_pool3d``) averages instead and fixes it. The
default is still ``interp`` so that runs recorded before the discovery stay
reproducible -- pass ``--resample pool`` for anything new.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

KINDS = ("linear", "mlp", "conv")


def _num_groups(channels: int, preferred: int = 32) -> int:
    """Largest divisor of ``channels`` no greater than ``preferred``.

    ``nn.GroupNorm`` requires the group count to divide the channel count, and
    ``--hidden`` is free-form -- ``--hidden 100`` would otherwise die inside
    torch rather than quietly using 25 groups.
    """
    groups = min(preferred, channels)
    while channels % groups:
        groups -= 1
    return groups


def resample_grid(x: torch.Tensor, target: Sequence[int], mode: str = "interp") -> torch.Tensor:
    """Resample ``(B, C, F, H, W)`` onto ``target`` = ``(F, H, W)``.

    See the module docstring for why ``mode`` matters more than it looks.
    """
    target = tuple(int(t) for t in target)
    if tuple(x.shape[2:]) == target:
        return x
    if mode == "pool":
        return F.adaptive_avg_pool3d(x, target)
    if mode == "interp":
        return F.interpolate(x, size=target, mode="trilinear", align_corners=False)
    raise ValueError(f"unknown resample mode: {mode!r}")


class Adaptor(nn.Module):
    """V-JEPA grid -> LTX-2 latent grid.

    Parameters
    ----------
    in_ch, out_ch:
        Channel counts. ``in_ch`` is V-JEPA's D (1024 for ViT-L); ``out_ch`` is
        the LTX VAE's latent channel count.
    target_grid:
        ``(F, H, W)`` of the LTX latent. Read off the data rather than assumed.
    kind:
        One of :data:`KINDS`.
    hidden:
        Width of the hidden layers. Defaults differ by kind so the ladder is a
        fair one -- see :func:`build_adaptor`.
    depth:
        Number of hidden blocks (``mlp`` and ``conv`` only).
    resample:
        ``"interp"`` or ``"pool"``.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        target_grid: Sequence[int],
        kind: str = "mlp",
        hidden: int = 768,
        depth: int = 2,
        resample: str = "interp",
    ) -> None:
        super().__init__()
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        self.kind = kind
        self.resample = resample
        self.target_grid = tuple(int(t) for t in target_grid)

        if kind == "linear":
            # 1x1x1 convolution == a positionwise affine map, written this way so
            # every head has the same (B, C, F, H, W) interface.
            self.net = nn.Conv3d(in_ch, out_ch, kernel_size=1)

        elif kind == "mlp":
            layers = [nn.Conv3d(in_ch, hidden, 1), nn.GELU()]
            for _ in range(max(0, depth - 1)):
                layers += [nn.Conv3d(hidden, hidden, 1), nn.GELU()]
            layers += [nn.Conv3d(hidden, out_ch, 1)]
            self.net = nn.Sequential(*layers)

        else:  # conv
            # GroupNorm rather than BatchNorm: batches here are small (16 clips)
            # and BatchNorm's running statistics would be noisy enough to matter.
            layers = [nn.Conv3d(in_ch, hidden, 1), nn.GELU()]
            for _ in range(max(1, depth)):
                layers += [
                    nn.Conv3d(hidden, hidden, kernel_size=3, padding=1),
                    nn.GroupNorm(_num_groups(hidden), hidden),
                    nn.GELU(),
                ]
            layers += [nn.Conv3d(hidden, out_ch, 1)]
            self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, D, Fv, Hv, Wv)`` -> ``(B, C, Fl, Hl, Wl)``."""
        x = resample_grid(x, self.target_grid, self.resample)
        return self.net(x)


def build_adaptor(
    kind: str,
    in_ch: int,
    out_ch: int,
    target_grid: Sequence[int],
    hidden: int = 0,
    depth: int = 0,
    resample: str = "interp",
) -> Adaptor:
    """Construct a head, filling in per-kind defaults for ``hidden`` and ``depth``.

    The defaults land ``mlp`` near 2M parameters and ``conv`` near 24M. That gap
    is deliberate and is the reason ``mlp`` was recommended first on ~850 clips:
    the effective number of independent training samples is closer to the *clip*
    count than to the position count, so a 24M-parameter head has far more
    capacity than the data supports. Revisit once the dataset scales.
    """
    if kind == "linear":
        return Adaptor(in_ch, out_ch, target_grid, "linear", resample=resample)
    if kind == "mlp":
        return Adaptor(
            in_ch, out_ch, target_grid, "mlp",
            hidden=hidden or 768, depth=depth or 2, resample=resample,
        )
    if kind == "conv":
        return Adaptor(
            in_ch, out_ch, target_grid, "conv",
            hidden=hidden or 512, depth=depth or 3, resample=resample,
        )
    raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")


# ---------------------------------------------------------------------------
# Metrics. All of these read standardized space -- see ChannelStats' docstring.
# ---------------------------------------------------------------------------


def retrieval_accuracy(pred: torch.Tensor, target: torch.Tensor, ks=(1, 5)) -> dict:
    """Can each predicted latent find its own true latent among the others?

    Judge by this, not by R^2 and not by eyeballed decodes. R^2 can look
    respectable while the model emits a generic average video, and retrieval
    cannot -- an average video is equally close to everything.

    The caveat, learned later and worth carrying: retrieval is only sensitive to
    what *discriminates* clips. Coarse colour and layout are shared across SSv2
    and carry no identity, so top1 tracks the detail band and not visual quality.
    A model can win 9 points of top1 on 1 point of full R^2. Quote it as evidence
    of information content, never as evidence the video looks right.
    """
    p = F.normalize(pred.flatten(1).float(), dim=1)
    t = F.normalize(target.flatten(1).float(), dim=1)
    sim = p @ t.T
    n = sim.shape[0]
    truth = torch.arange(n, device=sim.device)
    ranks = (sim > sim[truth, truth].unsqueeze(1)).sum(dim=1)
    out = {f"top{k}": float((ranks < k).float().mean()) for k in ks}
    out["median_rank"] = float(ranks.float().median()) + 1.0
    out["chance_top1"] = 1.0 / n
    return out


def standardized_spread(pred: torch.Tensor) -> float:
    """Per-channel standard deviation of predictions, in standardized space.

    1.0 means the predictions carry as much per-channel variance as the targets.
    A regressor trained on MSE drives this well below 1 -- that shrinkage *is*
    the collapse toward the conditional mean, measured directly.

    Read it alongside R^2, never instead of it. Spread is a magnitude, not an
    accuracy: correctly-scaled noise also scores 1.0.
    """
    flat = pred.reshape(pred.shape[0], pred.shape[1], -1).float()
    return float(flat.std(dim=2).mean())


def r2_standardized(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Fraction of the target's variance explained, in standardized space.

    Because the target is already unit-variance per channel, the residual mean
    square *is* ``1 - R^2``, which is why this is just ``1 - MSE``.
    """
    mse = float(((pred.float() - target.float()) ** 2).mean())
    return 1.0 - mse


def adaptor_loss(
    pred: torch.Tensor, target: torch.Tensor, cos_weight: float = 0.0
) -> torch.Tensor:
    """MSE, optionally with a cosine term.

    The cosine term is the cheap partial answer to the conditional-mean problem.
    MSE alone is minimized by the mean of the plausible targets, which is exactly
    what blurs the high-frequency band out. Cosine similarity is scale-invariant,
    so it rewards getting the *direction* right and does not pay the model to
    shrink toward the average. ``--cos-weight 3.0`` to ``5.0`` is the useful
    range; it sharpens, but it does not substitute for a decode-space or
    generative objective.
    """
    loss = F.mse_loss(pred, target)
    if cos_weight:
        p = F.normalize(pred.flatten(1), dim=1)
        t = F.normalize(target.flatten(1), dim=1)
        loss = loss + cos_weight * (1.0 - (p * t).sum(dim=1).mean())
    return loss


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def describe(model: Adaptor) -> str:
    """One-line summary for the run log."""
    n = count_parameters(model)
    return (
        f"{model.kind}: {n / 1e6:.2f}M parameters, "
        f"resample={model.resample} -> grid {model.target_grid}"
    )
