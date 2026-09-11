"""Scoring upgrade configuration loaded from the TOML single source of truth."""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SpatialScoringConfig(_Frozen):
    """B4 high-frequency spatial rolloff settings."""

    hf_rolloff_fc_hz: float = Field(gt=0.0)


class AsymmetryScoringConfig(_Frozen):
    """Peak/dip asymmetry (BLIND-3): which side of the target curve hurts more.

    A room response sits above the target curve in some bands (peaks) and below
    it in others (dips). This setting is the PO's value judgement about which is
    worse — the Toole/Olive position is that resonant peaks are more audible
    because they ring, the opposing position is that cancellation dips cannot be
    recovered by EQ at all.

    ``peak_dip_bias`` is the dial, not the internal weight:

    ==========  ==================================  ================
    bias        meaning                             peak : dip
    ==========  ==================================  ================
    +3          most Toole-leaning                  3 : 1
    +1.5        leans toward punishing peaks        2 : 1
     0          symmetric                           1 : 1
    -1.5        leans toward punishing dips         1 : 2
    -3          most "dips are unrecoverable"       1 : 3
    ==========  ==================================  ================

    The old key was ``w_dip``, bounded ``[0, 1]``, which could not express
    "punish dips more" at all. ``w_dip`` survives as the internal parameter of
    :func:`lib.scoring.core.target_curve_deviation`; derive it with
    :func:`w_dip_from_bias` rather than restating the mapping.
    """

    peak_dip_bias: float = Field(ge=-3.0, le=3.0)


def w_dip_from_bias(peak_dip_bias: float) -> float:
    """Convert the PO-facing bias dial to the internal negative-residual weight.

    `_asym_spread` computes ``sqrt(mean(pos**2 + w_dip * neg**2))``. The square
    root matters: ``w_dip`` weights the SQUARED residual, so the ratio observed on
    the returned penalty is ``1 : sqrt(w_dip)``, not ``1 : w_dip``. The dial is
    defined on the penalty the optimiser actually sees, so ``bias=+3`` really does
    make an isolated peak cost 3x an equal-sized dip.

    Caveat: that ratio is exact only for an isolated single-sign deviation.
    :func:`lib.scoring.core.target_curve_deviation` removes each receiver's mean
    first, which redistributes residuals across both signs, so a real mixed
    response shows a smaller effective ratio (~1.45 at bias=+3 on the probe in
    上一代那一棵樹的一份 cross-review 報告). The dial is a direction and
    a magnitude, not a guarantee about any particular response.

    >>> w_dip_from_bias(0.0)
    1.0
    >>> round(w_dip_from_bias(3.0), 6)   # peaks hurt 3x as much
    0.111111
    >>> w_dip_from_bias(-3.0)            # dips hurt 3x as much
    9.0
    """
    if not -3.0 <= peak_dip_bias <= 3.0:
        raise ValueError(f"peak_dip_bias must lie in [-3, 3], got {peak_dip_bias}")
    ratio = 1.0 + 2.0 * abs(peak_dip_bias) / 3.0
    # Square it. `_asym_spread` returns sqrt(mean(pos**2 + w_dip * neg**2)), so
    # w_dip scales the SQUARED term and the ratio the caller actually observes on
    # the returned penalty is 1 : sqrt(w_dip). Mapping the dial straight onto
    # w_dip therefore delivered sqrt(3) ~= 1.73x at bias=+3 while the docs
    # promised 3x (caught by Codex cross-review R20260720-1). Squaring makes the
    # dial mean what it says on the value the optimiser minimises.
    if peak_dip_bias > 0.0:
        return 1.0 / ratio**2
    if peak_dip_bias < 0.0:
        return ratio**2
    return 1.0


class DecayScoringConfig(_Frozen):
    """Decay penalty weights.

    ``w_decay`` — B1 low-frequency modal-decay penalty (modal.py, modes below
    Schroeder). ``w_rt60`` — M15-P4 broadband per-band T60 deadband-hinge penalty
    vs the ITU-R BS.1116-3 target (broadband_decay.py). Both default-off/placeholder
    until listening calibration (S4); ``w_rt60=0.0`` keeps goldens bit-identical.
    """

    w_decay: float = Field(ge=0.0)
    w_rt60: float = Field(ge=0.0, default=0.0)


class SmoothingScoringConfig(_Frozen):
    """Fractional-octave power smoothing settings."""

    frac_oct: float = Field(gt=0.0)


class ScoringConfig(_Frozen):
    """Validated scoring upgrade settings.

    Provenance: the previous generation's scoring-upgrade design paper §2(B3/B4)（不在這個 repo 裡）.
    """

    spatial: SpatialScoringConfig
    asymmetry: AsymmetryScoringConfig
    decay: DecayScoringConfig
    smoothing: SmoothingScoringConfig


def load_scoring(path: str | Path) -> ScoringConfig:
    """Load and validate the ``scoring.toml`` in this package's ``data`` directory."""
    resolved = Path(path)
    with resolved.open("rb") as file:
        data = tomllib.load(file)
    return ScoringConfig(**data)
