"""SCENE-1 v1 analytic loudspeaker-directivity SSOT.

This family is an explicitly versioned placeholder.  SCENE-3 replaces it with
measured Spinorama data.

Product default (SCENE-1.7, PO sign-off 2026-08-21): the CHORAS bridge resolves
directivity **ON** when ``DIRECTIVITY_ENABLED_KEY`` is absent from
``simulationSettings``.  The rollback is an explicit ``false`` in the payload,
which restores the complete pre-SCENE-1.7 OFF path; there is no hidden kill
switch further down the solver.  Lower-level library APIs are unaffected: they
keep their own ``directivity=None`` defaults.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

SpeakerType = Literal["bookshelf", "floorstanding", "inwall"]

DIRECTIVITY_MODEL_VERSION = "scene1-analytic-v1"
DIRECTIVITY_GOLDEN_VERSION = "scene1-directivity-golden-v3"
DIRECTIVITY_ENABLED_KEY = "aosr_speaker_directivity_enabled"
# SCENE-1.7 product default, PO sign-off 2026-08-21.  Single source of truth:
# the bridge must import this instead of writing a second boolean literal.
DIRECTIVITY_DEFAULT_ENABLED: bool = True
SPEAKER_TYPE_KEY = "aosr_speaker_type"
SPEAKER_WIDTH_OVERRIDE_KEY = "aosr_speaker_width_m"
PISTON_RADIUS_OVERRIDE_KEY = "aosr_speaker_piston_radius_m"
DEFAULT_SPEAKER_TYPE: SpeakerType = "bookshelf"
DIRECTIVITY_REAR_GAIN_MIN = 0.1
DIRECTIVITY_NORMALIZATION_GL_ORDER = 64
DIRECTIVITY_TOE_IN_ANCHOR = "fixed_receivers[0]"


@dataclass(frozen=True)
class SpeakerPreset:
    """D13a v1 enclosure and acoustic-centre preset, in metres."""

    width_m: float
    height_m: float
    depth_m: float
    acoustic_center_z_m: float
    piston_radius_m: float


@dataclass(frozen=True)
class SpeakerDirectivity:
    """The two resolved dimensions consumed by the analytic pressure pattern."""

    speaker_type: Literal["bookshelf", "floorstanding"]
    baffle_width_m: float
    piston_radius_m: float
    version: str = DIRECTIVITY_MODEL_VERSION


SPEAKER_PRESETS: Mapping[str, SpeakerPreset] = {
    "bookshelf": SpeakerPreset(0.20, 0.35, 0.28, 0.95, 0.065),
    "floorstanding": SpeakerPreset(0.25, 1.40, 0.35, 1.20, 0.083),
}
SPEAKER_TYPES: tuple[SpeakerType, ...] = ("bookshelf", "floorstanding", "inwall")


def _positive_override(value: float | None, *, label: str, fallback: float) -> float:
    if value is None:
        return fallback
    if isinstance(value, bool):
        raise ValueError(f"{label} 不接受布林值；必須是有限正數")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{label} must be finite and positive, got {value!r}")
    return resolved


def resolve_speaker_directivity(
    *,
    enabled: bool,
    speaker_type: str = DEFAULT_SPEAKER_TYPE,
    baffle_width_m: float | None = None,
    piston_radius_m: float | None = None,
) -> SpeakerDirectivity | None:
    """Resolve settings; OFF ignores unknown visuals, ON validates fail-closed.

    ``inwall`` is visual-only in v1.  An unknown type is ignored with a warning
    while directivity is OFF so a future visual preset cannot break legacy runs;
    the same value raises a Chinese fail-closed error when physics is ON.
    """

    normalized_type = str(speaker_type).strip().lower()
    if normalized_type not in SPEAKER_TYPES:
        if not enabled:
            warnings.warn(
                f"{SPEAKER_TYPE_KEY}={speaker_type!r} 不受支援；指向性開關為 OFF，已忽略",
                UserWarning,
                stacklevel=2,
            )
            return None
        raise ValueError(
            f"{SPEAKER_TYPE_KEY}={speaker_type!r} 不支援；指向性 ON 時必須是 {SPEAKER_TYPES}"
        )
    if not enabled or normalized_type == "inwall":
        return None
    preset = SPEAKER_PRESETS[normalized_type]
    return SpeakerDirectivity(
        speaker_type=normalized_type,
        baffle_width_m=_positive_override(
            baffle_width_m, label=SPEAKER_WIDTH_OVERRIDE_KEY, fallback=preset.width_m
        ),
        piston_radius_m=_positive_override(
            piston_radius_m,
            label=PISTON_RADIUS_OVERRIDE_KEY,
            fallback=preset.piston_radius_m,
        ),
    )
