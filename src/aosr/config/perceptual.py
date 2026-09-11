"""Perceptual thresholds layer — frozen, calibrated to measurements (design §8, §12.5).

These encode the subjective↔objective bridge (target curve, JND table, RFZ
window, RT60 model/tolerance, modal-decay and ITDG anchors). The TOML stores
scene profiles; :func:`load_perceptual` resolves one profile and returns the
same flat, caller-facing shape used by the rest of the config layer.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .early_reflection import EarlyReflectionConfig

PositiveFloat = Annotated[float, Field(gt=0.0)]
MATERIAL_RFZ_PROFILE = "material_rfz"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class TargetCurve(_Frozen):
    slope_db_oct: float = Field(description="House/Harman-style target slope (≈ -0.6..-0.8)")


class RfzProfileConfig(_Frozen):
    threshold_db: float
    role: Literal["soft_target", "hard_gate"]
    window_ms: PositiveFloat | None = None
    window_mode: Literal["dynamic"] | None = None

    @model_validator(mode="after")
    def validate_exactly_one_window(self) -> Self:
        if (self.window_ms is None) == (self.window_mode is None):
            raise ValueError("rfz requires exactly one of window_ms or window_mode")
        return self


class Rt60Target(_Frozen):
    """M15-P4 broadband decay target (ITU-R BS.1116-3 §8.2.3.1).

    A SINGLE nominal ``Tm = nominal_coeff * (V / volume_ref_m3)^(1/3)`` s (the
    200 Hz–4 kHz average), PLUS a frequency tolerance DEADBAND — NOT a per-octave
    target curve. The scored band at frequency ``f`` is
    ``[Tm - tol_lo_s(f), Tm + tol_hi_s(f)]``; ``tol_hi_s`` / ``tol_lo_s`` are
    POSITIVE magnitudes (added above / subtracted below ``Tm``), log-f interpolated
    from the ``hz`` anchors. Defaults reproduce ITU Figure 1: ±0.05 s in 200 Hz–4 kHz,
    upper widening to +0.30 s at 63 Hz (lower stays −0.05 s), ±0.10 s at 8 kHz.
    """

    nominal_coeff: PositiveFloat
    volume_ref_m3: PositiveFloat
    hz: tuple[PositiveFloat, ...] = Field(min_length=2)
    tol_hi_s: tuple[PositiveFloat, ...] = Field(min_length=2)
    tol_lo_s: tuple[PositiveFloat, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_axis_lengths(self) -> Self:
        if not (len(self.hz) == len(self.tol_hi_s) == len(self.tol_lo_s)):
            raise ValueError(
                "rt60.target hz, tol_hi_s and tol_lo_s must have equal lengths"
            )
        return self


class Rt60Config(_Frozen):
    model: str
    tol_s: float = Field(gt=0.0)
    target: Rt60Target


class JndConfig(_Frozen):
    edt_pct: float
    c80_db: float
    c50_db: float
    d50: float
    g_db: float
    sti: float


class ModalDecayCurve(_Frozen):
    hz: tuple[PositiveFloat, ...] = Field(min_length=1)
    t60_s: tuple[PositiveFloat, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_axis_lengths(self) -> Self:
        if len(self.hz) != len(self.t60_s):
            raise ValueError("modal-decay curve hz and t60_s must have equal lengths")
        return self


class ModalDecayConfig(_Frozen):
    curves: dict[str, ModalDecayCurve]


class ProfileConfig(_Frozen):
    rfz: RfzProfileConfig
    modal_decay_level: str

    @model_validator(mode="before")
    @classmethod
    def flatten_modal_decay_level(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value

        data: dict[str, object] = dict(value)
        selection = data.pop("modal_decay", None)
        if not isinstance(selection, dict) or set(selection) != {"level"}:
            raise ValueError("profile modal_decay must contain exactly one 'level'")
        data["modal_decay_level"] = selection["level"]
        return data


class ItdgConfig(_Frozen):
    target_ms: float | None = None


class PerceptualProfilesConfig(_Frozen):
    default_profile: str
    target_curve: TargetCurve
    rt60: Rt60Config
    jnd: JndConfig
    modal_decay: ModalDecayConfig
    itdg: ItdgConfig = ItdgConfig()
    early_reflection: EarlyReflectionConfig
    profiles: dict[str, ProfileConfig]

    @model_validator(mode="after")
    def validate_profile_references(self) -> Self:
        if self.default_profile not in self.profiles:
            raise ValueError(
                f"default_profile {self.default_profile!r} is not present in profiles"
            )
        curve_names = set(self.modal_decay.curves)
        unknown_levels = {
            profile.modal_decay_level
            for profile in self.profiles.values()
            if profile.modal_decay_level not in curve_names
        }
        if unknown_levels:
            raise ValueError(
                "profile modal_decay levels missing from modal_decay.curves: "
                + ", ".join(sorted(unknown_levels))
            )
        return self


class PerceptualConfig(_Frozen):
    """Resolved, flat perceptual config for one active scene profile."""

    active_profile: str
    target_curve: TargetCurve
    rfz: RfzProfileConfig
    rt60: Rt60Config
    jnd: JndConfig
    modal_decay: ModalDecayCurve
    itdg: ItdgConfig
    # P5.4 spec D6 — required, no default (a default would duplicate 15/10
    # outside the TOML SSOT). Model lives in lib.zoning (JAX-free chain).
    early_reflection: EarlyReflectionConfig


def load_perceptual(
    path: str | Path, *, profile: str | None = None
) -> PerceptualConfig:
    """Load, validate, and resolve one profile from this package's ``data`` directory (``perceptual.toml``)."""
    with open(path, "rb") as f:  # noqa: PTH123  # expires=2026-12-08 reason=與上一代逐字相同的開檔寫法——`open(None)` 的 TypeError 訊息是行為契約，`Path(path).open(...)` 講的是另一句（找碴第二輪）
        data = tomllib.load(f)
    source = PerceptualProfilesConfig(**data)

    active_profile = source.default_profile if profile is None else profile
    if active_profile not in source.profiles:
        available = ", ".join(sorted(source.profiles))
        raise ValueError(
            f"unknown perceptual profile {active_profile!r}; available profiles: {available}"
        )

    selected = source.profiles[active_profile]
    return PerceptualConfig(
        active_profile=active_profile,
        target_curve=source.target_curve,
        rfz=selected.rfz,
        rt60=source.rt60,
        jnd=source.jnd,
        modal_decay=source.modal_decay.curves[selected.modal_decay_level],
        itdg=source.itdg,
        early_reflection=source.early_reflection,
    )


def load_material_rfz_profile(path: str | Path) -> RfzProfileConfig:
    """Load the dedicated material-loss RFZ profile from the perceptual SSOT."""

    return load_perceptual(path, profile=MATERIAL_RFZ_PROFILE).rfz
