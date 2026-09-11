"""Versioned, probe-only configuration for the SCORING v2 sidecar."""

from __future__ import annotations

import hashlib
import json
import math
import tomllib
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from .paths import config_path
from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


SCORING_V2_SCHEMA_VERSION = "aosr.scoring.v2.alpha1"
LOCAL_CROSS_7_ORDER = ("center", "front", "back", "left", "right", "up", "down")
_DEFAULT_CONFIG_PATH = config_path("scoring_v2.toml")

FiniteTomlFloat = Annotated[float, Field(strict=True, allow_inf_nan=False)]


class ScoringV2ConfigError(ValueError):
    """Raised when the alpha1 scoring-v2 configuration is invalid."""


class ProbeOnlyScoringV2ConfigError(ScoringV2ConfigError):
    """Raised when a probe-only policy is requested in production mode."""


class DecisionState(StrEnum):
    """The four non-interchangeable states used by the R0 contract."""

    LOCKED = "locked"
    PROVISIONAL = "provisional"
    UNRESOLVED = "unresolved"
    UNAVAILABLE = "unavailable"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class ResponseConfig(_Frozen):
    equivalence_db: FiniteTomlFloat = Field(gt=0.0)
    equivalence_sweep_db: tuple[FiniteTomlFloat, ...]

    @model_validator(mode="after")
    def validate_sweep(self) -> ResponseConfig:
        if not self.equivalence_sweep_db or any(
            value <= 0.0 for value in self.equivalence_sweep_db
        ):
            raise ValueError("equivalence_sweep_db must contain positive values")
        if tuple(sorted(self.equivalence_sweep_db)) != self.equivalence_sweep_db:
            raise ValueError("equivalence_sweep_db must be strictly increasing")
        if len(set(self.equivalence_sweep_db)) != len(self.equivalence_sweep_db):
            raise ValueError("equivalence_sweep_db must not contain duplicates")
        if self.equivalence_db not in self.equivalence_sweep_db:
            raise ValueError("equivalence_db must be one of equivalence_sweep_db")
        return self


class FemConfig(_Frozen):
    low_hz: FiniteTomlFloat = Field(gt=0.0)
    high_hz: FiniteTomlFloat = Field(gt=0.0)
    evaluation_axis: Literal["log-1-24-octave"]


class IsmConfig(_Frozen):
    low_hz: FiniteTomlFloat = Field(gt=0.0)
    high_hz: FiniteTomlFloat = Field(gt=0.0)
    max_delay_ms: FiniteTomlFloat = Field(gt=0.0)
    samples_per_fringe: int = Field(strict=True, ge=1)


class ListeningCrossConfig(_Frozen):
    profile: Literal["local_cross_7"]
    radius_m: FiniteTomlFloat = Field(gt=0.0)
    order: tuple[Literal["center", "front", "back", "left", "right", "up", "down"], ...]
    frame_tolerance: FiniteTomlFloat = Field(gt=0.0)
    min_horizontal_projection_norm: FiniteTomlFloat = Field(gt=0.0)

    @model_validator(mode="after")
    def validate_local_cross(self) -> ListeningCrossConfig:
        if self.order != LOCAL_CROSS_7_ORDER:
            raise ValueError(f"local_cross_7 order must be {LOCAL_CROSS_7_ORDER}")
        return self


class DipEligibilityConfig(_Frozen):
    min_width_oct: FiniteTomlFloat = Field(gt=0.0)
    required_points: int = Field(strict=True, ge=1)
    require_center: bool

    @model_validator(mode="after")
    def validate_alpha1_rules(self) -> DipEligibilityConfig:
        if not math.isclose(self.min_width_oct, 1.0 / 6.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("dip_eligibility.min_width_oct must represent 1/6 octave")
        if self.required_points != 3:
            raise ValueError("dip_eligibility.required_points must be 3")
        if not self.require_center:
            raise ValueError("dip_eligibility.require_center must be true")
        return self


class RfzV2Config(_Frozen):
    window_ms: FiniteTomlFloat = Field(gt=0.0)
    receiver_role: Literal["center"]
    band_status: Literal["unresolved"]
    threshold_status: Literal["unresolved"]


class ScoringV2Config(_Frozen):
    """Validated alpha1 contract. It is deliberately not a production policy."""

    # 這一格是 Literal，而 Literal 只吃字面值（mypy 嚴格模式對 `Literal[常數]` 判
    # 「Parameter 1 of Literal[...] is invalid」）。所以字面寫在這裡，上面的常數
    # `SCORING_V2_SCHEMA_VERSION` 留著給呼叫端用；兩者一致由考卷
    # `tests/engine/test_config_cut2_scoring_v2.py` 釘住。
    schema_version: Literal["aosr.scoring.v2.alpha1"]
    status: Literal["probe-only"]
    response: ResponseConfig
    fem: FemConfig
    ism: IsmConfig
    listening_cross: ListeningCrossConfig
    dip_eligibility: DipEligibilityConfig
    rfz_v2: RfzV2Config

    _DECISION_STATES: ClassVar[Mapping[str, DecisionState]] = {
        "schema_version": DecisionState.LOCKED,
        "status": DecisionState.LOCKED,
        "fem.low_hz": DecisionState.LOCKED,
        "fem.high_hz": DecisionState.LOCKED,
        "ism.low_hz": DecisionState.LOCKED,
        "ism.high_hz": DecisionState.LOCKED,
        "listening_cross.profile": DecisionState.LOCKED,
        "listening_cross.radius_m": DecisionState.LOCKED,
        "listening_cross.order": DecisionState.LOCKED,
        "dip_eligibility.min_width_oct": DecisionState.LOCKED,
        "dip_eligibility.required_points": DecisionState.LOCKED,
        "dip_eligibility.require_center": DecisionState.LOCKED,
        "rfz_v2.window_ms": DecisionState.LOCKED,
        "rfz_v2.receiver_role": DecisionState.LOCKED,
        "response.equivalence_db": DecisionState.PROVISIONAL,
        "response.equivalence_sweep_db": DecisionState.PROVISIONAL,
        "fem.evaluation_axis": DecisionState.PROVISIONAL,
        "ism.samples_per_fringe": DecisionState.PROVISIONAL,
        "peak_deadband_db": DecisionState.UNRESOLVED,
        "dip_deadband_db": DecisionState.UNRESOLVED,
        "rfz_v2.band": DecisionState.UNRESOLVED,
        "rfz_v2.threshold": DecisionState.UNRESOLVED,
        "runtime.solver_path_coverage": DecisionState.UNAVAILABLE,
    }

    @model_validator(mode="after")
    def validate_alpha1_boundaries(self) -> ScoringV2Config:
        if self.fem.low_hz >= self.fem.high_hz:
            raise ValueError("FEM low_hz must be below high_hz")
        if self.ism.low_hz >= self.ism.high_hz:
            raise ValueError("ISM low_hz must be below high_hz")
        if not math.isclose(self.fem.high_hz, self.ism.low_hz, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("FEM and ISM frequency ranges must join at their shared seam")
        if not math.isclose(
            self.rfz_v2.window_ms, self.ism.max_delay_ms, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("RFZ window and ISM max_delay_ms must agree")
        return self

    def decision_state(self, field_name: str) -> DecisionState:
        """Return the explicit status of an alpha1 field or runtime outcome."""
        try:
            return self._DECISION_STATES[field_name]
        except KeyError as exc:
            raise KeyError(f"unknown scoring-v2 decision field: {field_name}") from exc

    def canonical_json(self) -> str:
        """Return a stable JSON representation suitable for provenance hashing."""
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    def fingerprint(self) -> str:
        """Return the SHA-256 fingerprint of this exact alpha1 config."""
        return "sha256:" + hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def load_scoring_v2(
    path: str | Path | None = None,
    *,
    mode: Literal["sidecar", "production"] = "sidecar",
) -> ScoringV2Config:
    """Load alpha1 and fail closed if a production caller requests it."""
    if mode not in ("sidecar", "production"):
        raise ScoringV2ConfigError(
            f"unsupported scoring-v2 load mode: {mode!r}; expected 'sidecar' or 'production'"
        )
    resolved = _DEFAULT_CONFIG_PATH if path is None else Path(path)
    try:
        with resolved.open("rb") as config_file:
            data = tomllib.load(config_file)
        config = ScoringV2Config.model_validate(data)
    except (OSError, tomllib.TOMLDecodeError, TypeError, ValueError) as exc:
        raise ScoringV2ConfigError(
            f"{resolved}: invalid scoring-v2 configuration: {exc}"
        ) from exc
    if mode == "production":
        raise ProbeOnlyScoringV2ConfigError(
            f"{resolved} has status={config.status!r}; "
            "probe-only config is not production-loadable"
        )
    return config
