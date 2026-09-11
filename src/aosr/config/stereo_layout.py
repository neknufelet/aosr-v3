"""Validated stereo-layout policy loaded from its TOML SSOT."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from .paths import config_path


@dataclass(frozen=True)
class StereoLayoutConfig:
    """Runtime policy values shared by stereo geometry consumers."""

    source_wall_margin_m: float
    receiver_wall_margin_m: float
    min_source_source_m: float
    min_source_seat_m: float
    min_seat_pair_center_m: float
    min_front_depth_m: float
    search_min_front_depth_m: float
    speaker_front_wall_min_m: float
    speaker_front_wall_max_m: float
    speaker_lateral_wall_margin_m: float
    seat_rear_wall_margin_frac: float
    receiver_centerline_tol_m: float
    horizontal_angle_min_deg: float
    horizontal_angle_target_deg: float
    horizontal_angle_max_deg: float
    search_horizontal_angle_max_deg: float
    horizontal_angle_target_scale_deg: float
    pair_center_offset_max_m: float
    distance_imbalance_max: float


_CONFIG_PATH = config_path("stereo_layout.toml")
_POLICY_KEYS = frozenset(field.name for field in fields(StereoLayoutConfig))


def load_stereo_layout(path: str | Path | None = None) -> StereoLayoutConfig:
    """Load finite stereo policy values and reject schema drift."""
    resolved = _CONFIG_PATH if path is None else Path(path)
    with resolved.open("rb") as config_file:
        data = tomllib.load(config_file)

    unknown_sections = set(data) - {"stereo"}
    if unknown_sections:
        raise ValueError(
            f"{resolved} has unknown top-level keys: {sorted(unknown_sections)}"
        )
    try:
        section = data["stereo"]
    except KeyError as exc:
        raise ValueError(f"{resolved} must define [stereo]") from exc
    if not isinstance(section, dict):
        raise ValueError(f"{resolved} [stereo] must be a table")

    unknown_keys = set(section) - _POLICY_KEYS
    if unknown_keys:
        raise ValueError(
            f"{resolved} [stereo] has unknown keys: {sorted(unknown_keys)}"
        )
    missing_keys = _POLICY_KEYS - set(section)
    if missing_keys:
        raise ValueError(
            f"{resolved} [stereo] is missing keys: {sorted(missing_keys)}"
        )

    values: dict[str, float] = {}
    for key in _POLICY_KEYS:
        raw_value = section[key]
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise ValueError(f"{resolved} [stereo].{key} must be numeric")
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"{resolved} [stereo].{key} must be finite")
        values[key] = value
    return StereoLayoutConfig(**values)


STEREO_LAYOUT = load_stereo_layout()
