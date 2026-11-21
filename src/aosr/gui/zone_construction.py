"""分區裡沒有材料代號的裸構造：換料、限制檢查與開跑都照同位置的原方案辨認。"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from aosr.geometry.shoebox import Room
from aosr.gui.labels import WALLS
from aosr.gui.zone_design import NAME
from aosr.gui.zone_result import BaselineRefused, OriginalWalls, original_material
from aosr.materials.construction_models import Assembly, SurfaceMaterial
from aosr.reporting.scheme import Scheme

Bounds = tuple[tuple[float, float], tuple[float, float]]


@dataclass(frozen=True)
class OriginalProblem:
    reason: str
    status: int = 400
    error: str = ""


ComparisonWalls = OriginalWalls | OriginalProblem
OriginalReader = Callable[[str, Mapping[str, object]], ComparisonWalls]


def read_original(data_dir: Path, name: str, lock: AbstractContextManager[object]) -> Scheme | OriginalProblem:
    """只有整份原方案通過驗證，才能用它的牆辨認構造。"""
    if not NAME.fullmatch(name):
        return OriginalProblem("原方案代號只收英數字、底線與連字號")
    try:
        with lock:
            document = (data_dir / "schemes" / f"{name}.json").read_bytes()
    except FileNotFoundError:
        return OriginalProblem(f"指的原方案「{name}」不在了；在上面「原方案」重選", 404)
    except OSError:
        return OriginalProblem(f"原方案「{name}」讀不進來")
    try:
        return Scheme.model_validate_json(document)
    except ValidationError as exc:
        reasons = "、".join(f"{'.'.join(map(str, error['loc']))} {error['msg']}".strip() for error in exc.errors(include_input=False))
        return OriginalProblem(f"原方案「{name}」讀不進來：{reasons}", error=str(exc).splitlines()[0])


def matching_construction(material: SurfaceMaterial | None, standalone: Mapping[str, object]) -> str | None:
    """只比材料庫裡剛背、空氣背的構造，含背後逐欄相同。重複的取材料庫順序第一筆。"""
    if isinstance(material, Assembly):
        return next((identifier for identifier, surface in standalone.items() if material == surface), None)
    return None


@dataclass(frozen=True)
class ConstructionComparison:
    state: Literal["picked", "unknown", "original"]
    original: SurfaceMaterial | None = None
    reason: str = ""


def different_from_original(original: ComparisonWalls | None, room: Room, wall: str, key: str,
                            box: Bounds | None, material: Assembly) -> ConstructionComparison:
    """比較裸構造與蓋住這一區的原牆。能比時帶回原牆材料，無法比較時帶回原因。"""
    if original is None:
        return ConstructionComparison("unknown", reason="沒指原方案")
    if isinstance(original, OriginalProblem):
        return ConstructionComparison("unknown", reason=original.reason)
    if original.room_m != room:
        return ConstructionComparison("unknown", reason="原方案的房間尺寸跟設計不同")
    if wall not in original.zones:
        return ConstructionComparison("unknown", reason=f"原方案缺少牆面「{WALLS[wall]}」")
    if box is None:
        return ConstructionComparison("unknown", reason="這一區的座標讀不出來")
    try:
        base = original_material(original, wall, key, box)
    except BaselineRefused as exc:
        return ConstructionComparison("unknown", reason=str(exc))
    return ConstructionComparison("original" if base == material else "picked", base)
