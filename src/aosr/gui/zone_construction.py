"""分區裡沒有材料代號的裸構造：換料、限制檢查與開跑都照同位置的原方案辨認。"""
from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import ValidationError

from aosr.geometry.shoebox import Room
from aosr.gui.labels import WALLS
from aosr.gui.problem_text import plain_problems
from aosr.gui.zone_design import NAME
from aosr.gui.zone_result import BaselineRefused, OriginalWalls, original_material
from aosr.materials.construction_models import Assembly, SurfaceMaterial
from aosr.reporting.scheme import Scheme
from aosr.reporting.validation import SchemeProblem

Bounds = tuple[tuple[float, float], tuple[float, float]]


@dataclass(frozen=True)
class OriginalProblem:
    reason: str
    status: int = 400
    error: str = ""


@dataclass(frozen=True)
class ImpedanceWalls:
    """原檔是六個阻抗數字；辨認裸構造只需要核對房間尺寸。"""

    room_m: Room


ComparisonWalls = OriginalWalls | ImpedanceWalls | OriginalProblem
OriginalReader = Callable[[str], ComparisonWalls]
_Walls = TypeVar("_Walls", OriginalWalls, OriginalWalls | ImpedanceWalls)


def comparison_walls(scheme: Scheme) -> OriginalWalls | ImpedanceWalls:
    zones = scheme.scene.surface_zones_by_wall
    if zones is None:
        numbers = scheme.scene.impedance_pa_s_per_m_by_wall or {}
        if numbers.keys() != WALLS.keys() or not all(math.isfinite(value) and value > 0 for value in numbers.values()):
            raise BaselineRefused(f"原方案「{scheme.scheme_id}」讀不進來：六面牆各要填一個大於零的有限阻抗")
        return ImpedanceWalls(scheme.scene.room_m)
    return OriginalWalls(scheme.scheme_id, scheme.scene.room_m, zones)


def read_original(data_dir: Path, name: str, lock: AbstractContextManager[object],
                  resolve: Callable[[Scheme], _Walls]) -> _Walls | OriginalProblem:
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
        scheme = Scheme.model_validate_json(document)
    except ValidationError as exc:
        try:
            fields = json.loads(document)
        except (ValueError, UnicodeError):
            return OriginalProblem(f"原方案「{name}」讀不進來：檔案格式不對", error=str(exc).splitlines()[0])
        problems = [SchemeProblem(".".join(map(str, error["loc"])), error["msg"].removeprefix("Value error, "))
                    for error in exc.errors(include_input=False)]
        reasons = "、".join(str(row["text"]) for row in plain_problems(problems, fields))
        return OriginalProblem(f"原方案「{name}」讀不進來：{reasons}", error=str(exc).splitlines()[0])
    try:
        return resolve(scheme)
    except BaselineRefused as exc:
        return OriginalProblem(str(exc), error=str(exc).partition("\n")[0])


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
                            box: Bounds | None, material: SurfaceMaterial | None) -> ConstructionComparison:
    """比較裸構造與蓋住這一區的原牆。能比時帶回原牆材料，無法比較時帶回原因。"""
    if original is None:
        return ConstructionComparison("unknown", reason="沒指原方案")
    if isinstance(original, OriginalProblem):
        return ConstructionComparison("unknown", reason=original.reason)
    if original.room_m != room:
        return ConstructionComparison("unknown", reason="原方案的房間尺寸跟設計不同")
    if isinstance(original, ImpedanceWalls):
        return ConstructionComparison("picked")
    if box is None:
        return ConstructionComparison("unknown", reason="這一區的座標讀不出來")
    try:
        base = original_material(original, wall, key, box)
    except BaselineRefused as exc:
        return ConstructionComparison("unknown", reason=str(exc))
    return ConstructionComparison("original" if base == material else "picked", base)


@dataclass(frozen=True)
class ConstructionPick:
    identifier: str | None
    comparison: ConstructionComparison
    mismatch: str = ""


def construction_pick(material: SurfaceMaterial | None, standalone: Mapping[str, object], recorded: str | None,
                      original: Callable[[], ComparisonWalls | None], room: Room, wall: str, key: str,
                      box: Bounds | None) -> ConstructionPick | None:
    """先比原牆。舊檔沒有選料代號也能據此認出更換過的裸構造。"""
    if recorded is None and not isinstance(material, Assembly):
        return None
    identifier = matching_construction(material, standalone)
    comparison = different_from_original(original(), room, wall, key, box, material)
    if comparison.state == "unknown" and recorded is None and identifier is None:
        return None
    if comparison.state == "unknown" and recorded is not None:
        if isinstance(material, Assembly) and standalone.get(recorded) == material:
            return ConstructionPick(recorded, ConstructionComparison("picked", reason=comparison.reason))
        return ConstructionPick(identifier, comparison, recorded)
    return ConstructionPick(identifier, comparison)


def mismatch_words(label: str) -> str:
    return f"設計設定記的那一筆「{label}」跟這一區現在的材料對不上"
