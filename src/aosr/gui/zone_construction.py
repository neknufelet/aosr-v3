"""分區裡沒有材料代號的裸構造：換料、限制檢查與開跑都照同位置的原方案辨認。"""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import TypeAdapter, ValidationError

from aosr.geometry.shoebox import Room
from aosr.gui.labels import WALLS
from aosr.gui.zone_design import NAME
from aosr.gui.zone_result import BaselineRefused, OriginalWalls, original_material
from aosr.materials.construction_models import Assembly, SurfaceMaterial
from aosr.physics.report_surfaces import ZONE_WALL_FIELD, ZonesByWall, validate_surface_partition
from aosr.reporting.scheme import Scheme

_ZONES = TypeAdapter(ZonesByWall)
Bounds = tuple[tuple[float, float], tuple[float, float]]


@dataclass(frozen=True)
class OriginalProblem:
    reason: str
    status: int = 400
    error: str = ""


ComparisonWalls = OriginalWalls | OriginalProblem
OriginalReader = Callable[[str], Scheme | ComparisonWalls | None]


@dataclass(frozen=True)
class SavedOriginal:
    scheme: Scheme | OriginalProblem
    walls: ComparisonWalls


def scheme_walls(scheme: Scheme, zones: ZonesByWall | None) -> ComparisonWalls:
    if zones is None:
        return OriginalProblem("原方案是六個數字的舊格式")
    return OriginalWalls(scheme.scheme_id, scheme.scene.room_m, zones)


def _remaining_walls(document: bytes, name: str) -> OriginalWalls:
    """原方案缺面時，只取留下的面；這些區仍須通過相同的分區格式、座標與覆蓋檢查。"""
    scene = json.loads(document)["scene"]
    room = TypeAdapter(Room).validate_python(scene["room_m"])
    zones = _ZONES.validate_python(scene[ZONE_WALL_FIELD])
    if not zones.keys() <= WALLS.keys():
        raise ValueError("原方案有不認識的牆面")
    validate_surface_partition(room, zones)
    return OriginalWalls(name, room, zones)


def read_original(data_dir: Path, name: str, lock: AbstractContextManager[object]) -> SavedOriginal | OriginalProblem:
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
        errors = exc.errors(include_input=False)
        reasons = "、".join(str(error["msg"]) for error in errors)
        problem = OriginalProblem(f"原方案「{name}」讀不進來：{reasons}", error=str(exc).splitlines()[0])
        if all(error["type"] == "wall_input" and error["msg"].startswith("缺少牆面") for error in errors):
            try:
                return SavedOriginal(problem, _remaining_walls(document, name))
            except (ValueError, KeyError):
                pass
        return problem
    zones = scheme.scene.surface_zones_by_wall
    if zones is None:
        # 驗方案會把六面整面固定阻抗折成六個數字。只有檔案原本寫了分區，才有原區可比。
        recorded = json.loads(document)["scene"].get(ZONE_WALL_FIELD)
        if recorded is not None:
            zones = _ZONES.validate_python(recorded)
    return SavedOriginal(scheme, scheme_walls(scheme, zones))


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


def _different_from_original(original: ComparisonWalls | None, room: Room, wall: str, key: str,
                             box: Bounds | None, material: Assembly) -> ConstructionComparison:
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
