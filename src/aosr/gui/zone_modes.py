"""每一區的做法換材料（#1105，#1082 第 1-3 段）：立面圖下的逐區表選了做法或材料，方案那一區該放什麼。

設計設定（``zone_design``）只記做法；計算與搜尋讀的是方案檔裡那一區的材料，所以選了之後方案那一區要跟著換：

- 「保留原牆」與「交給搜尋」：放原牆（搜尋區在方案裡先放原牆，見 1-6a ``gradient_plan``：固定與保留原牆都凍成
  ``scheme-material``）。
- 「固定這一組材料」選了材料庫的一筆：多層構造貼在原牆上（``MountedOnWall``，最後一層背後接原牆，1-5 的改善基準
  拆得回原牆）；固定阻抗與實測阻抗表本身就是整面的表面阻抗，直接換掉那一區，畫面點明「直接取代、不是貼上」。

原牆怎麼來（同 1-5 ``zone_result`` 那一條規則，只是這裡一區一區取）：現在是貼在原牆上的構造，取它記的原牆；
不是的話，設計設定指了原方案就照原方案蓋住這一區的那一塊；兩樣都沒有就看不出原牆，照現在的材料不動、照實說。
這一支只組材料、不算物理；材料庫讀不進來照講原因，不當成空的。
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from aosr.config.capabilities import CapabilityTable
from aosr.config.directivity_defaults import DirectivityDefaults
from aosr.gui.material_editor import plain_error, user_library_path
from aosr.gui.problem_text import checked_scheme
from aosr.gui.zone_design import NAME, WHOLE_WALL_KEY
from aosr.gui.zone_result import BaselineRefused, OriginalWalls, original_material, original_walls
from aosr.materials.construction_models import Assembly, MountedOnWall, SurfaceMaterial
from aosr.materials.library import FixedEntry, LibraryEntry, full_library
from aosr.physics.report_surfaces import RectangleZone
from aosr.physics.zone_fill import wall_size_m
from aosr.reporting.scheme import Scheme

_MATERIAL: TypeAdapter[SurfaceMaterial] = TypeAdapter(SurfaceMaterial)
WallName = Literal["floor", "ceiling", "x0", "xL", "y0", "yL"]
OriginSource = Literal["unwrapped", "original_scheme", "current", "unknown"]
UNKNOWN_ORIGIN = ("這一區現在的材料不是貼在原牆上的構造，看不出原牆是什麼；照現在的材料不動。"
                  "要換回原牆，請在上面「原方案」選切之前那一份方案")


class ZoneMaterialRequest(BaseModel):
    """換一區的材料：整份方案（網頁表單現在的樣子）、哪一面、哪一區、選了什麼、設計設定指的原方案。"""

    model_config = ConfigDict(extra="forbid")
    scheme: dict[str, object]
    wall: WallName
    zone: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    choice: Literal["original", "fixed"]
    material_id: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_]*$")
    original_scheme_id: str | None = None


class ZoneMaterialError(ValueError):
    """換不了：白話一句、HTTP 狀態碼。"""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class Origin:
    material: SurfaceMaterial
    source: OriginSource


def _zone_index(scheme: Scheme, wall: str, zone: str) -> tuple[int, tuple[tuple[float, float], tuple[float, float]]]:
    zones = (scheme.scene.surface_zones_by_wall or {}).get(wall)
    if zones is None:
        raise ZoneMaterialError(f"方案沒有{wall}的分區清單；先把這一面切過或重新打開方案")
    size = wall_size_m(scheme.scene.room_m, wall)
    for index, item in enumerate(zones):
        key = item.id if isinstance(item, RectangleZone) else WHOLE_WALL_KEY
        if key == zone:
            box = (item.u_m, item.v_m) if isinstance(item, RectangleZone) else ((0.0, size[0]), (0.0, size[1]))
            return index, box
    raise ZoneMaterialError(f"{wall}沒有「{zone}」這一區；畫面跟方案對不上，請重新打開方案")


def origin(current: SurfaceMaterial, original: OriginalWalls | None, wall: str, zone: str,
           box: tuple[tuple[float, float], tuple[float, float]], replacements: Sequence[SurfaceMaterial] = ()) -> Origin:
    """這一區的原牆：貼在原牆上的取它記的原牆；不是的話照原方案；都沒有就當現在的材料是原牆（``current``，
    剛切出來的區就是這樣）——除非現在的材料正是材料庫裡直接取代牆面的那幾種（``replacements``：固定阻抗、
    實測表），那就看不出原牆（``unknown``），照現在的材料不動、照實說。"""
    if isinstance(current, MountedOnWall):
        return Origin(current.wall, "unwrapped")
    if original is not None:
        try:
            return Origin(original_material(original, wall, zone, box), "original_scheme")
        except BaselineRefused as exc:
            raise ZoneMaterialError(str(exc)) from exc
    return Origin(current, "unknown" if current in replacements else "current")


def zone_material(current: SurfaceMaterial, base: Origin, entry: FixedEntry | None) -> tuple[SurfaceMaterial, str]:
    """選了之後這一區放什麼，和一句給畫面的說明。``entry`` 是 None 表示放原牆。"""
    if entry is None:
        return base.material, UNKNOWN_ORIGIN if base.source == "unknown" else "這一區放回原牆"
    chosen = entry.surface_material()
    if isinstance(chosen, Assembly):
        if isinstance(base.material, MountedOnWall):
            raise ZoneMaterialError("原方案這一區本身就是貼在原牆上的構造，不能再貼一層；原方案請選貼之前的那一份")
        mounted = MountedOnWall(layers=chosen.layers, wall=base.material)
        note = f"「{entry.label}」貼在原牆上（最後一層背後接原牆）"
        if base.source == "unknown":
            note += "；這一區現在是直接取代牆面的材料、看不出原牆，就把它當原牆；要換別的原牆，請在上面「原方案」選切之前那一份方案"
        return mounted, note
    return chosen, f"「{entry.label}」是整面的表面阻抗，直接取代這一區的牆（不是貼上去）"


class ZoneMaterialHandlers:
    """換一區材料的入口；方案與材料庫只讀，回新的材料由網頁原地換進方案。"""

    def __init__(self, data_dir: Path, lock: AbstractContextManager[object], *, capabilities: CapabilityTable,
                 directivity: DirectivityDefaults) -> None:
        self.data_dir = data_dir
        self.lock = lock
        self.capabilities = capabilities
        self.directivity = directivity

    def _library(self) -> tuple[LibraryEntry, ...]:
        try:
            return full_library(user_library_path(self.data_dir))
        except (OSError, ValueError, ValidationError) as exc:
            detail = plain_error(exc) if isinstance(exc, (ValidationError, json.JSONDecodeError)) else str(exc)
            raise ZoneMaterialError(f"使用者材料庫讀不進來，選不了材料：{detail}") from exc

    def _entry(self, entries: tuple[LibraryEntry, ...], material_id: str) -> FixedEntry:
        found = next((entry for entry in entries if entry.id == material_id), None)
        if found is None:
            raise ZoneMaterialError(f"材料庫裡沒有「{material_id}」這一筆", 404)
        if not isinstance(found, FixedEntry):
            raise ZoneMaterialError(f"「{found.label}」是可最佳化材料：只能放進「交給搜尋」的准用清單，不能指定成固定材料")
        return found

    def _original(self, name: str | None, design: Scheme) -> OriginalWalls | None:
        if name is None:
            return None
        if not NAME.fullmatch(name):
            raise ZoneMaterialError("原方案代號只收英數字、底線與連字號")
        path = self.data_dir / "schemes" / f"{name}.json"
        with self.lock:
            if not path.is_file():
                raise ZoneMaterialError(f"指的原方案「{name}」不在了；在上面「原方案」重選", 404)
            text = path.read_text(encoding="utf-8")
        try:
            return original_walls(Scheme.model_validate_json(text), design, capabilities=self.capabilities,
                                  directivity=self.directivity)
        except (ValueError, BaselineRefused) as exc:
            raise ZoneMaterialError(str(exc).splitlines()[0]) from exc

    def change(self, body: ZoneMaterialRequest) -> dict[str, object]:
        if body.choice == "fixed" and body.material_id is None:
            raise ZoneMaterialError("選固定材料要給材料代號")
        try:
            scheme = checked_scheme(body.scheme)
        except ValueError as exc:
            raise ZoneMaterialError("方案表單還有格子沒填好或填錯，先改好再選這一區的材料（按「檢查」看是哪一格）") from exc
        index, box = _zone_index(scheme, body.wall, body.zone)
        current = (scheme.scene.surface_zones_by_wall or {})[body.wall][index].material
        entries = self._library()
        entry = self._entry(entries, body.material_id) if body.choice == "fixed" and body.material_id else None
        replacements = [material for item in entries if isinstance(item, FixedEntry)
                        if not isinstance(material := item.surface_material(), Assembly)]
        base = origin(current, self._original(body.original_scheme_id, scheme), body.wall, body.zone, box, replacements)
        material, note = zone_material(current, base, entry)
        return {"material": _MATERIAL.dump_python(material, mode="json"), "note": note, "origin": base.source}


def zone_material_routes(data_dir: Path, lock: AbstractContextManager[object], *, capabilities: CapabilityTable,
                         directivity: DirectivityDefaults) -> list[Route]:
    handlers = ZoneMaterialHandlers(data_dir, lock, capabilities=capabilities, directivity=directivity)

    async def change(request: Request) -> Response:
        try:
            body = ZoneMaterialRequest.model_validate(await request.json())
        except (ValidationError, ValueError):
            return JSONResponse({"error": "換分區材料的請求格式不完整：要有方案、哪一面、哪一區與選了什麼"}, status_code=400)
        try:
            return JSONResponse(await run_in_threadpool(handlers.change, body))
        except ZoneMaterialError as exc:
            return JSONResponse({"error": str(exc)}, status_code=exc.status)

    return [Route("/api/zone-material", change, methods=["POST"])]
