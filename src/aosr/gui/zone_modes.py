"""每一區的做法換材料（#1105，#1082 第 1-3 段）：立面圖下的逐區表選了做法或材料，方案那一區該放什麼。

設計設定（``zone_design``）只記做法；計算與搜尋讀的是方案檔裡那一區的材料，所以選了之後方案那一區要跟著換：

- 「保留原牆」與「交給搜尋」：放原牆（搜尋區在方案裡先放原牆，見 1-6a ``gradient_plan``：固定與保留原牆都凍成
  ``scheme-material``）。
- 「固定這一組材料」選了材料庫的一筆：多層構造貼在原牆上（``MountedOnWall``，最後一層背後接原牆，1-5 的改善基準
  拆得回原牆）；固定阻抗與實測阻抗表本身就是整面的表面阻抗，直接換掉那一區，畫面點明「直接取代、不是貼上」。
  選的多層構造跟這一區原牆一模一樣（例如天花原牆本身就是那一套吊頂）：不貼上去疊兩層，放原牆、回
  ``same_as_original``，網頁把做法改成「不貼，保留原樣」（#1175、#1176）。

給畫面的那一句照面叫（#1176 用語表）：牆面「原牆」、地板「原本的地板」、天花「原本的天花」（天花地板不叫原牆）；
下面說明與程式裡的「原牆」是通稱，六面都算。

原牆怎麼來（同 1-5 ``zone_result`` 那一條規則、同一個先後，只是這裡一區一區取）：設計設定指了原方案就照原方案蓋住
這一區的那一塊；沒指時，現在是貼在原牆上的構造就取它記的原牆；都沒有就當現在的材料是原牆，只有它正是材料庫裡
直接取代牆面的那幾種時才說看不出原牆、照現在的材料不動。
這一支只組材料、不算物理；材料庫讀不進來照講原因，不當成空的。

一次換好幾區（#1176，``/api/zone-material/batch``）：方案、材料庫、原方案各讀一次，每一區照單筆同一套規則、都對送來的
那一份方案算；任何一區不成就整批不成（回哪一區、為什麼、「一區都沒改」），成了回每一區的結果，由網頁一口氣換進方案。
跟單筆一樣沒有副作用（不寫檔）。同一區在一批裡出現兩次不收（後面那一筆會蓋掉前面的，送的人多半弄錯了）。
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from aosr.config.capabilities import CapabilityTable
from aosr.config.directivity_defaults import DirectivityDefaults
from aosr.gui.material_editor import plain_error, user_library_path
from aosr.gui.problem_text import checked_scheme
from aosr.gui.wall_words import draft_wall_words
from aosr.gui.zone_design import NAME, WHOLE_WALL_KEY
from aosr.gui.zone_limits import zone_words
from aosr.gui.zone_result import BaselineRefused, OriginalWalls, original_material, original_walls
from aosr.materials.construction_models import Assembly, MountedOnWall, SurfaceMaterial
from aosr.materials.library import FixedEntry, LibraryEntry, full_library
from aosr.geometry.shoebox import Room
from aosr.physics.report_surfaces import RectangleZone, SurfaceZone
from aosr.physics.zone_fill import wall_size_m
from aosr.reporting.scheme import Scheme

_MATERIAL: TypeAdapter[SurfaceMaterial] = TypeAdapter(SurfaceMaterial)
WallName = Literal["floor", "ceiling", "x0", "xL", "y0", "yL"]
OriginSource = Literal["unwrapped", "original_scheme", "current", "unknown"]
# 這一區原本的面、這一面本身怎麼叫（同網頁 zone_modes.js 的 originalFace／faceWord）；沒給哪一面時照牆面叫。
ORIGINAL_FACE: Final = {"floor": "原本的地板", "ceiling": "原本的天花"}
FACE_ITSELF: Final = {"floor": "地板", "ceiling": "天花"}


def original_face(wall: str | None) -> str:
    return ORIGINAL_FACE.get(wall or "", "原牆")


def face_itself(wall: str | None) -> str:
    return FACE_ITSELF.get(wall or "", "牆面")


def unknown_origin(wall: str | None = None) -> str:
    face = original_face(wall)
    return (f"這一區現在的材料不是貼在{face}上的構造，看不出{face}是什麼；照現在的材料不動。"
            f"要換回{face}，請在上面「原方案」選切之前那一份方案")


class ZoneChoice(BaseModel):
    """一區選了什麼：哪一面、哪一區、放原牆（``original``）或指定材料庫的哪一筆（``fixed``＋代號）。"""

    model_config = ConfigDict(extra="forbid")
    wall: WallName
    zone: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    choice: Literal["original", "fixed"]
    material_id: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_]*$")


class ZoneMaterialRequest(ZoneChoice):
    """換一區的材料：整份方案（網頁表單現在的樣子）、哪一面、哪一區、選了什麼、設計設定指的原方案。"""

    scheme: dict[str, object]
    original_scheme_id: str | None = None


class ZoneMaterialBatch(BaseModel):
    """一次換好幾區（#1176）：整份方案、設計設定指的原方案、每一區選了什麼（至少一區）；全有或全無。"""

    model_config = ConfigDict(extra="forbid")
    scheme: dict[str, object]
    original_scheme_id: str | None = None
    changes: list[ZoneChoice] = Field(min_length=1)


class ZoneMaterialError(ValueError):
    """換不了：白話一句、HTTP 狀態碼。"""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


class ZoneBatchError(ZoneMaterialError):
    """整批不成：一句白話（哪一區、為什麼、一區都沒改）；是哪一區擋下的，``index``（第幾筆，從 0 數）、``wall``、``zone``、
    ``reason``（那一區自己的原因）回給網頁用它自己的號碼與位置名重寫；不是某一區的問題（方案沒填好、材料庫讀不進來）時是 None。"""

    def __init__(self, message: str, status: int, *, index: int | None = None, choice: ZoneChoice | None = None,
                 reason: str | None = None) -> None:
        super().__init__(message, status)
        self.index = index
        self.wall = choice.wall if choice is not None else None
        self.zone = choice.zone if choice is not None else None
        self.reason = reason

    def to_json(self) -> dict[str, object]:
        return {"error": str(self), "index": self.index, "wall": self.wall, "zone": self.zone, "reason": self.reason}


NOTHING_CHANGED = "一區都沒改"


@dataclass(frozen=True)
class Origin:
    material: SurfaceMaterial
    source: OriginSource


def _zone_index(zones: Mapping[str, tuple[SurfaceZone, ...]], room: Room, wall: str,
                zone: str) -> tuple[SurfaceZone, tuple[tuple[float, float], tuple[float, float]]]:
    size = wall_size_m(room, wall)
    for item in zones.get(wall, ()):
        key = item.id if isinstance(item, RectangleZone) else WHOLE_WALL_KEY
        if key == zone:
            box = (item.u_m, item.v_m) if isinstance(item, RectangleZone) else ((0.0, size[0]), (0.0, size[1]))
            return item, box
    raise ZoneMaterialError(f"{wall}沒有「{zone}」這一區；畫面跟方案對不上，請重新打開方案")


def origin(current: SurfaceMaterial, original: OriginalWalls | None, wall: str, zone: str,
           box: tuple[tuple[float, float], tuple[float, float]], replacements: Sequence[SurfaceMaterial] = ()) -> Origin:
    """這一區的原牆，順序同 1-5 的改善基準（``derive_baseline``）：指了原方案就照原方案；沒指時貼在原牆上的取它記的原牆；
    都沒有就當現在的材料是原牆（``current``，
    剛切出來的區就是這樣）——除非現在的材料正是材料庫裡直接取代牆面的那幾種（``replacements``：固定阻抗、
    實測表），那就看不出原牆（``unknown``），照現在的材料不動、照實說。"""
    if original is not None:
        try:
            return Origin(original_material(original, wall, zone, box), "original_scheme")
        except BaselineRefused as exc:
            raise ZoneMaterialError(str(exc)) from exc
    if isinstance(current, MountedOnWall):
        return Origin(current.wall, "unwrapped")
    return Origin(current, "unknown" if current in replacements else "current")


class ZoneChange(NamedTuple):
    """這一區放什麼、一句給畫面的說明、選的是不是跟原牆一模一樣（是的話放的就是原牆，網頁改成保留原牆）。"""

    material: SurfaceMaterial
    note: str
    same_as_original: bool = False


def zone_material(current: SurfaceMaterial, base: Origin, entry: FixedEntry | None, wall: str | None = None) -> ZoneChange:
    """選了之後這一區放什麼，和一句給畫面的說明（照 ``wall`` 那一面叫原本的面）。``entry`` 是 None 表示放原牆。"""
    face = original_face(wall)
    if entry is None:
        return ZoneChange(base.material, unknown_origin(wall) if base.source == "unknown" else f"這一區放回{face}")
    chosen = entry.surface_material()
    if isinstance(chosen, Assembly):
        # 跟原牆一模一樣（#1175：天花原牆本身就是那一套吊頂，再貼一次等於兩層吊頂疊在一起）：當成不貼。
        if base.material == chosen:
            return ZoneChange(base.material, f"選的「{entry.label}」跟這一區{face}一模一樣，當成不貼（保留原樣），不再疊一層", True)
        if isinstance(base.material, MountedOnWall):
            raise ZoneMaterialError(f"原方案這一區本身就是貼在{face}上的構造，不能再貼一層；原方案請選貼之前的那一份")
        mounted = MountedOnWall(layers=chosen.layers, wall=base.material)
        note = f"「{entry.label}」貼在{face}上（最後一層背後接{face}）"
        if base.source == "unknown":
            note += (f"；這一區現在是直接取代{face_itself(wall)}的材料、看不出{face}，就把它當{face}；"
                     f"要換別的{face}，請在上面「原方案」選切之前那一份方案")
        return ZoneChange(mounted, note)
    return ZoneChange(chosen, f"「{entry.label}」是整面的表面阻抗，直接取代這一區的{face_itself(wall)}（不是貼上去）")


@dataclass(frozen=True)
class ZoneScene:
    """一次請求（單筆或一批）共用的：驗過的方案、每一面的分區、材料庫、材料庫裡直接取代牆面的那幾種、原方案的原牆。"""

    scheme: Scheme
    walls: Mapping[str, tuple[SurfaceZone, ...]]
    entries: tuple[LibraryEntry, ...]
    replacements: tuple[SurfaceMaterial, ...]
    original: OriginalWalls | None


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

    def _walls(self, scheme: Scheme) -> Mapping[str, tuple[SurfaceZone, ...]]:
        """方案每一面的分區；六面都是整面固定阻抗的方案存成六個數字，走計算同一條驗證拿那六面材料
        （固定阻抗只在已驗輸入出生，見 ``zone_result.original_walls``），不在這裡自己建。"""
        if scheme.scene.surface_zones_by_wall is not None:
            return scheme.scene.surface_zones_by_wall
        try:
            return original_walls(scheme, scheme, capabilities=self.capabilities, directivity=self.directivity).zones
        except BaselineRefused as exc:
            raise ZoneMaterialError("方案表單還有格子沒填好或填錯，先改好再選這一區的材料（按「檢查」看是哪一格）") from exc

    def _scene(self, document: dict[str, object], original_scheme_id: str | None) -> ZoneScene:
        try:
            scheme = checked_scheme(document)
        except ValueError as exc:
            raise ZoneMaterialError("方案表單還有格子沒填好或填錯，先改好再選這一區的材料（按「檢查」看是哪一格）") from exc
        walls = self._walls(scheme)
        entries = self._library()
        replacements = tuple(material for item in entries if isinstance(item, FixedEntry)
                             if not isinstance(material := item.surface_material(), Assembly))
        return ZoneScene(scheme, walls, entries, replacements, self._original(original_scheme_id, scheme))

    def _one(self, scene: ZoneScene, choice: ZoneChoice) -> dict[str, object]:
        """一區選了之後放什麼（單筆與一批同一套）：原牆照 ``origin`` 的先後取，再照 ``zone_material`` 組。"""
        if choice.choice == "fixed" and choice.material_id is None:
            raise ZoneMaterialError("選固定材料要給材料代號")
        item, box = _zone_index(scene.walls, scene.scheme.scene.room_m, choice.wall, choice.zone)
        current = item.material
        entry = self._entry(scene.entries, choice.material_id) if choice.choice == "fixed" and choice.material_id else None
        base = origin(current, scene.original, choice.wall, choice.zone, box, scene.replacements)
        change = zone_material(current, base, entry, choice.wall)
        return {"material": _MATERIAL.dump_python(change.material, mode="json"), "note": change.note, "origin": base.source,
                "same_as_original": change.same_as_original}

    def change(self, body: ZoneMaterialRequest) -> dict[str, object]:
        if body.choice == "fixed" and body.material_id is None:
            raise ZoneMaterialError("選固定材料要給材料代號")
        return self._one(self._scene(body.scheme, body.original_scheme_id), body)

    def batch(self, body: ZoneMaterialBatch) -> dict[str, object]:
        """好幾區一起換：先查同一區有沒有出現兩次，再讀一次方案、材料庫、原方案，每一區照單筆同一套算；
        任何一區不成就整批不成。區名照限制檢查那一套（牆名照表單、區名寫代號），網頁另外用自己的號碼與位置名重寫。"""
        names = draft_wall_words(body.scheme)
        seen: set[tuple[str, str]] = set()
        for index, choice in enumerate(body.changes):
            if (choice.wall, choice.zone) in seen:
                reason = "同一區在這一批裡出現兩次"
                raise ZoneBatchError(f"{zone_words(choice.wall, choice.zone, names)}：{reason}；{NOTHING_CHANGED}", 400,
                                     index=index, choice=choice, reason=reason)
            seen.add((choice.wall, choice.zone))
        try:
            scene = self._scene(body.scheme, body.original_scheme_id)
        except ZoneMaterialError as exc:
            raise ZoneBatchError(f"{exc}；{NOTHING_CHANGED}", exc.status) from exc
        results = []
        for index, choice in enumerate(body.changes):
            try:
                results.append(self._one(scene, choice))
            except ZoneMaterialError as exc:
                raise ZoneBatchError(f"{zone_words(choice.wall, choice.zone, names)}：{exc}；{NOTHING_CHANGED}", exc.status,
                                     index=index, choice=choice, reason=str(exc)) from exc
        return {"changes": results}


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

    async def batch(request: Request) -> Response:
        try:
            body = ZoneMaterialBatch.model_validate(await request.json())
        except (ValidationError, ValueError):
            return JSONResponse({"error": "一次換好幾區的請求格式不完整：要有方案、至少一區，每一區要有哪一面、哪一區與選了什麼；"
                                 + NOTHING_CHANGED}, status_code=400)
        try:
            return JSONResponse(await run_in_threadpool(handlers.batch, body))
        except ZoneBatchError as exc:
            return JSONResponse(exc.to_json(), status_code=exc.status)

    return [Route("/api/zone-material", change, methods=["POST"]), Route("/api/zone-material/batch", batch, methods=["POST"])]
