"""每一區怎麼做的設計設定（#1083）：固定這一組材料／保留原牆／交給搜尋。

跟方案分開存在 ``data_dir/designs/<方案代號>.json``：方案檔格式不動（計算照舊只讀方案），
這一份給網頁畫立面圖、之後網頁開第三階段用。沒有列到的區一律當「固定」，也就是方案裡寫的那一組材料。

區用代號當鍵：矩形分區用它自己的代號，整面一區的牆用保留字 ``whole_wall``。自動補滿的代號
（``auto1``…）每次補滿都會重編，補滿後要重存這一份；寫入時對不上方案的牆或區一律拒收，
讀出時方案已經改過、對不上的就照實列出來，不靜默丟掉也不當錯誤擋住整頁。

設計設定不能比方案多活：沒有方案檔就不准寫；封存、還原跟著方案整包一起搬（見 ``scheme_archive``）。

``original_scheme_id``（選填）：這份設計是從哪一份方案切出來的。結果頁「改善多少」的原牆基準照它那一份的材料
鋪到這一套切法（見 ``zone_result``）；沒填就拆掉每一區加貼的層。寫入時那份方案要在、而且不能是自己。
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from aosr.gui.problem_text import checked_scheme
from aosr.physics.report_surfaces import RectangleZone
from aosr.reporting.scheme import Scheme

WHOLE_WALL_KEY = "whole_wall"
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,199}\Z")
ZoneMode = Literal["fixed", "keep_original", "search"]
MODE_WORDS: dict[str, str] = {"fixed": "固定這一組材料", "keep_original": "保留原牆", "search": "交給搜尋"}
DEFAULT_MODE: ZoneMode = "fixed"


class ZoneDesign(BaseModel):
    """一區的做法；之後的段（准用材料、深度上限）往這裡加欄位。"""

    model_config = ConfigDict(frozen=True, extra="forbid")
    mode: ZoneMode


class DesignSettings(BaseModel):
    """一份方案的設計設定：牆 → 區代號 → 做法。"""

    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal["aosr.zone_design.v1"] = "aosr.zone_design.v1"
    scheme_id: str
    zones: dict[str, dict[str, ZoneDesign]] = Field(default_factory=dict)
    original_scheme_id: str | None = Field(default=None, pattern="^" + NAME.pattern.removesuffix(r"\Z") + "$")


def design_relative(scheme_id: str) -> Path:
    return Path("designs") / f"{scheme_id}.json"


def zone_keys(scheme: Scheme) -> dict[str, list[str]]:
    """方案每一面牆的區代號，順序同方案；整面一區的牆只有一個保留字。

    六面都是整面固定阻抗的方案存檔時會折回六個數字（沒有分區清單），那六面也各算整面一區。
    """
    zones_by_wall = scheme.scene.surface_zones_by_wall
    if zones_by_wall is None:
        return {wall: [WHOLE_WALL_KEY] for wall in scheme.scene.impedance_pa_s_per_m_by_wall or {}}
    return {wall: [zone.id if isinstance(zone, RectangleZone) else WHOLE_WALL_KEY for zone in zones]
            for wall, zones in zones_by_wall.items()}


def mismatches(settings: DesignSettings, scheme: Scheme) -> list[str]:
    """設定裡對不上方案的牆或區，白話一條一句。"""
    keys = zone_keys(scheme)
    problems = []
    for wall, zones in settings.zones.items():
        if wall not in keys:
            problems.append(f"設計設定的牆 {wall} 方案裡沒有")
            continue
        problems.extend(f"設計設定的 {wall} 分區 {zone_id} 方案裡沒有" for zone_id in zones if zone_id not in keys[wall])
    return problems


def effective_modes(settings: DesignSettings | None, scheme: Scheme) -> dict[str, dict[str, ZoneMode]]:
    """方案每一區實際的做法：設定有寫就照寫，沒寫就固定；對不上的那幾條不算進來。"""
    listed = settings.zones if settings is not None else {}
    return {wall: {key: listed.get(wall, {}).get(key, ZoneDesign(mode=DEFAULT_MODE)).mode for key in keys}
            for wall, keys in zone_keys(scheme).items()}


def read_design(data_dir: Path, scheme_id: str) -> DesignSettings | None:
    path = data_dir / design_relative(scheme_id)
    if not path.is_file():
        return None
    return DesignSettings.model_validate_json(path.read_text(encoding="utf-8"))


def write_design(data_dir: Path, settings: DesignSettings, scheme: Scheme) -> None:
    """先核對方案再整份換上；對不上就丟 ValueError、不寫。"""
    problems = mismatches(settings, scheme)
    if problems:
        raise ValueError("；".join(problems))
    path = data_dir / design_relative(settings.scheme_id)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{settings.scheme_id}.", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(settings.model_dump_json())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _error(message: str, code: int) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=code)


class DesignHandlers:
    """設計設定的讀寫入口；方案只讀不改。

    ``lock`` 跟封存、還原用同一把：先讀完請求，再在鎖裡核方案、寫檔，免得核完方案、還在收請求時
    方案被封存走，留下包外的孤立設計檔（審查第 B 條，考卷 test_write_racing_an_archive_leaves_no_orphan）。
    """

    def __init__(self, data_dir: Path, lock: AbstractContextManager[object]) -> None:
        self.data_dir = data_dir
        self.lock = lock

    def _scheme(self, name: str) -> Scheme:
        if not NAME.fullmatch(name):
            raise ValueError("方案代號只收英數字、底線與連字號")
        path = self.data_dir / "schemes" / f"{name}.json"
        if not path.is_file():
            raise FileNotFoundError(f"沒有方案「{name}」，設計設定要跟著方案存")
        return checked_scheme(json.loads(path.read_text(encoding="utf-8")))

    async def item(self, request: Request) -> Response:
        name = request.path_params["name"]
        try:
            body: object = await request.json() if request.method == "PUT" else None
            with self.lock:
                scheme = self._scheme(name)
                if request.method == "GET":
                    settings = read_design(self.data_dir, name)
                    return JSONResponse({
                        "design": (settings or DesignSettings(scheme_id=name)).model_dump(mode="json"),
                        "modes": effective_modes(settings, scheme), "mode_words": MODE_WORDS,
                        "problems": mismatches(settings, scheme) if settings is not None else []})
                settings = DesignSettings.model_validate(body)
                if settings.scheme_id != name:
                    return _error("設計設定的方案代號跟網址不同", 400)
                if settings.original_scheme_id is not None:
                    if settings.original_scheme_id == name:
                        return _error("原方案不能是這一份自己", 400)
                    self._scheme(settings.original_scheme_id)
                write_design(self.data_dir, settings, scheme)
                return JSONResponse({"scheme_id": name, "message": "設計設定已儲存",
                                     "modes": effective_modes(settings, scheme)})
        except FileNotFoundError as exc:
            return _error(str(exc), 404)
        except ValidationError as exc:
            return _error(f"設計設定格式不對：{exc.error_count()} 處", 400)
        except (ValueError, OSError) as exc:
            return _error(str(exc), 400)


def design_routes(data_dir: Path, lock: AbstractContextManager[object]) -> list[Route]:
    handlers = DesignHandlers(data_dir, lock)
    return [Route("/api/designs/{name}", handlers.item, methods=["GET", "PUT"])]
