"""首頁第一屏（#1134）：最近算的那間房長怎樣（俯視圖要畫的東西）、三個入口連去哪、存著的方案每份一句白話。

只排字與座標、不改存檔（住在 gui 底下，不在計算指紋裡）。規則都借現有的：

- 牆名、前牆朝哪：``wall_words``／``front_wall``（跟 /search-start 前牆預設同一條）；定不出前牆時圖照 x 起點牆朝上、
  牆名照座標名，不猜。
- 哪裡「貼了吸音材料」：照 ``room_words.treated``（跟結果頁開頭那一句同一條）。顏色照結果頁同一支 ``material_facts`` 的標語：實測一色、其餘（模擬）一色。
- 自動存的名字：原牆基準 ``<名字>-orig-<8 碼>``（``zone_result.SUFFIX``）、第三階段搜尋選出來存的
  ``<名字>-search-<8 碼>``（``gradient_follow.adopted_name``）、擺位搜尋細算的 ``<搜尋代號>-…-verification``
  （``search.store.refine_scheme_id``）。
- /search-start 收不收：真的拒收那一條 ``Scene.require_whole_walls``。
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Final, Protocol
from urllib.parse import quote

from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from aosr.gui.gradient_follow import ADOPTED_WORD
from aosr.gui.labels import speaker_label
from aosr.gui.material_editor import user_library_path
from aosr.gui.modal_jobs import scheme_snapshot
from aosr.gui.problem_text import checked_scheme
from aosr.gui.result_list import ResultSummary
from aosr.gui.room_words import FOUR_WALLS, size_words, treated_words, treated_zones
from aosr.gui.wall_words import FACING, LEFT_OF, front_wall, wall_words
from aosr.gui.zone_result import SUFFIX, material_facts
from aosr.materials.library import LibraryEntry, builtin_entries, full_library
from aosr.physics.report_surfaces import RectangleZone
from aosr.reporting.scheme import Scheme

_BASELINE: Final = re.compile(rf"(?P<name>.+){re.escape(SUFFIX)}[0-9a-f]{{8}}\Z")
_ADOPTED: Final = re.compile(rf"(?P<name>.+)-{ADOPTED_WORD}-[0-9a-f]{{8}}\Z")
_REFINED: Final = re.compile(r"[0-9a-f]{32}-(?:baseline|trial-(?P<trial>[0-9]+))-verification\Z")
MATERIALS_NOTE: Final = "每一面牆、每一區貼哪種吸音材料、貼多厚，交給電腦挑"
LAYOUT_NOTE: Final = "喇叭和座位放哪裡最好，交給電腦試很多種擺法"
RESULT_NOTE: Final = "這一間算出來怎樣、比沒貼吸音材料的原牆好多少"
ZONED_LAYOUT_NOTE: Final = "這一間的牆面分了區，電腦試擺位目前只能用在牆面沒分區的方案；到那一頁選一份沒分區的"
NO_WHOLE_LAYOUT_NOTE: Final = "目前用不了：電腦試擺位只能用在牆面沒分區的方案，已存的每一份都分了區"


def name_words(scheme_id: str) -> tuple[str, bool]:
    """方案代號給人看的一句與它是不是電腦自動存的；不是自動存的照原名（使用者自己取的）。"""
    if found := _BASELINE.fullmatch(scheme_id):
        return f"「{found['name']}」拆掉貼上去的東西、只剩原牆的那一份（電腦自動算的，拿來比改善多少）", True
    if found := _ADOPTED.fullmatch(scheme_id):
        return f"「{found['name']}」第三階段搜尋選出來的那一套（電腦自動存的）", True
    if found := _REFINED.fullmatch(scheme_id):
        trial = found["trial"]
        return ("擺位搜尋：原本擺法的細算（電腦自動算的）" if trial is None
                else f"擺位搜尋：第 {int(trial)} 號候選擺法的細算（電腦自動算的）"), True
    return f"「{scheme_id}」", False


def _helper(scheme_id: str) -> bool:
    """只為了比較或細算才存的那幾種（原牆基準、擺位搜尋細算）；首頁「最近算的房間」跳過它們。"""
    return bool(_BASELINE.fullmatch(scheme_id) or _REFINED.fullmatch(scheme_id))


def _screen(scheme: Scheme) -> Callable[[float, float], tuple[float, float]]:
    """房間座標 → 俯視圖上的公尺（往右、往下，左上角是 0）：前牆朝上、面向前牆時的左手邊在左。"""
    front = front_wall(scheme) or "x0"
    fx, fy = FACING[front]
    lx, ly = FACING[LEFT_OF[front]]
    right, down = (-lx, -ly), (-fx, -fy)
    room = scheme.scene.room_m
    corners = [(0.0, 0.0), (room.Lx, 0.0), (0.0, room.Ly), (room.Lx, room.Ly)]
    left_edge = min(x * right[0] + y * right[1] for x, y in corners)
    top_edge = min(x * down[0] + y * down[1] for x, y in corners)

    def place(x: float, y: float) -> tuple[float, float]:
        return x * right[0] + y * right[1] - left_edge, x * down[0] + y * down[1] - top_edge

    return place


def room_overview(scheme: Scheme, library: Sequence[LibraryEntry]) -> dict[str, object]:
    """俯視圖要畫的東西，座標已經轉成畫面上的公尺（x 往右、y 往下）。"""
    names = wall_words(scheme)
    place = _screen(scheme)
    room = scheme.scene.room_m

    def point(x: float, y: float) -> dict[str, float]:
        across, down = place(x, y)
        return {"x": across, "y": down}

    ends: dict[str, Callable[[float], tuple[float, float]]] = {
        "x0": lambda u: (0.0, u), "xL": lambda u: (room.Lx, u), "y0": lambda u: (u, 0.0), "yL": lambda u: (u, room.Ly)}
    segments = []
    for wall, items in treated_zones(scheme).items():
        if wall in ends:
            for (u0, u1), material in items:
                measured = any(tag.code == "measured" for tag in material_facts(material, library).tags)
                segments.append({"wall": wall, "from": point(*ends[wall](u0)), "to": point(*ends[wall](u1)),
                                 "basis": "measured" if measured else "simulated"})
    middle = {"x0": (0.0, room.Ly / 2), "xL": (room.Lx, room.Ly / 2),
              "y0": (room.Lx / 2, 0.0), "yL": (room.Lx / 2, room.Ly)}
    corners = [place(x, y) for x in (0.0, room.Lx) for y in (0.0, room.Ly)]
    width, depth = max(across for across, _ in corners), max(down for _, down in corners)
    primary = scheme.receiver_set.primary.position_m
    return {"width_m": width, "depth_m": depth,
            "walls": [{"wall": wall, "name": names[wall], "at": point(*middle[wall])} for wall in FOUR_WALLS],
            "segments": segments,
            "speakers": [{"at": point(spot.x, spot.y), "name": speaker_label(name)}
                         for name, spot in sorted(scheme.speakers.items())],
            "primary": point(primary[0], primary[1]),
            "named": front_wall(scheme) is not None}


def zoned(scheme: Scheme) -> bool:
    """牆面有沒有切塊（/search-start 只收整面牆，同 ``require_whole_walls``）。"""
    try:
        scheme.scene.require_whole_walls()
    except ValueError:
        return True
    return False


def scheme_line(scheme: Scheme, saved: Mapping[str, Scheme]) -> dict[str, object]:
    """方案清單一列：白話說明、是不是自動存的、/search-start 收不收（不收寫原因）。"""
    title, auto = name_words(scheme.scheme_id)
    pieces = sum(1 for items in (scheme.scene.surface_zones_by_wall or {}).values()
                 for zone in items if isinstance(zone, RectangleZone))
    parts = [f"牆面分成 {pieces} 區"] if pieces else []
    parts.append(treated_words(scheme))
    if any(other.startswith(f"{scheme.scheme_id}-{ADOPTED_WORD}-") and _ADOPTED.fullmatch(other) for other in saved):
        parts.append("電腦挑過材料（第三階段搜尋）")
    return {"name": scheme.scheme_id, "title": title, "text": "；".join(parts), "auto": auto,
            "search_start_refused": "牆面分了區，電腦試擺位目前只能用在牆面沒分區的方案" if zoned(scheme) else ""}


def _with_scheme(url: str, name: str | None) -> str:
    return url if name is None else f"{url}?scheme={quote(name, safe='')}"


def latest_picks(rows: Sequence[tuple[str, str, str, str]]) -> Iterator[tuple[str, str]]:
    """結果清單（新的在前；每列方案代號、計算代號、結果頁網址、狀態）裡正常算完、不是只為比較或細算存的，新的先給；
    給計算代號與結果頁網址。"""
    for scheme_id, run_id, url, status in rows:
        if url and status in ("done", "none") and not _helper(scheme_id):
            yield run_id, url


def home_view(latest: tuple[str, Scheme] | None, saved: Mapping[str, Scheme],
              library: Sequence[LibraryEntry]) -> dict[str, object]:
    """首頁第一屏的整份資料。``latest`` 是最近一份正常算完的結果（結果頁網址、它的方案快照）。"""
    schemes = [scheme_line(scheme, saved) for _, scheme in sorted(saved.items())]
    if latest is None:
        return {"title": "還沒有算好的房間", "facts": "先在下面填房間、喇叭和座位，按「計算這個方案」。", "room": None,
                "links": {"result": {"url": "", "note": ""},
                          "materials": {"url": "/gradient", "note": MATERIALS_NOTE},
                          "layout": {"url": "/search-start", "note": LAYOUT_NOTE}},
                "schemes": schemes}
    url, scheme = latest
    name = scheme.scheme_id
    stored = name in saved
    cut = zoned(scheme)
    return {"title": f"最近算的房間：{name_words(name)[0]}",
            "facts": f"{size_words(scheme)}；{treated_words(scheme)}",
            "room": room_overview(scheme, library),
            "links": {"result": {"url": url, "note": RESULT_NOTE},
                      "materials": {"url": _with_scheme("/gradient", name if stored and cut else None),
                                    "note": MATERIALS_NOTE},
                      "layout": {"url": _with_scheme("/search-start", name if stored and not cut else None),
                                 "note": LAYOUT_NOTE if not cut else ZONED_LAYOUT_NOTE if any(
                                     not zoned(other) for other in saved.values()) else NO_WHOLE_LAYOUT_NOTE}},
            "schemes": schemes}


class HomeSource(Protocol):
    """首頁要讀的那幾樣（網頁伺服器的 GuiHandlers 有）：資料目錄、存著的方案名、結果清單（新的在前）。"""

    data_dir: Path

    def _scheme_names(self) -> list[str]: ...

    def _result_summaries(self, archived: bool = False) -> list[ResultSummary]: ...


def _latest(data_dir: Path, summaries: Sequence[ResultSummary]) -> tuple[str, Scheme] | None:
    """最近一份讀得出方案快照的；最新那份讀不出（例如格式改版）就往下一份找，不說成「還沒有算好的」。"""
    for run_id, url in latest_picks([(row.scheme_id, row.run_id, row.result_url, row.run_status) for row in summaries]):
        try:
            return url, scheme_snapshot(data_dir / "results" / f"{run_id}.json")
        except (ValueError, OSError):
            continue
    return None


def home_data(source: HomeSource) -> dict[str, object]:
    """讀存著的方案、結果清單與材料庫，排成首頁第一屏；讀不出的方案不列（下面「開舊方案」照樣列、打開時講原因）。"""
    data_dir = source.data_dir
    saved: dict[str, Scheme] = {}
    for name in source._scheme_names():
        try:
            saved[name] = checked_scheme(json.loads((data_dir / "schemes" / f"{name}.json").read_text(encoding="utf-8")))
        except (ValueError, OSError):
            continue
    latest = _latest(data_dir, source._result_summaries())
    try:
        library = full_library(user_library_path(data_dir))
    except (OSError, ValueError):
        library = builtin_entries()
    return home_view(latest, saved, library)


def home_routes(source: HomeSource) -> list[Route]:
    """GET /api/home；讀結果清單要拿工作鎖，所以放到執行緒（同 /api/schemes，#796）。"""
    async def home(request: Request) -> Response:
        return JSONResponse(await run_in_threadpool(home_data, source))

    return [Route("/api/home", home)]
