"""分區小工具（#1083）：一面牆上畫好的幾塊長方形之外，自動切成長方形補滿；不准重疊、不准留空。
另放網頁要用的兩支小查詢：牆面在分區座標裡的寬高、只取整面一區的牆。

這一支故意不改 ``report_surfaces``：那一支在物理身分的程式摘要裡，動一個字所有存下的物理結果都要重算；
這裡不在任何計算入口的匯入範圍內。

補滿的判法：把畫好的框所有的邊（加上牆的四邊）當格線切格，沒被框蓋到的格由下往上、由左往右
併成長方形（先沿 u 併成最長的一段，再往 v 疊同樣範圍的格）。邊一律直接取輸入的浮點數，不加容差；
補出來的整面放進方案時，由方案同一把精確判法（``report_surfaces.validate_surface_partition``）把關。

補出來比最小邊還窄的長條不當一區交出去：每一區都會變成網格的一塊與替身的一片，太碎會爆，
所以列成問題、連同那一條的位置交回去，讓畫的人把框貼齊牆邊或鄰框。
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from aosr.geometry.shoebox import Room
from aosr.materials.construction_models import SurfaceMaterial
from aosr.physics.report_surfaces import SurfaceZone, WholeWallZone

FILL_PREFIX = "auto"


class ZoneBox(Protocol):
    """只看位置的分區：代號與兩軸公尺座標；方案的矩形分區與補滿算出來的格子都符合。"""

    @property
    def id(self) -> str: ...
    @property
    def u_m(self) -> tuple[float, float]: ...
    @property
    def v_m(self) -> tuple[float, float]: ...


def wall_size_m(room: Room, wall: str) -> tuple[float, float]:
    """一面牆在分區座標裡的寬（u）與高（v），軸向同 RectangleZone 的說明（同 report_surfaces 的精確判法）。"""
    return {"floor": (room.Lx, room.Ly), "ceiling": (room.Lx, room.Ly),
            "x0": (room.Ly, room.Lz), "xL": (room.Ly, room.Lz),
            "y0": (room.Lx, room.Lz), "yL": (room.Lx, room.Lz)}[wall]


def whole_wall_entries(zones_by_wall: Mapping[str, Sequence[SurfaceZone]]) -> dict[str, SurfaceMaterial]:
    """只取整面一區的牆；有矩形分區的牆不在裡面。給只會畫整面表單、整面接縫的頁面用，分區牆另外照區列。"""
    return {name: zones[0].material for name, zones in zones_by_wall.items()
            if len(zones) == 1 and isinstance(zones[0], WholeWallZone)}


@dataclass(frozen=True)
class Box:
    """只有位置的一區：代號與兩軸公尺座標（軸向同方案的矩形分區）。"""

    id: str
    u_m: tuple[float, float]
    v_m: tuple[float, float]


@dataclass(frozen=True)
class FillResult:
    """補滿的結果。有問題時 zones 是空的，problems 寫原因，thin 是太窄的那幾條（給畫面標出來）。"""

    zones: tuple[Box, ...]
    filled_ids: tuple[str, ...]
    problems: tuple[str, ...]
    thin: tuple[Box, ...] = ()


def _drawn_problems(name: str, drawn: Sequence[ZoneBox], width: float, height: float) -> list[str]:
    """畫好的框自己的問題：代號重複、超出牆面、互相重疊（字樣同方案檢查）。留空是正常的，那是要補的地方。"""
    problems: list[str] = []
    seen: set[str] = set()
    for box in drawn:
        if box.id in seen:
            problems.append(f"牆面 {name} 分區 {box.id}：代號重複")
        seen.add(box.id)
        if not (0 <= box.u_m[0] < box.u_m[1] <= width and 0 <= box.v_m[0] < box.v_m[1] <= height):
            problems.append(f"牆面 {name} 分區 {box.id}：座標超出牆面範圍")
    if problems:
        return problems
    u_edges, v_edges = _edges(drawn, width, height)
    for u0, u1 in zip(u_edges, u_edges[1:]):
        for v0, v1 in zip(v_edges, v_edges[1:]):
            owners = [box.id for box in drawn if _covers(box, u0, u1, v0, v1)]
            if len(owners) > 1:
                message = f"牆面 {name} 分區 {'、'.join(owners)}：矩形重疊"
                if message not in problems:
                    problems.append(message)
    return problems


def _edges(boxes: Sequence[ZoneBox], width: float, height: float) -> tuple[list[float], list[float]]:
    u_edges = sorted({0.0, width, *(edge for box in boxes for edge in box.u_m)})
    v_edges = sorted({0.0, height, *(edge for box in boxes for edge in box.v_m)})
    return u_edges, v_edges


def _covers(box: ZoneBox, u0: float, u1: float, v0: float, v1: float) -> bool:
    return box.u_m[0] <= u0 and u1 <= box.u_m[1] and box.v_m[0] <= v0 and v1 <= box.v_m[1]


def _free_blocks(drawn: Sequence[ZoneBox], width: float, height: float) -> list[tuple[float, float, float, float]]:
    """沒被框蓋到的格併成長方形：由下往上、由左往右，先沿 u 併最長，再往 v 疊同樣範圍。"""
    u_edges, v_edges = _edges(drawn, width, height)
    nu, nv = len(u_edges) - 1, len(v_edges) - 1
    free = [[not any(_covers(box, u_edges[i], u_edges[i + 1], v_edges[j], v_edges[j + 1]) for box in drawn)
             for j in range(nv)] for i in range(nu)]
    taken = [[False] * nv for _ in range(nu)]
    blocks = []
    for j in range(nv):
        for i in range(nu):
            if not free[i][j] or taken[i][j]:
                continue
            i_end = i
            while i_end + 1 < nu and free[i_end + 1][j] and not taken[i_end + 1][j]:
                i_end += 1
            j_end = j
            while j_end + 1 < nv and all(free[k][j_end + 1] and not taken[k][j_end + 1] for k in range(i, i_end + 1)):
                j_end += 1
            for k in range(i, i_end + 1):
                for m in range(j, j_end + 1):
                    taken[k][m] = True
            blocks.append((u_edges[i], u_edges[i_end + 1], v_edges[j], v_edges[j_end + 1]))
    return blocks


def fill_wall(name: str, width: float, height: float, drawn: Sequence[ZoneBox], *, min_side_m: float) -> FillResult:
    """畫好的框原樣保留，空的地方補成長方形；沒畫任何框就不切（那面照舊整面一區）。

    補出來的代號是 ``auto1``、``auto2``…，跳過已經被畫的框用掉的代號。代號跟著每次補滿重編，
    所以別拿補出來的代號當長久的鍵（第 3 段的每區做法要記在畫的框或存檔後的代號上）。
    """
    if min_side_m < 0:
        raise ValueError("最小邊不可以是負的")
    if not drawn:
        return FillResult(zones=(), filled_ids=(), problems=())
    # 網頁送來的整數座標（JSON 的 2）一律轉成浮點數：方案的座標欄只收浮點數。
    boxes = [Box(id=box.id, u_m=(float(box.u_m[0]), float(box.u_m[1])), v_m=(float(box.v_m[0]), float(box.v_m[1])))
             for box in drawn]
    problems = _drawn_problems(name, boxes, width, height)
    if problems:
        return FillResult(zones=(), filled_ids=(), problems=tuple(problems))
    used = {box.id for box in boxes}
    filled: list[Box] = []
    counter = 0
    for u0, u1, v0, v1 in _free_blocks(boxes, width, height):
        counter += 1
        while f"{FILL_PREFIX}{counter}" in used:
            counter += 1
        filled.append(Box(id=f"{FILL_PREFIX}{counter}", u_m=(u0, u1), v_m=(v0, v1)))
    thin = tuple(box for box in filled if min(box.u_m[1] - box.u_m[0], box.v_m[1] - box.v_m[0]) < min_side_m)
    if thin:
        return FillResult(zones=(), filled_ids=(), thin=thin, problems=tuple(
            f"牆面 {name}：補滿後剩一條 u {box.u_m[0]:g}–{box.u_m[1]:g}、v {box.v_m[0]:g}–{box.v_m[1]:g} 公尺的窄條，"
            f"比最小邊 {min_side_m:g} 公尺還窄；請把畫的框貼齊牆邊或鄰框" for box in thin))
    zones = (*boxes, *filled)
    return FillResult(zones=zones, filled_ids=tuple(box.id for box in filled), problems=())
