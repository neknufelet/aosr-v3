"""擺位與硬限制的補考卷（#585 第 2 步審查補的盲點）：每一題寫明它防哪一種錯法、原本為什麼沒人防。"""

from __future__ import annotations

import ast
import math
import shutil
from pathlib import Path

import pytest

from aosr.geometry.shoebox import Point
from aosr.reporting.scheme import Scheme
from aosr.search.constraints import Reason, check
from aosr.search.layout import LayoutParams, Placement, params_from_unit, place, standard_start, to_scheme
from aosr.search.layout_settings import Box, LayoutSettings, Span
from tests.engine.test_search_layout import project, settings  # 共用暫存方案與型別設定 fixture（測試輸入）。

SEARCH_SOURCE = Path(__file__).resolve().parents[2] / "src" / "aosr" / "search"


def _changed(base: LayoutSettings, **fields: object) -> LayoutSettings:
    return LayoutSettings.model_validate(base.model_dump() | fields)


def _box(x: tuple[float, float], y: tuple[float, float], z: tuple[float, float]) -> Box:
    return Box(x=Span(low=x[0], high=x[1]), y=Span(low=y[0], high=y[1]), z=Span(low=z[0], high=z[1]))


def _placement(left: Point, right: Point, primary: Point) -> Placement:
    return Placement(left, right, primary, (("main", primary),), (-1.0, 0.0))


def test_to_scheme_writes_the_new_seats_and_speakers(tmp_path: Path, project: Scheme, settings: LayoutSettings) -> None:
    """參考參數會讓座位落回原位，那一題抓不到「沿用專案原本的座位」；這裡用別的參數逐席比。"""
    placement = place(project, settings, LayoutParams(0.8, 1.5, 1.8))
    scheme = to_scheme(project, placement, "moved")
    seats = dict(placement.receivers)
    originals = {point.receiver_id: point.position_m for point in project.receiver_set.points}
    for point in scheme.receiver_set.points:
        assert point.position_m == seats[point.receiver_id].as_tuple()
        assert point.position_m != originals[point.receiver_id]
    left_id, right_id = (next(c.speaker_id for c in project.channel_group.channels if c.role == role)
                         for role in ("left", "right"))
    assert scheme.speakers[left_id] == placement.left
    assert scheme.speakers[right_id] == placement.right
    assert tmp_path.is_dir()


def test_unit_mapping_keeps_each_quantity_on_its_own_range(tmp_path: Path, settings: LayoutSettings) -> None:
    """三個量各給不同範圍、不同比例：名字或範圍對調的寫法在「同範圍同比例」的題裡會照綠。"""
    chosen = _changed(settings, front_distance_m={"low": 0.2, "high": 1.0},
                      spacing_m={"low": 1.0, "high": 3.0}, listening_distance_m={"low": 2.0, "high": 4.0})
    params = params_from_unit({"front_distance": 0.0, "spacing": 1.0, "listening_distance": 0.5}, chosen)
    assert (params.front_distance_m, params.spacing_m, params.listening_distance_m) == (0.2, 3.0, 3.0)
    assert tmp_path.is_dir()


def _rotated_project(project: Scheme, turns: int) -> Scheme:
    """把參考方案放進 6×6 的房、繞房中心逆時針轉 turns 個直角：專案面向四個方向各一。"""
    side = project.scene.room_m.Lx
    room = {"Lx": side, "Ly": side, "Lz": project.scene.room_m.Lz}
    center = side / 2.0

    def turn(x: float, y: float) -> tuple[float, float]:
        for _ in range(turns):
            x, y = center - (y - center), center + (x - center)
        return x, y

    speakers = {key: dict(zip(("x", "y"), turn(p.x, p.y), strict=True)) | {"z": p.z}
                for key, p in project.speakers.items()}
    points = tuple(p.model_dump() | {"position_m": (*turn(p.position_m[0], p.position_m[1]), p.position_m[2])}
                   for p in project.receiver_set.points)
    document = project.model_dump()
    document["scene"]["room_m"] = room
    return Scheme.model_validate(document | {"speakers": speakers, "receiver_set": {"points": points}})


@pytest.mark.parametrize("turns", (0, 1, 2, 3))
def test_standard_start_reads_the_projects_own_front_wall_in_every_facing(
    tmp_path: Path, project: Scheme, settings: LayoutSettings, turns: int,
) -> None:
    """專案面向 −x、−y、+x、+y 各一：起點的離前牆是專案自己那面前牆的距離、間距是兩喇叭的水平距離。

    只在面向 −x 的參考方案上考，「軸寫死成 x」「永遠取座標 0 那一側」「間距只取 y 差」三種錯法都會照綠。
    """
    rotated = _rotated_project(project, turns)
    chosen = _changed(settings, front_distance_m={"low": 0.1, "high": 6.0})
    start = standard_start(rotated, chosen)
    assert start is not None
    assert start.front_distance_m == pytest.approx(1.0, abs=1e-12)
    assert start.spacing_m == pytest.approx(1.2, abs=1e-12)
    assert start.listening_distance_m == pytest.approx(1.2 * math.sqrt(3.0) / 2.0, abs=1e-12)
    assert tmp_path.is_dir()


def test_wall_gap_on_the_far_y_wall_and_ceiling_is_not_a_wall(
    tmp_path: Path, project: Scheme, settings: LayoutSettings,
) -> None:
    """yL 那一側的間隙要咬（原本只有 x 兩側的題）；天花板不算牆、出天花板算出房間。"""
    chosen = _changed(settings, wall_gap_m=0.1)
    # 右箱朝 +x 對準主位，寬沿 y：y 從 3.75 到 3.95，離 yL（4.0）0.05，缺 0.05。
    placed = _placement(Point(1.0, 1.0, 1.0), Point(1.0, 3.85, 1.0), Point(3.0, 3.85, 1.0))
    violations = check(project, chosen, placed)
    assert {v.reason for v in violations} == {Reason.WALL_GAP}
    assert violations[0].amount_m == pytest.approx(0.05, rel=1e-12)
    # 箱頂離天花板（3.0）只剩 0.05，間隙 0.1 也不咬：天花板不是牆。
    near_ceiling = _placement(Point(1.0, 1.0, 2.75), Point(1.0, 2.5, 2.75), Point(3.0, 1.75, 2.75))
    assert check(project, chosen, near_ceiling) == ()
    # 箱頂穿出天花板 0.1：算出房間；座位高過天花板 0.05：算座位出房間。
    through = _placement(Point(1.0, 1.0, 2.9), Point(1.0, 2.5, 2.9), Point(3.0, 1.75, 3.05))
    amounts = {v.reason: v.amount_m for v in check(project, settings, through)}
    assert amounts[Reason.CABINET_OUTSIDE_ROOM] == pytest.approx(0.1, rel=1e-12)
    assert amounts[Reason.SEAT_OUTSIDE_ROOM] == pytest.approx(0.05, rel=1e-12)
    assert tmp_path.is_dir()


def test_toed_in_cabinet_against_keep_out_uses_both_shapes_axes(
    tmp_path: Path, project: Scheme, settings: LayoutSettings,
) -> None:
    """轉 45° 的箱子碰軸對齊的禁區：只用一邊的分離軸，兩種「其實分開」的情形會被判成重疊。

    左箱聲學中心 (1, 1)、對準 (3, 3)，箱 0.2×0.2、聲學中心在前面板：前面板那條邊在 x+y=2 上，
    最右的角在 x = 1 + 0.1/√2。
    """
    placed = _placement(Point(1.0, 1.0, 1.0), Point(1.0, 3.0, 1.0), Point(3.0, 3.0, 1.0))
    corner_x = 1.0 + 0.1 / math.sqrt(2.0)
    # 禁區的角頂向箱子斜的那條邊、但在邊外（x+y=2.02）：只用禁區的軸會判重疊。
    corner_to_edge = _changed(settings, keep_out=(_box((1.01, 1.5), (1.01, 1.5), (0.0, 2.0)),))
    assert check(project, corner_to_edge, placed) == ()
    # 箱子的角頂向禁區的平邊、但在邊外：只用箱子的軸會判重疊。
    corner_to_face = _changed(settings, keep_out=(_box((corner_x + 0.01, 1.5), (0.5, 1.5), (0.0, 2.0)),))
    assert check(project, corner_to_face, placed) == ()
    # 平邊往左移到角的裡面 0.01：真的重疊，退出量就是 0.01（沿 x 退最短）。
    overlapping = _changed(settings, keep_out=(_box((corner_x - 0.01, 1.5), (0.5, 1.5), (0.0, 2.0)),))
    amounts = {v.reason: v.amount_m for v in check(project, overlapping, placed)}
    assert amounts == {Reason.CABINET_IN_KEEP_OUT: pytest.approx(0.01, rel=1e-9)}
    assert tmp_path.is_dir()


def _calls(path: Path, scratch: Path) -> set[str]:
    copied = scratch / path.name
    shutil.copyfile(path, copied)
    names: set[str] = set()
    for node in ast.walk(ast.parse(copied.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call):
            func = node.func
            names.add(func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else "")
    return names


def test_constraint_module_never_builds_a_scheme(tmp_path: Path) -> None:
    """判不合法的那一支程式不准建方案：替身擋得到 to_scheme，擋不到直接 Scheme.model_validate 的寫法。"""
    calls = _calls(SEARCH_SOURCE / "constraints.py", tmp_path)
    assert calls, "constraints.py 一個呼叫都沒讀到——這一題沒在考"
    assert not calls & {"Scheme", "ReceiverSet", "model_validate", "model_construct", "to_scheme"}


FORBIDDEN_MODULES = ("aosr.config.stereo_layout", "aosr.config.fem_lane", "aosr.config.default_geometry")
FORBIDDEN_NAMES = ("primary_cross", "rectangular_grid", "import_module", "__import__")


def _absolute(module: str | None, level: int, alias: str) -> str:
    """相對 import 換成絕對名字（search 層的模組都在 aosr.search 底下）。"""
    package = ["aosr", "search"][: max(0, 2 - (level - 1))] if level else []
    parts = [*package, *(module.split(".") if module else []), alias]
    return ".".join(part for part in parts if part)


def test_search_reads_no_default_layout_or_seat_generator(tmp_path: Path) -> None:
    """搜尋只看專案方案與搜尋設定：上一代的預設擺位、預設幾何、座位產生器都不准碰（紙第二節第 9 條）。

    原本那一題只擋兩支設定模組的絕對 import；相對 import、動態 import、預設幾何、座位產生器都會漏。
    """
    files = sorted(SEARCH_SOURCE.glob("*.py"))
    assert files, "search 層一支模組都沒掃到——這一題沒在考"
    for path in files:
        copied = tmp_path / path.name
        shutil.copyfile(path, copied)
        for node in ast.walk(ast.parse(copied.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                imported = {alias.name for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                imported = {_absolute(node.module, node.level, alias.name) for alias in node.names}
                imported |= {_absolute(node.module, node.level, "")}
            elif isinstance(node, (ast.Name, ast.Attribute)):
                name = node.id if isinstance(node, ast.Name) else node.attr
                assert name not in FORBIDDEN_NAMES, f"{path.name} 用到 {name}"
                continue
            else:
                continue
            for name in imported:
                assert not any(name == bad or name.startswith(bad + ".") for bad in FORBIDDEN_MODULES), (
                    f"{path.name} import 了 {name}"
                )
                assert name.rsplit(".", 1)[-1] not in FORBIDDEN_NAMES, f"{path.name} import 了 {name}"
