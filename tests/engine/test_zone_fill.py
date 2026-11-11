"""畫好的框之外自動補滿（#1083）：補完不重疊、不留空、方案收得下；太窄的長條當問題、不當一區。

牆的大小用 109 房間的實際尺寸（不是整數），邊一律取輸入的浮點數，抓「容差湊整」或「補到差一點」的寫法。
"""
from __future__ import annotations

from copy import deepcopy
from typing import cast

import pytest

from aosr.geometry.shoebox import Room
from aosr.materials.construction_models import ConstantImpedance
from aosr.physics.report_surfaces import RectangleZone, validate_surface_partition
from aosr.physics.zone_fill import Box, fill_wall, wall_size_m
from aosr.reporting.scheme import Scheme
from tests.engine._surface_zone_cases import NEW, rectangle_document

ROOM_109 = Room(Lx=6.98792656080558, Ly=4.41988145469311, Lz=3.1554034223396052)
MIN_SIDE = 0.2


def _exact(wall: str, boxes: tuple[Box, ...]) -> None:
    """交給方案同一把精確判法：不重疊、不留空、不超出，不看面積和。"""
    material = ConstantImpedance(impedance_pa_s_m=1000.0)
    validate_surface_partition(ROOM_109, {wall: tuple(RectangleZone(area="rectangle", id=box.id, u_m=box.u_m, v_m=box.v_m,
                                                                    material=material) for box in boxes)})


def _area(boxes: tuple[Box, ...]) -> float:
    return sum((box.u_m[1] - box.u_m[0]) * (box.v_m[1] - box.v_m[0]) for box in boxes)


@pytest.mark.parametrize("wall", ["x0", "y0", "ceiling"])
def test_one_drawn_panel_is_kept_and_the_rest_is_filled_exactly(wall: str) -> None:
    width, height = wall_size_m(ROOM_109, wall)
    panel = Box(id="panel", u_m=(1.0, 2.5), v_m=(0.5, 2.0))
    result = fill_wall(wall, width, height, [panel], min_side_m=MIN_SIDE)
    assert result.problems == () and result.thin == ()
    assert result.zones[0] == panel
    assert result.filled_ids == ("auto1", "auto2", "auto3", "auto4")
    assert {box.id for box in result.zones} == {"panel", *result.filled_ids}
    assert max(box.u_m[1] for box in result.zones) == width
    assert max(box.v_m[1] for box in result.zones) == height
    assert _area(result.zones) == pytest.approx(width * height, rel=1e-12)
    _exact(wall, result.zones)


def test_integer_coordinates_from_the_web_come_back_as_floats() -> None:
    """網頁 JSON 的整數座標（u 1–2）補完一律是浮點數，方案的座標欄才收得下。"""
    width, height = wall_size_m(ROOM_109, "x0")
    result = fill_wall("x0", width, height, [Box(id="p", u_m=(1, 2), v_m=(0, 1))], min_side_m=MIN_SIDE)
    assert all(isinstance(edge, float) for box in result.zones for edge in (*box.u_m, *box.v_m))
    _exact("x0", result.zones)


def test_panels_touching_the_edges_leave_fewer_blocks() -> None:
    """貼齊底邊、全寬的一條：上面剩一整塊。"""
    width, height = wall_size_m(ROOM_109, "y0")
    result = fill_wall("y0", width, height, [Box(id="dado", u_m=(0.0, width), v_m=(0.0, 1.2))], min_side_m=MIN_SIDE)
    assert result.problems == ()
    assert result.zones[1:] == (Box(id="auto1", u_m=(0.0, width), v_m=(1.2, height)),)


def test_filled_wall_goes_into_a_scheme_and_round_trips() -> None:
    """補完整面交給方案：方案的精確判法收得下，存出讀回每一區都在。"""
    document = rectangle_document()
    scene = cast(dict[str, object], document["scene"])
    room = Room(**cast(dict[str, float], scene["room_m"]))
    width, height = wall_size_m(room, "x0")
    drawn = [Box(id="left", u_m=(0.0, 1.0), v_m=(0.4, 2.1)), Box(id="right", u_m=(2.5, 3.3), v_m=(1.0, height))]
    result = fill_wall("x0", width, height, drawn, min_side_m=MIN_SIDE)
    assert result.problems == ()
    zones = cast(dict[str, list[dict[str, object]]], scene[NEW])
    material = deepcopy(zones["xL"][0]["material"])
    zones["x0"] = [{"area": "rectangle", "id": box.id, "u_m": list(box.u_m), "v_m": list(box.v_m),
                    "material": deepcopy(material)} for box in result.zones]
    scheme = Scheme.model_validate(document)
    again = Scheme.model_validate_json(scheme.model_dump_json())
    saved = (again.scene.surface_zones_by_wall or {})["x0"]
    assert [zone.model_dump(mode="json")["id"] for zone in saved] == [box.id for box in result.zones]


def test_thin_leftover_strip_is_a_problem_not_a_zone() -> None:
    """框離牆邊只差 1 公分：補出來那一條不交出去，列問題並把那一條的位置交回去。"""
    width, height = wall_size_m(ROOM_109, "x0")
    result = fill_wall("x0", width, height, [Box(id="big", u_m=(0.01, width), v_m=(0.0, height))], min_side_m=MIN_SIDE)
    assert result.zones == () and result.filled_ids == ()
    assert result.thin == (Box(id="auto1", u_m=(0.0, 0.01), v_m=(0.0, height)),)
    assert len(result.problems) == 1 and "窄條" in result.problems[0] and "最小邊 0.2 公尺" in result.problems[0]


@pytest.mark.parametrize(("drawn", "words"), [
    ([Box(id="a", u_m=(0.0, 2.0), v_m=(0.0, 2.0)), Box(id="b", u_m=(1.0, 3.0), v_m=(1.0, 3.0))], "牆面 x0 分區 a、b：矩形重疊"),
    ([Box(id="a", u_m=(0.0, 5.0), v_m=(0.0, 1.0))], "牆面 x0 分區 a：座標超出牆面範圍"),
    ([Box(id="a", u_m=(0.0, 1.0), v_m=(0.0, 1.0)), Box(id="a", u_m=(2.0, 3.0), v_m=(0.0, 1.0))], "牆面 x0 分區 a：代號重複"),
])
def test_bad_drawings_use_the_scheme_check_wording(drawn: list[Box], words: str) -> None:
    width, height = wall_size_m(ROOM_109, "x0")
    result = fill_wall("x0", width, height, drawn, min_side_m=MIN_SIDE)
    assert result.zones == () and words in result.problems


def test_filled_ids_skip_names_already_drawn_and_nothing_drawn_means_whole_wall() -> None:
    width, height = wall_size_m(ROOM_109, "x0")
    result = fill_wall("x0", width, height, [Box(id="auto1", u_m=(0.0, 1.0), v_m=(0.0, height))], min_side_m=MIN_SIDE)
    assert result.filled_ids == ("auto2",)
    assert fill_wall("x0", width, height, [], min_side_m=MIN_SIDE).zones == ()
    with pytest.raises(ValueError, match="最小邊"):
        fill_wall("x0", width, height, [], min_side_m=-1.0)
