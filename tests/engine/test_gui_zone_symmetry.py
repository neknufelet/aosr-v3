"""左右對稱的畫面那半邊（#1206 第 2 支）：配對純函式每條規則一題、設計設定那一格（缺＝開、關才寫、舊檔逐位不變）、
開跑計畫與 ``--tie``（只加在開跑參數、不進建替身核固定材料那一份；關掉與沒有一對時跟主線逐位相同）、分區頁從伺服器拿結果、
以及一題端到端：網頁算出的參數真的交給命令列 ``start``，引擎的開跑前檢查收下。

房間全是考卷自己編的（``_zone_symmetry_room``，5.0 × 4.0 × 2.7 m）。「跟主線逐位相同」比的是改動前主線
（e9c69c48）對同一份輸入算出來的雜湊，釘在這個檔裡。"""
from __future__ import annotations

import hashlib
import json
import random
import re
import subprocess
import sys
from collections.abc import Callable, Iterator
from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError
from starlette.testclient import TestClient

from aosr.config.material_gradient import load_material_gradient_settings
from aosr.config.paths import config_path
from aosr.gradient.cli import EXIT_STOPPED
from aosr.gui.app import GuiSettings, create_app
from aosr.gui.gradient_plan import GradientRequest, SearchPlan, TieInputs, configuration_arguments, plan_search, plan_symmetry, start_arguments
from aosr.gui.zone_design import DesignSettings, read_design, write_design
from aosr.gui.wall_words import wall_words
from aosr.gui.zone_edit import CutBox, WallName, cut_wall, snap_edges
from aosr.gui.zone_symmetry import PAIR_WITHIN_M, Symmetry, pair_zones
from aosr.reporting.scheme import Scheme
from aosr.runtime import child_process_env
from tests.engine._cpu_pins import forward_cpu_pins
from tests.engine._zone_symmetry_room import CENTRED, FRONT_PAIR, HEIGHT, LENGTH, NAME, SIDE, WIDTH, Box, Doc, searching, symmetry_document

SETTINGS = load_material_gradient_settings(config_path("material_gradient.toml"))
ALL_CUTS: dict[str, list[Box]] = {"x0": FRONT_PAIR, "y0": SIDE, "yL": SIDE, "ceiling": CENTRED}
ALL_ZONES = {"x0": ["b", "a"], "y0": ["s"], "yL": ["s"], "ceiling": ["c"]}
FRONT = ("x0@b", "x0@a")
SIDES = ("y0@s", "yL@s")
# 改動前主線（e9c69c48）對 ALL_CUTS＋ALL_ZONES 算出來的：開跑參數、建替身核固定材料那一份、計畫存檔，各自 JSON 的 SHA-256。
MAIN_START = "c4dab08ca28e3754ecda2dac75b411fe80984f92de55ccb02d04b113041d833b"
MAIN_CONFIGURATION = "09899cbca450c900343fc8387be0130e07c90e18baef4e490f6d0879517b0ea8"
MAIN_PLAN = "1e7e84aa21848465341cd5638630dbe9c7a8d4d9f4cd8297230f5bd6b172e6f3"
# 同一間房只有前牆 b 一區由程式最佳化（沒有對象）時主線算出來的。
MAIN_START_LONE = "9eae291c4c5f4976cf5d84459c8fffc9cf92ffb67e28a93ec2914415d8674013"
MAIN_PLAN_LONE = "ea9a5d85e45febd69cba1340a8c7f41b1cd04754e4ef4cbf9a70d72718dfd7e5"
# 主線寫出來的舊設計檔（沒有左右對稱那一格）。
OLD_DESIGN = ('{"schema_version":"aosr.zone_design.v1","scheme_id":"sym-sample","zones":{"x0":{"b":{"mode":"search"},'
              '"a":{"mode":"search"}},"y0":{"s":{"mode":"search"}},"yL":{"s":{"mode":"search"}},"ceiling":{"c":{"mode":"search"}}},'
              '"original_scheme_id":null}')
PASSED = "PASSED-THE-START-CHECK"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class _Check:
    """假的設定檢查：記下被問過哪幾對；``refuse`` 給了就一律用那一句拒收。"""

    def __init__(self, refuse: str | None = None) -> None:
        self.asked: list[tuple[str, str]] = []
        self.refuse = refuse

    def __call__(self, first: str, second: str) -> str | None:
        self.asked.append((first, second))
        return self.refuse


def _paired(cuts: dict[str, list[Box]], zones: dict[str, list[str]], check: Callable[[str, str], str | None] | None = None,
            design: Doc | None = None, *, across_shift_m: float = 0.0, seat_shift_m: float = 0.0, turned: bool = False) -> Symmetry:
    scheme = Scheme.model_validate(symmetry_document(cuts, across_shift_m=across_shift_m, seat_shift_m=seat_shift_m, turned=turned))
    settings = DesignSettings.model_validate(design if design is not None else searching(zones))
    return pair_zones(scheme, settings, check or _Check())


def _pairs(found: Symmetry) -> tuple[tuple[str, str], ...]:
    return tuple((pair.first, pair.second) for pair in found.pairs)


def _missed(found: Symmetry) -> list[tuple[str, str | None, str]]:
    return [(item.first, item.second, item.kind) for item in found.missed]


def _plan(zones: dict[str, list[str]] | None = None, design: Doc | None = None) -> SearchPlan:
    scheme = Scheme.model_validate(symmetry_document(ALL_CUTS))
    settings = DesignSettings.model_validate(design if design is not None else searching(zones or ALL_ZONES))
    return plan_search(scheme, settings, GradientRequest(), SETTINGS)


# ---- 配對規則（純函式）----

def test_mirrored_search_zones_pair_and_the_earlier_patch_comes_first() -> None:
    """前牆兩區對著軸鏡像、左右牆同位置各一區：兩對；每一對面序在前的那一片在前（前牆方案先列 b，字串序卻是 a 在前）。
    天花那一區自己跨在軸上、左右置中：不用配，也不算沒配成。"""
    check = _Check()
    found = _paired(ALL_CUTS, ALL_ZONES, check)
    assert found.enabled is True and found.reason is None
    assert found.searched == ("ceiling@c", "x0@b", "x0@a", "y0@s", "yL@s")
    assert _pairs(found) == (FRONT, SIDES) and found.missed == ()
    assert found.centred == ("ceiling@c",) and found.alone == 1
    assert check.asked == [FRONT, SIDES]
    assert found.axis is not None
    assert found.axis.along == "x" and found.axis.across_m == pytest.approx(2.0) and found.axis.centred is True


def test_only_zones_handed_to_the_program_take_part() -> None:
    """抓「只管由程式最佳化」被拿掉：鏡像位置上的對象是指定貼一種或保留原樣時，不配、不問引擎；那個對象自己不算
    由程式最佳化的區、也不掛沒配成（沒配成的是交給程式的那一區，見下一題）。"""
    design = searching({"x0": ["b"], "y0": ["s"]})
    cast(dict[str, dict[str, object]], design["zones"])["yL"] = {"s": {"mode": "keep_original"}}
    check = _Check()
    found = _paired(ALL_CUTS, {}, check, design=design)
    assert found.searched == ("x0@b", "y0@s") and found.pairs == () and found.alone == 2 and check.asked == []
    assert [item.first for item in found.missed] == ["x0@b", "y0@s"] and all(item.second is None for item in found.missed)


def test_half_a_pair_whose_mirror_zone_is_not_handed_to_the_program_is_missed() -> None:
    """半對之一：這一區由程式最佳化，鏡像位置那一區是指定貼一種（設計檔沒列）或保留原樣：算沒配成，
    原因講得出是哪一區不是由程式最佳化、怎麼改；不問引擎。"""
    design = searching({"x0": ["b"], "y0": ["s"]})
    cast(dict[str, dict[str, object]], design["zones"])["yL"] = {"s": {"mode": "keep_original"}}
    check = _Check()
    found = _paired(ALL_CUTS, {}, check, design=design)
    assert _missed(found) == [("x0@b", None, "not_search"), ("y0@s", None, "not_search")] and check.asked == []
    assert [item.other for item in found.missed] == ["x0@a", "yL@s"]
    for item in found.missed:
        assert item.reason == "鏡像位置那一區不是由程式最佳化" and item.cuts == () and item.figure is None
        assert "{first}" in item.detail and "{other}" in item.detail and "也改成由程式最佳化" in item.detail


def test_half_a_pair_with_no_zone_at_the_mirror_position_is_missed() -> None:
    """半對之二：鏡像位置根本沒有一樣大小的區（前牆只切了左邊那一塊）：算沒配成，原因寫鏡像位置在哪裡（離左牆幾公尺）。"""
    check = _Check()
    found = _paired({"x0": [FRONT_PAIR[0]]}, {"x0": ["b"]}, check)
    assert _missed(found) == [("x0@b", None, "no_zone")] and found.missed[0].other is None and check.asked == []
    assert found.missed[0].reason == "鏡像位置沒有一樣大小的區"
    assert "離左牆 2.50～3.60 公尺、離地 0.30～2.40 公尺" in found.missed[0].detail and "{first}" in found.missed[0].detail
    assert found.pairs == () and found.alone == 1 and found.centred == ()


@pytest.mark.parametrize(("shift_m", "pairs"), [(0.005, True), (-0.005, True), (0.05, False), (-0.05, False)])
def test_every_edge_must_be_within_one_centimetre(shift_m: float, pairs: bool) -> None:
    """抓容差被放寬（1 cm 變 1 m）：半公分還算一對，五公分不算。橫向與上下各考一次。"""
    for sideways in (True, False):
        moved: Box = (("a", (2.5 + shift_m, 3.6 + shift_m), (0.3, 2.4)) if sideways
                      else ("a", (2.5, 3.6), (0.3 + shift_m, 2.4 + shift_m)))
        found = _paired({"x0": [FRONT_PAIR[0], moved]}, {"x0": ["b", "a"]})
        assert (_pairs(found) == (FRONT,)) is pairs, (sideways, shift_m)
        assert _missed(found) == ([] if pairs else [("x0@b", "x0@a", "position")])


def test_side_walls_mirror_each_other_only_when_the_axis_is_in_the_middle() -> None:
    """抓左右牆不看軸在不在正中：喇叭與座位一起橫移 20 公分，軸離開房間正中，左右牆同位置的兩區不配、照實寫原因；
    前牆對著新的軸鏡像的兩區照配。"""
    cuts: dict[str, list[Box]] = {"x0": [("b", (0.6, 1.7), (0.3, 2.4)), ("a", (2.7, 3.8), (0.3, 2.4))], "y0": SIDE, "yL": SIDE}
    check = _Check()
    found = _paired(cuts, {"x0": ["b", "a"], "y0": ["s"], "yL": ["s"]}, check, across_shift_m=0.2)
    assert found.axis is not None
    assert found.axis.across_m == pytest.approx(2.2) and found.axis.centred is False
    assert _pairs(found) == (FRONT,) and check.asked == [FRONT]
    assert _missed(found) == [("y0@s", "yL@s", "side_walls")]
    assert "不在房間正中" in found.missed[0].reason and found.missed[0].cuts == ()
    # 軸不在正中時左右牆上落單的一區也是同一個原因。
    lone = _paired({"y0": SIDE}, {"y0": ["s"]}, across_shift_m=0.2)
    assert _missed(lone) == [("y0@s", None, "side_walls")] and "不在房間正中" in lone.missed[0].reason
    # 軸離正中半公分還算在正中。
    near = _paired({"y0": SIDE, "yL": SIDE}, {"y0": ["s"], "yL": ["s"]}, across_shift_m=0.005)
    assert _pairs(near) == (SIDES,)


def test_a_seat_off_the_speaker_midline_pairs_nothing() -> None:
    """主位橫向離兩支喇叭中點超過 1 公分：整張不配，原因照單上那一句；半公分照配。"""
    check = _Check()
    found = _paired(ALL_CUTS, ALL_ZONES, check, seat_shift_m=0.05)
    assert found.enabled is True and found.reason == "主位不在兩支喇叭中間，定不出對稱軸"
    assert found.axis is None and found.pairs == () and found.missed == () and check.asked == []
    assert found.alone == 5
    assert _pairs(_paired({"x0": FRONT_PAIR}, {"x0": ["b", "a"]}, seat_shift_m=0.005)) == (FRONT,)


def test_no_front_wall_pairs_nothing_and_says_so() -> None:
    """兩支喇叭的中點就在主位正上方：定不出前牆（``project_facing`` 拒收），整張不配、照實寫原因。"""
    document = symmetry_document(ALL_CUTS)
    document["speakers"] = {"left": {"x": 3.2, "y": 1.2, "z": 1.0}, "right": {"x": 3.2, "y": 2.8, "z": 1.0}}
    found = pair_zones(Scheme.model_validate(document), DesignSettings.model_validate(searching(ALL_ZONES)), _Check())
    assert found.axis is None and found.pairs == () and found.missed == ()
    assert found.reason is not None and "定不出對稱軸" in found.reason and "前牆" in found.reason


def test_the_other_facing_mirrors_the_other_coordinate() -> None:
    """同一間房轉 90 度（前牆 yL，軸沿 y）：前牆、地板對著 x = 2.0 鏡像，左右牆變成 x0／xL。"""
    cuts: dict[str, list[Box]] = {"yL": FRONT_PAIR, "x0": SIDE, "xL": SIDE, "ceiling": [("c", (1.2, 2.8), (1.5, 3.0))],
                                  "floor": [("p", (0.3, 1.3), (1.0, 2.0)), ("q", (2.7, 3.7), (1.0, 2.0))]}
    found = _paired(cuts, {"yL": ["b", "a"], "x0": ["s"], "xL": ["s"], "ceiling": ["c"], "floor": ["p", "q"]}, turned=True)
    assert found.axis is not None
    assert found.axis.along == "y" and found.axis.across_m == pytest.approx(2.0)
    assert _pairs(found) == (("floor@p", "floor@q"), ("x0@s", "xL@s"), ("yL@b", "yL@a"))
    assert found.centred == ("ceiling@c",) and found.missed == ()


def test_the_floor_mirrors_across_the_axis_in_the_default_facing() -> None:
    """預設朝向（軸沿 x）：地板的 v 是橫向，對著 y = 2.0 鏡像；只沿軸錯開的兩區不是一對。"""
    mirrored: list[Box] = [("p", (1.0, 2.0), (0.3, 1.3)), ("q", (1.0, 2.0), (2.7, 3.7))]
    assert _pairs(_paired({"floor": mirrored}, {"floor": ["p", "q"]})) == (("floor@p", "floor@q"),)
    along: list[Box] = [("p", (0.3, 1.3), (1.0, 2.0)), ("q", (3.7, 4.7), (1.0, 2.0))]
    assert _paired({"floor": along}, {"floor": ["p", "q"]}).pairs == ()


def test_whole_walls_pair_and_a_whole_floor_is_centred_only_under_a_middle_axis() -> None:
    """整面一區的牆照整面的範圍算：左右牆整面互為一對、地板整面自己置中；軸不在正中時地板不置中，鏡像位置沒有區。"""
    zones = {"y0": ["whole_wall"], "yL": ["whole_wall"], "floor": ["whole_wall"]}
    found = _paired({}, zones)
    assert _pairs(found) == (("y0", "yL"),) and found.centred == ("floor",) and found.alone == 1
    shifted = _paired({}, zones, across_shift_m=0.2)
    assert shifted.pairs == () and shifted.centred == ()
    assert _missed(shifted) == [("y0", "yL", "side_walls"), ("floor", None, "outside")]


def test_a_pair_the_engine_refuses_is_listed_as_missed_with_its_reason() -> None:
    """位置對上、設定檢查拒收：不配，列進沒配成，原因照檢查回的那一句，沒有切法可給。"""
    found = _paired({"x0": FRONT_PAIR}, {"x0": ["b", "a"]}, _Check("兩區的最深不一樣"))
    assert found.pairs == () and _missed(found) == [("x0@b", "x0@a", "settings")]
    assert found.missed[0].reason == "兩區的最深不一樣" and found.missed[0].cuts == () and found.alone == 2


def test_two_zones_that_almost_mirror_get_both_cuts_spelled_out() -> None:
    """前牆兩區寬度不同：沒配成，兩種切法都列，數字是離左牆幾公尺（照程式算）；不問引擎。"""
    check = _Check()
    found = _paired({"x0": [FRONT_PAIR[0], ("a", (2.5, 3.4), (0.3, 2.4))]}, {"x0": ["b", "a"]}, check)
    assert found.pairs == () and _missed(found) == [("x0@b", "x0@a", "position")] and check.asked == []
    missed = found.missed[0]
    assert missed.cuts == ("把 {second} 切到離左牆 2.50～3.60 公尺，就和 {first} 配成一對",
                           "把 {first} 切到離左牆 0.60～1.50 公尺，就和 {second} 配成一對")
    assert "1 公分" in missed.detail and "{first}" in missed.detail and "{second}" in missed.detail
    # 差幾公分直接寫（不讓人自己減）：靠中間那一邊對齊不提，靠右牆那一邊差 20 公分。
    assert "疊著比：靠右牆那一邊差 20 公分。" in missed.detail and "靠中間" not in missed.detail
    # 「左右對稱」那一塊直接寫的那一句（樣稿那一列就寫原因）。
    assert missed.summary == "大小不同，各自調；要配對，回「房間方案輸入」切到左右對稱的位置"
    assert found.axis is not None
    assert found.axis.words == "對稱軸跟著兩支喇叭的中點和主位走，現在離左牆 2.00 公尺、在房間正中；主位不在兩支喇叭中間時，就不配對。"
    assert missed.figure is not None
    assert missed.figure.wall == "x0" and missed.figure.mirrored_u_m == pytest.approx((2.5, 3.6))


def _wall_zones(document: Doc, wall: str) -> list[dict[str, object]]:
    return cast(dict[str, list[dict[str, object]]], cast(Doc, document["scene"])["surface_zones_by_wall"])[wall]


def _span(value: object) -> tuple[float, float]:
    low, high = cast(list[float], value)
    return (low, high)


def _found(document: Doc, zones: dict[str, list[str]]) -> Symmetry:
    return pair_zones(Scheme.model_validate(document), DesignSettings.model_validate(searching(zones)), _Check())


def test_distances_count_from_the_far_end_when_the_left_wall_is_the_far_one() -> None:
    """審查 A-K1（橫向）：整組轉 180 度，前牆是 xL、左牆是 yL（座標從另一端起算）。切法的「離左牆」要從 yL 那一端反算，
    不是照抄區的座標。"""
    found = _found(symmetry_document({"xL": [FRONT_PAIR[0], ("a", (2.5, 3.4), (0.3, 2.4))]}, flipped=True), {"xL": ["b", "a"]})
    assert found.axis is not None and (found.axis.front, found.axis.left) == ("xL", "yL")
    assert found.missed[0].cuts == ("把 {second} 切到離左牆 0.40～1.50 公尺，就和 {first} 配成一對",
                                    "把 {first} 切到離左牆 2.50～3.40 公尺，就和 {second} 配成一對")
    assert "靠左牆那一邊差 20 公分" in found.missed[0].detail


def test_distances_count_from_the_far_end_when_the_front_wall_is_the_far_one() -> None:
    """審查 A-K1（縱深）：轉 90 度的房間前牆是 yL。天花上一區的鏡像位置，「離前牆」要從 yL 那一端反算。"""
    found = _found(symmetry_document({"ceiling": [("p", (0.5, 1.5), (1.0, 2.0))]}, turned=True), {"ceiling": ["p"]})
    assert found.axis is not None and (found.axis.front, found.axis.left) == ("yL", "x0")
    assert _missed(found) == [("ceiling@p", None, "no_zone")]
    assert "離左牆 2.50～3.50 公尺、離前牆 3.00～4.00 公尺" in found.missed[0].detail


def test_zones_that_only_touch_the_mirror_position_are_not_a_near_miss() -> None:
    """審查 A-K4：軸偏 20 公分，b 的鏡像右緣剛好碰到 n 的左緣（浮點算出來多一點點重疊）。只是相接，不算「位置對不上」，
    不准給切法；兩區各自是「鏡像位置沒有一樣大小的區」。"""
    found = _found(symmetry_document({"x0": [("b", (0.6, 1.7), (0.3, 2.4)), ("n", (3.8, 4.0), (0.3, 2.4))]}, across_shift_m=0.2),
                   {"x0": ["b", "n"]})
    assert _missed(found) == [("x0@b", None, "no_zone"), ("x0@n", None, "no_zone")]
    assert all(item.cuts == () for item in found.missed)


def test_exactly_one_centimetre_off_still_pairs() -> None:
    """審查 A-K6：規格是「差在 1 公分內」含 1 公分。2.50 − 2.49 在浮點上比 0.01 多一點點，照樣算一對。"""
    found = _found(symmetry_document({"x0": [FRONT_PAIR[0], ("a", (2.49, 3.59), (0.3, 2.4))]}), {"x0": ["b", "a"]})
    assert _pairs(found) == (("x0@b", "x0@a"),) and found.missed == ()


def test_a_cut_that_would_leave_the_wall_is_not_offered() -> None:
    """切法不准落在牆外（審查 A）：軸偏 20 公分，e 的鏡像位置超出牆面（牆寬 4.0，鏡像到 4.40）。那一種切法不列；
    另一種（把 e 切到 f 的鏡像位置，在牆裡）照列。"""
    found = _found(symmetry_document({"x0": [("e", (0.0, 0.5), (0.3, 2.4)), ("f", (3.6, 4.0), (0.3, 2.4))]}, across_shift_m=0.2),
                   {"x0": ["e", "f"]})
    assert _missed(found) == [("x0@e", "x0@f", "position")]
    assert found.missed[0].cuts == ("把 {first} 切到離左牆 0.40～0.80 公尺，就和 {second} 配成一對",)
    assert "4.40" not in "".join(found.missed[0].cuts)


def test_when_no_cut_is_possible_it_says_the_two_zones_cannot_pair() -> None:
    """兩種切法都做不到：e 的鏡像超出牆面，f 跨在軸上、它的鏡像位置疊在 e 身上。不給切法，列上寫只搬一區切不出來，
    細節兩個方向各寫一句為什麼（不寫「這兩區沒辦法配成一對」：兩區都改有時配得成）。"""
    found = _found(symmetry_document({"x0": [("e", (0.0, 0.5), (0.3, 2.4)), ("f", (1.9, 4.0), (0.3, 2.4))]}, across_shift_m=0.2),
                   {"x0": ["e", "f"]})
    assert _missed(found) == [("x0@e", "x0@f", "position")] and found.missed[0].cuts == ()
    assert found.missed[0].summary == "只搬其中一區切不出鏡像的位置，各自調"
    assert found.missed[0].detail.endswith("只搬其中一區切不出鏡像的位置：把 {second} 切到離左牆 3.90～4.40 公尺，會超出牆面；"
                                           "把 {first} 切到離左牆 0.40～2.50 公尺，會疊在 {second} 上。")
    assert "沒辦法配成一對" not in found.missed[0].summary + found.missed[0].detail


def test_a_cut_the_wall_tool_would_refuse_is_not_offered() -> None:
    """審查 A 修補第 2 條：軸在 1.95，b 貼著左牆（0.00～1.00）、a 在 2.80～3.80。兩種切法都在牆裡、也不疊在另一區上，
    可是各有一邊離牆只差 10 公分：照打數字，網頁會先把那條邊吸到牆上（``elevation.js::snapEdges``），切出來差 10 公分，
    配不成。不列切法，兩個方向都寫吸邊（第三輪：不是「切牆工具不收的窄條」，網頁上窄條先被吸掉了）。"""
    found = _found(symmetry_document({"x0": [("b", (0.0, 1.0), (0.3, 2.4)), ("a", (2.8, 3.8), (0.3, 2.4))]}, across_shift_m=-0.05),
                   {"x0": ["b", "a"]})
    assert _missed(found) == [("x0@b", "x0@a", "position")] and found.missed[0].cuts == ()
    assert found.missed[0].summary == "只搬其中一區切不出鏡像的位置，各自調"
    assert found.missed[0].detail.endswith(f"只搬其中一區切不出鏡像的位置：把 {{second}} 切到離左牆 2.90～3.90 公尺，{SNAPPED_AWAY}；"
                                           f"把 {{first}} 切到離左牆 0.10～1.10 公尺，{SNAPPED_AWAY}。")


SNAPPED_AWAY = "照打數字，框邊會被吸到牆邊或旁邊那一區的邊上，切不到這個位置"
CEILING_PAIR: list[Box] = [("p", (1.0, 2.0), (0.5, 1.5)), ("q", (1.0, 2.0), (2.5, 3.55))]
# 另一欄一個保留原樣的框：它的一條邊離 q 的鏡像位置的邊只有 15 公分（兩框沒有相對，網頁的吸邊不管這個）。
CEILING_NEAR: Box = ("r", (3.0, 4.0), (3.35, 3.75))


def test_a_cut_the_page_would_snap_away_is_not_offered() -> None:
    """審查 A 第二輪：天花 p、q 差 5 公分，旁邊 r 的一條邊離「把 q 切到離左牆 2.50～3.50」的 3.50 只有 15 公分。照打數字，
    網頁會把那條邊吸到 3.35，配不成：這一句不列，搬 p 的那一句照列。再加一個 r2 把搬 p 那邊也吸走：兩種都不列，
    細節兩個方向都寫吸邊。沒有 r 的對照組兩種都列。"""
    plain = _found(symmetry_document({"ceiling": CEILING_PAIR}), {"ceiling": ["p", "q"]})
    assert plain.missed[0].cuts == ("把 {second} 切到離左牆 2.50～3.50 公尺，就和 {first} 配成一對",
                                    "把 {first} 切到離左牆 0.45～1.50 公尺，就和 {second} 配成一對")
    found = _found(symmetry_document({"ceiling": [*CEILING_PAIR, CEILING_NEAR]}), {"ceiling": ["p", "q"]})
    assert _missed(found) == [("ceiling@p", "ceiling@q", "position")]
    assert found.missed[0].cuts == ("把 {first} 切到離左牆 0.45～1.50 公尺，就和 {second} 配成一對",)
    both = _found(symmetry_document({"ceiling": [*CEILING_PAIR, CEILING_NEAR, ("r2", (3.0, 4.0), (0.3, 1.0))]}), {"ceiling": ["p", "q"]})
    assert both.missed[0].cuts == () and both.missed[0].summary == "只搬其中一區切不出鏡像的位置，各自調"
    assert both.missed[0].detail.endswith(f"只搬其中一區切不出鏡像的位置：把 {{second}} 切到離左牆 2.50～3.50 公尺，{SNAPPED_AWAY}；"
                                          f"把 {{first}} 切到離左牆 0.45～1.50 公尺，{SNAPPED_AWAY}。")


SWEEP_ROUNDS = 1500
_FACES = ("front", "back", "ceiling", "floor", "side")


def _wall_size(turned: bool, wall: str) -> tuple[float, float]:
    lx, ly = (WIDTH, LENGTH) if turned else (LENGTH, WIDTH)
    return (lx, ly) if wall in ("floor", "ceiling") else (ly if wall[0] == "x" else lx, HEIGHT)


def _random_boxes(rng: random.Random, size: tuple[float, float], count: int, step: float) -> list[Box]:
    """一面上亂擺幾個互不重疊的框（邊落在 ``step`` 的格點上；三成頂天立地）。"""
    width, height = size
    placed: list[Box] = []
    for index in range(count):
        for _ in range(20):
            u0 = round(rng.uniform(0, width - 0.2) / step) * step
            u1 = min(round(rng.uniform(u0 + 0.2, min(width, u0 + 2.5)) / step) * step, width)
            v0, v1 = 0.0, height
            if rng.random() >= 0.3:
                v0 = round(rng.uniform(0, height - 0.2) / step) * step
                v1 = min(round(rng.uniform(v0 + 0.2, height) / step) * step, height)
            if u1 - u0 < 0.2 or v1 - v0 < 0.2:
                continue
            if any(not (u1 <= low or high <= u0 or v1 <= bottom or top <= v0) for _name, (low, high), (bottom, top) in placed):
                continue
            placed.append((f"z{index}", (round(u0, 4), round(u1, 4)), (round(v0, 4), round(v1, 4))))
            break
    return placed


def _random_rooms(count: int, seed: int) -> Iterator[tuple[str, Doc, Doc]]:
    """固定種子亂擺每一面（前牆、後牆、天花、地板、左右牆；三種朝向；軸在正中或偏一邊），挑兩三區交給搜尋
    （畫的框、自動補滿的、整面一區都可能）。回（哪一類面, 方案, 設計設定）。"""
    rng = random.Random(seed)
    for _ in range(count):
        face = rng.choice((*_FACES, "side_whole"))
        turned = rng.random() < 0.25
        flipped = (not turned) and rng.random() < 0.25
        shift = 0.0 if face.startswith("side") else rng.choice((0.0, 0.0, 0.1, -0.05, 0.25, 0.37))
        step = rng.choice((0.05, 0.1, 0.01))
        front = "yL" if turned else "xL" if flipped else "x0"
        back = {"x0": "xL", "xL": "x0", "yL": "y0"}[front]
        left = {"x0": "y0", "xL": "yL", "yL": "x0"}[front]
        right = {"y0": "yL", "yL": "y0", "x0": "xL"}[left]
        cuts: dict[str, list[Box]] = {}
        if face in ("front", "back", "ceiling", "floor"):
            wall = {"front": front, "back": back}.get(face, face)
            cuts[wall] = _random_boxes(rng, _wall_size(turned, wall), rng.choice((1, 2, 2, 3)), step)
        elif face == "side":
            cuts[left] = _random_boxes(rng, _wall_size(turned, left), rng.choice((1, 2)), step)
            cuts[right] = _random_boxes(rng, _wall_size(turned, right), rng.choice((1, 2)), step)
        else:
            cuts[rng.choice((left, right))] = _random_boxes(rng, _wall_size(turned, left), rng.choice((1, 2)), step)
        document = symmetry_document(cuts, across_shift_m=shift, turned=turned, flipped=flipped)
        pool = [(wall, cast(str, zone["id"])) for wall in cuts for zone in _wall_zones(document, wall) if zone["area"] == "rectangle"]
        if face == "side_whole":
            pool.append((right if left in cuts else left, "whole_wall"))
        if len(pool) < 2:
            continue
        zones: dict[str, list[str]] = {}
        for wall, key in rng.sample(pool, min(len(pool), rng.choice((2, 2, 3)))):
            zones.setdefault(wall, []).append(key)
        yield face.removesuffix("_whole"), document, searching(zones)


def _box_of(document: Doc, surface: str) -> tuple[str, tuple[float, float], tuple[float, float]]:
    """一片現在在哪：（牆, u, v）；整面一區就是整面。"""
    wall, _, key = surface.partition("@")
    for zone in _wall_zones(document, wall):
        if key and zone.get("id") == key:
            return wall, _span(zone["u_m"]), _span(zone["v_m"])
    room = cast(dict[str, float], cast(Doc, document["scene"])["room_m"])
    width, height = (room["Lx"], room["Ly"]) if wall in ("floor", "ceiling") else (room["Ly"] if wall[0] == "x" else room["Lx"], room["Lz"])
    return wall, (0.0, width), (0.0, height)


def _typed_box(found: Symmetry, document: Doc, moved: str, words: str) -> tuple[str, tuple[float, float], tuple[float, float]]:
    """照那一句切法，人會打成哪個框：只拿句子印出來的數字（「離左牆 a～b 公尺」「離前牆…」「離地…」），句子沒寫到的
    那一邊沿用要搬的那一區現在的位置。「離○牆」從座標終點那一面起算時反算回座標。"""
    axis = found.axis
    assert axis is not None
    wall, u, v = _box_of(document, moved)
    scheme = Scheme.model_validate(document)
    names = wall_words(scheme)
    room = scheme.scene.room_m
    across_length, along_length = (room.Ly, room.Lx) if axis.along == "x" else (room.Lx, room.Ly)
    spans = {"u": u, "v": v}
    flat, end = wall in ("floor", "ceiling"), wall in axis.end_walls
    printed = re.findall(r"離(\S+?) ([0-9.]+)～([0-9.]+) 公尺", words.split("，就和")[0])
    assert printed, words
    for name, low_text, high_text in printed:
        low, high = float(low_text), float(high_text)
        if name == "地":
            spans["v"] = (low, high)
            continue
        across = name == names[axis.left]
        assert across or name == names[axis.front], words
        from_wall, length = (axis.left, across_length) if across else (axis.front, along_length)
        span = (low, high) if from_wall.endswith("0") else (length - high, length - low)
        # 天花地板：u 是 x、v 是 y；前後牆的 u 是橫向；左右牆的 u 是縱深。
        along_x = axis.along == "x"
        spans[("v" if across == along_x else "u") if flat else "u"] = span
        assert flat or across == end, words
    return wall, spans["u"], spans["v"]


def _do_the_cut(found: Symmetry, document: Doc, design: Doc, moved: str, stays: str, words: str) -> None:
    """照那一句切法、照網頁真的做的那樣做一遍，再配一次：``moved`` 要配成一對，另一片就在 ``stays`` 原本的位置。"""
    wall, u, v = _typed_box(found, document, moved, words)
    key = moved.partition("@")[2]
    # 網頁送的框：這一面的矩形區扣掉自動補滿的（elevation.js::drawnBoxes）、扣掉要搬的那一區；要搬的框先吸邊（snapEdges）。
    others = [CutBox(id=cast(str, zone["id"]), u_m=_span(zone["u_m"]), v_m=_span(zone["v_m"])) for zone in _wall_zones(document, wall)
              if zone["area"] == "rectangle" and not cast(str, zone["id"]).startswith("auto") and zone["id"] != key]
    size = _box_of(document, wall)
    sent = snap_edges(CutBox(id=key or "new", u_m=u, v_m=v), others, size[1][1], size[2][1])
    cut = cut_wall(document, cast(WallName, wall), [*others, sent])
    after = deepcopy(document)
    cast(dict[str, object], cast(Doc, after["scene"])["surface_zones_by_wall"])[wall] = cut["zones"]
    # 設計設定跟著走：新的每一塊沿用 inherited 指的那一塊舊設定（zone_modes.js::remapZoneDesign）。
    old = cast(dict[str, dict[str, object]], design["zones"])[wall]
    moved_design = deepcopy(design)
    cast(dict[str, object], moved_design["zones"])[wall] = {
        name: deepcopy(old[source]) for name, source in cast(dict[str, str], cut["inherited"]).items() if source in old}
    again = pair_zones(Scheme.model_validate(after), DesignSettings.model_validate(moved_design), _Check())
    mine = f"{wall}@{sent.id}"
    partners = [({pair.first, pair.second} - {mine}).pop() for pair in again.pairs if mine in (pair.first, pair.second)]
    told = (document["scene"], design, words, sent, again.pairs, [item.summary for item in again.missed])
    assert len(partners) == 1, told
    was, now = _box_of(document, stays), _box_of(after, partners[0])
    assert was[0] == now[0] and all(abs(new - old_edge) <= PAIR_WITHIN_M + 1e-9 for new, old_edge in zip(
        (*now[1], *now[2]), (*was[1], *was[2]), strict=True)), told


def test_every_offered_cut_really_makes_the_two_zones_a_pair() -> None:
    """畫面那句是「切到…就和○配成一對」：固定種子亂擺每一面（前牆、後牆、天花、地板、左右牆；畫的框、自動補滿的、
    整面一區、第三區），每一條列出來的切法都照網頁真的做的那樣做一遍——從句子印出來的數字取框、吸邊（``snap_edges``）、
    交給切牆工具、設計設定照 ``inherited`` 跟著走——再配一次，這兩區要配成一對。"""
    listed, unlisted = dict.fromkeys(_FACES, 0), dict.fromkeys(_FACES, 0)
    snapped = offered = 0
    for face, document, design in _random_rooms(SWEEP_ROUNDS, seed=1206):
        try:
            found = pair_zones(Scheme.model_validate(document), DesignSettings.model_validate(design), _Check())
        except ValidationError:  # 喇叭被挪出房間之類，這一種擺法不算
            continue
        for item in found.missed:
            if item.kind != "position":
                continue
            assert item.second is not None
            (listed if item.cuts else unlisted)[face] += 1
            snapped += "框邊會被吸到" in item.detail
            offered += len(item.cuts)
            for words in item.cuts:
                moved, stays = (item.second, item.first) if words.startswith("把 {second}") else (item.first, item.second)
                _do_the_cut(found, document, design, moved, stays, words)
    print(f"掃描 {SWEEP_ROUNDS} 輪：有列 {listed}；不列 {unlisted}；切法 {offered} 條；細節寫到吸邊 {snapped} 筆")
    # 不寫死數目：每一面「有列切法」「不列切法」都要掃到，因為吸邊不列的也要有（數目寫在修補報告）。
    assert all(listed[face] and unlisted[face] for face in _FACES), (listed, unlisted)
    assert snapped


FULL = (0.0, HEIGHT)


@pytest.mark.parametrize(("boxes", "other", "shift", "second_way", "first_way"), [
    ([("p", (3.8, 4.0), FULL)], "auto1", 0.0,
     "把 {second} 切到離左牆 0.00～0.20 公尺，它是自動補滿的那一塊，要先在立面圖畫一個框，才搬得動", "把 {first} 切到離左牆 0.20～4.00 公尺，會疊在 {second} 上"),
    ([("p", (1.2, 2.6), FULL)], "auto2", 0.1,
     "把 {second} 切到離左牆 1.60～3.00 公尺，會疊在 {first} 上", "把 {first} 切到離左牆 0.20～1.60 公尺，補滿之後 {second} 那一塊的形狀會跟著變"),
    ([("p", (0.4, 1.4), FULL), ("q", (3.0, 3.8), FULL), ("r", (2.6, 3.0), FULL), ("t", (0.0, 0.4), FULL)], "q", 0.0,
     "把 {second} 切到離左牆 2.60～3.60 公尺，會疊在這一面別的區上", "把 {first} 切到離左牆 0.20～1.00 公尺，會疊在這一面別的區上"),
    # 不是網頁畫得出來的方案（r 只有 10 公分寬，網頁不收這種框）：切牆工具整面拒收，照實寫一句，不亂猜原因。
    ([("p", (1.5, 1.9), FULL), ("q", (2.2, 3.1), FULL), ("r", (0.0, 0.1), FULL)], "q", 0.0,
     "把 {second} 切到離左牆 2.10～2.50 公尺，切牆工具不收這個框", "把 {first} 切到離左牆 0.90～1.80 公尺，切牆工具不收這個框"),
])
def test_each_reason_a_cut_is_not_offered_is_written_out(boxes: list[Box], other: str, shift: float, second_way: str, first_way: str) -> None:
    """兩種切法都列不出來時，細節兩個方向各寫一句為什麼，每一種原因一個固定的例子：要搬的是自動補滿的那一塊、
    會疊在留著那一區上、補滿之後留著那一塊（自動的）形狀跟著變、會疊在別的區上（前三列都是網頁畫得出來的擺法），
    以及不是網頁畫的方案被切牆工具拒收。超出牆面在 ``test_when_no_cut_is_possible…``，吸邊在 ``…snap_away…``。"""
    found = _found(symmetry_document({"x0": boxes}, across_shift_m=shift), {"x0": ["p", other]})
    assert _missed(found) == [("x0@p", f"x0@{other}", "position")] and found.missed[0].cuts == ()
    assert found.missed[0].summary == "只搬其中一區切不出鏡像的位置，各自調"
    assert found.missed[0].detail.endswith(f"只搬其中一區切不出鏡像的位置：{second_way}；{first_way}。")


def test_a_drawn_zone_can_be_cut_to_pair_with_an_auto_filled_one_that_stays_put() -> None:
    """留著的那一區是自動補滿的也列得出切法：p、r 兩個畫的框，另一區是 r 右邊補滿的那一塊（3.00～4.00 之外的 2.60～4.00）。
    把 p 切到鏡像位置之後那一塊原封不動（代號會重編，照位置找），照做真的配成一對；反方向（搬自動那一塊）不列。"""
    document = symmetry_document({"x0": [("p", (1.3, 1.6), FULL), ("r", (2.1, 2.6), FULL)]})
    design = searching({"x0": ["p", "auto3"]})
    found = pair_zones(Scheme.model_validate(document), DesignSettings.model_validate(design), _Check())
    assert _missed(found) == [("x0@p", "x0@auto3", "position")]
    assert found.missed[0].cuts == ("把 {first} 切到離左牆 0.00～1.40 公尺，就和 {second} 配成一對",)
    _do_the_cut(found, document, design, "x0@p", "x0@auto3", found.missed[0].cuts[0])


def test_a_whole_wall_zone_can_be_cut_down_to_pair_with_the_box_on_the_other_side() -> None:
    """左右牆：一面整面一區、另一面只有一個框，兩種切法都列。整面那一面還沒有框，照做就是在鏡像位置畫第一個框
    （設定沿用整面那一區的）：真的切、設定照 ``inherited`` 跟著走、再配一次，這個框和另一面那個框配成一對。"""
    document = symmetry_document({"yL": SIDE})
    design = searching({"y0": ["whole_wall"], "yL": ["s"]})
    found = pair_zones(Scheme.model_validate(document), DesignSettings.model_validate(design), _Check())
    assert _missed(found) == [("y0", "yL@s", "position")]
    assert found.missed[0].cuts == ("把 {second} 切到離前牆 0.00～5.00 公尺、離地 0.00～2.70 公尺，就和 {first} 配成一對",
                                    "把 {first} 切到離前牆 1.00～3.00 公尺、離地 0.50～2.20 公尺，就和 {second} 配成一對")
    cut = cut_wall(document, "y0", [CutBox(id="new", u_m=SIDE[0][1], v_m=SIDE[0][2])])
    assert cast(dict[str, str], cut["inherited"])["new"] == "whole_wall"
    after = deepcopy(document)
    cast(dict[str, object], cast(Doc, after["scene"])["surface_zones_by_wall"])["y0"] = cut["zones"]
    moved_design = deepcopy(design)
    cast(dict[str, object], moved_design["zones"])["y0"] = {key: {"mode": "search"} for key in cast(dict[str, str], cut["inherited"])}
    again = pair_zones(Scheme.model_validate(after), DesignSettings.model_validate(moved_design), _Check())
    assert ("y0@new", "yL@s") in [(pair.first, pair.second) for pair in again.pairs]


def test_a_mirror_that_lands_a_hair_outside_the_wall_still_counts_as_inside() -> None:
    """審查 A-N1：軸在 1.55、e 的右緣在 3.10，鏡像左緣算出來是 −4e-16（浮點）。那是貼著牆，不是超出牆面。"""
    found = _found(symmetry_document({"x0": [("e", (2.2, 3.1), (0.3, 2.4))]}, across_shift_m=-0.45), {"x0": ["e"]})
    assert _missed(found) == [("x0@e", None, "no_zone")]
    assert found.missed[0].summary.startswith("對稱軸另一邊沒有一樣大小的區，各自調")
    assert "超出牆面" not in found.missed[0].summary + found.missed[0].detail


def test_same_size_zones_off_by_five_centimetres_read_as_misplaced_not_missized() -> None:
    """審查 A-N2：兩區一樣大、位置偏 5 公分：列上寫「位置沒對齊」，不是「大小不同」；大小真的不同才寫大小。"""
    found = _found(symmetry_document({"x0": [FRONT_PAIR[0], ("a", (2.55, 3.65), (0.3, 2.4))]}), {"x0": ["b", "a"]})
    assert _missed(found) == [("x0@b", "x0@a", "position")]
    assert found.missed[0].summary.startswith("位置沒對齊，各自調")
    sized = _found(symmetry_document({"x0": [FRONT_PAIR[0], ("a", (2.5, 3.4), (0.3, 2.4))]}), {"x0": ["b", "a"]})
    assert sized.missed[0].summary.startswith("大小不同，各自調")


def test_front_and_back_swap_in_the_wording_when_the_front_wall_is_the_far_one() -> None:
    """審查 A-N3：前牆是 xL（整組轉 180 度）時，x 小的那一邊靠的是後牆 x0。天花兩區縱深差 10 公分，差的是靠後牆那一邊。"""
    found = _found(symmetry_document({"ceiling": [("p", (0.5, 1.5), (0.3, 1.3)), ("q", (0.6, 1.5), (2.7, 3.7))]}, flipped=True),
                   {"ceiling": ["p", "q"]})
    assert found.axis is not None and found.axis.front == "xL"
    assert "疊著比：靠後牆那一邊差 10 公分。" in found.missed[0].detail



def test_a_lone_zone_whose_mirror_leaves_the_wall_gets_no_cut() -> None:
    """落單的區，鏡像位置超出牆面（軸偏 30 公分，貼著左邊牆角的一區）：不叫人「在那裡切出一區」。"""
    found = _found(symmetry_document({"x0": [("e", (0.0, 0.3), (0.3, 2.4))]}, across_shift_m=0.3), {"x0": ["e"]})
    assert _missed(found) == [("x0@e", None, "outside")] and found.missed[0].cuts == ()
    assert "切出" not in found.missed[0].detail and "超出牆面" in found.missed[0].detail and "4.30" not in found.missed[0].detail
    assert found.missed[0].summary == "鏡像過去會超出牆面，沒辦法配成一對，各自調"


def test_a_zone_across_the_axis_but_off_centre_is_not_told_to_cut_on_itself() -> None:
    """跨在軸上但沒有置中的區：鏡像位置就是它自己那一帶，不叫人「在那裡切出一區」。"""
    found = _found(symmetry_document({"x0": [("m", (1.0, 3.1), (0.3, 2.4))]}), {"x0": ["m"]})
    assert _missed(found) == [("x0@m", None, "straddle")] and found.missed[0].cuts == ()
    assert "切出" not in found.missed[0].detail and "沒有置中" in found.missed[0].detail


def test_switched_off_pairs_nothing_and_a_missing_field_means_on() -> None:
    """抓缺欄位讀成關、或關掉還在配：設計設定沒有那一格＝開；寫了「關」就一對都不配、也不問引擎。"""
    check = _Check()
    off = _paired(ALL_CUTS, {}, check, design=searching(ALL_ZONES, left_right_symmetry="off"))
    assert off.enabled is False and off.pairs == () and off.missed == () and check.asked == [] and off.alone == 5
    on = _paired(ALL_CUTS, {}, design=json.loads(OLD_DESIGN))
    assert on.enabled is True and _pairs(on) == (FRONT, SIDES)


# ---- 設計設定那一格 ----

def test_old_design_files_keep_their_bytes_and_only_off_is_written(tmp_path: Path) -> None:
    """舊設計檔讀進來再寫回去逐位相同；開著不寫這一格，關掉才寫；只收「關」這一個值。"""
    scheme = Scheme.model_validate(symmetry_document(ALL_CUTS))
    (tmp_path / "designs").mkdir()
    path = tmp_path / "designs" / f"{NAME}.json"
    path.write_text(OLD_DESIGN, encoding="utf-8")
    old = read_design(tmp_path, NAME)
    assert old is not None
    write_design(tmp_path, old, scheme)
    assert path.read_bytes() == OLD_DESIGN.encode()
    assert "left_right_symmetry" not in old.model_dump(mode="json")
    off = DesignSettings.model_validate(json.loads(OLD_DESIGN) | {"left_right_symmetry": "off"})
    write_design(tmp_path, off, scheme)
    assert json.loads(path.read_text(encoding="utf-8"))["left_right_symmetry"] == "off"
    for bad in ("on", True, False, ""):
        with pytest.raises(ValidationError):
            DesignSettings.model_validate(json.loads(OLD_DESIGN) | {"left_right_symmetry": bad})


def test_the_design_route_round_trips_an_old_file_byte_for_byte(tmp_path: Path) -> None:
    """網頁那條路：讀舊設計檔（GET）、原樣存回（PUT），檔案逐位相同。"""
    (tmp_path / "schemes").mkdir()
    (tmp_path / "designs").mkdir()
    (tmp_path / "schemes" / f"{NAME}.json").write_text(Scheme.model_validate(symmetry_document(ALL_CUTS)).model_dump_json())
    path = tmp_path / "designs" / f"{NAME}.json"
    path.write_text(OLD_DESIGN, encoding="utf-8")
    client = TestClient(create_app(GuiSettings(engine_commit="a" * 40, data_dir=tmp_path)), base_url="http://127.0.0.1")
    design = client.get(f"/api/designs/{NAME}").json()["design"]
    assert "left_right_symmetry" not in design
    assert client.put(f"/api/designs/{NAME}", json=design).status_code == 200
    assert path.read_bytes() == OLD_DESIGN.encode()


# ---- 開跑計畫與參數 ----

def test_each_pair_becomes_one_tie_after_the_other_start_arguments() -> None:
    """有一對就多一個 ``--tie A=B``（A 是面序在前那一片），接在原本的參數後面；拿掉它們，剩下的跟主線逐位相同。"""
    plan = _plan()
    assert plan.ties == (FRONT, SIDES)
    arguments = start_arguments(plan)
    assert arguments[-4:] == ["--tie", "x0@b=x0@a", "--tie", "y0@s=yL@s"]
    assert _sha(json.dumps(arguments[:-4])) == MAIN_START and "--tie" not in arguments[:-4]


def test_ties_never_reach_the_arguments_the_surrogate_build_checks_with() -> None:
    """抓 ``--tie`` 被移進建替身核固定材料那一份：有兩對時那一份照樣跟主線逐位相同。"""
    plan = _plan()
    assert plan.ties != ()
    configuration = configuration_arguments(plan)
    assert "--tie" not in configuration and _sha(json.dumps(configuration)) == MAIN_CONFIGURATION


def test_switched_off_or_nothing_to_pair_is_byte_for_byte_the_main_branch() -> None:
    """抓關掉時也送 ``--tie``、或沒有一對時計畫多寫東西：開跑參數與計畫存檔都跟主線逐位相同。"""
    off = _plan(design=searching(ALL_ZONES, left_right_symmetry="off"))
    assert off.ties == () and "--tie" not in start_arguments(off)
    assert _sha(json.dumps(start_arguments(off))) == MAIN_START and _sha(off.model_dump_json()) == MAIN_PLAN
    lone = _plan({"x0": ["b"]})
    assert _sha(json.dumps(start_arguments(lone))) == MAIN_START_LONE and _sha(lone.model_dump_json()) == MAIN_PLAN_LONE
    assert SearchPlan.model_validate_json(off.model_dump_json()).ties == ()


def test_the_real_engine_check_decides_and_its_refusal_is_said_plainly() -> None:
    """抓不經引擎檢查就配：前牆兩區位置對上，但一區多准一種材料、或兩區最深不同，引擎的開跑前檢查拒收，
    那一對不進計畫、畫面上算沒配成（原因是白話）；左右牆那一對照配。"""
    scheme = Scheme.model_validate(symmetry_document(ALL_CUTS))
    for change, word in (({"allowed": ["glass_wool_48k", "none"]}, "材料"), ({"max_depth_m": 0.2}, "最深")):
        document = searching(ALL_ZONES)
        cast(dict[str, dict[str, dict[str, object]]], document["zones"])["x0"]["a"].update(change)
        design = DesignSettings.model_validate(document)
        plan = plan_search(scheme, design, GradientRequest(), SETTINGS)
        assert plan.ties == (SIDES,) and start_arguments(plan)[-2:] == ["--tie", "y0@s=yL@s"]
        found = plan_symmetry(scheme, design, plan, SETTINGS)
        assert _pairs(found) == (SIDES,) and _missed(found) == [("x0@b", "x0@a", "settings")]
        assert word in found.missed[0].reason and "綁定" not in found.missed[0].reason and "x0@" not in found.missed[0].reason


def test_starting_from_the_page_sends_one_tie_per_pair(tmp_path: Path) -> None:
    """從網頁開跑那條路整段（設計檔沒有那一格＝開）：交給命令列的參數帶兩對 ``--tie``，存下來的計畫也記著。"""
    from tests.engine.test_gui_gradient_jobs import FakeLauncher, _jobs, _place_surrogate

    data_dir = tmp_path / "data"
    (data_dir / "schemes").mkdir(parents=True)
    (data_dir / "designs").mkdir()
    scheme = Scheme.model_validate(symmetry_document(ALL_CUTS))
    (data_dir / "schemes" / f"{NAME}.json").write_text(scheme.model_dump_json(), encoding="utf-8")
    (data_dir / "designs" / f"{NAME}.json").write_text(OLD_DESIGN, encoding="utf-8")
    _place_surrogate(data_dir, scheme)
    launcher = FakeLauncher()
    view = _jobs(data_dir, launcher).start(NAME, GradientRequest())
    argv = launcher.launched[-1][1]
    assert [argv[index + 1] for index, word in enumerate(argv) if word == "--tie"] == ["x0@b=x0@a", "y0@s=yL@s"]
    stored = json.loads((data_dir / "gradient" / str(view["job_id"]) / "job.json").read_text(encoding="utf-8"))
    assert stored["plan"]["ties"] == [list(FRONT), list(SIDES)]


_CANNOT_START = "位置對上了；這份設定現在開不了搜尋，等開得了再看兩區的設定一不一樣"
_NOT_CHECKED = "{first} 和 {second} 的位置對上了，但這份設定開跑前檢查沒過，先看不了兩區的設定一不一樣。"


def test_a_start_check_that_cannot_even_load_its_inputs_pairs_nothing_instead_of_crashing(tmp_path: Path) -> None:
    """引擎檢查要的檔讀不到（能力表不在）：不丟例外（分區頁的限制檢查約定一律回得出結果），那幾對照原因算沒配成。"""
    scheme = Scheme.model_validate(symmetry_document(ALL_CUTS))
    design = DesignSettings.model_validate(searching(ALL_ZONES))
    plan = plan_search(scheme, design, GradientRequest(), SETTINGS, pair=False)
    found = plan_symmetry(scheme, design, plan, SETTINGS, TieInputs(capabilities=tmp_path / "missing.toml"))
    assert found.pairs == () and [item.kind for item in found.missed] == ["settings", "settings"]
    # 整份設定核不了，不是這一對兩區設定不一樣：不叫人把兩區「設成一樣」，例外原文（路徑）不上畫面。
    for item in found.missed:
        assert (item.summary, item.reason, item.detail) == (_CANNOT_START, "這份設定開跑前檢查沒過", _NOT_CHECKED)
        assert "missing.toml" not in item.summary + item.reason + item.detail
    assert plan_search(scheme, design, GradientRequest(), SETTINGS, tie_inputs=TieInputs(capabilities=tmp_path / "missing.toml")).ties == ()


def test_resuming_a_search_started_before_ties_sends_no_tie(tmp_path: Path) -> None:
    """預設開不准動到已經開跑的搜尋：一場沒有綁定的舊搜尋（工作記錄裡的計畫沒有 ties 那一格），方案現在剛好配得上兩對；
    接續送出的只有「resume 資料夾」（讀搜尋資料夾裡存的，不重算計畫），沒有 ``--tie``，存著的計畫也沒被改寫。"""
    from tests.engine.test_gui_gradient_jobs import STATUS, FakeLauncher, _finish, _folder, _jobs, _place_surrogate

    data_dir = tmp_path / "data"
    (data_dir / "schemes").mkdir(parents=True)
    (data_dir / "designs").mkdir()
    scheme = Scheme.model_validate(symmetry_document(ALL_CUTS))
    (data_dir / "schemes" / f"{NAME}.json").write_text(scheme.model_dump_json(), encoding="utf-8")
    # 這一支之前開跑的樣子：計畫沒有綁定（用「關」開跑，存下來的計畫跟主線逐位相同）。
    write_design(data_dir, DesignSettings.model_validate(searching(ALL_ZONES, left_right_symmetry="off")), scheme)
    _place_surrogate(data_dir, scheme)
    launcher = FakeLauncher()
    jobs = _jobs(data_dir, launcher)
    view = jobs.start(NAME, GradientRequest())
    job_file = data_dir / "gradient" / str(view["job_id"]) / "job.json"
    stored = job_file.read_text(encoding="utf-8")
    assert "--tie" not in launcher.launched[-1][1] and "ties" not in json.loads(stored)["plan"]
    assert _sha(json.dumps(json.loads(stored)["plan"], separators=(",", ":"))) == MAIN_PLAN
    folder = _folder(data_dir, view, STATUS)
    jobs.stop(str(view["job_id"]))
    _finish(data_dir, launcher, view, EXIT_STOPPED)
    # 現在的設計檔沒有那一格（預設開），同一份方案新開一場會配出兩對。
    (data_dir / "designs" / f"{NAME}.json").write_text(OLD_DESIGN, encoding="utf-8")
    assert jobs.plan(scheme, read_design(data_dir, NAME), GradientRequest())[0].ties == (FRONT, SIDES)
    jobs.resume(str(view["job_id"]))
    argv = launcher.launched[-1][1]
    assert argv[-2:] == ["resume", str(folder)] and "--tie" not in argv
    assert "ties" not in json.loads(job_file.read_text(encoding="utf-8"))["plan"]


_NO_JAX_AFTER_A_CALL = """
import sys, threading
from pathlib import Path
from starlette.testclient import TestClient
from aosr.config.paths import config_path
from aosr.gui.app import GuiSettings, create_app
from aosr.gui.gradient_jobs import GradientJobs
from aosr.gui.gradient_plan import GradientRequest
from aosr.gui.zone_design import DesignSettings
from aosr.reporting.scheme import Scheme
from tests.engine._zone_symmetry_room import CENTRED, FRONT_PAIR, SIDE, searching, symmetry_document
from tests.engine.test_gui_gradient_jobs import FakeLauncher

assert "jax" not in sys.modules, "載入期就有 JAX（另一題在守）"
data = Path(sys.argv[1])
document = symmetry_document({"x0": FRONT_PAIR, "y0": SIDE, "yL": SIDE, "ceiling": CENTRED})
design = searching({"x0": ["b", "a"], "y0": ["s"], "yL": ["s"]})
if sys.argv[2] == "fixed":  # 准用清單帶一筆固定材料：引擎檢查會多走「把固定材料排進構造清單」那一支
    from aosr.config.construction_presets import load_construction_presets
    fixed = load_construction_presets(config_path("construction_presets.toml"))[0].id
    for wall, zone in (("x0", "b"), ("x0", "a"), ("y0", "s"), ("yL", "s")):
        design["zones"][wall][zone]["allowed"] = ["glass_wool_48k", "none", fixed]
client = TestClient(create_app(GuiSettings(engine_commit="a" * 40, data_dir=data / "page")), base_url="http://127.0.0.1")
body = client.post("/api/zone-limits", json={"scheme": document, "design": design}).json()
assert len(body["symmetry"]["pairs"]) == 2, body["symmetry"]
after_page = "jax" in sys.modules
jobs = GradientJobs(data / "jobs", launcher=FakeLauncher(), engine_commit="c" * 40, capabilities=config_path("capabilities.toml"),
                    scheme_lock=threading.RLock())
plan, _, _ = jobs.plan(Scheme.model_validate(document), DesignSettings.model_validate(design), GradientRequest())
assert len(plan.ties) == 2, plan.ties
print(f"zone-limits 後 jax={after_page} plan 後 jax={'jax' in sys.modules}")
raise SystemExit(1 if "jax" in sys.modules else 0)
"""


@pytest.mark.parametrize("allowed", ["plain", "fixed"])
def test_the_web_process_still_has_no_jax_after_really_pairing(tmp_path: Path, allowed: str) -> None:
    """網頁行程真的配過一次之後還是沒有 JAX（審查 A 第 1 條）：另開乾淨行程，叫一次分區頁的限制檢查（回兩對）、
    一次帶兩對的 ``GradientJobs.plan``（預覽與開跑共用），之後 ``sys.modules`` 裡不准有 ``jax``。
    准用清單兩種都跑：不寫（全部可調材料）與帶一筆固定材料（引擎檢查多走一支）。
    ``test_web_process_never_imports_jax`` 只看載入那一刻，引擎檢查是第一次被叫才載入它要的模組，那一題抓不到。"""
    env = forward_cpu_pins(child_process_env())
    env["TMPDIR"] = str(tmp_path)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    done = subprocess.run([sys.executable, "-c", _NO_JAX_AFTER_A_CALL, str(tmp_path), allowed], env=env, cwd=tmp_path,
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stdout[-300:] + done.stderr[-1500:]


# ---- 分區頁從伺服器拿結果 ----

def _client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(GuiSettings(engine_commit="a" * 40, data_dir=tmp_path)), base_url="http://127.0.0.1")


def test_the_zone_page_gets_the_pairing_from_the_limits_route(tmp_path: Path) -> None:
    """限制檢查那一支順便回配對（同一支純函式）：開著列兩對，關掉一對都沒有，草稿還不是合法方案時照講、不回 500。"""
    client = _client(tmp_path)
    document = symmetry_document(ALL_CUTS)
    body = client.post("/api/zone-limits", json={"scheme": document, "design": searching(ALL_ZONES)}).json()
    found = body["symmetry"]
    assert found["available"] is True and found["enabled"] is True
    assert [(pair["first"], pair["second"]) for pair in found["pairs"]] == [FRONT, SIDES]
    assert found["centred"] == ["ceiling@c"] and found["alone"] == 1 and found["missed"] == []
    off = client.post("/api/zone-limits", json={"scheme": document, "design": searching(ALL_ZONES, left_right_symmetry="off")}).json()
    assert off["symmetry"]["enabled"] is False and off["symmetry"]["pairs"] == []
    none = client.post("/api/zone-limits", json={"scheme": document, "design": None}).json()["symmetry"]
    assert none["available"] is True and none["enabled"] is True and none["searched"] == [] and none["pairs"] == []
    half = deepcopy(document)
    del half["speakers"]
    response = client.post("/api/zone-limits", json={"scheme": half, "design": searching(ALL_ZONES)})
    assert response.status_code == 200 and response.json()["symmetry"]["available"] is False
    assert response.json()["symmetry"]["message"]


def test_a_design_that_cannot_start_a_search_still_answers_the_limits_route(tmp_path: Path) -> None:
    """審查 A-K2：兩區最深都設 0.03 m（放不下，開不了搜尋），但位置對上。限制檢查照回 200，那一對算沒配成、講原因；
    不准因為配對那一段就整支回 500（原本的限制紅字也會跟著沒了）。"""
    design = searching({"x0": ["b", "a"]})
    for zone in ("b", "a"):
        cast(dict[str, dict[str, dict[str, object]]], design["zones"])["x0"][zone]["max_depth_m"] = 0.03
    response = _client(tmp_path).post("/api/zone-limits", json={"scheme": symmetry_document({"x0": FRONT_PAIR}), "design": design})
    assert response.status_code == 200
    found = response.json()["symmetry"]
    assert found["available"] is True and found["pairs"] == []
    assert [item["kind"] for item in found["missed"]] == ["settings"]
    missed = found["missed"][0]
    assert (missed["summary"], missed["reason"]) == (_CANNOT_START, "這份設定現在開不了搜尋")
    opening = "{first} 和 {second} 的位置對上了，但這份設定現在開不了搜尋：有超過限制的區"
    assert missed["detail"].startswith(opening) and missed["detail"].endswith("。") and not missed["detail"].endswith("。。")
    assert "0.03 公尺" in missed["detail"] and "設成一樣" not in missed["summary"] + missed["reason"] + missed["detail"]


def test_a_start_the_engine_refuses_outright_pairs_nothing_instead_of_crashing() -> None:
    """審查 A-K3：起跑配置本身引擎就不收（構造清單是空的，起跑樣板不認識）：不丟例外，那一對照原因算沒配成。"""
    scheme = Scheme.model_validate(symmetry_document({"x0": FRONT_PAIR}))
    design = DesignSettings.model_validate(searching({"x0": ["b", "a"]}))
    plan = plan_search(scheme, design, GradientRequest(), SETTINGS, pair=False)
    found = plan_symmetry(scheme, design, plan, SETTINGS, TieInputs(presets=()))
    assert found.pairs == () and [item.kind for item in found.missed] == ["settings"]
    item = found.missed[0]
    assert (item.summary, item.reason, item.detail) == (_CANNOT_START, "這份設定開跑前檢查沒過", _NOT_CHECKED)
    assert "preset" not in item.summary + item.reason + item.detail and "glass_wool" not in item.summary + item.reason + item.detail


def test_the_page_lists_a_refused_pair_with_the_engine_reason(tmp_path: Path) -> None:
    """分區頁那條路也過引擎：兩區最深不同，回的是沒配成與白話原因。"""
    document = searching({"x0": ["b", "a"]})
    cast(dict[str, dict[str, dict[str, object]]], document["zones"])["x0"]["a"]["max_depth_m"] = 0.2
    found = _client(tmp_path).post("/api/zone-limits", json={"scheme": symmetry_document(ALL_CUTS), "design": document}).json()["symmetry"]
    assert found["pairs"] == [] and [item["kind"] for item in found["missed"]] == ["settings"]
    assert "最深" in found["missed"][0]["reason"]


# ---- 端到端：網頁算的參數交給命令列 start ----

_PROBE = f"""
import sys
from aosr.reporting import fem_reduced_scheme


def reached(*args, **kwargs):
    raise RuntimeError("{PASSED}")


fem_reduced_scheme.export_operators = reached
from aosr.gradient import cli
raise SystemExit(cli.main(sys.argv[1:]))
"""


def _start(tmp_path: Path, design_arguments: list[str]) -> subprocess.CompletedProcess[str]:
    """真的跑命令列 ``start``（另一個行程）：參數解析、起跑配置、綁定表的開跑前檢查都是產品那一份；
    走到匯出完整模型（之後才是替身與搜尋本體）就停，停在那裡表示開跑前檢查收下了。"""
    scheme = tmp_path / "scheme.json"
    scheme.write_text(Scheme.model_validate(symmetry_document(ALL_CUTS)).model_dump_json(), encoding="utf-8")
    env = forward_cpu_pins(child_process_env())
    env["TMPDIR"] = str(tmp_path)
    arguments = ["start", "--scheme", str(scheme), "--surrogate", str(tmp_path / "surrogate.npz"),
                 "--purpose", "dedicated_two_channel_listening_room", "--root", str(tmp_path / "root"), *design_arguments]
    return subprocess.run([sys.executable, "-c", _PROBE, *arguments], env=env, cwd=tmp_path, capture_output=True, text=True,
                          timeout=300)


def test_the_command_line_accepts_the_ties_the_web_page_computed(tmp_path: Path) -> None:
    """自編房間配出兩對，網頁算出的開跑參數原樣交給命令列 ``start``：引擎的開跑前檢查收下（走到匯出完整模型那一步）。
    對照：同一份參數把其中一對換成設定不同的兩片，同一條路在開跑前檢查就被點名拒收。"""
    arguments = [word if word != "gpu" else "cpu" for word in start_arguments(_plan())]
    assert arguments.count("--tie") == 2
    done = _start(tmp_path, arguments)
    assert PASSED in done.stderr and "綁定" not in done.stderr, done.stderr
    wrong = [word if word != "x0@b=x0@a" else "x0@b=x0@auto1" for word in arguments]
    refused = _start(tmp_path, wrong)
    assert refused.returncode == 1 and PASSED not in refused.stderr and "綁定 x0@auto1=x0@b" in refused.stderr, refused.stderr
