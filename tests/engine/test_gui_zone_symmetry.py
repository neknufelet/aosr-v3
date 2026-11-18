"""左右對稱的畫面那半邊（#1206 第 2 支）：配對純函式每條規則一題、設計設定那一格（缺＝開、關才寫、舊檔逐位不變）、
開跑計畫與 ``--tie``（只加在開跑參數、不進建替身核固定材料那一份；關掉與沒有一對時跟主線逐位相同）、分區頁從伺服器拿結果、
以及一題端到端：網頁算出的參數真的交給命令列 ``start``，引擎的開跑前檢查收下。

房間全是考卷自己編的（``_zone_symmetry_room``，5.0 × 4.0 × 2.7 m）。「跟主線逐位相同」比的是改動前主線
（e9c69c48）對同一份輸入算出來的雜湊，釘在這個檔裡。"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Callable
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
from aosr.gui.zone_symmetry import Symmetry, pair_zones
from aosr.reporting.scheme import Scheme
from aosr.runtime import child_process_env
from tests.engine._cpu_pins import forward_cpu_pins
from tests.engine._zone_symmetry_room import CENTRED, FRONT_PAIR, NAME, SIDE, Box, Doc, searching, symmetry_document

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
    assert _missed(shifted) == [("y0", "yL", "side_walls"), ("floor", None, "no_zone")]


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
    assert missed.figure is not None
    assert missed.figure.wall == "x0" and missed.figure.mirrored_u_m == pytest.approx((2.5, 3.6))


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


def test_a_start_check_that_cannot_even_load_its_inputs_pairs_nothing_instead_of_crashing(tmp_path: Path) -> None:
    """引擎檢查要的檔讀不到（能力表不在）：不丟例外（分區頁的限制檢查約定一律回得出結果），那幾對照原因算沒配成。"""
    scheme = Scheme.model_validate(symmetry_document(ALL_CUTS))
    design = DesignSettings.model_validate(searching(ALL_ZONES))
    plan = plan_search(scheme, design, GradientRequest(), SETTINGS, pair=False)
    found = plan_symmetry(scheme, design, plan, SETTINGS, TieInputs(capabilities=tmp_path / "missing.toml"))
    assert found.pairs == () and [item.kind for item in found.missed] == ["settings", "settings"]
    assert all("開跑前檢查沒過" in item.reason for item in found.missed)
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


def test_the_web_process_still_has_no_jax_after_really_pairing(tmp_path: Path) -> None:
    """網頁行程真的配過一次之後還是沒有 JAX（審查 A 第 1 條）：另開乾淨行程，叫一次分區頁的限制檢查（回兩對）、
    一次帶兩對的 ``GradientJobs.plan``（預覽與開跑共用），之後 ``sys.modules`` 裡不准有 ``jax``。
    ``test_web_process_never_imports_jax`` 只看載入那一刻，引擎檢查是第一次被叫才載入它要的模組，那一題抓不到。"""
    env = forward_cpu_pins(child_process_env())
    env["TMPDIR"] = str(tmp_path)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    done = subprocess.run([sys.executable, "-c", _NO_JAX_AFTER_A_CALL, str(tmp_path)], env=env, cwd=tmp_path,
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
