"""每一區自己的可調範圍（#1199 範圍第 1 支）——格式、對材料庫核、收窄後的深度判法。

數字全是考卷自己編的：房間是 ``_zone_symmetry_room`` 的假房間（5.0 × 4.0 × 2.7 m），材料是內建家族
（設定檔 ``material_gradient.toml``：48K 產品厚度 0.05／0.1／0.2、空腔 0～0.25、總深 0.3；24K 不調空腔）、
``_sample_material_library`` 那一筆可最佳化材料（棉厚不調 0.075、空腔 0～0.12、總深 0.18），加一筆這裡編的
（棉厚產品 0.03／0.05／0.08／0.12、空腔 0～0.1、總深 0.2）。「跟主線逐位相同」比的是改動前主線（46b1ea3d）
對同一份輸入算出來的雜湊（``_zone_range_control``），釘在這個檔裡。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from pydantic import TypeAdapter, ValidationError

from aosr.config.material_gradient import MaterialGradientSettings
from aosr.gradient import depth_fit
from aosr.gradient.depth_fit import ZoneRange, narrowed_family, range_problem, start_problem, target_fit
from aosr.gradient.start_config import presets_table
from aosr.gui.gradient_plan import builtin_depth_targets, library_depth_targets
from aosr.gui.zone_design import DesignSettings, ZoneDesign, write_design
from aosr.gui.zone_limits import DepthTargets, zone_limits
from aosr.materials import library
from aosr.materials.search_library import with_user_materials
from tests.engine._sample_material_library import SAMPLE_USER_ENTRIES
from tests.engine._zone_range_control import SETTINGS, WOOL, control_design, control_hashes, control_scheme, limits_bytes, sha

_ENTRY: TypeAdapter[library.LibraryEntry] = TypeAdapter(library.LibraryEntry)
SAMPLE = "demo109_rockwool60k_capped"
STEPS = "demo_wool_steps"
# 改動前主線（46b1ea3d）對 ``_zone_range_control`` 那一份沒寫範圍的輸入算出來的（計畫存檔＋開跑參數＋建替身那一份參數、
# 起跑設計、兩份帳本標頭＋資料夾名＋project.json、網頁預覽回覆、分區頁限制檢查回覆），各自位元組的 SHA-256。
MAIN = {"plan": "5ad63e325f1be6ebe6e2667ff0aae32fcdd0f5e566363606787b82d79a6490aa",
        "designs": "5e1e59739d1601558b2d67efc73e45180e1ec00039deb2fb1c6b453261a4d3c2",
        "ledger": "baaf346edde5911d0b575359266304d780aed23729712464d7f60b8e4c6df079",
        "preview": "226720d6dae164a14d098a27c2dea515afa39e8b8efe7e9e6b91c1b341d152f8",
        "limits": "634c0333b9cdda95b933e4c21ffc2ecbd07e6f047b8265a2f5fa48ee0d008aa8"}


def _steps_entry() -> library.LibraryEntry:
    """考卷自己編的一筆可最佳化材料：棉厚四個產品厚度、空腔 0～0.1、總深 0.2。"""
    return _ENTRY.validate_python({
        "kind": "optimizable", "id": STEPS, "label": "示範階梯棉（自編）", "provenance": "model", "source": "考卷自編",
        "porous": {"kind": "porous", "thickness_m": 0.05, "flow_resistivity_pa_s_m2": 17500.0, "material_class": "fibrous"},
        "porous_thickness": {"adjustable": True, "range_m": [0.03, 0.12], "choices_m": [0.03, 0.05, 0.08, 0.12], "initial_m": 0.05},
        "cavity": {"adjustable": True, "range_m": [0.0, 0.1], "initial_m": 0.02}, "max_total_depth_m": 0.2})


def _entries() -> tuple[library.LibraryEntry, ...]:
    return (*(_ENTRY.validate_python(entry) for entry in SAMPLE_USER_ENTRIES), _steps_entry())


def _library_settings() -> MaterialGradientSettings:
    return with_user_materials(SETTINGS, presets_table(), _entries()).settings


def _targets() -> DepthTargets:
    return library_depth_targets(_library_settings(), _entries())


def _ranged(zone_fields: dict[str, dict[str, object]], *, base: dict[str, object] | None = None) -> dict[str, object]:
    """控制組那一份設計設定，再把 ``zone_fields``（"牆@區" → 那一區多的欄位）蓋上去。"""
    document = json.loads(json.dumps(base if base is not None else control_design()))
    zones = cast(dict[str, dict[str, dict[str, object]]], document["zones"])
    for key, fields in zone_fields.items():
        wall, _, zone = key.partition("@")
        zones.setdefault(wall, {}).setdefault(zone, {"mode": "search"}).update(fields)
    return cast(dict[str, object], document)


def _limits(document: dict[str, object], targets: DepthTargets | None = None) -> dict[str, object]:
    found = targets if targets is not None else builtin_depth_targets(SETTINGS)
    return zone_limits(control_scheme().model_dump(mode="json"), document, depth_targets=lambda: found)


def _red(reply: dict[str, object], wall: str, zone: str) -> list[dict[str, str]]:
    zones = cast(dict[str, dict[str, list[dict[str, str]]]], reply["zones"])
    return [line for line in zones.get(wall, {}).get(zone, []) if line["level"] == "red"]


# ---- 格式（設計設定那一格） ----

def test_zone_range_is_written_only_when_given_and_old_design_files_keep_their_bytes(tmp_path: Path) -> None:
    """抓沒給範圍也寫一格（舊設計設定變了位元組）、或給了卻沒存下來。"""
    (tmp_path / "designs").mkdir()
    old = DesignSettings.model_validate(control_design())
    write_design(tmp_path, old, control_scheme())
    path = tmp_path / "designs" / f"{old.scheme_id}.json"
    written = path.read_bytes()
    assert b"ranges" not in written and written == old.model_dump_json().encode()
    assert DesignSettings.model_validate_json(written).model_dump_json().encode() == written
    given = {"porous_m": [0.05, 0.1], "cavity_m": [0.0, 0.03]}
    ranged = DesignSettings.model_validate(_ranged({"x0@b": {"ranges": {WOOL: given}}}))
    write_design(tmp_path, ranged, control_scheme())
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["zones"]["x0"]["b"]["ranges"] == {WOOL: given}
    assert "ranges" not in saved["zones"]["x0"]["a"]
    only_porous = ZoneDesign.model_validate({"mode": "search", "ranges": {WOOL: {"porous_m": [0.05, 0.1]}}})
    assert only_porous.model_dump_json() == '{"mode":"search","ranges":{"glass_wool_48k":{"porous_m":[0.05,0.1]}}}'


@pytest.mark.parametrize("mode", ["fixed", "keep_original"])
def test_zone_range_only_on_zones_handed_to_search(mode: str) -> None:
    with pytest.raises(ValidationError, match="可調範圍只給交給搜尋的區"):
        ZoneDesign.model_validate({"mode": mode, "ranges": {WOOL: {"porous_m": [0.05, 0.1]}}})


def test_zone_range_key_must_be_in_the_allowed_list_when_one_is_given() -> None:
    with pytest.raises(ValidationError, match="不在這一區的准用清單"):
        ZoneDesign.model_validate({"mode": "search", "allowed": [WOOL, "none"],
                                   "ranges": {"glass_wool_24k": {"porous_m": [0.05, 0.1]}}})
    listed = ZoneDesign.model_validate({"mode": "search", "allowed": [WOOL, "none"], "ranges": {WOOL: {"porous_m": [0.05, 0.1]}}})
    assert listed.ranges == {WOOL: ZoneRange(porous_m=(0.05, 0.1))}


@pytest.mark.parametrize(("ranges", "message"), [
    ({WOOL: {"porous_m": [0.1, 0.05]}}, "下限要不大於上限"),
    ({WOOL: {"porous_m": [0.05, 0.1], "cavity_m": [0.03, 0.01]}}, "下限要不大於上限"),
    ({WOOL: {"porous_m": [-0.05, 0.1]}}, "不能是負的"),
    ({"none": {"porous_m": [0.05, 0.1]}}, "不貼"),
    ({}, "at least 1"),
    ({WOOL: {"porous_m": [0.05, 0.1], "extra": 1}}, "Extra inputs"),
])
def test_zone_range_shape_is_checked_before_the_library(ranges: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        ZoneDesign.model_validate({"mode": "search", "ranges": ranges})


# ---- 對材料庫核（只准收窄、產品厚度、空腔） ----

def test_zone_range_wider_than_the_material_page_is_refused_and_says_where_to_widen() -> None:
    """抓「只准收窄」那條拿掉：棉厚或空腔超出材料頁的範圍都拒收，句子寫材料頁的範圍、叫人去材料頁改。"""
    family = SETTINGS.family_by_id(WOOL)
    porous = range_problem(family, ZoneRange(porous_m=(0.05, 0.4)))
    assert porous is not None and "要放寬請到材料頁改那份材料" in porous and "材料頁的產品厚度是 0.05、0.1、0.2 公尺" in porous
    assert range_problem(family, ZoneRange(porous_m=(0.03, 0.1))) == porous.replace("0.05～0.4", "0.03～0.1")
    cavity = range_problem(family, ZoneRange(porous_m=(0.05, 0.1), cavity_m=(0.0, 0.3)))
    assert cavity == "空腔 0～0.3 公尺超出材料頁的範圍（0～0.25 公尺）；這裡只准收窄，要放寬請到材料頁改那份材料"
    sample = _library_settings().family_by_id(SAMPLE)
    over = range_problem(sample, ZoneRange(porous_m=(0.075, 0.075), cavity_m=(0.01, 0.13)))
    assert over is not None and "（0～0.12 公尺）" in over and "材料頁" in over
    assert range_problem(sample, ZoneRange(porous_m=(0.075, 0.075), cavity_m=(0.01, 0.09))) is None
    reply = _limits(_ranged({"x0@b": {"ranges": {WOOL: {"porous_m": [0.05, 0.4]}}}}))
    lines = _red(reply, "x0", "b")
    assert [line["code"] for line in lines] == ["range_invalid"] and "材料頁" in lines[0]["text"]
    assert any("「b」" in text and "48K 玻璃棉" in text and "材料頁" in text for text in cast(list[str], reply["refusals"]))


@pytest.mark.parametrize("porous", [(0.05, 0.15), (0.06, 0.1), (0.1, 0.12)])
def test_zone_range_porous_ends_must_be_product_thicknesses(porous: tuple[float, float]) -> None:
    problem = range_problem(SETTINGS.family_by_id(WOOL), ZoneRange(porous_m=porous))
    assert problem is not None and "要是產品厚度之一" in problem and "材料頁" not in problem.split("（")[0]


def test_zone_range_legal_thicknesses_are_the_product_thicknesses_inside_the_range() -> None:
    """抓不切合法厚度：收窄後那一族的產品厚度只剩範圍裡那幾個（取原本那幾個數字本身）、空腔界換成這一區的。"""
    narrowed = narrowed_family(_library_settings().family_by_id(STEPS), ZoneRange(porous_m=(0.05, 0.08), cavity_m=(0.01, 0.04)))
    assert narrowed.thicknesses_m == (0.05, 0.08) and narrowed.cavity_range_m == (0.01, 0.04)
    assert narrowed.id == STEPS and narrowed.max_total_depth_m == 0.2
    keep_cavity = narrowed_family(SETTINGS.family_by_id(WOOL), ZoneRange(porous_m=(0.1, 0.2)))
    assert keep_cavity.thicknesses_m == (0.1, 0.2) and keep_cavity.cavity_range_m == (0.0, 0.25)


def test_zone_range_cavity_is_refused_for_a_material_that_does_not_tune_its_cavity() -> None:
    problem = range_problem(SETTINGS.family_by_id("glass_wool_24k"), ZoneRange(porous_m=(0.05, 0.1), cavity_m=(0.0, 0.03)))
    assert problem == "這種材料搜尋不調空腔（樣板沒有空氣層），不能給空腔範圍"
    assert range_problem(SETTINGS.family_by_id("glass_wool_24k"), ZoneRange(porous_m=(0.05, 0.1))) is None


def test_zone_range_on_a_fixed_product_or_an_unknown_code_is_red() -> None:
    document = _ranged({"y0@s": {"ranges": {"hall_door_hollow_core": {"porous_m": [0.05, 0.1]}}},
                        "yL@s": {"ranges": {"no_such_wool": {"porous_m": [0.05, 0.1]}}}})
    reply = _limits(document, _targets())
    fixed, unknown = _red(reply, "y0", "s"), _red(reply, "yL", "s")
    assert [line["code"] for line in fixed] == ["range_unknown"] and "固定成品" in fixed[0]["text"]
    assert [line["code"] for line in unknown] == ["range_unknown"] and "no_such_wool" in unknown[0]["text"]


def test_zone_range_on_a_user_material_is_checked_against_that_material() -> None:
    """使用者可最佳化材料的代號就是家族代號：範圍對它自己的產品厚度與空腔核，合的不紅。"""
    good = _ranged({"y0@s": {"ranges": {STEPS: {"porous_m": [0.05, 0.08], "cavity_m": [0.0, 0.04]}}}})
    assert _red(_limits(good, _targets()), "y0", "s") == []
    bad = _ranged({"y0@s": {"ranges": {STEPS: {"porous_m": [0.04, 0.08]}}}})
    lines = _red(_limits(bad, _targets()), "y0", "s")
    assert [line["code"] for line in lines] == ["range_invalid"] and "0.03、0.05、0.08、0.12" in lines[0]["text"]


# ---- 深度：收窄後的最淺排法（分區頁與開跑同一把） ----

def test_zone_range_narrowed_shallowest_that_does_not_fit_the_cap_is_red_on_the_zone_page() -> None:
    """抓深度判斷用沒收窄的那一族：48K 不收窄時最淺 0.05 放得進這一塊的 0.2，收窄到棉 0.1～0.2、空腔 0.12 起
    最淺 0.22 放不下，標紅、點名材料；沒寫範圍的 24K 照舊只列「不可用」以外的判決。"""
    given = {"porous_m": [0.1, 0.2], "cavity_m": [0.12, 0.2]}
    reply = _limits(_ranged({"y0@s": {"ranges": {WOOL: given}}}))
    lines = _red(reply, "y0", "s")
    assert [line["code"] for line in lines] == ["range_depth"]
    assert "48K 玻璃棉" in lines[0]["text"] and "0.22" in lines[0]["text"] and "收窄" in lines[0]["text"]
    caps = cast(dict[str, dict[str, dict[str, list[dict[str, object]]]]], reply["caps"])
    usable = {line["target"]: line["usable"] for line in caps["y0"]["s"]["targets"]}
    assert usable[WOOL] is False and usable["glass_wool_24k"] is True
    plain = _limits(control_design())
    assert _red(plain, "y0", "s") == [] and caps["y0"]["s"]["targets"][1] == cast(
        dict[str, dict[str, dict[str, list[dict[str, object]]]]], plain["caps"])["y0"]["s"]["targets"][1]


def test_zone_range_depth_judgment_is_the_same_float_rule_as_the_search() -> None:
    """抓判法換一套（例如十進位相加）：棉 0.1 加空腔 0.05 十進位剛好 0.15，照搜尋取規格那一把（θ 的精度）放不進 0.15；
    分區頁照同一支 ``depth_fit.target_fit`` 判，0.15 紅、0.16 不紅。"""
    given = ZoneRange(porous_m=(0.1, 0.2), cavity_m=(0.05, 0.2))
    edge = target_fit(WOOL, 0.15, SETTINGS, {}, {WOOL: given})
    assert edge.usable is False and edge.shallowest_m == pytest.approx(0.15)
    assert target_fit(WOOL, 0.15, SETTINGS, {}).usable is True
    assert target_fit(WOOL, 0.16, SETTINGS, {}, {WOOL: given}).usable is True
    for cap, red in ((0.15, True), (0.16, False)):
        document = _ranged({"y0@s": {"max_depth_m": cap, "ranges": {WOOL: {"porous_m": [0.1, 0.2], "cavity_m": [0.05, 0.2]}}}})
        assert bool(_red(_limits(document), "y0", "s")) is red


def test_zone_range_narrowed_shallowest_beyond_the_material_own_depth_is_red_without_a_cap() -> None:
    """沒有個案上限也要放得進材料自己的總深：48K 收窄到棉 0.2、空腔 0.15 起，最淺 0.35 超過 0.3。"""
    given = ZoneRange(porous_m=(0.2, 0.2), cavity_m=(0.15, 0.25))
    problem = range_problem(SETTINGS.family_by_id(WOOL), given)
    assert problem is not None and "0.35" in problem and "0.3 公尺" in problem
    lines = _red(_limits(_ranged({"ceiling@c": {"ranges": {WOOL: given.model_dump(mode="json")}}})), "ceiling", "c")
    assert [line["code"] for line in lines] == ["range_invalid"]


# ---- 起跑值（給第 2 單元用的同一條規則） ----

def test_zone_range_start_value_outside_the_range_is_named() -> None:
    family = SETTINGS.family_by_id(WOOL)
    assert start_problem(family, ZoneRange(porous_m=(0.05, 0.1)), 0.05, 0.05) is None
    assert start_problem(family, ZoneRange(porous_m=(0.1, 0.2)), 0.05, 0.05) == "起跑值棉厚 0.05 公尺不在這一區的範圍 0.1～0.2 公尺內"
    assert (start_problem(family, ZoneRange(porous_m=(0.05, 0.1), cavity_m=(0.0, 0.03)), 0.05, 0.05)
            == "起跑值空腔 0.05 公尺不在這一區的範圍 0～0.03 公尺內")
    assert "初值不在範圍內" in depth_fit.START_ADVICE


# ---- 控制組：沒寫範圍跟主線逐位相同 ----

def test_zone_range_absent_keeps_the_zone_page_reply_byte_for_byte() -> None:
    assert sha(limits_bytes()) == MAIN["limits"]


def test_zone_range_absent_keeps_plan_designs_ledger_and_preview_byte_for_byte(tmp_path: Path) -> None:
    """抓沒寫範圍時計畫、開跑參數、起跑設計、帳本標頭、資料夾名、project.json 或網頁預覽多了東西。"""
    assert control_hashes(tmp_path) == MAIN
