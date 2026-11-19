"""結果頁標「含 300 Hz 以下」（#1208 第 2 支）函式層考卷：存的方案裡有「鋼板直接貼在棉上」的構造時，結果頁回覆多一格
（表上方那一句、為什麼、小標籤的字、要標的類別）；沒有時回覆跟原來一樣、不讀界線。

資料全是考卷自己的：結果是分區結果頁考卷那一份真算的小房間（``_zone_result_cases.real_design``，同一次 pytest 只算一次），
各題只把存的方案換成帶這類構造的版本再排字（結果頁排字不重算）；構造用 ``_bonded_backing_cases`` 的原始欄位。
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import cast

import pytest

from aosr.config.paths import config_path
from aosr.config.quality_targets import QualityTargets, load_quality_targets
from aosr.gui import reference_only
from aosr.gui.app import STATIC
from aosr.gui.compare_view import build_compare_view, summary_csv
from aosr.gui.result_view import ResultView, build_result_view
from aosr.reporting import evaluation as reporting_evaluation
from aosr.reporting.result import SchemeResult
from aosr.reporting.scheme import Scheme
from aosr.scoring import channel_matching_settings, reflections, reverberation_cost, stage_one_score, timbre
from aosr.scoring.contract import QualityCategory
from tests.engine._bonded_backing_cases import BONDED_VPR, MELAMINE_50, MELAMINE_BONDED, STEEL_1MM
from tests.engine._zone_result_cases import real_design

Doc = dict[str, object]
TARGETS = config_path("quality_targets.toml")
PURPOSE = "dedicated_two_channel_listening_room"
# 預期被標的五類（寫死，不拿規則跟自己比）；低頻拖尾不計分、空間感尚未評估、反射與回聲下界剛好等於界線，都不標。
MARKED = ["timbre_balance", "listening_area_stability", "low_frequency_response", "reverberation", "channel_matching"]
WHY = ("程式算這類構造用的是簡化算法（一維、鋼板當成整塊一起動）。跟 Fraunhofer 的實測比，算出來的共振偏高、頻寬偏窄。"
       "數字沒有改，照舊算。")
# 沒黏住的同一塊鋼板與棉（背襯沒填骨架兩格）：不是這一類。
LOOSE: Doc = {"kind": "assembly", "backing": "rigid", "layers": [STEEL_1MM, MELAMINE_50]}
PLAIN_WALL: Doc = {"kind": "constant_impedance", "impedance_pa_s_m": 1646.4}
# 主線（a4952bac）結果頁排字回覆的每一個鍵（選填的三格沒有內容時不出現）。
MAIN_KEYS = {
    "material_research_notice", "measured_walls", "speaker_setup_line", "room_line", "furniture", "furniture_reason", "furniture_notes", "construction_notes",
    "furniture_flutter_text", "frequency_note", "ranking_approximation_text", "not_modeled", "manual_checks", "scheme_id",
    "engine_commit", "engine_commit_text", "run_date", "timings", "timing_texts", "labels", "speaker_names", "point_names",
    "frequency_responses", "frequency_plot_data", "ranking_status", "ranking_reasons", "missing_categories", "ranking_text",
    "ideal_source_note", "cost_note", "categories", "alerts", "flutter_groups", "reverberation", "reflections", "listening_area"}


@pytest.fixture(scope="module")
def plain(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> SchemeResult:
    """沒有這類構造的結果：前牆切兩區，一區在原牆上貼一層棉、一區不動，其餘五面整面固定阻抗。"""
    return SchemeResult.model_validate_json(real_design(tmp_path_factory, worker_id))


def _zone(material: Doc) -> list[Doc]:
    return [{"area": "whole_wall", "material": material}]


def _mounted(layers: list[Doc], wall: Doc) -> Doc:
    return {"kind": "mounted_on_wall", "layers": layers, "wall": wall}


def _saved(result: SchemeResult, walls: dict[str, list[Doc]] | None = None, front: dict[str, Doc] | None = None) -> SchemeResult:
    """同一份結果，存的方案換成：``walls`` 指名的面整面換一區、``front`` 指名的前牆矩形區換材料。"""
    document = result.scheme.model_dump(mode="json")
    zones = cast(dict[str, list[Doc]], cast(Doc, document["scene"])["surface_zones_by_wall"])
    zones.update(walls or {})
    for zone in zones["x0"]:
        if zone.get("id") in (front or {}):
            zone["material"] = (front or {})[cast(str, zone["id"])]
    return result.model_copy(update={"scheme": Scheme.model_validate(document)})


def _block(result: SchemeResult, targets: Path = TARGETS) -> Doc | None:
    return cast("Doc | None", build_result_view(result, quality_targets_path=targets).model_dump(mode="json").get("reference_only"))


def _sentence(walls: str) -> str:
    return (f"{walls}有 VPR（鋼板共振吸音）這類「鋼板直接貼在棉上」的構造：300 Hz 以下的數字只能參考；"
            "下表標了「含 300 Hz 以下」的幾列都受影響。")


def test_a_whole_wall_of_such_a_construction_adds_the_block_with_every_word(plain: SchemeResult) -> None:
    """後牆整面是這類構造：多的那一格剛好五樣——那一句（點名後牆）、「為什麼」三個字與展開的原因、小標籤的字、
    要標的五類（照表上的順序）。"""
    assert _block(_saved(plain, {"xL": _zone(BONDED_VPR)})) == {
        "sentence": _sentence("後牆"), "why_label": "為什麼", "why": WHY, "tag": "含 300 Hz 以下", "categories": MARKED}
    assert reference_only.words()["why_only"] == WHY  # 展開的原因就是第 1 支那一句


@pytest.mark.parametrize(("saved", "walls"), [
    ({"front": {"b": BONDED_VPR}}, "前牆"),                                              # 分區裡一區
    ({"front": {"a": _mounted([MELAMINE_50], BONDED_VPR)}}, "前牆"),                     # 原牆是這一類，上面貼一般的棉
    ({"front": {"a": _mounted([STEEL_1MM, MELAMINE_BONDED], PLAIN_WALL)}}, "前牆"),      # 貼上去的那幾層是這一類
    ({"walls": {"ceiling": _zone(BONDED_VPR)}}, "天花"),
    ({"walls": {"floor": _zone(BONDED_VPR)}}, "地板"),
    ({"walls": {"y0": _zone(BONDED_VPR)}}, "左牆"),
    ({"walls": {"yL": _zone(BONDED_VPR)}}, "右牆"),
], ids=["one_zone", "original_wall_under_plain_layers", "pasted_layers", "ceiling", "floor", "left_wall", "right_wall"])
def test_wherever_the_saved_scheme_keeps_such_a_construction_the_sentence_names_that_face(
        plain: SchemeResult, saved: dict[str, dict[str, object]], walls: str) -> None:
    """存的方案裡材料的每一種放法：矩形分區的一區、貼在原牆上的（原牆是這一類／貼上去的是這一類）、天花、地板、左右牆整面。
    句子只寫牆名，要標的還是那五類。"""
    block = _block(_saved(plain, cast("dict[str, list[Doc]] | None", saved.get("walls")), cast("dict[str, Doc] | None", saved.get("front"))))
    assert block is not None and block["sentence"] == _sentence(walls) and block["categories"] == MARKED


def test_several_faces_are_named_once_each_in_the_order_of_the_opening_line(plain: SchemeResult) -> None:
    """地板、天花、右牆、前牆兩區都是這一類：一面只寫一次，面序同結果頁開頭「貼了吸音材料」那一句
    （前後左右、天花、地板），不照方案裡寫的先後。"""
    result = _saved(plain, {"floor": _zone(BONDED_VPR), "ceiling": _zone(BONDED_VPR), "yL": _zone(BONDED_VPR)},
                    {"a": BONDED_VPR, "b": BONDED_VPR})
    block = _block(result)
    assert block is not None and block["sentence"] == _sentence("前牆、右牆、天花、地板")
    view = build_result_view(result, quality_targets_path=TARGETS)
    assert view.room_line.endswith("貼了吸音材料：前牆 2 塊、右牆整面、天花整面、地板整面")


def _fixed_only(result: SchemeResult) -> SchemeResult:
    """六面只給固定阻抗、沒有分區那一格的方案。"""
    document = result.scheme.model_dump(mode="json")
    scene = cast(Doc, document["scene"])
    del scene["surface_zones_by_wall"]
    scene["impedance_pa_s_per_m_by_wall"] = dict.fromkeys(("floor", "ceiling", "x0", "xL", "y0", "yL"), 1646.4)
    return result.model_copy(update={"scheme": Scheme.model_validate(document)})


def test_without_such_a_construction_the_reply_is_the_ordinary_one_and_nothing_is_read(
        plain: SchemeResult, monkeypatch: pytest.MonkeyPatch) -> None:
    """控制組：沒有這類構造（貼一般的棉、同樣鋼板與棉但沒黏住、六面只有固定阻抗）——回覆沒有那一格、鍵就是主線那幾個；
    把界線與範圍換成讀不出來，回覆一模一樣（沒命中不靠登記簿那幾格）。對照：有命中的同樣植法會炸（真的植到了）。"""
    cases = (plain, _saved(plain, {"xL": _zone(LOOSE)}), _fixed_only(plain))
    before = [build_result_view(result, quality_targets_path=TARGETS).model_dump(mode="json") for result in cases]
    for reply in before:
        assert "reference_only" not in reply and set(reply) <= MAIN_KEYS
    assert MAIN_KEYS == set(ResultView.model_fields) - {"reference_only"}

    def broken(*args: object) -> float:
        raise ValueError("讀不出來（考卷植的）")
    monkeypatch.setattr(reference_only, "registry_band_hz", broken)
    monkeypatch.setattr(reference_only, "band_hz", broken)
    monkeypatch.setattr(reference_only, "marked_categories", broken)
    assert [build_result_view(result, quality_targets_path=TARGETS).model_dump(mode="json") for result in cases] == before
    with pytest.raises(ValueError, match="考卷植的"):
        build_result_view(_saved(plain, {"xL": _zone(BONDED_VPR)}), quality_targets_path=TARGETS)


def test_the_block_sits_beside_the_rows_and_changes_nothing_else(plain: SchemeResult, monkeypatch: pytest.MonkeyPatch) -> None:
    """有命中：回覆只多那一格。拿掉它，等於同一份結果把判斷換成「永遠不命中」的回覆——各類那幾列、數字都沒被動到。"""
    hit = _saved(plain, {"xL": _zone(BONDED_VPR)})
    real = build_result_view(hit, quality_targets_path=TARGETS).model_dump(mode="json")
    monkeypatch.setattr(reference_only, "reference_only", lambda material: False)
    off = build_result_view(hit, quality_targets_path=TARGETS).model_dump(mode="json")
    assert set(real) - set(off) == {"reference_only"} and {key: value for key, value in real.items() if key != "reference_only"} == off


def test_the_compare_page_and_its_text_export_are_the_same_with_or_without_the_block(plain: SchemeResult) -> None:
    """比較頁不改：一邊有命中、一邊沒有，比較頁的回覆與文字匯出，跟把那一格拿掉的結果頁資料算出來的一模一樣。"""
    hit = _saved(plain, {"xL": _zone(BONDED_VPR)})
    view_hit = build_result_view(hit, quality_targets_path=TARGETS)
    view_plain = build_result_view(plain, quality_targets_path=TARGETS)
    assert view_hit.reference_only is not None

    def compared(view: ResultView) -> tuple[Doc, str]:
        page = build_compare_view(a_run_id="a" * 32, a=hit, view_a=view, b_run_id="b" * 32, b=plain, view_b=view_plain,
                                  quality_targets=load_quality_targets(TARGETS), run_date=date(2026, 11, 20))
        return page.model_dump(mode="json"), summary_csv(page)
    assert compared(view_hit) == compared(view_hit.model_copy(update={"reference_only": None}))


# ---- 哪幾列要標 ----

ALL_COSTED = {category.value: "costed" for category in QualityCategory}


def _registry(tmp_path: Path, key: str, value: str) -> QualityTargets:
    """登記簿複製一份到暫存資料夾，把用途那一段指名的那一格的值換掉。"""
    text = TARGETS.read_text(encoding="utf-8")
    start = text.index(f'key = "{key}"', text.index("[[purpose]]"))
    at = text.index("value = ", start)
    end = text.index("\n", at)
    changed = tmp_path / "quality_targets.toml"
    changed.write_text(f"{text[:at]}value = {value}{text[end:]}", encoding="utf-8")
    return load_quality_targets(changed)


def test_a_category_is_marked_only_when_it_has_a_cost(plain: SchemeResult) -> None:
    """有算出代價（已量、已算代價）才標；不可估、尚未評估的不標，就算它的範圍低於界線。這一份結果表上的狀態：
    低頻拖尾、空間感是尚未評估，其餘已算代價。"""
    registry = load_quality_targets(TARGETS)
    view = build_result_view(plain, quality_targets_path=TARGETS)
    assert {row.category: row.state for row in view.categories} == {
        **ALL_COSTED, "low_frequency_decay": "not_evaluated", "spatial_impression": "not_evaluated"}
    assert reference_only.marked_categories(ALL_COSTED, registry, PURPOSE, 300.0) == MARKED
    states = {**ALL_COSTED, "timbre_balance": "unavailable", "reverberation": "measured", "channel_matching": "not_evaluated"}
    assert reference_only.marked_categories(states, registry, PURPOSE, 300.0) == [
        "listening_area_stability", "low_frequency_response", "reverberation"]


def test_a_lower_edge_exactly_at_the_band_is_not_marked_and_one_just_below_is(tmp_path: Path) -> None:
    """反射與回聲的範圍下界現在剛好等於界線（300）：不標。登記簿複製一份、下界改成 250：多標反射與回聲。
    （這一題考的是決定標哪幾列的那一支：改過的登記簿拿去排整張結果頁，反射評估器會先因量法跟存檔不同而拒收。）"""
    registry = load_quality_targets(TARGETS)
    assert reference_only.registry_band_hz(registry) == 300.0
    assert "reflections_and_echo" not in reference_only.marked_categories(ALL_COSTED, registry, PURPOSE, 300.0)
    lowered = _registry(tmp_path, "reflections_and_echo.frequency_range_hz", "[250.0, 8000.0]")
    assert reference_only.marked_categories(ALL_COSTED, lowered, PURPOSE, 300.0) == [
        "timbre_balance", "listening_area_stability", "low_frequency_response", "reflections_and_echo", "reverberation",
        "channel_matching"]
    # 界線自己往下移到 250：下界剛好 250 的又不標了。
    assert "reflections_and_echo" not in reference_only.marked_categories(ALL_COSTED, lowered, PURPOSE, 250.0)


def test_any_one_span_below_the_band_is_enough_and_all_at_or_above_is_not(tmp_path: Path) -> None:
    """一類讀好幾段：任何一段下界低於界線就標。聲道匹配兩段都從 20 起；把起伏那一段改成 300 起，另一段還低於界線，照標；
    殘響的頻帶中心最低 125，界線在 125 就不標、在 126 就標。"""
    one_high = _registry(tmp_path, "channel_matching.ripple_range_hz", "[300.0, 8000.0]")
    assert "channel_matching" in reference_only.marked_categories(ALL_COSTED, one_high, PURPOSE, 300.0)
    registry = load_quality_targets(TARGETS)
    assert "reverberation" not in reference_only.marked_categories(ALL_COSTED, registry, PURPOSE, 125.0)
    assert "reverberation" in reference_only.marked_categories(ALL_COSTED, registry, PURPOSE, 126.0)


def test_the_table_names_the_keys_the_scoring_code_reads() -> None:
    """對照表每一類的鍵名，就是評分程式讀登記簿用的那幾個常數；每一類都在表上，不計分、尚未評估的兩類寫空的。"""
    table = {category: [(section, key) for section, key, _ in spans] for category, spans in reference_only.RESULT_RANGES.items()}
    timbre_ranges = [key for key in timbre.SETTING_KEYS if key.endswith("_range_hz")]
    assert len(timbre_ranges) == 4
    assert table == {
        "timbre_balance": [("purpose", key) for key in timbre_ranges],
        "listening_area_stability": [("purpose", channel_matching_settings.BROADBAND_KEY),
                                     *(("purpose", key) for key in timbre_ranges if "coverage" not in key)],
        "low_frequency_response": [("stage_one", stage_one_score.SCORED_RANGE_KEY)],
        "low_frequency_decay": [],
        "reflections_and_echo": [("purpose", reflections._FREQUENCY_KEY)],
        "reverberation": [("purpose", reverberation_cost._CENTERS_KEY)],
        "channel_matching": [("purpose", channel_matching_settings.RIPPLE_KEY), ("purpose", channel_matching_settings.BROADBAND_KEY)],
        "spatial_impression": [],
    }
    assert list(table) == [category.value for category in QualityCategory]
    # 聆聽區那一段寬頻範圍，報表層讀的就是聲道匹配設定檔那個常數（同一個物件）。
    assert vars(reporting_evaluation)["BROADBAND_KEY"] is channel_matching_settings.BROADBAND_KEY
    assert {read for spans in reference_only.RESULT_RANGES.values() for _, _, read in spans} == {"range", "centers"}


@pytest.mark.parametrize(("ranges", "error"), [
    ({"spatial_impression": None}, ValueError),                                                         # 少一類
    ({"made_up_category": ()}, ValueError),                                                             # 多一類
    ({"timbre_balance": (("purpose", "timbre_balance.no_such_range_hz", "range"),)}, KeyError),         # 登記簿沒有這一格
    ({"timbre_balance": (("stage_one", "timbre_balance.coverage_range_hz", "range"),)}, KeyError),      # 住錯段
    ({"timbre_balance": (("purpose", "reflections_and_echo.window_upper_ms", "range"),)}, TypeError),   # 是一個數，不是一段
    ({"timbre_balance": (("purpose", "reflections_and_echo.window_upper_ms", "centers"),)}, TypeError),
    ({"timbre_balance": (("purpose", "reverberation.target_band_centers_hz", "range"),)}, TypeError),   # 七個數，不是一段
    ({"timbre_balance": (("purpose", "timbre_balance.target_tilt_db_per_octave", "range"),)}, TypeError),  # 是目標，不是量法設定
], ids=["one_category_missing", "one_category_extra", "no_such_key", "wrong_section", "a_number_read_as_range",
        "a_number_read_as_centers", "centers_read_as_range", "a_target_entry"])
def test_a_table_that_does_not_fit_the_registry_is_an_error_and_not_a_silent_no_mark(
        monkeypatch: pytest.MonkeyPatch, ranges: dict[str, object], error: type[Exception]) -> None:
    """對照表少一類、多一類、鍵名在登記簿讀不到、讀不出一段頻率：一律報錯，不默默不標。就算那一類沒有算出代價也照查。"""
    changed = {**reference_only.RESULT_RANGES, **ranges}
    monkeypatch.setattr(reference_only, "RESULT_RANGES", {key: value for key, value in changed.items() if value is not None})
    states = {**ALL_COSTED, "timbre_balance": "unavailable"}
    with pytest.raises(error):
        reference_only.marked_categories(states, load_quality_targets(TARGETS), PURPOSE, 300.0)


def test_the_result_page_script_writes_none_of_these_words_itself() -> None:
    """頁面只照那一格畫：腳本裡沒有那幾句字，也沒有界線數字的判斷。"""
    script = (STATIC / "results.js").read_text(encoding="utf-8")
    assert "view.reference_only" in script
    for words in ("以下", "為什麼", "只能參考", "鋼板"):
        assert words not in script.split("const reference = view.reference_only;")[1].split("// 評估器版本")[0].replace(
            "// 方案裡有「鋼板直接貼在棉上」的構造", "")
