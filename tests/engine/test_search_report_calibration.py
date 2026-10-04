"""品質段的尺校準進度（#585 第 3 題）：答案另外算（直接讀登記簿 TOML、網頁的類名、手造升級的登記簿），不照抄被測的分類程式。"""

import dataclasses
import tomllib
from pathlib import Path
from typing import cast

import pytest

from aosr.config.paths import config_path
from aosr.config.quality_targets import QualityPurpose, QualityTargets, WeightTable
from aosr.gui.result_view import LABELS as GUI_LABELS
from aosr.scoring.category_registry import CATEGORY_REGISTRY
from aosr.scoring.contract import QualityCategory
from aosr.search.report import build_report, render_text
from aosr.search.report_calibration import CATEGORY_LABELS, calibration_lines, calibration_progress
from aosr.search.run import SearchStatus
from tests.engine import _search_run_cases as run_cases
from tests.engine._search_run_cases import RUN_DATE, make_store
from tests.engine.test_search_report import _sections

PURPOSE = "dedicated_two_channel_listening_room"
NONE_FULL = "沒有任何一類用到的尺（含評估器與排名層讀到的別類尺）全部校準完，所以還沒有「已校準項目通過幾項」可以報（等 #358）"
# 評估器那一層跨類讀的尺（#633，兩位查核員從 evaluate_parts 與每一處 QualityPurpose.entry 各追一次的結果），手列不照抄宣告。
TIMBRE_EVALUATOR = tuple(f"timbre_balance.{name}" for name in (
    "coverage_range_hz", "tilt_fit_range_hz", "ripple_range_hz", "smoothing_width_octave_tilt",
    "smoothing_width_octave_ripple", "feature_min_width_octave", "min_points", "target_tilt_db_per_octave"))
REFLECTIONS_EVALUATOR = ("reflections_and_echo.window_upper_ms", "reflections_and_echo.frequency_range_hz",
                         "reflections_and_echo.flutter_decay_db", "reflections_and_echo.flutter_alert_band_centers_hz",
                         "direction_zones.vertical_min_abs_elevation_deg", "direction_zones.front_max_abs_azimuth_deg",
                         "direction_zones.rear_min_abs_azimuth_deg")
CROSS_READ = {"listening_area_stability": (*TIMBRE_EVALUATOR, "channel_matching.broadband_range_hz"),
              "channel_matching": (*TIMBRE_EVALUATOR, *REFLECTIONS_EVALUATOR)}


Raw = dict[str, object]


def _raw_purpose() -> Raw:
    with config_path("quality_targets.toml").open("rb") as handle:
        document: Raw = tomllib.load(handle)
    return next(purpose for purpose in cast(list[Raw], document["purpose"]) if purpose["name"] == PURPOSE)


def _items(raw: Raw, kind: str) -> list[Raw]:
    return cast(list[Raw], raw[kind])


def _raw_counts(raw: Raw) -> dict[str, tuple[int, int]]:
    """直接讀 TOML 另算：設定、目標、資格規則一條一筆，權重表一項一筆，類別取鍵名第一段。"""
    rows = [(str(entry["key"]), str(entry["status"]))
            for kind in ("setting", "target", "qualification") for entry in _items(raw, kind)]
    rows += [(str(table["key"]), str(item["status"])) for table in _items(raw, "weight") for item in _items(table, "item")]
    counts: dict[str, tuple[int, int]] = {}
    for key, status in rows:
        calibrated, total = counts.get(key.split(".")[0], (0, 0))
        counts[key.split(".")[0]] = (calibrated + int(status == "calibrated"), total + 1)
    return counts


def _promoted(prefixes: tuple[str, ...], baseline: str = "") -> QualityPurpose:
    """把鍵名前綴命中的每一條升成 calibrated，結構化出處照抄一條已校準的；鍵名等於 baseline 的那一條反過來
    設成基線值（今天已校準的也一樣）；仍走登記簿的驗證。"""
    raw = _raw_purpose()
    template = next(entry for entry in _items(raw, "setting") if entry["status"] == "calibrated")
    receipt = {key: value for key, value in template.items()
               if key not in ("key", "value", "unit", "status")}

    def promote(entry: Raw, key: str) -> None:
        if key == baseline:
            entry["status"] = "baseline"
        elif key.startswith(prefixes):
            entry.update(receipt | {"status": "calibrated"})
    for kind in ("setting", "target", "qualification"):
        for entry in _items(raw, kind):
            promote(entry, str(entry["key"]))
    for table in _items(raw, "weight"):
        for item in _items(table, "item"):
            # 權重項在排名層的依賴清單裡寫成「表的鍵．項名」，兩種寫法都要命中得到。
            promote(item, f"{table['key']}.{item['name']}")
    return QualityTargets.model_validate({"schema_version": 1, "purpose": [raw]}).purpose(PURPOSE)


def test_every_category_count_matches_the_raw_registry() -> None:
    progress = calibration_progress(QualityTargets.model_validate(
        {"schema_version": 1, "purpose": [_raw_purpose()]}).purpose(PURPOSE))
    assert {item.category: (item.calibrated, item.total) for item in progress.categories} == _raw_counts(_raw_purpose())


def test_category_names_match_the_result_page() -> None:
    judged = {"timbre_balance", "channel_matching", "reverberation", "reflections_and_echo", "listening_area_stability"}
    assert {name: CATEGORY_LABELS[name] for name in judged} == {name: GUI_LABELS[name] for name in judged}
    assert set(CATEGORY_LABELS) >= set(_raw_counts(_raw_purpose()))


def test_today_no_category_uses_only_calibrated_rulers() -> None:
    progress = calibration_progress(QualityTargets.model_validate(
        {"schema_version": 1, "purpose": [_raw_purpose()]}).purpose(PURPOSE))
    assert calibration_lines(progress)[2] == NONE_FULL
    assert "判定共用的尺：方向分區" in calibration_lines(progress)[1]


@pytest.mark.parametrize("prefixes,named", [
    (("reflections_and_echo.",), None),
    (("reflections_and_echo.", "direction_zones."), "反射與回聲"),
    (("direction_zones.",), None),
    (("ranking.",), None),
])
def test_full_calibration_counts_dependencies(prefixes: tuple[str, ...], named: str | None) -> None:
    """反射與回聲自己全校準、依賴的方向分區沒校準 → 不算全部校準；方向分區、排名規則不是品質類，全校準也不點名。"""
    closing = calibration_lines(calibration_progress(_promoted(prefixes)))[2]
    if named is None:
        assert closing == NONE_FULL
    else:
        assert closing == f"用到的尺（含評估器與排名層讀到的別類尺）全部校準完的類別：{named}；這幾類過不過還沒接進報告（等 #358）"


def _ranking_sources(category: str) -> tuple[str, ...]:
    purpose = QualityTargets.model_validate({"schema_version": 1, "purpose": [_raw_purpose()]}).purpose(PURPOSE)
    return tuple(key for key, _ in CATEGORY_REGISTRY[QualityCategory(category)].registry_sources(purpose))


@pytest.mark.parametrize("category,extra,named", [
    ("listening_area_stability", (), None),
    ("listening_area_stability", TIMBRE_EVALUATOR, None),
    ("listening_area_stability", ("channel_matching.broadband_range_hz",), None),
    ("listening_area_stability", (*TIMBRE_EVALUATOR, "channel_matching.broadband_range_hz"), "聆聽區穩定性"),
    ("channel_matching", (), None),
    ("channel_matching", TIMBRE_EVALUATOR, None),
    ("channel_matching", REFLECTIONS_EVALUATOR, None),
    ("channel_matching", (*TIMBRE_EVALUATOR, *REFLECTIONS_EVALUATOR), "聲道匹配"),
])
def test_full_calibration_counts_what_the_evaluators_read(
        category: str, extra: tuple[str, ...], named: str | None) -> None:
    """自己的尺與排名層登記的都校準了，評估器跨類讀的尺（逐座位音色、寬頻範圍、反射設定）還有基線值 → 不算全部校準（#633）。"""
    closing = calibration_lines(calibration_progress(_promoted((f"{category}.", *_ranking_sources(category), *extra))))[2]
    if named is None:
        assert closing == NONE_FULL
    else:
        assert closing == f"用到的尺（含評估器與排名層讀到的別類尺）全部校準完的類別：{named}；這幾類過不過還沒接進報告（等 #358）"


def test_progress_counts_the_search_snapshot_not_the_current_registry(tmp_path: Path) -> None:
    store, registry = make_store(tmp_path)
    store.status_path.write_text(SearchStatus().model_dump_json())
    renamed = tmp_path / "renamed-registry"
    renamed.write_text(registry.read_text(encoding="utf-8").replace(f'name = "{PURPOSE}"', 'name = "some_other_purpose"'),
                       encoding="utf-8")
    section = _sections(render_text(build_report(store, quality_targets_path=renamed, run_date=RUN_DATE)))["品質合不合格"]
    snapshot = QualityPurpose.model_validate(store.identity.purpose_settings.content)
    lines = section.splitlines()
    assert lines[0] == "判定：未判定合格"
    assert lines[3:6] == list(calibration_lines(calibration_progress(snapshot)))


def test_invalid_snapshot_content_says_unreadable_and_keeps_the_report(tmp_path: Path) -> None:
    """快照內容真的不是合法的用途設定（不是換掉函式假裝失敗）：品質段寫讀不回，其他段照印（複查）。"""
    store, registry = make_store(tmp_path)
    store.status_path.write_text(SearchStatus().model_dump_json())
    identity = store.identity
    broken = identity.purpose_settings.model_copy(update={"content": {"name": PURPOSE, "bogus": True}})
    store._identity = dataclasses.replace(identity, purpose_settings=broken)
    sections = _sections(render_text(build_report(store, quality_targets_path=registry, run_date=RUN_DATE)))
    assert "尺的校準進度：這次搜尋快照裡的登記簿讀不回，數不出來" in sections["品質合不合格"].splitlines()
    assert "搜尋停了沒" in sections


def _without_entry(text: str, key: str) -> str:
    """從登記簿 TOML 拿掉含這個鍵的那一整個條目區塊。"""
    blocks = text.split("\n[[")
    kept = [block for block in blocks if f'key = "{key}"' not in block]
    assert len(kept) == len(blocks) - 1
    return "\n[[".join(kept)


@pytest.mark.parametrize("missing", [
    "reflections_and_echo.flutter_decay_db",  # 排名層會讀
    "timbre_balance.coverage_range_hz",  # 只有評估器讀（#633）
])
def test_old_snapshot_missing_a_ruler_keeps_the_report(
        missing: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """搜尋開跑後登記簿加了一條會讀的尺：舊快照少這條，報告照常出、照快照數、明說判不出（複查）。"""
    original = run_cases.registry_copy

    def older(directory: Path) -> Path:
        path = original(directory)
        path.write_text(_without_entry(path.read_text(encoding="utf-8"), missing), encoding="utf-8")
        return path

    monkeypatch.setattr(run_cases, "registry_copy", older)
    store, _ = make_store(tmp_path / "search")
    store.status_path.write_text(SearchStatus().model_dump_json())
    current = original(tmp_path)
    section = _sections(render_text(build_report(store, quality_targets_path=current, run_date=RUN_DATE)))["品質合不合格"]
    lines = section.splitlines()
    raw_total = sum(total for _, total in _raw_counts(_raw_purpose()).values())
    assert lines[3].startswith(f"尺的校準進度（這次搜尋快照裡的登記簿）：共 {raw_total - 1} 條")
    assert lines[5].startswith("這次搜尋快照裡的登記簿跟現在的評分程式（評估器與排名層）對不上（") and missing in lines[5]
    assert "評分設定跟搜尋快照不同" in section


def test_own_rulers_count_not_only_ranking_dependencies() -> None:
    """只升音色平衡排名層登記的那幾條、它自己其他量法設定還是基線值：不准說音色平衡全部校準完（複查）。"""
    purpose = QualityTargets.model_validate({"schema_version": 1, "purpose": [_raw_purpose()]}).purpose(PURPOSE)
    sources = CATEGORY_REGISTRY[QualityCategory("timbre_balance")].registry_sources(purpose)
    promoted = _promoted(tuple(key for key, _ in sources))
    promoted_sources = CATEGORY_REGISTRY[QualityCategory("timbre_balance")].registry_sources(promoted)
    assert all(status == "calibrated" for _, status in promoted_sources)
    assert any(entry.status != "calibrated" for entry in promoted.setting if entry.key.startswith("timbre_balance."))
    assert calibration_lines(calibration_progress(promoted))[2] == NONE_FULL


@pytest.mark.parametrize("category,key", [
    (category, key) for category, keys in CROSS_READ.items() for key in keys])
def test_every_cross_read_ruler_alone_blocks_full_calibration(category: str, key: str) -> None:
    """評估器跨類讀的尺逐條留一條是基線值、其餘全部校準：那一條就足以讓這一類不算全部校準（複查：整包一起升考不出少宣告一條）。"""
    promoted = _promoted((f"{category}.", *_ranking_sources(category), *CROSS_READ[category]), baseline=key)
    left = promoted.entry(key)
    assert not isinstance(left, WeightTable) and left.status == "baseline"
    assert calibration_lines(calibration_progress(promoted))[2] == NONE_FULL


@pytest.mark.parametrize("promoted,named", [
    (("timbre_balance.",), None),
    (("timbre_balance.", "reverberation.within_category_weights"), "音色平衡"),
    # 只校準其中一項、另一項還是基線值：兩種都要擋（不管表內哪一項排第一，複查植錯「只看第一項」）。
    (("timbre_balance.", "reverberation.within_category_weights.t20_target_interval"), None),
    (("timbre_balance.", "reverberation.within_category_weights.adjacent_t20_jump"), None),
])
def test_a_declared_weight_table_counts_every_item(
        promoted: tuple[str, ...], named: str | None, monkeypatch: pytest.MonkeyPatch) -> None:
    """評估器宣告指到別類一整張權重表：照排名層的寫法逐項算，每一項都校準才算（複查；今天沒有評估器讀權重表）。"""
    category = QualityCategory("timbre_balance")
    registration = CATEGORY_REGISTRY[category]
    monkeypatch.setitem(CATEGORY_REGISTRY, category, dataclasses.replace(
        registration, evaluator_keys=(*registration.evaluator_keys, "reverberation.within_category_weights")))
    closing = calibration_lines(calibration_progress(_promoted(promoted)))[2]
    assert closing == (NONE_FULL if named is None else
                       f"用到的尺（含評估器與排名層讀到的別類尺）全部校準完的類別：{named}；這幾類過不過還沒接進報告（等 #358）")
