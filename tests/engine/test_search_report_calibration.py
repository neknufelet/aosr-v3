"""品質段的尺校準進度（#585 第 3 題）：答案另外算（直接讀登記簿 TOML、網頁的類名、手造升級的登記簿），不照抄被測的分類程式。"""

import tomllib
from pathlib import Path
from typing import cast

import pytest

from aosr.config.paths import config_path
from aosr.config.quality_targets import QualityPurpose, QualityTargets
from aosr.gui.result_view import LABELS as GUI_LABELS
from aosr.search import report as report_module
from aosr.search.report import build_report, render_text
from aosr.search.report_calibration import CATEGORY_LABELS, calibration_lines, calibration_progress
from aosr.search.run import SearchStatus
from tests.engine._search_run_cases import RUN_DATE, make_store
from tests.engine.test_search_report import _sections

PURPOSE = "dedicated_two_channel_listening_room"
NONE_FULL = "沒有任何一類用到的尺（含它依賴的別類尺，照排名層登記）全部校準完，所以還沒有「已校準項目通過幾項」可以報（等 #358）"


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


def _promoted(prefixes: tuple[str, ...]) -> QualityPurpose:
    """把鍵名前綴命中的每一條升成 calibrated，結構化出處照抄一條已校準的；仍走登記簿的驗證。"""
    raw = _raw_purpose()
    template = next(entry for entry in _items(raw, "setting") if entry["status"] == "calibrated")
    receipt = {key: value for key, value in template.items()
               if key not in ("key", "value", "unit", "status")}

    def promote(entry: Raw, key: str) -> None:
        if key.startswith(prefixes):
            entry.update(receipt | {"status": "calibrated"})
    for kind in ("setting", "target", "qualification"):
        for entry in _items(raw, kind):
            promote(entry, str(entry["key"]))
    for table in _items(raw, "weight"):
        for item in _items(table, "item"):
            promote(item, str(table["key"]))
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
        assert closing == f"用到的尺（含依賴，照排名層登記）全部校準完的類別：{named}；這幾類過不過還沒接進報告（等 #358）"


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


def test_unreadable_snapshot_says_so_and_keeps_the_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, registry = make_store(tmp_path)
    store.status_path.write_text(SearchStatus().model_dump_json())

    def broken(purpose: QualityPurpose) -> object:
        raise ValueError("快照壞了")

    monkeypatch.setattr(report_module, "calibration_progress", broken)
    sections = _sections(render_text(build_report(store, quality_targets_path=registry, run_date=RUN_DATE)))
    assert "尺的校準進度：這次搜尋快照裡的登記簿讀不回，數不出來" in sections["品質合不合格"].splitlines()
    assert "搜尋停了沒" in sections
