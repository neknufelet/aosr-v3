"""#558 第二段 B：顯示答案從原始量與登記簿推，排名除分項展示外逐格不變。"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from aosr.config.frequency_axis import LowFrequencyAxis, low_frequency_axis_frequencies
from aosr.config.paths import config_path
from aosr.config.quality_targets import (
    QualityPurpose, QualityTargets, SettingEntry, TargetEntry, WeightTable, load_quality_targets,
)
from aosr.gui.result_view import build_result_view
from aosr.reporting import scheme_cli
from aosr.reporting.compare import compare_results
from aosr.reporting.result import SchemeResult
from aosr.scoring import ranking
from aosr.scoring.contract import CategoryEvaluation, EvaluationState, QualityCategory
from aosr.scoring.contract_base import ReasonCode
from aosr.scoring.low_frequency_response import evaluate_low_frequency_response
from aosr.scoring.low_frequency_response_contract import LowFrequencyResponsePayload
from aosr.scoring.ranking_models import ComponentLine
from aosr.scoring.stage_one_low_frequency_contract import StageOneSpeakerCurves
from aosr.scoring.stage_one_score import PRINCIPAL, ROBUSTNESS, SCORED_RANGE_KEY, WEIGHTS_KEY
from tests.engine._cross_cpu_tolerance import SCORING_ABS, SCORING_REL
from tests.engine._stage_one_low_frequency_curves import DIRECTIONS, candidate, speaker
from tests.engine.test_gui_compare_view import _renamed, _view
from tests.engine.test_low_frequency_response_reload import saved as saved

TARGETS = config_path("quality_targets.toml")
LF = QualityCategory.LOW_FREQUENCY_RESPONSE
NAMES: dict[str, str] = {PRINCIPAL: "主位起伏", ROBUSTNESS: "座位差異"}
# 最差聲道照喇叭代號印中文；代號不在表裡就照印代號。
SPEAKER_NAMES: dict[str, str] = {"left": "左聲道喇叭", "right": "右聲道喇叭"}
ZEROS = np.zeros(len(low_frequency_axis_frequencies(LowFrequencyAxis.SEARCH_48)[0]))


def low_frequency(result: SchemeResult) -> CategoryEvaluation:
    return next(item for item in result.candidate.evaluations if item.category is LF)


def measured(result: SchemeResult, *speakers: StageOneSpeakerCurves) -> SchemeResult:
    """換上用造好曲線量出的低頻評估；場景、擺位與出處照抄原評估，其他類不動。"""
    old = low_frequency(result)
    curves = candidate(*speakers, axis=LowFrequencyAxis.SEARCH_48).model_copy(
        update={"candidate_id": result.scheme.scheme_id})
    changed = evaluate_low_frequency_response(curves, registry=load_quality_targets(TARGETS),
        purpose=result.scheme.purpose, scene_fingerprint=old.scene_fingerprint,
        placement=old.placement, provenance=old.provenance)
    return result.model_copy(update={"candidate": result.candidate.model_copy(update={"evaluations": tuple(
        changed if item.category is LF else item for item in result.candidate.evaluations)})})


def expected_description(registry: QualityTargets, purpose: str) -> str:
    """說明句的答案直接讀登記簿兩格，不呼叫被測的說明函式。"""
    bounds = registry.stage_one(purpose).entry(SCORED_RANGE_KEY)
    weights = registry.purpose(purpose).entry("ranking.category_weights")
    assert isinstance(bounds, SettingEntry) and isinstance(bounds.value, tuple)
    assert isinstance(weights, WeightTable)
    weight = next(item.value for item in weights.item if item.name == LF.value)
    low, high = bounds.value
    return (f"{low:g}～{high:g} Hz 的主位起伏與座位差異，量法沿用第一階段；"
            f"類權重 {weight:g} 是暫定工程選擇，不宣稱聽感校準")


def test_shared_labels_and_description_read_registry(saved: SchemeResult, tmp_path: Path) -> None:
    from aosr.reporting.display import LABELS, low_frequency_response_description
    from tests.engine.test_stage_one_score import _rewrite
    assert {key: LABELS[key] for key in NAMES} == NAMES
    paths = (TARGETS, _rewrite(tmp_path, f'key = "{SCORED_RANGE_KEY}"', "value", [35.0, 240.0]))
    registries = [load_quality_targets(path) for path in paths]
    original = registries[0]
    purpose = original.purpose(saved.scheme.purpose)
    table = purpose.entry("ranking.category_weights")
    assert isinstance(table, WeightTable)
    changed_table = table.model_copy(update={"item": tuple(item.model_copy(update={"value": 0.375})
        if item.name == LF.value else item for item in table.item)})
    changed_purpose = purpose.model_copy(update={"weight": tuple(changed_table if item.key == table.key else item
        for item in purpose.weight)})
    registries.append(original.model_copy(update={"purposes": (changed_purpose,)}))
    for registry in registries:
        assert low_frequency_response_description(registry, saved.scheme.purpose) == expected_description(
            registry, saved.scheme.purpose)


@pytest.mark.parametrize("own_weight", (None, 2.0), ids=("registry", "own-weight-2"))
def test_component_values_weights_and_unchanged_ranking(saved: SchemeResult, monkeypatch: pytest.MonkeyPatch,
                                                         tmp_path: Path, own_weight: float | None) -> None:
    from tests.engine.test_stage_one_score import _rewrite
    path = TARGETS if own_weight is None else _rewrite(tmp_path, f'name = "{PRINCIPAL}"', "value", own_weight)
    registry = load_quality_targets(path)
    pair = (saved, _renamed(saved, "other"))
    after = compare_results(pair, quality_targets=registry, run_date=saved.run_date)
    section = registry.stage_one(saved.scheme.purpose)
    table = section.entry(WEIGHTS_KEY)
    assert isinstance(table, WeightTable)
    weights = {item.name: item.value for item in table.item}
    if own_weight is not None:
        assert weights[PRINCIPAL] == own_weight
    references: dict[str, float] = {}
    for name in NAMES:
        entry = section.entry(f"stage_one_low_frequency.{name}")
        assert isinstance(entry, TargetEntry) and entry.cost_shape == "less_is_better" and entry.value == 0.0
        references[name] = entry.worse_reference
    for row in after.rankable:
        line = next(item for item in row.categories if item.identity.category is LF)
        raw = {item.name: item.value for item in line.evaluation.raw_quantities}
        assert {item.name: (item.raw_value, item.raw_unit, item.weight, item.role) for item in line.components} == {
            name: (value, "dB", weights[name], "principal") for name, value in raw.items()}
        # 分項代價未乘權重（原始 dB ÷ 較差參考），乘上權重加總才是類代價；跟別類同一個寫法，權重不是 1 也成立。
        assert {item.name: item.cost for item in line.components} == pytest.approx(
            {name: value / references[name] for name, value in raw.items()}, rel=SCORING_REL, abs=SCORING_ABS)
        assert math.fsum(item.cost * weights[item.name] for item in line.components) == pytest.approx(
            line.category_cost, rel=SCORING_REL, abs=SCORING_ABS)
    original = ranking._component_lines

    def before(evaluation: CategoryEvaluation, purpose: QualityPurpose) -> tuple[ComponentLine, ...]:
        if evaluation.category is not LF:
            return original(evaluation, purpose)
        assert evaluation.category_cost is not None
        return tuple(ComponentLine(name=name, role="reported", raw_value=None, raw_unit="1", cost=value, weight=None)
                     for name, value in sorted(evaluation.category_cost.components.items()))

    with monkeypatch.context() as patch:
        patch.setattr(ranking, "_component_lines", before)
        previous = compare_results(pair, quality_targets=registry, run_date=saved.run_date)
    # 把新增展示欄位還原成改前的通用列，其餘整份排名（含總分、類代價與名次）逐位相同。
    document = after.model_dump(mode="json")
    for row in document["rankable"]:
        for line in row["categories"]:
            if line["identity"]["category"] == LF.value:
                line["components"] = sorted([
                    {**item, "role": "reported", "raw_value": None, "raw_unit": "1", "weight": None}
                    for item in line["components"]], key=lambda item: item["name"])
    assert document == previous.model_dump(mode="json")


def assert_details(text: str, evaluation: CategoryEvaluation) -> None:
    for item in evaluation.raw_quantities:
        assert f"{NAMES[item.name]} {item.value:.3f} dB（逐聲道平均）" in text
    payload = evaluation.payload
    assert isinstance(payload, LowFrequencyResponsePayload)
    worst = payload.worst_channel.speaker_id
    assert f"最差聲道 {SPEAKER_NAMES.get(worst, worst)}（只提醒、不淘汰）" in text
    low, high = payload.scored_range_hz
    assert f"計分範圍 {low:g}～{high:g} Hz" in text
    assert payload.channel_summation_text in text


def test_result_and_comparison_details(saved: SchemeResult) -> None:
    result = build_result_view(saved, quality_targets_path=TARGETS)
    row = next(item for item in result.categories if item.category == LF.value)
    assert row.state_label == "已算代價"
    assert row.cost is not None
    assert_details(row.note, low_frequency(saved))
    description = expected_description(load_quality_targets(TARGETS), saved.scheme.purpose)
    assert row.note.split("\n")[0] == description
    view = _view((saved, _renamed(saved, "other")))
    compared = next(item for item in view.categories if item.category == LF.value)
    assert compared.better == "same" and compared.better_text == "相同"
    for side in (compared.a, compared.b):
        assert_details(side.note, low_frequency(saved))
    assert {line for line in compared.note_text.split("\n") if description in line} == {
        f"A：{description}", f"B：{description}"}


def test_worst_channel_names_the_right_speaker_when_it_is_worse(saved: SchemeResult) -> None:
    # 右聲道六方向偏得比左聲道多，最差聲道就該印右聲道；只造左聲道最差的題分不出「永遠印左」（審查補）。
    changed = measured(saved, speaker(ZEROS, {name: ZEROS + 1.0 for name in DIRECTIONS}, speaker_id="left"),
                       speaker(ZEROS, {name: ZEROS + 6.0 for name in DIRECTIONS}, speaker_id="right"))
    payload = low_frequency(changed).payload
    assert isinstance(payload, LowFrequencyResponsePayload) and payload.worst_channel.speaker_id == "right"
    row = next(item for item in build_result_view(changed, quality_targets_path=TARGETS).categories
               if item.category == LF.value)
    assert "最差聲道 右聲道喇叭（只提醒、不淘汰）" in row.note
    assert_details(row.note, low_frequency(changed))


@pytest.mark.parametrize("reason,chinese", [
    (ReasonCode.MISSING_DECLARED_POINT, "低頻軸缺少有限元素點"),
    (ReasonCode.INSUFFICIENT_POINTS, "低頻資料點不足"),
    (ReasonCode.SCORING_RANGE_GAP, "低頻計分範圍有缺段"),
    (ReasonCode.LOW_FREQUENCY_SEAT_LAYOUT_INVALID, "低頻量法需要主位加前後左右上下六方向"),
])
def test_unavailable_reason_visible(saved: SchemeResult, reason: ReasonCode, chinese: str,
                                    capsys: pytest.CaptureFixture[str]) -> None:
    unavailable = low_frequency(saved).model_copy(update={"state": EvaluationState.UNAVAILABLE,
        "payload": None, "raw_quantities": (), "category_cost": None, "reason_codes": (reason,)})
    changed = saved.model_copy(update={"candidate": saved.candidate.model_copy(update={"evaluations": tuple(
        unavailable if item.category is LF else item for item in saved.candidate.evaluations)})})
    row = next(item for item in build_result_view(changed, quality_targets_path=TARGETS).categories
               if item.category == LF.value)
    assert row.state_label == "不可估" and row.cost is None
    assert chinese in row.note and "不可估" in row.note
    compared = next(item for item in _view((changed, _renamed(saved, "other"))).categories
                    if item.category == LF.value)
    assert not compared.better and chinese in compared.note_text
    ranked = compare_results((changed,), quality_targets=load_quality_targets(TARGETS), run_date=saved.run_date)
    scheme_cli._print_result(changed, ranked)
    assert f"不可估：{chinese}" in capsys.readouterr().out


def test_cli_single_and_compare_details(saved: SchemeResult, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    registry = load_quality_targets(TARGETS)
    ranked = compare_results((saved,), quality_targets=registry, run_date=saved.run_date)
    scheme_cli._print_result(saved, ranked)
    text = capsys.readouterr().out
    assert "低頻頻響" in text
    assert_details(text, low_frequency(saved))
    assert expected_description(registry, saved.scheme.purpose) in text
    paths = [tmp_path / "a.json", tmp_path / "b.json"]
    for path, result in zip(paths, (saved, _renamed(saved, "other")), strict=True):
        path.write_text(result.model_dump_json(), encoding="utf-8")
    assert not scheme_cli.main(["compare", *map(str, paths), "--capabilities", str(config_path("capabilities.toml"))])
    text = capsys.readouterr().out
    assert_details(text, low_frequency(saved))
    assert "low_frequency_response |" in text


def test_comparison_uses_same_rule_for_low_frequency_costs_and_split_tables(saved: SchemeResult) -> None:
    other = measured(_renamed(saved, "other"), speaker(ZEROS, {name: ZEROS + 6.0 for name in DIRECTIONS}))
    changed = low_frequency(other)
    for pair, better in (((saved, other), "a"), ((other, saved), "b")):
        view = _view(pair)
        row = next(item for item in view.categories if item.category == LF.value)
        assert row.better == better
        assert_details(row.b.note if better == "a" else row.a.note, changed)
        # 合起來的說明欄：A 那段是第一份、B 那段是第二份；兩份數字不同，前綴對調就會紅（審查補）。
        a_part, b_part = row.note_text.split("\nB：")
        assert a_part.startswith("A：")
        assert_details(a_part, low_frequency(pair[0]))
        assert_details(b_part, low_frequency(pair[1]))
    versioned = changed.model_copy(update={"evaluator_version": changed.evaluator_version + ".other"})
    split = other.model_copy(update={"candidate": other.candidate.model_copy(update={"evaluations": tuple(
        versioned if item.category is LF else item for item in other.candidate.evaluations)})})
    view = _view((saved, split))
    row = next(item for item in view.categories if item.category == LF.value)
    assert not view.table.same_table
    assert not row.better and "評分條件不同" in row.comparison_text


def test_capability_description_and_outputs_include_low_frequency() -> None:
    from aosr.config.capabilities import load_capabilities
    entry = load_capabilities(config_path("capabilities.toml")).for_entry("scheme_pipeline")
    assert "六類評估（音色、聆聽區、反射、殘響、左右匹配、低頻頻響）" in entry.note
    expected = {QualityCategory.TIMBRE_BALANCE.value, QualityCategory.LISTENING_AREA_STABILITY.value,
        QualityCategory.REFLECTIONS_AND_ECHO.value, QualityCategory.REVERBERATION.value,
        QualityCategory.CHANNEL_MATCHING.value, LF.value}
    for row in entry.capability:
        assert set(row.outputs) == expected
        assert row.status == "experimental"
        assert "低頻拖尾不計分" in row.note
        assert "低頻頻響沿用第一階段量法與合分，以暫定類權重參與第二階段" in row.note


def test_missing_raw_quantity_leaves_the_component_raw_value_blank(saved: SchemeResult) -> None:
    # 原始量缺一項時分項列留空，不讓展示欄擋住整次排名；類代價照第一階段合分（審查補）。
    evaluation = low_frequency(saved)
    partial = evaluation.model_copy(update={"raw_quantities": tuple(
        item for item in evaluation.raw_quantities if item.name == PRINCIPAL)})
    changed = saved.model_copy(update={"candidate": saved.candidate.model_copy(update={"evaluations": tuple(
        partial if item.category is LF else item for item in saved.candidate.evaluations)})})
    registry = load_quality_targets(TARGETS)
    full, cut = (next(line for line in compare_results((result,), quality_targets=registry,
                                                       run_date=saved.run_date).rankable[0].categories
                      if line.identity.category is LF) for result in (saved, changed))
    assert cut.category_cost == full.category_cost
    assert {item.name: item.raw_value for item in cut.components} == {
        PRINCIPAL: next(item.value for item in evaluation.raw_quantities if item.name == PRINCIPAL), ROBUSTNESS: None}
