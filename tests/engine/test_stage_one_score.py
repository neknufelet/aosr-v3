"""第一階段低音篩選分數考卷（票 #558 PR-2）：合分形狀、參考尺度、排序方向、左右平均與登記簿讀法。

答案由考卷自己算：權重、較差參考與計分範圍從登記簿 TOML（設定表）直接解析（``tomllib``），不經被測的讀法；
不寫死權重數字，臨時登記簿一律寫在 ``tmp_path``。曲線用程式造，不跑整房求解。
"""
from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest

from aosr.config.paths import config_path
from aosr.config.quality_targets import TargetEntry, load_quality_targets
from aosr.scoring.recommendation import NotFinalReason
from aosr.scoring.stage_one_low_frequency_contract import (
    StageOneLowFrequencyEvaluation,
    StageOneReasonCode,
)
from aosr.scoring.stage_one_score import (
    COMPONENT_NAMES,
    ComponentName,
    StageOneScore,
    StageOneScoreSettings,
    score_stage_one,
    score_stage_one_input,
    stage_one_score_settings,
)
from tests.engine._stage_one_low_frequency_curves import (
    AXIS_HZ,
    candidate,
    demo_curve_db,
    evaluate,
    gaussian_db,
    speaker,
    with_energy,
)

_REGISTRY = config_path("quality_targets.toml")
_PURPOSE = "dedicated_two_channel_listening_room"
_PRINCIPAL = "own_line_ripple_rms_db"
_ROBUSTNESS = "seat_robustness_rms_db"
_PREFIX = "stage_one_low_frequency."


class _Raw(NamedTuple):
    """考卷自己從 TOML 解析的第一階段那一段：每個分項的權重與較差參考，外加計分範圍。"""

    weight: dict[str, float]
    reference: dict[str, float]
    scored_range_hz: tuple[float, float]


def _section(path: Path = _REGISTRY) -> dict[str, list[dict[str, object]]]:
    section, = (item for item in tomllib.loads(path.read_text(encoding="utf-8"))["stage_one"]
                if item["name"] == _PURPOSE)
    return {name: section[name] for name in ("setting", "target", "weight")}


def _raw(path: Path = _REGISTRY) -> _Raw:
    section = _section(path)
    references = {str(item["key"]).removeprefix(_PREFIX): float(str(item["worse_reference"]))
                  for item in section["target"]}
    table, = (item for item in section["weight"] if item["key"] == _PREFIX + "component_weights")
    rows = table["item"]
    assert isinstance(rows, list)
    weights = {str(item["name"]): float(str(item["value"])) for item in rows}
    scored, = (item for item in section["setting"] if item["key"] == _PREFIX + "scored_range_hz")
    bounds = scored["value"]
    assert isinstance(bounds, list)
    return _Raw(weights, references, (float(bounds[0]), float(bounds[1])))


def _settings(path: Path = _REGISTRY) -> StageOneScoreSettings:
    return stage_one_score_settings(load_quality_targets(path), _PURPOSE)


def _scored_range(path: Path = _REGISTRY) -> tuple[float, float]:
    return _raw(path).scored_range_hz


def _expected_total(values: dict[str, float], path: Path = _REGISTRY) -> float:
    """考卷自己的合分：Σ 權重 × 分項 dB ÷ 較差參考。"""
    raw = _raw(path)
    return sum(raw.weight[name] * values[name] / raw.reference[name] for name in COMPONENT_NAMES)


def _two_speakers() -> StageOneLowFrequencyEvaluation:
    """左右兩支喇叭、周圍點有差的真評估結果（分項不是 0），當改數字的底稿。"""
    curve = demo_curve_db()
    shifted = {"front": curve + gaussian_db(AXIS_HZ, 80.0, 3.0, 0.2)}
    return evaluate(candidate(speaker(curve, shifted, speaker_id="left"),
                              speaker(curve + gaussian_db(AXIS_HZ, 120.0, -4.0, 0.1), speaker_id="right")),
                    _scored_range())


def _with_values(base: StageOneLowFrequencyEvaluation, values: dict[str, dict[str, float]],
                 candidate_id: str = "candidate-a") -> StageOneLowFrequencyEvaluation:
    """把每支喇叭 payload 的兩個分項換成指定值（其餘欄位照底稿）。"""
    speakers = []
    for item in base.speakers:
        assert item.payload is not None
        speakers.append(item.model_copy(update={"payload": item.payload.model_copy(update=values[item.speaker_id])}))
    return base.model_copy(update={"speakers": tuple(speakers), "candidate_id": candidate_id})


def _both(principal: float, robustness: float) -> dict[str, dict[str, float]]:
    return {side: {_PRINCIPAL: principal, _ROBUSTNESS: robustness} for side in ("left", "right")}


def _total(score: StageOneScore) -> float:
    assert score.state == "scored", score.reason_codes
    assert score.screening_total is not None
    return score.screening_total


def _rewrite_many(tmp_path: Path, changes: list[tuple[str, str, object]]) -> Path:
    """臨時登記簿：第一階段那一段裡，每筆（含 ``block_marker`` 的那一塊、``field``、新值）換掉那一行。"""
    text = _REGISTRY.read_text(encoding="utf-8")
    head, marker, rest = text.partition("\n[[stage_one]]\n")
    section, purposes, tail = rest.partition("\n[[purpose]]\n")
    assert marker and purposes
    blocks = section.split("\n[[")
    for block_marker, field, new_value in changes:
        hit, = (index for index, block in enumerate(blocks) if block_marker in block)
        blocks[hit], count = re.subn(rf"(?m)^{field} = .*$", f"{field} = {new_value!r}", blocks[hit])
        assert count == 1, (block_marker, field)
    path = tmp_path / "quality_targets.toml"
    path.write_text(head + marker + "\n[[".join(blocks) + purposes + tail, encoding="utf-8")
    return path


def _rewrite(tmp_path: Path, block_marker: str, field: str, new_value: object) -> Path:
    """只換一格的 ``_rewrite_many``。"""
    return _rewrite_many(tmp_path, [(block_marker, field, new_value)])


def test_smaller_principal_ranks_first_at_equal_robustness() -> None:
    base = _two_speakers()
    settings = _settings()
    low = score_stage_one(_with_values(base, _both(1.0, 2.0)), settings)
    high = score_stage_one(_with_values(base, _both(1.5, 2.0)), settings)
    assert _total(low) < _total(high)


def test_smaller_robustness_ranks_first_at_equal_principal() -> None:
    base = _two_speakers()
    settings = _settings()
    low = score_stage_one(_with_values(base, _both(2.0, 1.0)), settings)
    high = score_stage_one(_with_values(base, _both(2.0, 1.5)), settings)
    assert _total(low) < _total(high)


def test_identical_curves_at_two_placements_score_identically() -> None:
    """兩個擺位交出一模一樣的曲線，分數一模一樣（不要求移動就變分）。"""
    curves = candidate(speaker(demo_curve_db(), {"up": demo_curve_db() + 1.0}))
    settings = _settings()
    first = score_stage_one_input(curves, settings)
    second = score_stage_one_input(curves.model_copy(update={"candidate_id": "candidate-b"}), settings)
    assert _total(first) == _total(second)
    assert first.components == second.components


def test_one_db_on_either_component_adds_the_same_cost() -> None:
    """老闆原話要求核參考尺度：照正式登記簿，兩項各加 1 dB，合計的增量相同（權重 1:1 不被不同尺度改成別的比例）。"""
    raw = _raw()
    # 正式登記簿兩項權重相等：比的是兩格彼此，不寫死數字。
    assert raw.weight[_PRINCIPAL] == raw.weight[_ROBUSTNESS]
    base = _two_speakers()
    settings = _settings()
    start = _total(score_stage_one(_with_values(base, _both(2.0, 3.0)), settings))
    principal = _total(score_stage_one(_with_values(base, _both(3.0, 3.0)), settings)) - start
    robustness = _total(score_stage_one(_with_values(base, _both(2.0, 4.0)), settings)) - start
    assert principal > 0.0
    assert principal == pytest.approx(robustness, rel=1e-12)


# 考卷自己的較差參考：兩項相等但不是 1，「整個漏掉除以較差參考」才分得出來（正式值是 1 時除了等於沒除）。
_OWN_WORSE_REFERENCE_DB = 2.0


def test_equal_references_other_than_one_still_normalize(tmp_path: Path) -> None:
    """臨時登記簿兩個較差參考都改成同一個不是 1 的值：分項、每支小計、合計與 +1 dB 增量都照「dB ÷ 較差參考」。

    答案從考卷寫的那份 TOML 直接算；同一組數字不除較差參考會得到另一個合計，這裡先確認兩者真的分得開。
    """
    path = _rewrite_many(tmp_path, [
        (f'key = "{_PREFIX}{name}"', "worse_reference", _OWN_WORSE_REFERENCE_DB) for name in (_PRINCIPAL, _ROBUSTNESS)
    ])
    raw = _raw(path)
    assert raw.reference == {_PRINCIPAL: _OWN_WORSE_REFERENCE_DB, _ROBUSTNESS: _OWN_WORSE_REFERENCE_DB}
    values = {_PRINCIPAL: 2.0, _ROBUSTNESS: 4.0}
    expected_parts = {name: raw.weight[name] * values[name] / raw.reference[name] for name in COMPONENT_NAMES}
    expected = sum(expected_parts.values())
    unnormalized = sum(raw.weight[name] * values[name] for name in COMPONENT_NAMES)
    assert expected != pytest.approx(unnormalized)
    settings = _settings(path)
    base = _two_speakers()
    score = score_stage_one(_with_values(base, {side: values for side in ("left", "right")}), settings)

    assert {item.name: item.weighted_cost for item in score.components} == pytest.approx(expected_parts, rel=1e-12)
    assert {item.name: item.worse_reference_db for item in score.components} == raw.reference
    assert {channel.speaker_id: channel.screening_subtotal for channel in score.channels} == pytest.approx(
        {"left": expected, "right": expected}, rel=1e-12)
    assert _total(score) == pytest.approx(expected, rel=1e-12)
    for name in COMPONENT_NAMES:
        bumped = {**values, name: values[name] + 1.0}
        increment = _total(score_stage_one(
            _with_values(base, {side: bumped for side in ("left", "right")}), settings)) - _total(score)
        assert increment == pytest.approx(raw.weight[name] / raw.reference[name], rel=1e-12), name


def test_unequal_worse_references_are_rejected(tmp_path: Path) -> None:
    """有人只改一格較差參考，權重 1:1 就變成別的比例：讀登記簿時直接報錯。"""
    reference = _raw().reference[_ROBUSTNESS]
    path = _rewrite(tmp_path, f'key = "{_PREFIX}{_ROBUSTNESS}"', "worse_reference", reference * 2.0)
    with pytest.raises(ValueError, match="較差參考必須相等"):
        _settings(path)


def _rebuilt(
    settings: StageOneScoreSettings, *, targets: tuple[tuple[ComponentName, TargetEntry], ...] | None = None,
    weights: tuple[tuple[ComponentName, float], ...] | None = None,
    scored_range_hz: tuple[float, float] | None = None,
) -> StageOneScoreSettings:
    """照原設定逐欄重建，只換指定的那一欄；讀回結果檔建設定也是走建構子、不經讀登記簿那一支。"""
    return StageOneScoreSettings(
        purpose=settings.purpose, scored_range_hz=scored_range_hz or settings.scored_range_hz,
        targets=settings.targets if targets is None else targets,
        weights=settings.weights if weights is None else weights,
        status=settings.status, fingerprint=settings.fingerprint,
    )


@pytest.mark.parametrize(("damage", "reason"), [
    ("missing-weight", "各一次"), ("duplicate-weight", "各一次"), ("repeated-weight", "各一次"), ("missing-target", "各一次"),
    ("duplicate-target", "各一次"), ("cost-shape", "less_is_better"), ("unequal-references", "較差參考必須相等"),
    ("reversed-range", "遞增"),
])
def test_settings_guard_their_own_shape(damage: str, reason: str) -> None:
    """設定自己守形狀（審查第三輪：讀回結果檔不經讀登記簿那一支）；控制組原樣重建照常建得起來。"""
    settings = _settings()
    assert _rebuilt(settings) == settings
    (name, entry), *others = settings.targets
    with pytest.raises(ValueError, match=reason):
        if damage == "missing-weight":
            _rebuilt(settings, weights=settings.weights[1:])
        elif damage == "duplicate-weight":
            _rebuilt(settings, weights=(*settings.weights, (name, 0.0)))
        elif damage == "repeated-weight":
            # 總數照樣兩個，但同一個分項出現兩次、另一個不見。
            _rebuilt(settings, weights=(settings.weights[0], settings.weights[0]))
        elif damage == "missing-target":
            _rebuilt(settings, targets=tuple(others))
        elif damage == "duplicate-target":
            _rebuilt(settings, targets=(*settings.targets, (name, entry)))
        elif damage == "cost-shape":
            _rebuilt(settings, targets=((name, entry.model_copy(update={"cost_shape": "beyond_threshold_only"})),
                                        *others))
        elif damage == "unequal-references":
            _rebuilt(settings, targets=((name, entry.model_copy(update={"worse_reference": entry.worse_reference * 2.0})),
                                        *others))
        else:
            lower, upper = settings.scored_range_hz
            _rebuilt(settings, scored_range_hz=(upper, lower))


def test_total_follows_weights_in_a_temporary_registry(tmp_path: Path) -> None:
    """臨時登記簿把穩健性權重改掉，合計照新權重算（權重讀登記簿，不寫在程式裡）。"""
    weight = _raw().weight[_ROBUSTNESS]
    path = _rewrite(tmp_path, f'name = "{_ROBUSTNESS}"', "value", weight * 3.0)
    values = {_PRINCIPAL: 1.25, _ROBUSTNESS: 2.5}
    evaluation = _with_values(_two_speakers(), {side: values for side in ("left", "right")})
    changed = _total(score_stage_one(evaluation, _settings(path)))
    original = _total(score_stage_one(evaluation, _settings()))
    assert changed == pytest.approx(_expected_total(values, path), rel=1e-12)
    assert original == pytest.approx(_expected_total(values), rel=1e-12)
    assert changed != pytest.approx(original)


def test_channels_are_averaged_and_the_worst_is_only_listed() -> None:
    """左右兩支分項取等權平均算合計；小計較大的那一支另列，不拿它當合計。"""
    left = {_PRINCIPAL: 1.0, _ROBUSTNESS: 0.5}
    right = {_PRINCIPAL: 3.0, _ROBUSTNESS: 2.5}
    score = score_stage_one(_with_values(_two_speakers(), {"left": left, "right": right}), _settings())
    mean = {name: (left[name] + right[name]) / 2.0 for name in COMPONENT_NAMES}
    assert _total(score) == pytest.approx(_expected_total(mean), rel=1e-12)
    assert _total(score) < _expected_total(right)
    assert {item.name: item.value_db for item in score.components} == pytest.approx(mean, rel=1e-12)
    assert score.worst_channel is not None
    assert score.worst_channel.speaker_id == "right"
    assert score.worst_channel.screening_subtotal == pytest.approx(_expected_total(right), rel=1e-12)
    assert score.principal_only_sort_key_db == pytest.approx(mean[_PRINCIPAL], rel=1e-12)


@pytest.mark.parametrize("order", ["left_first", "right_first"])
def test_worse_left_channel_is_listed_whatever_the_input_order(order: str) -> None:
    """左聲道比較差時最差那一支是左聲道；喇叭交進來的先後換過也一樣（不是固定挑第一支或最後一支）。"""
    left = {_PRINCIPAL: 3.0, _ROBUSTNESS: 2.5}
    right = {_PRINCIPAL: 1.0, _ROBUSTNESS: 0.5}
    evaluation = _with_values(_two_speakers(), {"left": left, "right": right})
    if order == "right_first":
        evaluation = evaluation.model_copy(update={"speakers": tuple(reversed(evaluation.speakers))})
    assert [item.speaker_id for item in evaluation.speakers][-1] == ("right" if order == "left_first" else "left")
    score = score_stage_one(evaluation, _settings())
    assert {channel.speaker_id: channel.screening_subtotal for channel in score.channels} == pytest.approx(
        {"left": _expected_total(left), "right": _expected_total(right)}, rel=1e-12)
    assert score.worst_channel is not None
    assert score.worst_channel.speaker_id == "left"
    assert score.worst_channel.screening_subtotal == pytest.approx(_expected_total(left), rel=1e-12)
    mean = {name: (left[name] + right[name]) / 2.0 for name in COMPONENT_NAMES}
    assert _total(score) == pytest.approx(_expected_total(mean), rel=1e-12)


def test_principal_only_key_orders_by_the_principal_alone() -> None:
    """主項單獨排序的鍵只看主項：合計排前的那一個，主項鍵可以排後。"""
    base = _two_speakers()
    settings = _settings()
    a = score_stage_one(_with_values(base, _both(2.0, 0.5)), settings)
    b = score_stage_one(_with_values(base, _both(1.5, 2.0)), settings)
    assert _total(a) < _total(b)
    assert a.principal_only_sort_key_db is not None and b.principal_only_sort_key_db is not None
    assert b.principal_only_sort_key_db < a.principal_only_sort_key_db


def test_one_unavailable_channel_makes_the_candidate_unavailable() -> None:
    """右聲道缺一個宣告點：整個候選不可估，不拿左聲道湊；逐聲道結果照留。"""
    curve = demo_curve_db()
    right = with_energy(speaker(curve, speaker_id="right"), {len(AXIS_HZ) // 2: None})
    score = score_stage_one_input(candidate(speaker(curve, speaker_id="left"), right), _settings())
    assert score.state == "unavailable"
    assert score.screening_total is None and score.principal_only_sort_key_db is None
    assert score.reason_codes == (StageOneReasonCode.MISSING_DECLARED_POINT,)
    assert {channel.speaker_id: channel.state for channel in score.channels} == {
        "left": "scored", "right": "unavailable",
    }


def test_score_carries_components_labels_and_not_final_reasons() -> None:
    """分項剛好兩個具名分項（拖尾不在內）；假設標示照抄；登記簿是基線所以帶「校準仍是基線」的非最終原因。"""
    evaluation = _two_speakers()
    score = score_stage_one(evaluation, _settings())
    assert {item.name for item in score.components} == {_PRINCIPAL, _ROBUSTNESS}
    assert score.score_kind == "screening"
    assert (score.assumption, score.assumption_text) == (evaluation.assumption, evaluation.assumption_text)
    assert score.channel_summation == "left_right_coherent_sum_not_evaluated"
    statuses = {item["status"] for item in _section_records()}
    assert score.settings_status == ("baseline" if "baseline" in statuses else "calibrated")
    assert NotFinalReason.NO_FINALIZING_PROCESS in score.not_final_reasons
    assert NotFinalReason.EXTERNAL_NOT_CHECKED in score.not_final_reasons
    assert (NotFinalReason.CALIBRATION_BASELINE in score.not_final_reasons) == ("baseline" in statuses)


def _section_records() -> list[dict[str, object]]:
    section = _section()
    tables = [table["item"] for table in section["weight"]]
    return [*section["setting"], *section["target"],
            *(row for rows in tables if isinstance(rows, list) for row in rows)]


@pytest.mark.parametrize("change", ["added", "renamed"])
def test_components_other_than_the_two_named_are_rejected(tmp_path: Path, change: str) -> None:
    """權重表多一項（例如拖尾超出量）或某一項換了名字都拒收：第一階段分數只有兩個具名分項。"""
    text = _REGISTRY.read_text(encoding="utf-8")
    marker = f'[[stage_one.weight.item]]\nname = "{_ROBUSTNESS}"\n'
    head, found, tail = text.partition(marker)
    block, purposes, rest = tail.partition("\n[[purpose]]\n")
    assert found and purposes
    decay = '[[stage_one.weight.item]]\nname = "low_frequency_decay_excess"\n'
    if change == "added":
        changed = head + found + block + "\n" + decay + block + purposes + rest
    else:
        changed = head + decay + block + purposes + rest
    path = tmp_path / "quality_targets.toml"
    path.write_text(changed, encoding="utf-8")
    with pytest.raises(ValueError, match="分項必須剛好是"):
        _settings(path)


def test_registered_scored_range_drives_the_evaluator_and_must_match() -> None:
    """計分範圍讀登記簿那一格；拿別的範圍量出來的結果不准合分。"""
    curves = candidate(speaker(demo_curve_db()))
    settings = _settings()
    assert settings.scored_range_hz == _scored_range()
    assert score_stage_one_input(curves, settings).scored_range_hz == _scored_range()
    lower, upper = _scored_range()
    narrower = evaluate(curves, (lower, float(np.sqrt(lower * upper))))
    with pytest.raises(ValueError, match="計分範圍"):
        score_stage_one(narrower, settings)


def test_fingerprint_hashes_only_the_cells_it_reads(tmp_path: Path) -> None:
    """改第二階段的一格，第一階段合分的設定指紋不變；改第一階段的權重就變。"""
    original = _settings().fingerprint
    text = _REGISTRY.read_text(encoding="utf-8")
    stage_two = text.replace('key = "timbre_balance.min_points"\nvalue = ', 'key = "timbre_balance.min_points"\nvalue = 1', 1)
    assert stage_two != text
    stage_two_path = tmp_path / "stage_two_changed.toml"
    stage_two_path.write_text(stage_two, encoding="utf-8")
    assert load_quality_targets(stage_two_path).fingerprint != load_quality_targets(_REGISTRY).fingerprint
    assert _settings(stage_two_path).fingerprint == original
    weight = _raw().weight[_PRINCIPAL]
    stage_one_path = _rewrite(tmp_path, f'name = "{_PRINCIPAL}"', "value", weight * 2.0)
    assert _settings(stage_one_path).fingerprint != original
