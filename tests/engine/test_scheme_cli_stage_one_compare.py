"""第一階段命令列：從曲線重評、完整列出分項，身分不同就不印名次。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from aosr.config.paths import config_path
from aosr.reporting import fem_slices, scheme_cli, stage_one_result as results
from aosr.scoring.stage_one_low_frequency import evaluate_stage_one_low_frequency
from aosr.scoring.stage_one_low_frequency_contract import StageOneLowFrequencyInput
from aosr.scoring.stage_one_score import score_stage_one
from tests.engine import _stage_one_cases as cases
from tests.engine._fem_cli_cases import forbid_fem
from tests.engine.test_scheme_pipeline import _run_control


def _args(*paths: Path) -> list[str]:
    return ["stage-one-compare", *map(str, paths), "--capabilities", str(config_path("capabilities.toml"))]


@pytest.fixture
def pair(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, results.StageOneResult, results.StageOneResult]:
    item = cases.scheme()
    first = results.run_stage_one(item, shards=cases.shards(monkeypatch, item), **cases.run_kwargs())
    document = first.model_dump()
    curves = list(first.curves)
    primary = curves[0].primary
    energies = tuple(value * (100.0 if index % 2 else 1.0) if value is not None else None
                     for index, value in enumerate(primary.fem_energy))
    curves[0] = curves[0].model_copy(update={"primary": primary.model_copy(update={"fem_energy": energies})})
    evaluation = evaluate_stage_one_low_frequency(StageOneLowFrequencyInput(candidate_id=item.scheme_id,
        low_frequency_axis=first.low_frequency_axis, speakers=tuple(curves)), scoring_range_hz=first.score_settings.scored_range_hz)
    second = results.StageOneResult.model_validate({**document, "curves": tuple(curves), "evaluation": evaluation,
        "score": score_stage_one(evaluation, first.score_settings)})
    assert first.score.screening_total is not None and second.score.screening_total is not None
    assert first.score.screening_total < second.score.screening_total
    a, b = tmp_path / "better.json", tmp_path / "worse.json"
    results.save_stage_one_result(first, a)
    results.save_stage_one_result(second, b)
    forbid_fem(monkeypatch)
    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("比較只讀曲線")
    monkeypatch.setattr(fem_slices, "solve_slice", forbidden)
    monkeypatch.setattr(fem_slices, "energies_from_shards", forbidden)
    monkeypatch.setattr("aosr.reporting.physics_identity.physics_identity", lambda **kwargs: first.physics_identity)
    return a, b, first, second


def test_comparable_results_print_standings_components_and_sorted_table(
        pair: tuple[Path, Path, results.StageOneResult, results.StageOneResult], capsys: pytest.CaptureFixture[str]) -> None:
    a, b, first, second = pair
    assert not scheme_cli.main(_args(b, a))
    text = capsys.readouterr().out
    for path, result in ((a, first), (b, second)):
        assert f"{path.name} | 讀回等級 current" in text
        assert f"方案 {result.scheme.scheme_id}" in text
        assert f"第一階段合分 {result.score.screening_total}" in text
        for speaker, channel in zip(result.evaluation.speakers, result.score.channels, strict=True):
            assert speaker.payload is not None
            assert (f"{speaker.speaker_id}：主項 {speaker.payload.own_line_ripple_rms_db} dB；"
                    f"穩健性 {speaker.payload.seat_robustness_rms_db} dB；合分小計 {channel.screening_subtotal}") in text
    assert "名次 | 檔名 | 方案代號 | 合分 | 主項（逐聲道平均） | 穩健性（逐聲道平均） | 最差聲道" in text
    first_row = f"1 | {a.name} | {first.scheme.scheme_id} | {first.score.screening_total}"
    second_row = f"2 | {b.name} | {second.scheme.scheme_id} | {second.score.screening_total}"
    assert text.index(first_row) < text.index(second_row)
    for result, row in ((first, first_row), (second, second_row)):
        principal = next(component.value_db for component in result.score.components if component.name == "own_line_ripple_rms_db")
        robustness = next(component.value_db for component in result.score.components if component.name == "seat_robustness_rms_db")
        assert result.score.worst_channel is not None
        assert f"{row} | {principal} | {robustness} | {result.score.worst_channel.speaker_id}" in text
    assert text.splitlines().count(first.wall_assumption_text) == len({first.wall_assumption_text})
    assert text.splitlines().count(first.score.assumption_text) == len({first.score.assumption_text})


def test_different_identity_refuses_table_with_success_exit(
        pair: tuple[Path, Path, results.StageOneResult, results.StageOneResult], capsys: pytest.CaptureFixture[str]) -> None:
    a, b, _, second = pair
    changed = second.model_copy(update={"physics_identity": "phys-v1:" + "1" * 64})
    results.save_stage_one_result(changed, b)
    assert not scheme_cli.main(_args(a, b))
    text = capsys.readouterr().out
    assert f"{b.name} | 讀回等級 needs_physics" in text
    assert f"不能同表：{b.name} 跟 {a.name} 的 物理身分 不同" in text
    assert "名次 |" not in text
    assert f"第一階段合分 {second.score.screening_total}" in text


def test_unavailable_result_has_reason_and_no_rank(
        pair: tuple[Path, Path, results.StageOneResult, results.StageOneResult], capsys: pytest.CaptureFixture[str]) -> None:
    a, b, first, _ = pair
    document = json.loads(b.read_text())
    document["curves"][0]["primary"]["fem_energy"] = [None for _ in first.curves[0].primary.fem_energy]
    b.write_text(json.dumps(document))
    assert not scheme_cli.main(_args(b, a))
    text = capsys.readouterr().out
    assert f"{b.name} | 讀回等級 remeasured" in text
    assert "不可估" in text and "missing_declared_point" in text
    ranked = f"1 | {a.name} |"
    unavailable = f"— | {b.name} | {first.scheme.scheme_id} | 不可估"
    assert text.index(ranked) < text.index(unavailable)
    assert "第一階段合分 不可估" in text


def test_compare_requires_two_results(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        scheme_cli.main(_args(tmp_path / "one.json"))
    assert error.value.code
    assert "stage-one-compare 至少需要兩份結果" in capsys.readouterr().err


@pytest.mark.parametrize("kind,reason", [("stage-two", "第二階段結果檔，請用 compare"),
    ("unknown", "認不得的結果檔格式"), ("missing", "No such file"), ("truncated", "")])
def test_bad_input_reports_stderr_and_failure(tmp_path: Path, capsys: pytest.CaptureFixture[str], kind: str, reason: str) -> None:
    path = tmp_path / "bad.json"
    if kind == "stage-two":
        path.write_text(_run_control(cases.scheme()).model_dump_json())
    elif kind == "unknown":
        path.write_text(json.dumps({"schema_version": "unknown"}))
    elif kind == "truncated":
        path.write_text("{")
    code = scheme_cli.main(_args(path, tmp_path / "other.json"))
    assert code == 1
    error = capsys.readouterr().err
    assert error.strip() and reason in error
