"""方案存讀、結果交叉身分與無求解邊界的修補考卷。"""
from __future__ import annotations

import json
import fcntl
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path

import pytest

from aosr.config.capabilities import load_capabilities
from aosr.config.paths import config_path
from aosr.geometry.shoebox import Point
from aosr.config.frequency_axis import LowFrequencyAxis
from aosr.physics import report_io, three_lane_report
from aosr.physics.report_source import SourceModelKind
from aosr.reporting import pipeline
from aosr.reporting import scheme_cli
from aosr.reporting.compare import compare_results
from aosr.reporting.result import (RESULT_SCHEMA_VERSION, SchemeResult, Timings,
                                   load_result, read_registry_settings, reevaluate, save_result)
from aosr.reporting.scheme import Scheme, load_scheme, pair_input_document
from aosr.scoring.contract import (CandidateEvaluation, CategoryEvaluation,
                                   EvaluationState, QualityCategory, ReasonCode)
from aosr.config.quality_targets import QualityTargets, load_quality_targets
from aosr.scoring.ranking_models import RankingResult
from aosr.scoring.ranking_models import EliminatedRow
from aosr.scoring.category_registry import EliminationReason
from tests.engine._directivity import DIRECTIVITY
from tests.engine import _scoring_source_model_control as control
from tests.engine.test_scheme_pipeline import _many_fem, _scheme


@pytest.fixture(scope="module")
def result(tmp_path_factory: pytest.TempPathFactory) -> SchemeResult:
    base = tmp_path_factory.getbasetemp()
    shared = base.parent if base.name.startswith("popen-gw") else base
    path = shared / "repair-wall-1.json"
    with (shared / "repair-wall-1.lock").open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if path.exists():
            return SchemeResult.model_validate_json(path.read_text(encoding="utf-8"))
        with pytest.MonkeyPatch.context() as patch:
            for module, name, fake in control.STAND_INS:
                patch.setattr(module, name, fake)
            patch.setattr(three_lane_report, "_solve_fem_energies", _many_fem)
            patch.setattr(pipeline, "report_capability",
                          lambda table: three_lane_report._unchecked_capability())
            calculated = pipeline.run_scheme(
                _scheme("wall-1"), capabilities=load_capabilities(config_path("capabilities.toml")),
                directivity=DIRECTIVITY, quality_targets_path=control.TARGETS,
                engine_commit="control", run_date=date(2026, 9, 27))
        save_result(calculated, path)
        return calculated


@pytest.fixture(scope="module")
def second_result(result: SchemeResult) -> SchemeResult:
    original = result.model_dump_json()
    copied = original.replace('"wall-1"', '"copy-2"')
    assert copied != original
    renamed = SchemeResult.model_validate_json(copied)
    candidate = reevaluate(renamed, quality_targets_path=control.TARGETS)
    return SchemeResult.model_validate(renamed.model_copy(update={"candidate": candidate}))


@pytest.fixture(scope="module")
def wall_2() -> SchemeResult:
    scheme = _scheme("wall-2")
    points = tuple(point.model_copy(update={"position_m": (
        point.position_m[0] + 0.1, *point.position_m[1:])})
        for point in scheme.receiver_set.points)
    receivers = scheme.receiver_set.model_copy(update={"points": points})
    scheme = Scheme.model_validate(scheme.model_copy(update={"receiver_set": receivers}))
    with pytest.MonkeyPatch.context() as patch:
        for module, name, fake in control.STAND_INS:
            patch.setattr(module, name, fake)
        patch.setattr(three_lane_report, "_solve_fem_energies", _many_fem)
        patch.setattr(pipeline, "report_capability",
                      lambda table: three_lane_report._unchecked_capability())
        return pipeline.run_scheme(scheme,
            capabilities=load_capabilities(config_path("capabilities.toml")),
            directivity=DIRECTIVITY, quality_targets_path=control.TARGETS,
            engine_commit="control", run_date=date(2026, 9, 27))


def _document(result: SchemeResult) -> dict[str, object]:
    return result.model_dump(mode="json")


def _pairs(document: dict[str, object]) -> list[dict[str, object]]:
    pairs = document["pairs"]
    assert isinstance(pairs, list)
    return pairs


def _reject(document: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SchemeResult.model_validate(document)


@pytest.mark.parametrize("scattering_present", [False, True])
def test_scheme_json_round_trip_produces_identical_accepted_inputs(
    tmp_path: Path, scattering_present: bool,
) -> None:
    document = _scheme("wall-1").model_dump(mode="json")
    if not scattering_present:
        del document["scene"]["scattering_by_wall"]
    original = Scheme.model_validate(document)
    path = tmp_path / "scheme.json"
    path.write_text(original.model_dump_json(), encoding="utf-8")
    loaded = load_scheme(path)
    assert loaded == original
    table = load_capabilities(config_path("capabilities.toml"))
    for scheme in (original, loaded):
        for channel in scheme.channel_group.channels:
            for receiver in scheme.receiver_set.points:
                source = scheme.speakers[channel.speaker_id]
                target = Point(*receiver.position_m)
                current = pair_input_document(scheme, source, target,
                                              {"kind": "omnidirectional"})
                scene = {key: value for key, value in original.scene.model_dump(mode="json").items()
                         if value is not None}
                expected = scene | {
                    "source_m": {"x": source.x, "y": source.y, "z": source.z},
                    "receiver_m": {"x": target.x, "y": target.y, "z": target.z},
                    "source_model": {"kind": "omnidirectional"},
                }
                assert json.dumps(current, sort_keys=True) == json.dumps(expected, sort_keys=True)
                assert all(current.get(key) is not None for key in current)
                report_io.load_input_document(current, table, DIRECTIVITY)


def test_result_requires_schema_version(result: SchemeResult) -> None:
    document = _document(result)
    del document["schema_version"]
    _reject(document, "schema_version")


def test_result_candidate_id_matches_scheme(result: SchemeResult) -> None:
    document = _document(result)
    candidate = document["candidate"]
    assert isinstance(candidate, dict)
    candidate["candidate_id"] = "other"
    candidate["evaluations"] = []
    _reject(document, "candidate_id")


@pytest.mark.parametrize("change", ["missing", "extra", "duplicate", "role"])
def test_result_pairs_exact_cross_product(result: SchemeResult, change: str) -> None:
    document = _document(result)
    pairs = _pairs(document)
    if change == "missing":
        pairs.pop()
    elif change in {"extra", "duplicate"}:
        new_pair = dict(pairs[0])
        if change == "extra":
            new_pair["receiver_id"] = "unknown"
        pairs.append(new_pair)
    else:
        pairs[0]["role"] = "right"
    _reject(document, "pairs")


@pytest.mark.parametrize("field", ["listening_area_channel_role", "listening_area_speaker_id"])
def test_result_listening_channel_matches_first_declared(result: SchemeResult, field: str) -> None:
    document = _document(result)
    document[field] = "right"
    _reject(document, field)


def test_result_after_validator_runs_on_existing_model(result: SchemeResult) -> None:
    changed = result.model_copy(update={"listening_area_speaker_id": "right"})
    with pytest.raises(ValueError, match="listening_area_speaker_id"):
        SchemeResult.model_validate(changed)


def test_result_listening_payload_speaker_matches_selection(result: SchemeResult) -> None:
    document = _document(result)
    candidate = document["candidate"]
    assert isinstance(candidate, dict)
    evaluations = candidate["evaluations"]
    assert isinstance(evaluations, list)
    listening = next(item for item in evaluations
                     if item["category"] == QualityCategory.LISTENING_AREA_STABILITY.value)
    listening["payload"]["speaker_id"] = "right"
    _reject(document, "聆聽區.*speaker_id")


def test_result_listening_provenance_speaker_matches_selection(result: SchemeResult) -> None:
    document = _document(result)
    candidate = document["candidate"]
    assert isinstance(candidate, dict)
    evaluations = candidate["evaluations"]
    assert isinstance(evaluations, list)
    listening = next(item for item in evaluations
                     if item["category"] == QualityCategory.LISTENING_AREA_STABILITY.value)
    listening["provenance"]["speaker_id"] = "right"
    _reject(document, "聆聽區.*speaker_id")


@pytest.mark.parametrize("field", [
    "source_m.x", "source_m.y", "source_m.z",
    "receiver_m.x", "receiver_m.y", "receiver_m.z",
    "room_m", "sound_speed_m_s", "density_kg_m3",
    "impedance_pa_s_per_m_by_wall", "scattering_by_wall",
    "reflection_order_k", "low_frequency_axis", "source_model",
])
def test_result_pair_document_matches_scheme(result: SchemeResult, field: str) -> None:
    document = _document(result)
    pair_input = _pairs(document)[0]["input_document"]
    assert isinstance(pair_input, dict)
    if field == "source_model":
        pair_input[field] = {"kind": SourceModelKind.ANALYTIC_AXISYMMETRIC_TWO_PARAMETER_V1.value}
    elif field == "room_m":
        pair_input[field] = {"Lx": 8.0, "Ly": 4.1, "Lz": 2.9}
    elif field.startswith(("source_m.", "receiver_m.")):
        name, axis = field.split(".")
        point = dict(pair_input[name])
        point[axis] += 0.1
        pair_input[name] = point
    elif field == "impedance_pa_s_per_m_by_wall":
        walls = dict(pair_input[field])
        walls["x0"] += 1.0
        pair_input[field] = walls
    elif field == "scattering_by_wall":
        scattering = dict(pair_input[field])
        scattering["x0"] += 0.01
        pair_input[field] = scattering
    elif field == "reflection_order_k":
        pair_input[field] = 2
    elif field == "low_frequency_axis":
        pair_input[field] = LowFrequencyAxis.VERIFICATION.value
    else:
        pair_input[field] = 344.0
    _reject(document, "input_document")


def test_result_product_default_requires_analytic_input_kind(result: SchemeResult) -> None:
    document = _document(result)
    scheme = document["scheme"]
    assert isinstance(scheme, dict)
    scheme["source_model"] = "product_default"
    for pair in _pairs(document):
        pair_input = pair["input_document"]
        assert isinstance(pair_input, dict)
        pair_input["source_model"] = {
            "kind": SourceModelKind.ANALYTIC_AXISYMMETRIC_TWO_PARAMETER_V1.value}
    SchemeResult.model_validate(document)
    first_input = _pairs(document)[0]["input_document"]
    assert isinstance(first_input, dict)
    first_input["source_model"] = {"kind": SourceModelKind.OMNIDIRECTIONAL.value}
    _reject(document, "source_model.kind")


def test_load_result_rechecks_report_input(result: SchemeResult, tmp_path: Path) -> None:
    document = _document(result)
    pair_input = _pairs(document)[0]["input_document"]
    assert isinstance(pair_input, dict)
    pair_input["source_model"] = {"kind": "omnidirectional", "parameters": {"bogus": 1}}
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError):
        load_result(path, capabilities=load_capabilities(config_path("capabilities.toml")),
                    directivity=DIRECTIVITY, quality_targets_path=control.TARGETS)


@pytest.mark.parametrize("change", ["swap_reports", "drop_category", "empty",
                                    "report_id", "engine_commit"])
def test_load_result_rejects_candidate_that_parts_cannot_reproduce(
    result: SchemeResult, tmp_path: Path, change: str,
) -> None:
    document = _document(result)
    pairs = _pairs(document)
    candidate = document["candidate"]
    assert isinstance(candidate, dict)
    evaluations = candidate["evaluations"]
    assert isinstance(evaluations, list)
    if change == "swap_reports":
        pairs[0]["report"], pairs[1]["report"] = pairs[1]["report"], pairs[0]["report"]
    elif change == "drop_category":
        candidate["evaluations"] = [item for item in evaluations
                                    if item["category"] != QualityCategory.REFLECTIONS_AND_ECHO.value]
    elif change == "empty":
        candidate["evaluations"] = []
    elif change == "report_id":
        pairs[0]["report_id"] = "report-forged"
    else:
        document["engine_commit"] = "forged"
    path = tmp_path / "forged.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="候選包跟存下來的零件對不上"):
        load_result(path, capabilities=load_capabilities(config_path("capabilities.toml")),
                    directivity=DIRECTIVITY, quality_targets_path=control.TARGETS)


def test_load_result_requires_quality_targets(result: SchemeResult, tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    save_result(result, path)
    loader: Callable[..., SchemeResult] = load_result
    with pytest.raises(TypeError, match="quality_targets_path"):
        loader(path, capabilities=load_capabilities(config_path("capabilities.toml")),
               directivity=DIRECTIVITY)


@pytest.mark.parametrize("change", ["source", "receiver", "report_id"])
def test_result_pair_report_identity_matches_input(
    result: SchemeResult, change: str,
) -> None:
    document = _document(result)
    pair = _pairs(document)[0]
    if change == "report_id":
        pair["report_id"] = "wrong"
        message = "report_id"
    else:
        report = pair["report"]
        assert isinstance(report, dict)
        scene = report["scene"]
        assert isinstance(scene, dict)
        point = scene[f"{change}_m"]
        assert isinstance(point, dict)
        point["x"] += 0.1
        message = f"report.scene.{change}_m"
    _reject(document, message)


@pytest.mark.parametrize("field", ["solve_s", "output_s", "evaluate_s", "total_s"])
def test_timings_reject_negative_seconds(field: str) -> None:
    values = {key: 0.0 for key in ("solve_s", "output_s", "evaluate_s", "total_s")}
    values[field] = -0.1
    with pytest.raises(ValueError, match=field):
        Timings.model_validate(values)


def test_result_schema_version_is_explicit(result: SchemeResult) -> None:
    assert result.schema_version == RESULT_SCHEMA_VERSION


def test_explicit_scene_options_reach_solver(monkeypatch: pytest.MonkeyPatch) -> None:
    class ReachedSolver(Exception):
        pass

    scheme = _scheme("wall-1")
    changed_scene = scheme.scene.model_copy(update={
        "reflection_order_k": 2, "low_frequency_axis": LowFrequencyAxis.VERIFICATION})
    scheme = scheme.model_copy(update={"scene": changed_scene})
    received: dict[str, object] = {}

    def stop_at_solver(**kwargs: object) -> None:
        received.update(kwargs)
        raise ReachedSolver

    monkeypatch.setattr(three_lane_report, "solve_three_lane_reports", stop_at_solver)
    monkeypatch.setattr(pipeline, "report_capability",
                        lambda table: three_lane_report._unchecked_capability())
    with pytest.raises(ReachedSolver):
        pipeline.run_scheme(scheme, capabilities=load_capabilities(config_path("capabilities.toml")),
                            directivity=DIRECTIVITY, quality_targets_path=control.TARGETS,
                            engine_commit="control", run_date=date(2026, 9, 27))
    assert received["reflection_order_k"] == 2
    assert received["low_frequency_axis"] is LowFrequencyAxis.VERIFICATION


def test_pipeline_timing_boundaries(result: SchemeResult, monkeypatch: pytest.MonkeyPatch) -> None:
    table = load_capabilities(config_path("capabilities.toml"))
    inputs = {(pair.speaker_id, pair.receiver_id): (
        pair.input_document, report_io.load_input_document(pair.input_document, table, DIRECTIVITY))
        for pair in result.pairs}
    pairs = {(pair.speaker_id, pair.receiver_id): pair for pair in result.pairs}
    monkeypatch.setattr(pipeline, "_inputs", lambda *args: inputs)
    monkeypatch.setattr(pipeline, "_pair", lambda scheme, key, *args: pairs[key])
    monkeypatch.setattr(pipeline, "evaluate_parts", lambda *args: result.candidate)
    monkeypatch.setattr(pipeline, "report_capability",
                        lambda table: three_lane_report._unchecked_capability())
    monkeypatch.setattr(three_lane_report, "solve_three_lane_reports",
                        lambda **kwargs: dict.fromkeys(inputs))
    instants = iter((10.0, 12.0, 17.0, 20.0, 29.0))
    monkeypatch.setattr("aosr.reporting.pipeline.time.perf_counter", lambda: next(instants))
    run_date = date(2026, 9, 26)
    produced = pipeline.run_scheme(result.scheme, capabilities=table, directivity=DIRECTIVITY,
        quality_targets_path=control.TARGETS, engine_commit="control",
        run_date=run_date)
    assert produced.run_date == run_date
    measured = produced.timings
    assert measured.solve_s == 17.0 - 12.0
    assert measured.output_s == 20.0 - 17.0
    assert measured.evaluate_s == 29.0 - 20.0
    assert measured.total_s == 29.0 - 10.0


def test_pipeline_rejects_divergent_scene_fingerprints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = iter(("a" * 64, "b" * 64, "c" * 64, "d" * 64))
    monkeypatch.setattr(report_io, "scene_fingerprint", lambda inputs: next(values))
    monkeypatch.setattr(three_lane_report, "solve_three_lane_reports",
                        lambda **kwargs: pytest.fail("指紋未核對就求解"))
    with pytest.raises(ValueError, match="場景指紋不同"):
        pipeline.run_scheme(_scheme("wall-1"),
            capabilities=load_capabilities(config_path("capabilities.toml")),
            directivity=DIRECTIVITY, quality_targets_path=control.TARGETS,
            engine_commit="control", run_date=date(2026, 9, 27))


@pytest.mark.parametrize("key,old,new,message", [
    ("reflections_and_echo.window_upper_ms", 'unit = "ms"', 'unit = "s"', "ms"),
    ("reflections_and_echo.window_upper_ms", "value = 15.0", "value = [15.0, 20.0]", "單值"),
    ("reverberation.adjacent_t20_logarithm_base", "value = 2.0", "value = [2.0, 3.0]", "單值"),
    ("channel_matching.broadband_range_hz", "value = [20.0, 8000.0]",
     "value = [20.0, 1000.0, 8000.0]", "兩端點"),
])
def test_registry_rejects_wrong_shape_or_unit(
    tmp_path: Path, key: str, old: str, new: str, message: str,
) -> None:
    original = control.TARGETS.read_text(encoding="utf-8")
    marker = f'key = "{key}"'
    head, body = original.split(marker, maxsplit=1)
    section, tail = body.split("[[purpose.", maxsplit=1)
    assert old in section
    changed = head + marker + section.replace(old, new, 1) + "[[purpose." + tail
    assert changed != original
    path = tmp_path / "quality_targets.toml"
    path.write_text(changed, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        read_registry_settings(path, control.PURPOSE)


@pytest.mark.parametrize("field,message", [
    ("purpose", "purpose"), ("channel_group", "聲道組指紋"),
    ("listening_area_channel_role", "聆聽區角色"),
    ("listening_area_speaker_id", "聆聽區角色"),
    ("engine_commit", "engine_commit"), ("duplicate", "候選代號重複"),
])
def test_compare_names_second_incompatible_result(
    result: SchemeResult, second_result: SchemeResult, field: str, message: str,
) -> None:
    variant = second_result
    if field == "purpose":
        variant = variant.model_copy(update={"scheme": variant.scheme.model_copy(
            update={"purpose": "other"})})
    elif field == "channel_group":
        group = variant.scheme.channel_group.model_copy(update={
            "feature_match_tolerance_hz": 9.0})
        variant = variant.model_copy(update={"scheme": variant.scheme.model_copy(
            update={"channel_group": group})})
    elif field == "listening_area_channel_role":
        variant = variant.model_copy(update={field: "right"})
    elif field == "listening_area_speaker_id":
        variant = variant.model_copy(update={field: "right"})
    elif field == "engine_commit":
        variant = variant.model_copy(update={field: "other"})
    else:
        variant = result
    with pytest.raises(ValueError, match=f"第 2 份 {variant.scheme.scheme_id}.*{message}"):
        compare_results([result, variant], quality_targets=load_quality_targets(control.TARGETS),
                        run_date=date(2026, 9, 27))


def test_compare_rejects_empty_results() -> None:
    with pytest.raises(ValueError, match="至少需要一份"):
        compare_results([], quality_targets=load_quality_targets(control.TARGETS),
                        run_date=date(2026, 9, 27))


@pytest.mark.parametrize("arguments", [[], ["one.json"]])
def test_compare_cli_rejects_less_than_two_files(arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        scheme_cli.main(["compare", *arguments,
                         "--capabilities", str(config_path("capabilities.toml"))])
    exit_code = exc.value.code
    assert exit_code == 2


def test_cli_run_writes_result_and_prints_ranked_costs(
    result: SchemeResult, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    scheme_path = tmp_path / "scheme.json"
    scheme_path.write_text(result.scheme.model_dump_json(), encoding="utf-8")
    out = tmp_path / "result.json"
    dates: list[date] = []
    run_dates: list[date] = []
    original_compare = compare_results

    def record_compare(results: Sequence[SchemeResult], *, quality_targets: QualityTargets,
                       run_date: date) -> RankingResult:
        dates.append(run_date)
        return original_compare(results, quality_targets=quality_targets, run_date=run_date)

    def record_run(*args: object, **kwargs: object) -> SchemeResult:
        received = kwargs["run_date"]
        assert isinstance(received, date)
        run_dates.append(received)
        return result

    monkeypatch.setattr(scheme_cli, "run_scheme", record_run)
    monkeypatch.setattr(scheme_cli, "compare_results", record_compare)
    exit_code = scheme_cli.main(["run", str(scheme_path), "--out", str(out),
        "--capabilities", str(config_path("capabilities.toml")),
        "--engine-commit", "control", "--run-date", "2026-09-26"])
    assert exit_code == 0
    assert load_result(out, capabilities=load_capabilities(config_path("capabilities.toml")),
                       directivity=DIRECTIVITY, quality_targets_path=control.TARGETS) == result
    assert dates == [date(2026, 9, 26)]
    assert run_dates == [date(2026, 9, 26)]
    printed = capsys.readouterr().out
    ranking = compare_results([result], quality_targets=load_quality_targets(control.TARGETS),
                              run_date=date(2026, 9, 26))
    for line in ranking.rankable[0].categories:
        assert f"{line.identity.category.value} | measured | 代價 {line.category_cost}" in printed
    for label, value in (("求解", result.timings.solve_s),
                         ("輸出與反射", result.timings.output_s),
                         ("評估", result.timings.evaluate_s),
                         ("全程", result.timings.total_s)):
        assert f"{label} {value:.3f}" in printed
    assert "低頻拖尾：尚未評估" in printed


def test_cli_compare_prints_costs_unassessed_and_each_file(
    result: SchemeResult, wall_2: SchemeResult, second_result: SchemeResult, tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    items = (result, wall_2, second_result)
    paths = (tmp_path / "wall-1.json", tmp_path / "wall-2.json", tmp_path / "copy-2.json")
    for item, path in zip(items, paths, strict=True):
        save_result(item, path)
    exit_code = scheme_cli.main(["compare", *(str(path) for path in paths),
        "--capabilities", str(config_path("capabilities.toml")),
        "--run-date", "2026-09-27"])
    assert exit_code == 0
    printed = capsys.readouterr().out
    ranking = compare_results(items,
                              quality_targets=load_quality_targets(control.TARGETS),
                              run_date=date(2026, 9, 27))
    for row in ranking.rankable:
        assert row.candidate_id in printed
    costs = {row.candidate_id: {line.identity.category: line.category_cost
                                for line in row.categories} for row in ranking.rankable}
    assert any(costs[result.scheme.scheme_id][category] != costs[wall_2.scheme.scheme_id][category]
               for category in costs[result.scheme.scheme_id])
    for category in costs[result.scheme.scheme_id]:
        expected = " | ".join(str(costs[item.scheme.scheme_id][category]) for item in items)
        assert f"{category.value} | {expected}" in printed
    assert "spatial_impression | 未評估 | 未評估 | 未評估" in printed
    assert "表頭座位組指紋" not in printed
    for item, path in zip(items, paths, strict=True):
        assert (f"{path.name} | 座位組指紋 {item.scheme.receiver_set.fingerprint} | "
                in printed)
        assert f"{item.scheme.scheme_id}：rankable" in printed
    assert f"{paths[1].name} | 座位組指紋 {wall_2.scheme.receiver_set.fingerprint} | 與第一份不同" in printed
    assert f"{paths[2].name} | 座位組指紋 {second_result.scheme.receiver_set.fingerprint} | 與第一份相同" in printed


def test_comparison_table_names_unavailable_category_reason(
    result: SchemeResult, capsys: pytest.CaptureFixture[str],
) -> None:
    document = result.candidate.model_dump(mode="json")
    evaluations = document["evaluations"]
    reflection = next(item for item in evaluations
                      if item["category"] == QualityCategory.REFLECTIONS_AND_ECHO.value)
    reflection.update(state=EvaluationState.UNAVAILABLE.value, payload=None,
                      category_cost=None, raw_quantities=[],
                      reason_codes=[ReasonCode.LISTENING_AXIS_UNDEFINED.value])
    candidate = CandidateEvaluation.model_validate(document)
    changed = SchemeResult.model_validate(result.model_copy(update={"candidate": candidate}))
    ranking = compare_results([changed], quality_targets=load_quality_targets(control.TARGETS),
                              run_date=date(2026, 9, 27))
    assert ranking.rankable[0].candidate_id == changed.scheme.scheme_id
    scheme_cli._comparison_table([changed], ranking)
    printed = capsys.readouterr().out
    assert "reflections_and_echo | unavailable：listening_axis_undefined" in printed
    assert "reflections_and_echo | rankable" not in printed


def test_compare_prints_missing_category_reason(
    result: SchemeResult, second_result: SchemeResult, tmp_path: Path,
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = second_result.candidate.model_copy(update={"evaluations": tuple(
        item for item in second_result.candidate.evaluations
        if item.category is not QualityCategory.TIMBRE_BALANCE)})
    changed = second_result.model_copy(update={"candidate": candidate})
    ranking = compare_results([result, changed],
                              quality_targets=load_quality_targets(control.TARGETS),
                              run_date=date(2026, 9, 27))
    reason = ranking.not_evaluated[0].missing[0].reason.value
    paths = (tmp_path / "first.json", tmp_path / "second.json")
    for item, path in zip((result, changed), paths, strict=True):
        save_result(item, path)
    monkeypatch.setattr(scheme_cli, "load_result", lambda path, **kwargs: (
        result if path == paths[0] else changed))
    exit_code = scheme_cli.main(["compare", *(str(path) for path in paths),
        "--capabilities", str(config_path("capabilities.toml")),
        "--run-date", "2026-09-27"])
    assert exit_code == 0
    printed = capsys.readouterr().out
    assert f"not_evaluated | 原因 timbre_balance:{reason}" in printed
    timbre_cost = next(line.category_cost for line in ranking.rankable[0].categories
                       if line.identity.category is QualityCategory.TIMBRE_BALANCE)
    assert f"timbre_balance | {timbre_cost} | not_evaluated" in printed


def test_compare_prints_elimination_reason_from_row(
    result: SchemeResult, second_result: SchemeResult, tmp_path: Path,
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = compare_results([result, second_result],
        quality_targets=load_quality_targets(control.TARGETS), run_date=date(2026, 9, 27))
    row = next(item for item in original.rankable
               if item.candidate_id == second_result.scheme.scheme_id)
    eliminated = EliminatedRow.model_validate({
        "status": "eliminated", "candidate_id": row.candidate_id,
        "scene_fingerprint": row.scene_fingerprint,
        "reasons": [EliminationReason.EXTERNAL_FLOOR_FAILED.value], "missing": [],
        "evaluations": second_result.candidate.evaluations,
        "external_acceptance": row.external_acceptance,
        "review_alerts": row.review_alerts,
    })
    ranking = original.model_copy(update={
        "rankable": tuple(item for item in original.rankable if item.candidate_id != row.candidate_id),
        "eliminated": (eliminated,)})
    paths = (tmp_path / "first.json", tmp_path / "second.json")
    monkeypatch.setattr(scheme_cli, "load_result", lambda path, **kwargs: (
        result if path == paths[0] else second_result))
    monkeypatch.setattr(scheme_cli, "compare_results", lambda *args, **kwargs: ranking)
    exit_code = scheme_cli.main(["compare", *(str(path) for path in paths),
        "--capabilities", str(config_path("capabilities.toml")),
        "--run-date", "2026-09-27"])
    assert exit_code == 0
    assert "eliminated | 原因 external_floor_failed" in capsys.readouterr().out


def test_cli_compare_passes_run_date_to_ranking(
    result: SchemeResult, second_result: SchemeResult, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """compare 的 --run-date 要原樣進排名，不准被今天蓋掉。"""
    paths = (tmp_path / "wall-1.json", tmp_path / "copy-2.json")
    for item, path in zip((result, second_result), paths, strict=True):
        save_result(item, path)
    seen: list[date] = []
    original = compare_results

    def spy(results: list[SchemeResult], *, quality_targets: QualityTargets,
            run_date: date) -> RankingResult:
        seen.append(run_date)
        return original(results, quality_targets=quality_targets, run_date=run_date)

    monkeypatch.setattr(scheme_cli, "compare_results", spy)
    exit_code = scheme_cli.main(["compare", *(str(path) for path in paths),
        "--capabilities", str(config_path("capabilities.toml")),
        "--run-date", "2031-02-03"])
    assert exit_code == 0
    assert seen == [date(2031, 2, 3)]


def test_compare_real_relative_layout_selects_actual_main_table(
    result: SchemeResult, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    scheme = _scheme("wall-2")
    points = tuple(point.model_copy(update={"position_m": (
        point.position_m[0] + 0.1 if point.role.value == "surrounding"
        else point.position_m[0], *point.position_m[1:])})
        for point in scheme.receiver_set.points)
    changed = scheme.receiver_set.model_copy(update={"points": points})
    scheme = Scheme.model_validate(scheme.model_copy(update={"receiver_set": changed}))
    with pytest.MonkeyPatch.context() as patch:
        for module, name, fake in control.STAND_INS:
            patch.setattr(module, name, fake)
        patch.setattr(three_lane_report, "_solve_fem_energies", _many_fem)
        patch.setattr(pipeline, "report_capability",
                      lambda table: three_lane_report._unchecked_capability())
        relative = pipeline.run_scheme(scheme,
            capabilities=load_capabilities(config_path("capabilities.toml")),
            directivity=DIRECTIVITY, quality_targets_path=control.TARGETS,
            engine_commit="control", run_date=date(2026, 9, 27))
    assert relative.scheme.receiver_set.layout_fingerprint != result.scheme.receiver_set.layout_fingerprint
    paths = {"wall-1": tmp_path / "wall-1.json",
             "wall-2": tmp_path / "wall-2-relative.json"}
    for item in (result, relative):
        save_result(item, paths[item.scheme.scheme_id])
    first_outside = False
    for ordered in ((result, relative), (relative, result)):
        ranking = compare_results(ordered,
            quality_targets=load_quality_targets(control.TARGETS),
            run_date=date(2026, 9, 27))
        main_id = ranking.rankable[0].candidate_id
        outside_id = ranking.not_comparable.rows[0].candidate_id
        assert {main_id, outside_id} == {item.scheme.scheme_id for item in ordered}
        exit_code = scheme_cli.main(["compare", *(str(paths[item.scheme.scheme_id])
            for item in ordered), "--capabilities", str(config_path("capabilities.toml")),
            "--run-date", "2026-09-27"])
        assert exit_code == 0
        printed = capsys.readouterr().out
        for index, item in enumerate(ordered):
            status = "rankable" if item.scheme.scheme_id == main_id else "not_comparable"
            relation = "相同" if index == 0 else "不同"
            assert (f"{paths[item.scheme.scheme_id].name} | 座位組指紋 "
                    f"{item.scheme.receiver_set.fingerprint} | 與第一份{relation} | {status}") in printed
            assert f"方案 {item.scheme.scheme_id}：{status}" in printed
        if ordered[0].scheme.scheme_id == outside_id:
            first_outside = True
            assert f"{paths[outside_id].name} | 座位組指紋 " in printed
            assert "與第一份相同 | not_comparable | 原因 與主表的比較身分不同" in printed
            different = [part.category.value for part in ranking.not_comparable.rows[0].identity
                         if part not in ranking.header.main_table_identity]
            assert "與主表的比較身分不同：" + ",".join(different) in printed
            assert f"方案 {main_id}：rankable" in printed
    assert first_outside
