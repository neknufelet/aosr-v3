"""#577 保存物理、重評候選與讀回四級的考卷。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from aosr.config.capabilities import load_capabilities
from aosr.config.paths import config_path
from aosr.config.quality_targets import load_quality_targets
from aosr.reporting.evaluation import (
    LoadedResult, ResultStanding, _same_measurement, load_result, purpose_settings,
)
from aosr.reporting.result import PurposeSettings, ResultOrigin, SchemeResult, save_result
from tests.engine import _scoring_source_model_control as control
from tests.engine._directivity import DIRECTIVITY
from tests.engine.test_scheme_repair import result


def load_saved(tmp_path: Path, saved: SchemeResult, *, registry: Path = control.TARGETS,
               physics: str | None = None) -> LoadedResult:
    path = tmp_path / f"saved{'.json'}"
    save_result(saved, path)
    return load_result(path, capabilities=load_capabilities(config_path("capabilities.toml")),
                       directivity=DIRECTIVITY, quality_targets_path=registry,
                       physics_identity=physics or saved.physics_identity)


def registry_change(tmp_path: Path, old: str, new: str) -> Path:
    text = control.TARGETS.read_text(encoding="utf-8")
    assert old in text
    path = tmp_path / f"targets{'.toml'}"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return path


def test_current_preserves_candidate(tmp_path: Path, result: SchemeResult) -> None:
    loaded = load_saved(tmp_path, result)
    assert loaded.standing is ResultStanding.CURRENT
    assert loaded.result.candidate == loaded.stored_candidate == result.candidate


def test_program_change_only_stays_current(tmp_path: Path, result: SchemeResult) -> None:
    changed = result.model_copy(update={"program_fingerprint": "calc-v1:" + "f" * 64})
    loaded = load_saved(tmp_path, changed)
    assert loaded.standing is ResultStanding.CURRENT
    assert loaded.result.program_fingerprint == changed.program_fingerprint
    assert loaded.stored_candidate is not None
    assert _same_measurement(loaded.result.candidate, loaded.stored_candidate)


def test_physics_change_still_remeasures(tmp_path: Path, result: SchemeResult) -> None:
    loaded = load_saved(tmp_path, result, physics="phys-v1:" + "f" * 64)
    assert loaded.standing is ResultStanding.NEEDS_PHYSICS
    assert loaded.result.candidate == result.candidate


def test_snapshot_digest_and_purpose_rejected(tmp_path: Path, result: SchemeResult) -> None:
    content = result.purpose_settings.model_dump(mode="json")
    content["fingerprint"] = "f" * 64
    with pytest.raises(ValueError, match="fingerprint"):
        PurposeSettings.model_validate(content)
    document = result.model_dump(mode="json")
    document["purpose_settings"]["purpose"] = "other_purpose"
    with pytest.raises(ValueError, match="purpose"):
        SchemeResult.model_validate(document)
    assert purpose_settings(control.TARGETS, result.scheme.purpose) == result.purpose_settings
    purpose = load_quality_targets(control.TARGETS).purpose(result.scheme.purpose)
    reordered = purpose.model_copy(update={"target": tuple(reversed(purpose.target))})
    assert reordered.canonical() == purpose.canonical()
    assert reordered.fingerprint == purpose.fingerprint


@pytest.mark.parametrize(("kind", "fields", "valid"), [
    ("run", {}, True), ("run", {"search_id": "search", "trial_number": 0}, False),
    ("search_candidate", {"search_id": "search", "trial_number": 0}, True),
    ("search_candidate", {}, False),
    ("search_candidate", {"search_id": " ", "trial_number": 0}, False),
    ("search_candidate", {"search_id": "search", "trial_number": -1}, False),
    ("run", {"search_id": "search"}, False),
    ("run", {"trial_number": 0}, False),
    ("search_candidate", {"search_id": "search"}, False),
    ("search_candidate", {"trial_number": 0}, False),
    ("search_baseline", {"search_id": "search"}, True),
    ("search_baseline", {}, False),
    ("search_baseline", {"search_id": " "}, False),
    ("search_baseline", {"search_id": "search", "trial_number": 0}, False),
])
def test_origin_rules(tmp_path: Path, kind: str, fields: dict[str, object], valid: bool) -> None:
    document = {"kind": kind, **fields}
    if valid:
        assert ResultOrigin.model_validate(document).kind == kind
    else:
        with pytest.raises(ValueError):
            ResultOrigin.model_validate(document)


@pytest.mark.parametrize(("version", "message"), [(3, "舊版"), (99, "認不得")])
def test_format_version_messages(tmp_path: Path, result: SchemeResult,
                                 version: int, message: str) -> None:
    path = tmp_path / f"old{'.json'}"
    path.write_text(json.dumps({"schema_version": f"aosr.scheme_result.v{version}"}))
    with pytest.raises(ValueError, match=message):
        load_result(path, capabilities=load_capabilities(config_path("capabilities.toml")),
                    directivity=DIRECTIVITY, quality_targets_path=control.TARGETS,
                    physics_identity=result.physics_identity)


def test_weight_change_reranks(tmp_path: Path, result: SchemeResult) -> None:
    registry = registry_change(tmp_path, 'name = "timbre_balance"\nvalue = 1.0',
                               'name = "timbre_balance"\nvalue = 2.0')
    loaded = load_saved(tmp_path, result, registry=registry)
    assert loaded.standing is ResultStanding.RERANKED
    assert loaded.stored_candidate is not None
    assert _same_measurement(loaded.stored_candidate, loaded.result.candidate)


def test_target_change_remeasures(tmp_path: Path, result: SchemeResult) -> None:
    registry = registry_change(tmp_path, 'key = "timbre_balance.target_tilt_db_per_octave"\nvalue = 0.0',
                               'key = "timbre_balance.target_tilt_db_per_octave"\nvalue = -1.0')
    loaded = load_saved(tmp_path, result, registry=registry)
    assert loaded.standing is ResultStanding.REMEASURED
    assert loaded.result.candidate != loaded.stored_candidate
    assert loaded.stored_candidate is not None
    assert not _same_measurement(loaded.result.candidate, loaded.stored_candidate)
    # 物理已改時仍先重新量，且物理等級優先。
    physical = load_saved(tmp_path, result, registry=registry, physics="phys-v1:" + "f" * 64)
    assert physical.standing is ResultStanding.NEEDS_PHYSICS
    assert physical.result.candidate == loaded.result.candidate


def test_unrelated_purpose_stays_current(tmp_path: Path, result: SchemeResult) -> None:
    original = control.TARGETS.read_text(encoding="utf-8")
    purpose = original[original.index("[[purpose]]"):]
    purpose = purpose.replace(f'name = "{result.scheme.purpose}"', 'name = "new_purpose"', 1)
    registry = tmp_path / f"expanded{'.toml'}"
    registry.write_text(original + "\n" + purpose, encoding="utf-8")
    expanded = load_quality_targets(registry)
    assert expanded.fingerprint != result.quality_targets_fingerprint
    assert expanded.purpose(result.scheme.purpose).fingerprint == result.purpose_settings.fingerprint
    loaded = load_saved(tmp_path, result, registry=registry)
    assert loaded.standing is ResultStanding.CURRENT
    assert loaded.stored_candidate is not None
    assert _same_measurement(loaded.result.candidate, loaded.stored_candidate)


def test_source_text_does_not_remeasure(tmp_path: Path, result: SchemeResult) -> None:
    registry = registry_change(tmp_path, 'source = "測試基線，未查證；正式值等 #358"',
                               'source = "出處文字補充，未查證；正式值等 #358"')
    loaded = load_saved(tmp_path, result, registry=registry)
    assert loaded.standing is not ResultStanding.REMEASURED
    assert loaded.stored_candidate is not None
    assert _same_measurement(loaded.result.candidate, loaded.stored_candidate)


def test_tampered_score_is_repaired(tmp_path: Path, result: SchemeResult) -> None:
    evaluation = result.candidate.evaluations[0]
    raw = evaluation.raw_quantities[0]
    changed = evaluation.model_copy(update={"raw_quantities": (
        raw.model_copy(update={"value": raw.value + 1.0}), *evaluation.raw_quantities[1:])})
    stored = result.model_copy(update={"candidate": result.candidate.model_copy(update={
        "evaluations": (changed, *result.candidate.evaluations[1:])})})
    loaded = load_saved(tmp_path, stored)
    assert loaded.standing is ResultStanding.REMEASURED
    assert loaded.result.candidate == result.candidate
    assert loaded.stored_candidate == stored.candidate


def test_tampered_physics_changes_measurement(tmp_path: Path, result: SchemeResult) -> None:
    pair = result.pairs[0]
    points = pair.report.points
    assert points is not None
    point = points[0].model_copy(update={"total_energy": points[0].total_energy * 2.0})
    report = pair.report.model_copy(update={"points": (point, *points[1:])})
    stored = result.model_copy(update={"pairs": (pair.model_copy(update={"report": report}),
                                                *result.pairs[1:])})
    loaded = load_saved(tmp_path, stored)
    assert loaded.result.candidate != result.candidate
    assert not _same_measurement(loaded.result.candidate, result.candidate)


def test_measurement_folds_versions_and_settings(tmp_path: Path, result: SchemeResult) -> None:
    from aosr.scoring.contract import CandidateEvaluation
    document = result.candidate.model_dump_json()
    first = result.candidate.evaluations[0]
    version = first.evaluator_version
    changed = CandidateEvaluation.model_validate_json(document.replace(version, version.rsplit(".v", 1)[0] + ".v999"))
    assert _same_measurement(result.candidate, changed)
    fingerprint = first.settings_fingerprint
    changed = CandidateEvaluation.model_validate_json(document.replace(fingerprint, "altered-settings"))
    assert _same_measurement(result.candidate, changed)
    assert not _same_measurement(result.candidate, result.candidate.model_copy(update={
        "evaluations": (first.model_copy(update={"raw_quantities": (
            first.raw_quantities[0].model_copy(update={"value": first.raw_quantities[0].value + 1.0}),
            *first.raw_quantities[1:])}), *result.candidate.evaluations[1:])}))


@pytest.mark.parametrize("field", ["state", "reason_codes", "scene_fingerprint"])
def test_measurement_preserves_status_reason_and_other_identity(
    tmp_path: Path, result: SchemeResult, field: str,
) -> None:
    from aosr.scoring.contract import EvaluationState, ReasonCode
    first = result.candidate.evaluations[0]
    changes: dict[str, object] = {"state": EvaluationState.UNAVAILABLE,
                                 "reason_codes": (ReasonCode.ZERO_TOTAL_IMPORTANCE,),
                                 "scene_fingerprint": "another-scene"}
    changed = result.candidate.model_copy(update={"evaluations": (
        first.model_copy(update={field: changes[field]}), *result.candidate.evaluations[1:])})
    assert not _same_measurement(result.candidate, changed)


def test_cli_refuses_physics_change_midrun(tmp_path: Path, result: SchemeResult,
                                          monkeypatch: pytest.MonkeyPatch,
                                          capsys: pytest.CaptureFixture[str]) -> None:
    from aosr.reporting import scheme_cli
    scheme = tmp_path / f"scheme{'.json'}"
    scheme.write_text(result.scheme.model_dump_json())
    out = tmp_path / f"result{'.json'}"
    values = iter((result.physics_identity, "phys-v1:" + "f" * 64))
    monkeypatch.setattr("aosr.reporting.physics_identity.physics_identity", lambda **kwargs: next(values))
    monkeypatch.setattr("aosr.reporting.pipeline.run_scheme", lambda *args, **kwargs: result)
    code = scheme_cli.main(["run", str(scheme), "--out", str(out), "--capabilities",
                            str(config_path("capabilities.toml")), "--engine-commit", "test"])
    assert code == scheme_cli.PROGRAM_CHANGED_EXIT
    assert not out.exists()
    assert "物理" in capsys.readouterr().err
