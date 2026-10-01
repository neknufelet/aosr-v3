"""存檔候選包改版不阻擋物理零件讀回；非候選檢查仍完整。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from aosr.config.capabilities import load_capabilities
from aosr.config.paths import config_path
from aosr.reporting import evaluation
from aosr.reporting.evaluation import LoadedResult, ResultStanding, load_result
from aosr.reporting.result import SchemeResult
from tests.engine import _scoring_source_model_control as control
from tests.engine._directivity import DIRECTIVITY
from tests.engine.test_scheme_repair import result


def load_document(tmp_path: Path, document: dict[str, object],
                  physics: str) -> LoadedResult:
    path = tmp_path / f"saved{'.json'}"
    path.write_text(json.dumps(document), encoding="utf-8")
    return load_result(path, capabilities=load_capabilities(config_path("capabilities.toml")),
                       directivity=DIRECTIVITY, quality_targets_path=control.TARGETS,
                       physics_identity=physics)


@pytest.mark.parametrize("change", ["extra", "missing", "version", "category"])
@pytest.mark.parametrize("old_physics", [False, True])
def test_obsolete_candidate_remeasures_without_rejecting_parts(
    tmp_path: Path, result: SchemeResult, change: str, old_physics: bool,
) -> None:
    document = result.model_dump(mode="json")
    candidate = document["candidate"]
    if change == "extra":
        candidate["retired_field"] = "old contract"
    elif change == "missing":
        candidate.pop("evaluations")
    elif change == "version":
        candidate["schema_version"] = "aosr.candidate_evaluation.v999"
    else:
        candidate["evaluations"][0]["category"] = "renamed_category"
    physics = "phys-v1:" + "f" * 64 if old_physics else result.physics_identity
    loaded = load_document(tmp_path, document, physics)
    assert loaded.stored_candidate is None
    assert loaded.standing is (ResultStanding.NEEDS_PHYSICS if old_physics
                               else ResultStanding.REMEASURED)
    assert loaded.result.candidate == result.candidate
    assert SchemeResult.model_validate_json(loaded.result.model_dump_json()) == loaded.result


@pytest.mark.parametrize("candidate", [None, [], "broken", {"candidate_id": "wrong"}, {}])
def test_corrupt_candidate_envelope_is_rejected(
    tmp_path: Path, result: SchemeResult, candidate: object,
) -> None:
    document = result.model_dump(mode="json")
    document["candidate"] = candidate
    with pytest.raises(ValueError):
        load_document(tmp_path, document, result.physics_identity)


@pytest.mark.parametrize(("change", "message"), [
    ("pairs", "pairs 必須"), ("role", "pairs 必須"), ("report_id", "report_id"),
    ("input", "report.scene.receiver_m"), ("digest", "purpose_settings.fingerprint"),
    ("purpose", "purpose_settings.purpose"), ("origin", "run 不可"),
    ("scope", "scope"), ("extra", "retired_field"),
])
def test_obsolete_candidate_does_not_bypass_other_checks(
    tmp_path: Path, result: SchemeResult, change: str, message: str,
) -> None:
    document = result.model_dump(mode="json")
    document["candidate"]["retired_field"] = "old contract"
    if change == "pairs":
        document["pairs"].pop()
    elif change == "role":
        document["pairs"][0]["role"] = "wrong-role"
    elif change == "report_id":
        document["pairs"][0]["report_id"] = "wrong-report"
    elif change == "input":
        document["pairs"][0]["input_document"]["receiver_m"]["x"] += 0.1
    elif change == "digest":
        document["purpose_settings"]["fingerprint"] = "f" * 64
    elif change == "purpose":
        document["purpose_settings"]["purpose"] = "wrong-purpose"
    elif change == "origin":
        document["origin"]["search_id"] = "not-a-run"
    elif change == "scope":
        document["scope"] = "wrong-scope"
    else:
        document["retired_field"] = "not a candidate field"
    with pytest.raises(ValueError, match=message):
        load_document(tmp_path, document, result.physics_identity)


def test_missing_registry_purpose_has_actionable_value_error(
    tmp_path: Path, result: SchemeResult,
) -> None:
    document = result.model_dump(mode="json")
    document["scheme"]["purpose"] = "removed_purpose"
    document["purpose_settings"]["purpose"] = "removed_purpose"
    with pytest.raises(ValueError, match="品質登記簿沒有這個用途（removed_purpose），要先把用途加回登記簿"):
        load_document(tmp_path, document, result.physics_identity)


def test_equal_candidates_do_not_fold_measurements(
    tmp_path: Path, result: SchemeResult, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_fold(*args: object) -> bool:
        raise AssertionError("全等的候選包不需要摺疊")

    monkeypatch.setattr(evaluation, "_same_measurement", unexpected_fold)
    loaded = load_document(tmp_path, result.model_dump(mode="json"), result.physics_identity)
    assert loaded.standing is ResultStanding.CURRENT
    assert loaded.result.candidate == loaded.stored_candidate


def test_listening_area_channels_version_change_does_not_remeasure(
    tmp_path: Path, result: SchemeResult,
) -> None:
    original = result.model_dump_json()
    changed = original.replace("aosr.scoring.listening_area_channels.v1",
                               "aosr.scoring.listening_area_channels.v999")
    assert changed != original
    loaded = load_document(tmp_path, json.loads(changed), result.physics_identity)
    assert loaded.stored_candidate is not None
    assert loaded.result.candidate != loaded.stored_candidate
    assert loaded.standing is ResultStanding.CURRENT


def test_only_remeasured_listening_payload_is_checked(
    tmp_path: Path, result: SchemeResult,
) -> None:
    document = result.model_dump(mode="json")
    listening = next(item for item in document["candidate"]["evaluations"]
                     if item["category"] == "listening_area_stability")
    listening["payload"]["channel_group_fingerprint"] = "f" * 64
    loaded = load_document(tmp_path, document, result.physics_identity)
    assert loaded.result.candidate == result.candidate
    assert loaded.standing is ResultStanding.REMEASURED


def test_remeasured_listening_payload_gets_full_result_validation(
    tmp_path: Path, result: SchemeResult, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aosr.scoring.contract import QualityCategory

    listening = next(item for item in result.candidate.evaluations
                     if item.category is QualityCategory.LISTENING_AREA_STABILITY)
    bad = listening.model_copy(update={"provenance": listening.provenance.model_copy(
        update={"receiver_id": "wrong-seat"})})
    candidate = result.candidate.model_copy(update={"evaluations": tuple(
        bad if item is listening else item for item in result.candidate.evaluations)})
    monkeypatch.setattr(evaluation, "reevaluate", lambda *args, **kwargs: candidate)
    with pytest.raises(ValueError, match="聆聽區彙總 provenance"):
        load_document(tmp_path, result.model_dump(mode="json"), result.physics_identity)
