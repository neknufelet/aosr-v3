"""#577 網頁四級白話、物理比較與用途設定清單。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from tests.engine._gui_cache import gui_startup_identity_memo
from aosr.gui.app import STATIC, GuiSettings, create_app
from aosr.gui.result_list import summarize_result
from aosr.gui.result_view import build_result_view
from tests.engine import _scoring_source_model_control as control
from aosr.reporting.result import PurposeSettings, SchemeResult, save_result
from tests.engine.test_scheme_repair import result, second_result


TEXT = {
    "current": "物理與評分設定都跟現在相同",
    "reranked": "評分的權重或設定改過：已用存下的指標重新排名，數字跟存檔時相同，不用重算",
    "remeasured": "評分的量法或設定改過：已用存下的物理結果重新量過再排名，不用重算物理",
    "needs_physics": "物理計算的程式或設定改過：畫面上是舊的物理結果配現在的評分，要重算物理（約 6 分鐘）才能跟現在算的結果比較",
}


def variant_result(result: SchemeResult, standing: str) -> SchemeResult:
    if standing == "needs_physics":
        return result.model_copy(update={"physics_identity": "phys-v1:" + "f" * 64})
    if standing == "reranked":
        return result.model_copy(update={"purpose_settings": changed_snapshot(result)})
    if standing == "remeasured":
        first = result.candidate.evaluations[0]
        raw = first.raw_quantities[0]
        changed = first.model_copy(update={"raw_quantities": (
            raw.model_copy(update={"value": raw.value + 1.0}), *first.raw_quantities[1:])})
        return result.model_copy(update={"candidate": result.candidate.model_copy(update={
            "evaluations": (changed, *result.candidate.evaluations[1:])})})
    return result


def changed_snapshot(result: SchemeResult) -> PurposeSettings:
    from hashlib import sha256
    content = json.loads(json.dumps(result.purpose_settings.content))
    content["weight"][0]["item"][0]["value"] += 1.0
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return PurposeSettings(purpose=result.scheme.purpose, content=content,
                           fingerprint=sha256(canonical.encode()).hexdigest())


def client_for(tmp_path: Path, result: SchemeResult) -> TestClient:
    return TestClient(create_app(GuiSettings(engine_commit="a" * 40, data_dir=tmp_path,
                                            startup_fingerprint=result.program_fingerprint,
                                            startup_physics_identity=result.physics_identity)),
                      base_url="http://localhost")


@pytest.mark.parametrize("standing", tuple(TEXT))
def test_result_standing_text_and_rerun(tmp_path: Path, result: SchemeResult,
                                       standing: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("aosr.gui.app.calculation_fingerprint", lambda **kwargs: result.program_fingerprint)
    with client_for(tmp_path, result) as client:
        run_id = "b" * 32
        save_result(variant_result(result, standing), tmp_path / "results" / f"{run_id}{'.json'}")
        response = client.get(f"/api/results/{run_id}")
        assert response.status_code == 200
        body = response.json()
        assert body["standing"] == standing
        assert body["standing_text"] == TEXT[standing]
        assert ("rerun_url" in body) == (standing == "needs_physics")
        assert "fingerprint_relation" not in body and "fingerprint_notice" not in body
        if standing == "remeasured":
            assert body["categories"] == build_result_view(
                result, quality_targets_path=control.TARGETS).model_dump(mode="json")["categories"]


def test_comparison_reranks_settings_and_marks_only_changed_physics(
    tmp_path: Path, result: SchemeResult, second_result: SchemeResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("aosr.gui.app.calculation_fingerprint", lambda **kwargs: result.program_fingerprint)
    a_id, b_id = "b" * 32, "c" * 32
    with client_for(tmp_path, result) as client:
        save_result(result, tmp_path / "results" / f"{a_id}{'.json'}")
        other = second_result.model_copy(update={"purpose_settings": changed_snapshot(second_result)})
        path = tmp_path / "results" / f"{b_id}{'.json'}"
        save_result(other, path)
        response = client.get(f"/api/compare/{a_id}/{b_id}")
        assert response.status_code == 200
        assert not response.json()["outdated_schemes"]
        assert "rerun_urls" not in response.json()
        save_result(variant_result(other, "needs_physics"), path)
        response = client.get(f"/api/compare/{a_id}/{b_id}")
        assert response.status_code == 409
        assert response.json()["outdated_sides"] == ["b"]
        assert set(response.json()["rerun_urls"]) == {"b"}


@pytest.mark.parametrize(("physical", "settings"), [(True, True), (True, False),
                                                     (False, True), (False, False)])
def test_summary_physics_and_purpose_settings(tmp_path: Path, result: SchemeResult,
                                              physical: bool, settings: bool) -> None:
    saved = result if physical else variant_result(result, "needs_physics")
    if not settings:
        saved = saved.model_copy(update={"purpose_settings": changed_snapshot(result)})
    path = tmp_path / f"summary{'.json'}"
    save_result(saved, path)
    row = summarize_result(path, result.physics_identity,
                           {result.scheme.purpose: result.purpose_settings.fingerprint})
    assert row.calculation_text == ("跟現在相同" if physical else "物理改過，要重算")
    assert row.registry_text == ("跟現在相同" if settings else
                                  "改過：打開時自動重新排名或重量，不用重算")
    assert saved.physics_identity.split(":")[1][:12] in row.calculation_detail
    assert saved.engine_commit[:7] in row.calculation_detail


def test_summary_v3_is_old_format(tmp_path: Path, result: SchemeResult) -> None:
    path = tmp_path / f"old{'.json'}"
    document = result.model_dump(mode="json")
    document["schema_version"] = "aosr.scheme_result.v3"
    path.write_text(json.dumps(document))
    row = summarize_result(path, result.physics_identity,
                           {result.scheme.purpose: result.purpose_settings.fingerprint})
    assert row.calculation_text == "舊格式（程式更新前算的），要重算"


def tampered_timbre(result: SchemeResult) -> SchemeResult:
    """竄改代價評估會讀的傾斜，結果頁與比較頁都顯示音色代價。"""
    document = result.model_dump(mode="json")
    timbre = next(item for item in document["candidate"]["evaluations"]
                  if item["category"] == "timbre_balance")
    timbre["payload"]["channels"][0]["payload"]["tilt_db_per_octave"] += 50.0
    return SchemeResult.model_validate(document)


def timbre_cost(result: SchemeResult) -> float:
    view = build_result_view(result, quality_targets_path=control.TARGETS)
    cost = next(item.cost for item in view.categories if item.category == "timbre_balance")
    assert cost is not None
    return cost


def test_result_page_uses_remeasured_visible_timbre_cost(
    tmp_path: Path, result: SchemeResult, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("aosr.gui.app.calculation_fingerprint", lambda **kwargs: result.program_fingerprint)
    tampered = tampered_timbre(result)
    correct, wrong = timbre_cost(result), timbre_cost(tampered)
    assert correct != wrong
    with client_for(tmp_path, result) as client:
        run_id = "b" * 32
        save_result(tampered, tmp_path / "results" / f"{run_id}{'.json'}")
        response = client.get(f"/api/results/{run_id}")
    assert response.status_code == 200
    category = next(item for item in response.json()["categories"]
                    if item["category"] == "timbre_balance")
    assert category["cost"] == correct
    assert category["cost"] != wrong


def test_compare_page_uses_remeasured_visible_timbre_cost(
    tmp_path: Path, result: SchemeResult, second_result: SchemeResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("aosr.gui.app.calculation_fingerprint", lambda **kwargs: result.program_fingerprint)
    a_id, b_id = "b" * 32, "c" * 32
    tampered = tampered_timbre(result)
    correct, wrong = timbre_cost(result), timbre_cost(tampered)
    assert correct != wrong
    with client_for(tmp_path, result) as client:
        save_result(tampered, tmp_path / "results" / f"{a_id}{'.json'}")
        save_result(second_result, tmp_path / "results" / f"{b_id}{'.json'}")
        response = client.get(f"/api/compare/{a_id}/{b_id}")
    assert response.status_code == 200
    category = next(item for item in response.json()["categories"]
                    if item["category"] == "timbre_balance")
    assert category["a"]["cost"] == correct
    assert category["a"]["cost"] != wrong
    assert category["b"]["cost"] == timbre_cost(second_result)


def test_removed_registry_purpose_is_rejected_without_server_error(
    tmp_path: Path, result: SchemeResult, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("aosr.gui.app.calculation_fingerprint", lambda **kwargs: result.program_fingerprint)
    removed = result.model_copy(update={
        "scheme": result.scheme.model_copy(update={"purpose": "removed_purpose"}),
        "purpose_settings": result.purpose_settings.model_copy(update={"purpose": "removed_purpose"})})
    with client_for(tmp_path, result) as client:
        run_id = "b" * 32
        save_result(removed, tmp_path / "results" / f"{run_id}{'.json'}")
        response = client.get(f"/api/results/{run_id}")
    assert response.status_code == 409
    assert response.json()["reason"] == "品質登記簿沒有這個用途（removed_purpose），要先把用途加回登記簿"


def test_results_script_preserves_technical_details_explanation(tmp_path: Path) -> None:
    script = (STATIC / f"results{'.js'}").read_text(encoding="utf-8")
    assert "技術細節只在讀到結果時才有東西：拒收頁不顯示這個空的摺疊區。" in script
