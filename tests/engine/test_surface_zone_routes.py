"""構造輸入的能力、模態、伺服器與比較顯示；不開瀏覽器。"""
from pathlib import Path
from typing import cast

import pytest

from aosr.config.capabilities import CapabilityTable
from aosr.geometry.shoebox import Wall
from aosr.gui.compare_view import scheme_differences
from aosr.gui.modal_jobs import ModalJobs
from aosr.gui.modal_view import build_modal_view
from aosr.gui.problem_text import field_name, plain_problem
from aosr.physics import report_io, report_output, three_lane_report as report
from aosr.reporting.modal_diagnosis_model import ModalDiagnosisState
from aosr.reporting.modal_lookup import diagnose_scheme, key_from_scheme, scheme_scope_reason
from aosr.reporting.scheme import Scheme
from aosr.reporting.validation import SchemeProblem
from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine._directivity import DIRECTIVITY
from tests.engine._furniture_cases import CAPABILITIES, pair, reference_document
from tests.engine._surface_zone_cases import NEW, invalid_scene, mixed_scheme, zone_document
from tests.engine.test_gui_app import _app


@pytest.mark.parametrize("mixed,code", [(False, "real_frequency_independent_impedance"), (True, "assembly_by_wall")])
def test_capability_uses_actual_materials(mixed: bool, code: str) -> None:
    inputs = report_io.load_input_document(pair(Scheme.model_validate(zone_document(mixed=mixed))), CAPABILITIES, DIRECTIVITY)
    solved = report_io.solver_inputs(inputs)
    capability = report_output.report_capability(CAPABILITIES, solved.materials_by_wall)
    assert capability.materials == code
    assert capability.record is not None and capability.record.status == "experimental"
    assert report._unchecked_capability(solved.materials_by_wall).materials == code


def test_fixed_input_does_not_look_up_missing_construction_row() -> None:
    document = CAPABILITIES.model_dump(mode="python")
    for entry in document["entry"]:
        if entry["name"] == "three_lane_report":
            entry["capability"] = [row for row in entry["capability"] if row["materials"] != "assembly_by_wall"]
    table = CapabilityTable.model_validate(document)
    inputs = report_io.load_input_document(pair(Scheme.model_validate(reference_document())), table, DIRECTIVITY)
    assert report_output.report_capability(table, report_io.solver_inputs(inputs).materials_by_wall).materials == "real_frequency_independent_impedance"


def test_unsupported_construction_row_is_refused_with_its_note() -> None:
    document = CAPABILITIES.model_dump(mode="python")
    for entry in document["entry"]:
        if entry["name"] == "three_lane_report":
            for row in entry["capability"]:
                if row["materials"] == "assembly_by_wall":
                    row.update(status="unsupported", note="構造牆這一版不支援（考卷改的表）", evidence=[])
    table = CapabilityTable.model_validate(document)
    inputs = report_io.load_input_document(pair(mixed_scheme()), CAPABILITIES, DIRECTIVITY)
    with pytest.raises(ValueError, match=r"assembly_by_wall：構造牆這一版不支援（考卷改的表）"):
        report_output.report_capability(table, report_io.solver_inputs(inputs).materials_by_wall)


def test_construction_modal_is_outside_scope_before_wall_numbers(tmp_path: Path) -> None:
    scheme = mixed_scheme()
    assert scheme_scope_reason(scheme) == "construction_walls"
    assert key_from_scheme(scheme) is None
    diagnosis = diagnose_scheme(scheme, cache_dir=tmp_path, cache_only=True)
    assert diagnosis.state is ModalDiagnosisState.OUT_OF_SCOPE
    assert diagnosis.reason_code == "construction_walls"
    view = build_modal_view(diagnosis, scheme)
    assert view["state_text"] == "範圍外" and view["can_calculate"] is False
    assert "構造" in view["reason_text"] and "六面固定阻抗" in view["reason_text"]
    assert ModalJobs._room_label(object.__new__(ModalJobs), scheme)


def test_validate_and_save_construction_scheme_without_500(tmp_path: Path) -> None:
    with _app(tmp_path) as client:
        document = zone_document(mixed=True)
        response = client.post("/api/validate", json=document)
        assert response.status_code == 200, response.text
        data = response.json()
        assert not data["problems"]
        assert "floor" not in data["impedance_multiples"]
        assert data["impedance_multiples"]["ceiling"] == pytest.approx(4.0)
        saved = client.put("/api/schemes/construction", json=document | {"scheme_id": "construction"})
        assert saved.status_code == 200, saved.text
        loaded = client.get("/api/schemes/construction").json()["scheme"]
        assert loaded == mixed_scheme().model_copy(update={"scheme_id": "construction"}).model_dump(mode="json")
        fixed = client.post("/api/validate", json=zone_document()).json()
        assert fixed["impedance_multiples"]["floor"] == pytest.approx(4.0)


def test_input_page_refuses_construction_with_specific_wall_reason(tmp_path: Path) -> None:
    with _app(tmp_path) as client:
        response = client.post("/api/input-preview", json=zone_document(mixed=True))
        assert response.status_code == 400
        text = response.json()["error"]
        assert "地板" in text and "這一版輸入頁還不能改構造牆" in text
        assert "#309 第六步" in text and "計算與比較照常" in text
        response = client.post("/api/input-edit", json={"scheme": zone_document(mixed=True), "action": "shortcut", "value": "listening"})
        assert response.status_code == 400 and "構造牆" in response.json()["error"]


@pytest.mark.parametrize("case", ["both", "missing", "extra", "two", "area", "measured", "layer"])
def test_input_page_bad_zone_is_plain_400_naming_the_problem(tmp_path: Path, case: str) -> None:
    scene, _path, hint = invalid_scene(case)
    document = reference_document() | {"scene": scene}
    with _app(tmp_path) as client:
        for route, body in (("/api/input-preview", document),
                            ("/api/input-edit", {"scheme": document, "action": "shortcut", "value": "listening"})):
            response = client.post(route, json=body)
            assert response.status_code == 400, (route, response.text)
            assert hint in response.json()["error"], (route, response.text)


def test_input_page_all_fixed_zones_match_old_format_when_half_filled(tmp_path: Path) -> None:
    # 只正規化牆面欄位：房間半填仍交正式驗證列原因，回應跟舊寫法逐字相同。
    old, new = reference_document(), zone_document()
    for document in (old, new):
        cast(dict[str, object], cast(dict[str, object], document["scene"])["room_m"])["Lz"] = None
    with _app(tmp_path) as client:
        before = client.post("/api/input-preview", json=old)
        after = client.post("/api/input-preview", json=new)
    assert before.status_code == 200, before.text
    assert after.status_code == 200, after.text
    assert after.json() == before.json()


def test_construction_comparison_includes_each_wall_material_summary() -> None:
    original = mixed_scheme()
    document = zone_document(mixed=True)
    scene = cast(dict[str, object], document["scene"])
    zones = cast(dict[str, list[dict[str, object]]], scene[NEW])
    material = cast(dict[str, object], zones["floor"][0]["material"])
    layers = cast(list[dict[str, object]], material["layers"])
    layers[1]["thickness_m"] = 0.2
    rows = scheme_differences(original, Scheme.model_validate(document))
    path = f"scene.{NEW}.floor"
    assert {row.path for row in rows} == {path}
    assert rows[0].label == field_name(path, {})
    assert "0.43" in str(rows[0]) and "0.2" in str(rows[0])
    switched = scheme_differences(Scheme.model_validate(reference_document()), original)
    assert {row.path for row in switched} == {f"scene.{NEW}.floor"}


def test_material_leaf_labels_are_model_descriptions_and_errors_are_plain() -> None:
    from aosr.materials.construction_models import AirLayer, BoardLayer, ConstantImpedance, PorousLayer

    for model in (AirLayer, BoardLayer, ConstantImpedance, PorousLayer):
        for name in model.model_fields:
            label = field_name(f"scene.{NEW}.floor.0.material.assembly.layers.0.{name}", {})
            assert label is not None
            assert "地板" in label and name not in label
    for message in ("實測阻抗表這一版還不收：請改填構造或固定阻抗", "這一版每面只收整面一區；矩形分區是 #309 第二步"):
        _, text = plain_problem(SchemeProblem(f"scene.{NEW}.floor.0.material", message), zone_document(mixed=True))
        assert message in text


def test_search_modal_attachment_saves_out_of_scope_without_running_solver(tmp_path: Path) -> None:
    from aosr.reporting.modal_lookup import placement_digest
    from aosr.search.modal_attach import _diagnose
    from aosr.search.modal_record import ModalRole, RoleInput, summary_path

    scheme = mixed_scheme()
    summary_path(tmp_path).parent.mkdir()
    role = ModalRole(role="baseline", placement_digest=placement_digest(scheme))
    item = RoleInput(role, scheme, tmp_path / "scheme.json", tmp_path / "result.json")
    record = _diagnose(tmp_path, item, cache_dir=tmp_path / "cache", identity="unused",
                       lock_fd=None, runner=("must-not-run",))
    assert record.state == "out_of_scope" and "構造" in record.reason_text
    assert record.diagnosis_file is not None
    text = (summary_path(tmp_path).parent / record.diagnosis_file).read_text()
    assert "construction_walls" in text


def test_prepare_accepts_missing_legacy_numbers_but_checks_fixed_materials() -> None:
    from aosr.materials.construction_models import ConstantImpedance
    from aosr.physics.three_lane_report_batch import _checked_material_walls

    inputs = report_io.load_input_document(pair(mixed_scheme()), CAPABILITIES, DIRECTIVITY)
    materials = report_io.solver_inputs(inputs).materials_by_wall
    _checked_material_walls(materials)
    for invalid in (0.0, float("inf"), True):
        broken = materials | {Wall.CEILING: ConstantImpedance.model_construct(impedance_pa_s_m=invalid)}
        with pytest.raises(ValueError, match="ceiling.*正有限"):
            _checked_material_walls(broken)
    with pytest.raises(ValueError, match="六面牆"):
        _checked_material_walls({Wall.FLOOR: materials[Wall.FLOOR]})


def test_gui_construction_modal_route_is_out_of_scope_without_500(tmp_path: Path) -> None:
    import json
    import time

    from starlette.testclient import TestClient
    from aosr.gui.app import GuiSettings, create_app
    from aosr.reporting.modal_diagnosis_model import ModalDiagnosis
    from tests.engine._modal_cases import runner
    from tests.engine.test_gui_app import COMMIT

    diagnosis = ModalDiagnosis(state=ModalDiagnosisState.OUT_OF_SCOPE, reason_code="construction_walls")
    command = runner(tmp_path / "runner", diagnosis)
    with TestClient(create_app(GuiSettings(engine_commit=COMMIT, data_dir=tmp_path, modal_runner=command)),
                    base_url="http://localhost") as client:
        run_id = "b" * 32
        # 模態端點的輸入契約是 scheme_snapshot（方案快照），不重評報表物理數字。
        (tmp_path / "results" / f"{run_id}.json").write_text(json.dumps({"scheme": mixed_scheme().model_dump(mode="json")}))
        response = client.get(f"/api/results/{run_id}/modal")
        assert response.status_code == 200, response.text
        job = response.json()["job"]
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            response = client.get(f"/api/results/{run_id}/modal", params={"job_id": job["run_id"]})
            assert response.status_code == 200, response.text
            body = response.json()
            if body.get("view") is not None:
                assert body["view"]["state_text"] == "範圍外"
                assert body["view"]["can_calculate"] is False
                assert "構造" in body["view"]["reason_text"]
                return
            time.sleep(0.02)
        pytest.fail("構造範圍外診斷沒有回來")
