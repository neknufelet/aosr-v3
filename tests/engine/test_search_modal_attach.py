"""收尾附件不改搜尋、按文件判四態、固定房間防線與人手重跑。"""
from __future__ import annotations

from pathlib import Path

import pytest

from aosr.reporting.modal_diagnosis_model import ModalDiagnosis, ModalDiagnosisState
from aosr.search.modal_attach import attach_modal
from aosr.search.modal_record import read_summary
from aosr.search.outer_status import OUTER_MESSAGES, OuterStatus
from tests.engine._modal_cases import runner, sample
from tests.engine._search_modal_cases import prepared, protected, scheme_for


@pytest.mark.parametrize("state,reason", [
    (ModalDiagnosisState.DIAGNOSED_NOT_SCORED, ""),
    (ModalDiagnosisState.NOT_COMPUTED, "尚未算"),
    (ModalDiagnosisState.FAILED, "solver 原文錯誤"),
    (ModalDiagnosisState.OUT_OF_SCOPE, "unsupported_impedance"),
])
def test_attachment_reads_document_state_and_preserves_search(tmp_path: Path,
                                                             state: ModalDiagnosisState, reason: str) -> None:
    store, _, status = prepared(tmp_path)
    diagnosis = sample(store.project)[0] if state is ModalDiagnosisState.DIAGNOSED_NOT_SCORED else ModalDiagnosis(
        state=state, reason_code=reason if state is ModalDiagnosisState.OUT_OF_SCOPE else None,
        reason_text=reason if state is not ModalDiagnosisState.OUT_OF_SCOPE else None)
    before = protected(store)
    attach_modal(store, status=status, cache_dir=tmp_path / "cache", runner=runner(tmp_path / "runner", diagnosis))
    summary = read_summary(store.path)
    assert summary is not None and summary.completed
    assert summary.roles[0].state == state.value
    assert protected(store) == before
    assert summary.cache_dir == str((tmp_path / "cache").resolve())
    if reason and state is not ModalDiagnosisState.OUT_OF_SCOPE:
        assert reason in summary.roles[0].reason_text


@pytest.mark.parametrize("conclusion", ["user_stopped", "search_failed", "search_interrupted", "refine_failed", "refine_interrupted"])
def test_abnormal_conclusion_skips_with_reason(tmp_path: Path, conclusion: str) -> None:
    from aosr.search.outer_status import OuterStatus
    store, _, status = prepared(tmp_path)
    status = status.model_copy(update={"outer": OuterStatus.model_validate({"conclusion": conclusion})})
    before = protected(store)
    attach_modal(store, status=status, cache_dir=tmp_path / "cache", runner=("must-not-run",))
    summary = read_summary(store.path)
    assert summary is not None and summary.completed
    assert "搜尋沒有正常收尾" in summary.reason_text and "接續跑完後會補" in summary.reason_text
    assert all(role.state == "skipped" for role in summary.roles)
    assert protected(store) == before


@pytest.mark.parametrize("change", ["room", "material", "scope"])
def test_changed_room_material_or_scope_skips_every_role(tmp_path: Path, change: str) -> None:
    import json
    from aosr.geometry.shoebox import Room
    store, _, status = prepared(tmp_path)
    scheme = scheme_for(store, "search_best")
    if change == "scope":
        path = store.candidate_path(0)
        path.write_text(json.dumps(json.loads(path.read_bytes()) | {"scope": "future_stage"}))
    else:
        scene = scheme.scene.model_copy(update={"room_m": Room(7, 4, 3)} if change == "room" else {
            "impedance_pa_s_per_m_by_wall": {wall: value * 2 for wall, value in scheme.scene.impedance_pa_s_per_m_by_wall.items()}})
        store.scheme_path_for(store.candidate_path(0)).write_text(scheme.model_copy(update={"scene": scene}).model_dump_json())
    attach_modal(store, status=status, cache_dir=tmp_path / "cache", runner=("must-not-run",))
    summary = read_summary(store.path)
    assert summary is not None and summary.completed
    assert "房間或材料不固定，這次不補（成本另評估）" in summary.reason_text
    assert all(role.state == "skipped" for role in summary.roles)


def test_missing_scheme_does_not_block_other_roles(tmp_path: Path) -> None:
    store, _, status = prepared(tmp_path)
    store.scheme_path_for(store.candidate_path(0)).unlink()
    attach_modal(store, status=status, cache_dir=tmp_path / "cache",
                 runner=runner(tmp_path / "runner", ModalDiagnosis(state=ModalDiagnosisState.NOT_COMPUTED)))
    summary = read_summary(store.path)
    assert summary is not None
    roles = {role.role: role for role in summary.roles}
    assert roles["search_best"].state == "failed" and "方案檔讀不回" in roles["search_best"].reason_text
    assert roles["search_best"].diagnosis_file is not None
    assert roles["baseline"].state == roles["refine_best"].state == "not_computed"


def test_diagnosed_wrong_placement_is_failure(tmp_path: Path) -> None:
    store, _, status = prepared(tmp_path)
    attach_modal(store, status=status, cache_dir=tmp_path / "cache",
                 runner=runner(tmp_path / "runner", sample(scheme_for(store, "refine_best"))[0]))
    summary = read_summary(store.path)
    assert summary is not None
    assert summary.roles[0].state == "failed" and "擺位" in summary.roles[0].reason_text


@pytest.mark.parametrize("refined", [False, True])
def test_selected_roles_and_duplicate_placement(tmp_path: Path, refined: bool) -> None:
    store, _, status = prepared(tmp_path, refined=refined)
    # 搜尋首名方案改成原擺位，保留試算代號；細算首名維持帳本的 1 號。
    store.scheme_path_for(store.candidate_path(0)).write_text(store.project.model_dump_json())
    attach_modal(store, status=status, cache_dir=tmp_path / "cache",
                 runner=runner(tmp_path / "runner", sample(store.project)[0]))
    summary = read_summary(store.path)
    assert summary is not None
    roles = {role.role: role for role in summary.roles}
    assert roles["search_best"].trial_number == 0 and roles["search_best"].duplicate_of == "baseline"
    assert roles["search_best"].diagnosis_file == roles["baseline"].diagnosis_file
    assert (store.path / "modal-diagnosis" / str(roles["search_best"].diagnosis_file)).read_bytes() == (
        store.path / "modal-diagnosis" / str(roles["baseline"].diagnosis_file)).read_bytes()
    assert ("refine_best" in roles) is refined
    if refined:
        assert roles["refine_best"].trial_number == 1 and roles["refine_best"].temporary


@pytest.mark.parametrize("initial", [ModalDiagnosisState.FAILED, ModalDiagnosisState.NOT_COMPUTED])
def test_manual_rerun_retries_unsuccessful_and_reuses_diagnosed(tmp_path: Path, initial: ModalDiagnosisState) -> None:
    store, _, status = prepared(tmp_path, refined=False)
    store.scheme_path_for(store.candidate_path(0)).unlink()
    attach_modal(store, status=status, cache_dir=tmp_path / "cache",
                 runner=runner(tmp_path / "first", ModalDiagnosis(state=initial, reason_text="上一次沒有算完")))
    attach_modal(store, status=status, cache_dir=tmp_path / "cache",
                 runner=runner(tmp_path / "second", sample(store.project)[0]))
    summary = read_summary(store.path)
    assert summary is not None and summary.roles[0].state == "diagnosed_not_scored"
    saved = summary.roles[0].diagnosis_file
    attach_modal(store, status=status, cache_dir=tmp_path / "cache", runner=("must-not-run",))
    again = read_summary(store.path)
    assert again is not None and again.roles[0].diagnosis_file == saved


def test_search_and_refine_first_share_placement_only_one_child(tmp_path: Path) -> None:
    import json
    store, _, status = prepared(tmp_path)
    value = scheme_for(store, "search_best")
    store.scheme_path_for(store.refine_result_path(1)).write_text(value.model_dump_json())
    fake = runner(tmp_path / "runner", ModalDiagnosis(state=ModalDiagnosisState.NOT_COMPUTED))
    summary = attach_modal(store, status=status, cache_dir=tmp_path / "cache", runner=fake)
    roles = {role.role: role for role in summary.roles}
    assert roles["refine_best"].duplicate_of == "search_best"
    args_files = tuple((store.path / "modal-diagnosis").glob("*.args.json"))
    invoked = {Path(json.loads(path.read_text())[1]).name for path in args_files}
    assert invoked == {"project.json", "trial-000000-scheme.json"}


def test_refine_first_can_be_original_placement(tmp_path: Path) -> None:
    from aosr.search.refine import RefineLedger
    store, _, status = prepared(tmp_path)
    header, rows = RefineLedger.read(store.refine_ledger_path)
    store.refine_ledger_path.unlink()
    book = RefineLedger.create(store.refine_ledger_path, header)
    for row in rows:
        book.append(row.model_copy(update={"total_cost": 0.01}) if row.trial_number is None else row)
    summary = attach_modal(store, status=status, cache_dir=tmp_path / "cache",
                           runner=runner(tmp_path / "runner", ModalDiagnosis(state=ModalDiagnosisState.NOT_COMPUTED)))
    role = next(role for role in summary.roles if role.role == "refine_best")
    assert role.trial_number is None and role.duplicate_of == "baseline"


def test_unsupported_original_is_out_of_scope_without_child(tmp_path: Path) -> None:
    from aosr.reporting.scheme import Scheme
    store, _, status = prepared(tmp_path, refined=False)
    document = store.project.model_dump(mode="json")
    document["scene"]["impedance_pa_s_per_m_by_wall"]["floor"] = 0
    value = Scheme.model_validate(document)
    (store.path / "project.json").write_text(value.model_dump_json())
    store.scheme_path_for(store.candidate_path(0)).write_text(value.model_dump_json())
    opened = type(store).open(store.path)
    summary = attach_modal(opened, status=status, cache_dir=tmp_path / "cache", runner=("must-not-run",))
    assert all(role.state == "out_of_scope" for role in summary.roles)
    assert all("有限正實數" in role.reason_text for role in summary.roles)


@pytest.mark.parametrize("conclusion", [reason for reason in OUTER_MESSAGES if reason not in {
    "user_stopped", "search_failed", "search_interrupted", "refine_failed", "refine_interrupted"}])
def test_all_normal_writer_conclusions_attach(tmp_path: Path, conclusion: str) -> None:
    store, _, status = prepared(tmp_path)
    status = status.model_copy(update={"outer": OuterStatus.model_validate({"conclusion": conclusion})})
    summary = attach_modal(store, status=status, cache_dir=tmp_path / "cache",
                           runner=runner(tmp_path / "runner", ModalDiagnosis(state=ModalDiagnosisState.NOT_COMPUTED)))
    assert summary.completed and not summary.reason_text
    assert all(role.state == "not_computed" for role in summary.roles)


def test_missing_scope_fails_only_that_role_without_solving_it(tmp_path: Path) -> None:
    import json
    store, _, status = prepared(tmp_path)
    path = store.candidate_path(0)
    document = json.loads(path.read_bytes())
    del document["scope"]
    path.write_text(json.dumps(document))
    fake = runner(tmp_path / "runner", ModalDiagnosis(state=ModalDiagnosisState.NOT_COMPUTED))
    summary = attach_modal(store, status=status, cache_dir=tmp_path / "cache", runner=fake)
    roles = {r.role: r for r in summary.roles}
    assert roles["search_best"].state == "failed"
    assert "結果範圍讀不回" in roles["search_best"].reason_text
    assert not summary.reason_text
    assert roles["baseline"].state == roles["refine_best"].state == "not_computed"
    invoked = {Path(json.loads(p.read_bytes())[1]).name for p in (store.path / "modal-diagnosis").glob("*.args.json")}
    assert "trial-000000-scheme.json" not in invoked


@pytest.mark.parametrize("state", [ModalDiagnosisState.DIAGNOSED_NOT_SCORED, ModalDiagnosisState.FAILED])
def test_five_reruns_keep_only_referenced_owned_files(tmp_path: Path, state: ModalDiagnosisState) -> None:
    import re
    store, _, status = prepared(tmp_path, refined=False)
    store.scheme_path_for(store.candidate_path(0)).write_text(store.project.model_dump_json())
    folder = store.path / "modal-diagnosis"
    folder.mkdir()
    other = {folder / "manual.json": b"other", folder / "baseline-not-owned.stderr": b"other error"}
    for path, data in other.items():
        path.write_bytes(data)
    diagnosis = sample(store.project)[0] if state is ModalDiagnosisState.DIAGNOSED_NOT_SCORED else ModalDiagnosis(state=state, reason_text="失敗原文")
    fake = runner(tmp_path / "runner", diagnosis)
    script = Path(fake[-1])
    script.write_text(script.read_text().replace("for target in (shared, out.with_suffix('.args.json')):", "for target in (shared,):"))
    expected_count: int | None = None
    for _ in range(5):
        summary = attach_modal(store, status=status, cache_dir=tmp_path / "cache", runner=fake)
        referenced = {r.diagnosis_file for r in summary.roles}
        owned = {p.name for p in folder.iterdir() if re.fullmatch(r"(?:baseline|search_best|refine_best)-[0-9a-f]{32}\.(?:json|stderr)", p.name)}
        assert {name for name in owned if name.endswith(".json")} == referenced
        assert len(owned) <= len(referenced) * 2
        assert all((folder / str(name)).is_file() for name in referenced)
        count = len(tuple(folder.iterdir()))
        if expected_count is None:
            expected_count = count
        assert count == expected_count
        assert {p: p.read_bytes() for p in other} == other


@pytest.mark.parametrize("change", ["role", "trial", "placement", "missing", "bad", "none"])
def test_abnormal_conclusion_keeps_only_same_role_readable_diagnosis(tmp_path: Path, change: str) -> None:
    from aosr.search.modal_record import write_summary
    store, _, status = prepared(tmp_path, refined=False)
    store.scheme_path_for(store.candidate_path(0)).write_text(store.project.model_dump_json())
    previous = attach_modal(store, status=status, cache_dir=tmp_path / "cache",
                            runner=runner(tmp_path / "runner", sample(store.project)[0]))
    if change in ("role", "trial", "placement"):
        saved = previous.roles[1]
        changes: dict[str, object] = {"trial_number": 2} if change == "trial" else {"placement_digest": "bad"} if change == "placement" else {"role": "refine_best"}
        changed = saved.model_copy(update=changes)
        previous = previous.model_copy(update={"roles": (previous.roles[0], changed)})
        write_summary(store.path, previous)
    elif change in ("missing", "bad"):
        path = store.path / "modal-diagnosis" / str(previous.roles[0].diagnosis_file)
        if change == "missing":
            path.unlink()
        else:
            path.write_text("{")
    status = status.model_copy(update={"outer": OuterStatus(conclusion="user_stopped")})
    summary = attach_modal(store, status=status, cache_dir=tmp_path / "cache", runner=("must-not-run",))
    if change == "none":
        assert summary.roles == previous.roles
    elif change == "trial":
        assert summary.roles[1].state == "diagnosed_not_scored"
        assert summary.roles[1].diagnosis_file == previous.roles[1].diagnosis_file
        assert summary.roles[1].trial_number == 0
    elif change in ("role", "placement"):
        assert summary.roles[0] == previous.roles[0] and summary.roles[1].state == "skipped"
    else:
        assert all(r.state == "skipped" for r in summary.roles)


def test_cross_role_rerun_points_to_original_document(tmp_path: Path) -> None:
    from aosr.search.modal_record import write_summary
    store, _, status = prepared(tmp_path, refined=False)
    store.scheme_path_for(store.candidate_path(0)).write_text(store.project.model_dump_json())
    previous = attach_modal(store, status=status, cache_dir=tmp_path / "cache",
                            runner=runner(tmp_path / "runner", sample(store.project)[0]))
    source = previous.roles[1].model_copy(update={"diagnosis_file": previous.roles[0].diagnosis_file})
    write_summary(store.path, previous.model_copy(update={"roles": (source,)}))
    summary = attach_modal(store, status=status, cache_dir=tmp_path / "cache", runner=("must-not-run",))
    assert all(r.state == "diagnosed_not_scored" and r.diagnosis_file == source.diagnosis_file for r in summary.roles)
