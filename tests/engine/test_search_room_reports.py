"""房間搜尋串上細算與診斷；核對完整房間鍵、欄位與網頁中文。"""
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from aosr.geometry.shoebox import Wall
from aosr.gui.search_view import build_search_view
from aosr.reporting.modal_diagnosis_model import ModalKey
from aosr.reporting.scheme import load_scheme
from aosr.scoring.low_frequency_decay_reference import load_reference
from aosr.search import ledger
from aosr.search.layout import room_from_unit
from aosr.search.run import CandidateJob
from aosr.search.stage_one_worker import StageOneComputed
from aosr.search.stage_one_report import build_stage_one_report, render_stage_one_text
from tests.engine import _search_stage_one_cases as search
from tests.engine._search_room_loop_cases import ROOM_COLUMN, ROOM_KEYS, layout_for
from tests.engine._stage_one_auto_cases import invoke, read, running
from tests.engine._stage_one_post_check_cases import diagnosis, runner
from tests.engine._stage_one_refine_cases import snapshot


def test_room_auto_reports_group_by_exact_modal_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = running(tmp_path, monkeypatch, layout=layout_for())
    root = tmp_path / "child"
    command = runner(root, store)
    fake = search.FakeCompute(store, monkeypatch)
    def compute(jobs: Sequence[CandidateJob], workers: int) -> Iterator[StageOneComputed]:
        for item in fake(jobs, workers):
            if item.job.result_path.parent == store.refine_dir:
                number = item.job.trial_number
                assert number is not None
                (root / f"{number}.json").write_text(diagnosis(store, number).model_dump_json())
            yield item
    assert not invoke(store, tmp_path, monkeypatch, compute=compute, runner=command)
    assert read(store).outer.conclusion == "complete"
    report = build_stage_one_report(store)
    text = render_stage_one_text(report)
    rows = {row.trial_number: row for row in ledger.read_for(store).rows}
    expected_keys: set[str] = set()
    for result in store.refine_dir.glob("*-scheme.json"):
        scheme = load_scheme(result)
        scene = scheme.scene
        assert scene.impedance_pa_s_per_m_by_wall is not None
        key = ModalKey.from_inputs(room=scene.room_m,
            wall_impedances={Wall.from_name(wall): value for wall, value in scene.impedance_pa_s_per_m_by_wall.items()},
            density_kg_m3=scene.density_kg_m3, sound_speed_m_s=scene.sound_speed_m_s)
        expected_keys.add(key.digest)
        number = int(result.name.split("-")[1])
        row = rows[number]
        expected_room = room_from_unit({name: float.fromhex(value) for name, value in row.unit_params_hex.items()},
            store.settings.layout, store.project)
        assert scene.room_m == expected_room
        room_text = "×".join(f"{row.params_m[name]:.6f}" for name in ROOM_KEYS)
        assert any(f"試算 {number}｜" in line and room_text in line for line in report.refine_lines)
        assert any(f"試算 {number}｜" in line and room_text in line for line in text.splitlines())
    assert len(expected_keys) > 1  # 假診斷的頻率跟房間無關；沒有直接核鑰匙就會假綠。
    assert {line.removeprefix("房間鍵：") for line in report.post_check_lines if line.startswith("房間鍵：")} == expected_keys
    assert sum(line == "超過聽感參考線的共振（照超出量排序）" for line in report.post_check_lines) == len(expected_keys)
    assert ROOM_COLUMN in text and ROOM_COLUMN in "\n".join(report.refine_lines)
    assert "房間隨候選而變，面積對照不適用" in text
    assert "房間固定，面積對照不適用" not in text and "這一席的房間層跟上面那份不同" not in text
    before = snapshot(store)
    view = build_search_view(store.path, server_physics=store.identity.physics_identity,
        server_program=store.identity.program_fingerprint)
    best = next(block for block in view.blocks if block.key == "search-best")
    best_trial = read(store).best_trial
    assert best_trial is not None
    best_row = rows[best_trial]
    for name, label in zip(ROOM_KEYS, ("房間長（x 向）", "房間寬（y 向）", "房間高"), strict=True):
        assert f"{label}：{best_row.params_m[name]:.3f} 公尺" in best.lines
    assert snapshot(store) == before


def test_rounded_room_dimensions_cannot_merge_different_keys() -> None:
    import math
    from aosr.geometry.shoebox import Room
    from aosr.reporting.modal_diagnosis_readout import read_modal_diagnosis
    from aosr.search.layout import project_in
    from aosr.search.stage_one_post_check_report import _RoomGroup, _group_room
    from tests.engine import _modal_cases as modal, _stage_one_cases as stage

    project = stage.scheme()
    reference = load_reference(stage.run_kwargs()["registry"].purpose(project.purpose))
    first = project.scene.room_m
    second = Room(Lx=math.nextafter(first.Lx, math.inf), Ly=first.Ly, Lz=first.Lz)
    assert f"{first.Lx:.6f}" == f"{second.Lx:.6f}" and first.Lx.hex() != second.Lx.hex()
    diagnoses = [modal.sample(project_in(project, room))[0] for room in (first, second)]
    groups: list[_RoomGroup] = []
    for diagnosis in diagnoses:
        assert diagnosis.key is not None
        _group_room(groups, diagnosis.key, read_modal_diagnosis(diagnosis, reference), "合成參考線", variable=True)
    assert [group.key for group in groups] == [diagnosis.key for diagnosis in diagnoses]
    for group in groups:
        assert f"房間鍵：{group.key.digest}" in group.lines
    first_diagnosis = diagnoses[0]
    assert first_diagnosis.key is not None
    _group_room(groups, first_diagnosis.key, read_modal_diagnosis(first_diagnosis, reference), "合成參考線", variable=True)
    assert [group.key for group in groups] == [diagnosis.key for diagnosis in diagnoses]


def test_same_room_key_still_rejects_changed_resonances() -> None:
    from aosr.reporting.modal_diagnosis_model import ModalDiagnosis
    from aosr.reporting.modal_diagnosis_readout import read_modal_diagnosis
    from aosr.search.stage_one_post_check_report import _RoomGroup, _group_room
    from tests.engine import _modal_cases as modal, _stage_one_cases as stage

    reference = load_reference(stage.run_kwargs()["registry"].purpose(stage.scheme().purpose))
    diagnosis = modal.sample(stage.scheme())[0]
    assert diagnosis.key is not None
    groups: list[_RoomGroup] = []
    _group_room(groups, diagnosis.key, read_modal_diagnosis(diagnosis, reference), "合成參考線", variable=True)
    document = diagnosis.model_dump(mode="json")
    document["room_layer"]["modes"][1]["t60_s"] = 20.
    changed = ModalDiagnosis.model_validate(document)
    with pytest.raises(ValueError, match="這一席的房間層跟上面那份不同"):
        _group_room(groups, diagnosis.key, read_modal_diagnosis(changed, reference), "合成參考線", variable=True)


def test_nonrankable_room_rows_still_show_dimensions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = search.make_store(tmp_path, layout=layout_for())
    search.run(store, search.FakeCompute(store, monkeypatch, unavailable=frozenset({1})))
    report = build_stage_one_report(store)
    assert report.excluded
    text = render_stage_one_text(report)
    for item in report.excluded:
        room = "×".join(f"{item.row.params_m[name]:.6f}" for name in ROOM_KEYS)
        assert any(line.startswith(f"試算 {item.trial_number}：") and room in line for line in text.splitlines())
