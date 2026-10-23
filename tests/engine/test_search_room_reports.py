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
from aosr.search.store import SearchStore
from tests.engine import _search_stage_one_cases as search
from tests.engine._search_room_loop_cases import ROOM_COLUMN, ROOM_KEYS, layout_for
from tests.engine._stage_one_auto_cases import invoke, read, running
from tests.engine._stage_one_post_check_cases import diagnosis, runner
from tests.engine._stage_one_refine_cases import snapshot


def _finish(store: SearchStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """跑完搜尋、細算與入選後檢查；每席細算時順手放一份假診斷給檢查讀。"""
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


def test_room_auto_reports_group_by_exact_modal_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = running(tmp_path, monkeypatch, layout=layout_for())
    _finish(store, tmp_path, monkeypatch)
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
    assert "這次搜尋設定沒有面積對照的兩個門檻（設定早於面積對照），不開這一席" in text
    assert "房間固定，面積對照不適用" not in text and "這一席的房間層跟上面那份不同" not in text
    assert "面積對照只用來比較空間代價" not in text
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


def test_room_auto_area_control_rides_refine_shortlist_and_post_check(tmp_path: Path,
                                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    # 跨度門檻放寬到一定會開：答案照帳本自己重算（前 2 名的面積範圍外、合分最好的那一題），不從挑席程式抄。
    store = running(tmp_path, monkeypatch, layout=layout_for(),
                    area={"area_control_max_spread": 10., "area_control_min_difference": 1e-6})
    _finish(store, tmp_path, monkeypatch)
    scored = sorted((row for row in ledger.read_for(store).rows if row.outcome == "scored"),
                    key=lambda row: (row.score if row.score is not None else float("inf"), row.trial_number))
    def area(row: ledger.LedgerRow) -> float:
        return row.params_m["room_Lx"] * row.params_m["room_Ly"]
    low, high = min(map(area, scored[:2])), max(map(area, scored[:2]))
    expected = next(row.trial_number for row in scored[2:] if (low - area(row)) / low >= 1e-6 or (area(row) - high) / high >= 1e-6)
    report = build_stage_one_report(store)
    text = render_stage_one_text(report)
    assert f"面積對照門檻：前 N 面積跨度（最大減最小，再除以最小）{(high - low) / low * 100:.2f}%，不超過 1000% 才開；" in text
    assert any(line.startswith(f"面積對照＝試算 {expected}：") and "不宣稱已證明哪種大小或比例最好" in line for line in report.refine_lines)
    assert any(line.split("｜")[0] == f"試算 {expected}" and "面積對照" in line.split("｜")[1] for line in report.shortlist_lines)
    assert "面積對照只用來比較空間代價，不宣稱已證明哪種大小或比例最好" in report.shortlist_lines
    assert any(line.startswith(f"試算 {expected}／") and "面積對照：已診斷不計分" in line for line in report.post_check_lines)
    view = build_search_view(store.path, server_physics=store.identity.physics_identity,
        server_program=store.identity.program_fingerprint)
    block = next(block for block in view.blocks if block.key == "shortlist")
    assert "面積對照只用來比較空間代價，不宣稱已證明哪種大小或比例最好" in block.lines
    assert block.table is not None and any(row[0] == f"試算 {expected}" and "面積對照" in row[1] for row in block.table.rows)
    assert not any(line.startswith("本次未找到面積對照") for line in block.lines)


def test_room_auto_area_control_not_found_is_shown_on_web(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 「至少 99%」在房間範圍內不可能做到：照老闆「沒有就明列本次未找到」，網頁說明行要寫，不留面積對照的角色與提醒。
    store = running(tmp_path, monkeypatch, layout=layout_for(),
                    area={"area_control_max_spread": 10., "area_control_min_difference": .99})
    _finish(store, tmp_path, monkeypatch)
    reason = "本次未找到面積對照：前 N 以外沒有面積比前 N 最小再小至少 99%、或比最大再大至少 99% 的候選"
    report = build_stage_one_report(store)
    assert reason in report.refine_lines and reason not in report.shortlist_lines
    view = build_search_view(store.path, server_physics=store.identity.physics_identity,
        server_program=store.identity.program_fingerprint)
    block = next(block for block in view.blocks if block.key == "shortlist")
    assert reason in block.lines and "面積對照只用來比較空間代價，不宣稱已證明哪種大小或比例最好" not in block.lines
    assert block.table is not None and all("面積對照" not in row[1] for row in block.table.rows)


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


@pytest.mark.parametrize("kind,zone", [("unavailable", "unassessed"), ("different", "incomparable")])
def test_unassessed_and_incomparable_room_rows_show_dimensions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                               kind: str, zone: str) -> None:
    store = search.make_store(tmp_path, layout=layout_for())
    search.run(store, search.FakeCompute(store, monkeypatch,
        unavailable=frozenset({1}) if kind == "unavailable" else frozenset(),
        different=frozenset({1}) if kind == "different" else frozenset()))
    report = build_stage_one_report(store)
    assert any(item.trial_number == 1 and item.row.reason == zone for item in report.excluded)
    text = render_stage_one_text(report)
    for item in report.excluded:
        room = "×".join(f"{item.row.params_m[name]:.6f}" for name in ROOM_KEYS)
        assert any(line.startswith(f"試算 {item.trial_number}：") and room in line for line in text.splitlines())


@pytest.mark.parametrize("changed", [False, True])
def test_fixed_room_keeps_one_table_compared_by_resonances(changed: bool) -> None:
    # 固定房間照改動前的規則：不看鑰匙，只有第一份房間層，之後每席跟它比共振。
    import math
    from aosr.geometry.shoebox import Room
    from aosr.reporting.modal_diagnosis_model import ModalDiagnosis
    from aosr.reporting.modal_diagnosis_readout import read_modal_diagnosis
    from aosr.search.layout import project_in
    from aosr.search.stage_one_post_check_report import _RoomGroup, _group_room
    from tests.engine import _modal_cases as modal, _stage_one_cases as stage

    project = stage.scheme()
    reference = load_reference(stage.run_kwargs()["registry"].purpose(project.purpose))
    first = project.scene.room_m
    other = Room(Lx=math.nextafter(first.Lx, math.inf), Ly=first.Ly, Lz=first.Lz)
    base, second = (modal.sample(project_in(project, room))[0] for room in (first, other))
    base_key, second_key = base.key, second.key
    assert base_key is not None and second_key is not None and base_key != second_key
    if changed:
        document = second.model_dump(mode="json")
        document["room_layer"]["modes"][1]["t60_s"] = 20.
        second = ModalDiagnosis.model_validate(document)
    groups: list[_RoomGroup] = []
    _group_room(groups, base_key, read_modal_diagnosis(base, reference), "合成參考線", variable=False)
    if changed:
        with pytest.raises(ValueError, match="這一席的房間層跟上面那份不同"):
            _group_room(groups, second_key, read_modal_diagnosis(second, reference), "合成參考線", variable=False)
    else:
        _group_room(groups, second_key, read_modal_diagnosis(second, reference), "合成參考線", variable=False)
    assert [group.key for group in groups] == [base_key]
    assert not any(line.startswith("房間鍵：") for line in groups[0].lines)


def test_search_table_prints_room_on_header_and_each_ranked_row(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 細算那段也印房間欄；這題逐字核搜尋表自己那一行表頭與每一名次列，不靠整份文字裡出現過。
    store = search.make_store(tmp_path, layout=layout_for())
    search.run(store, search.FakeCompute(store, monkeypatch))
    report = build_stage_one_report(store)
    assert report.ranked
    lines = render_stage_one_text(report).splitlines()
    assert "名次｜試算編號｜離前牆／喇叭間距／聆聽距離（公尺）｜房間 長×寬×高（公尺）｜合分" in lines
    for rank, item in enumerate(report.ranked, 1):
        meters = item.row.params_m
        room = "×".join(f"{meters[name]:.6f}" for name in ROOM_KEYS)
        assert (f"{rank}｜試算 {item.trial_number}｜{meters['front_distance']:.6f}／{meters['spacing']:.6f}／"
                f"{meters['listening_distance']:.6f}｜{room}｜{item.row.score}") in lines
