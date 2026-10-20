"""第一階段搜尋：合分交回、同表釘住、拒建與拒接命令。"""
from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Mapping, Sequence
from aosr.search.sampler import Outcome, Proposal

import pytest

from aosr.config.paths import config_path
from aosr.reporting.stage_one_result import load_stage_one_result
from aosr.search import ledger
from aosr.search.run import CandidateJob
from aosr.search.settings import SearchSettings
from aosr.search.store import SearchIdentity, SearchStore
from tests.engine import _search_stage_one_cases as cases
from tests.engine import _stage_one_cases as stage
from tests.engine._search_store_cases import settings_document, purpose_settings


def test_old_settings_fingerprint_and_snapshot_stay_identical() -> None:
    # 主線未加 stage 前，以 settings_document() 的 canonical 緊湊排序 JSON 算 SHA-256。
    settings = SearchSettings.model_validate(settings_document())
    assert settings.fingerprint == "211ef3f9969fd541d729c9ea08ef09efc65ae2ec3acbfd40f430190fe3b5c203"
    assert json.loads(settings.model_dump_json()) == settings_document()
    changed = SearchSettings.model_validate(settings_document() | {"stage": "stage_one_fem_only"})
    assert changed.fingerprint != settings.fingerprint


def test_sampler_receives_saved_screening_total(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.search.sampler import SamplerAdapter, Scored
    received: dict[int, Outcome] = {}
    tell = SamplerAdapter.tell_batch
    def capture(self: SamplerAdapter, outcomes: Mapping[int, Outcome]) -> None:
        received.update(outcomes)
        return tell(self, outcomes)
    monkeypatch.setattr(SamplerAdapter, "tell_batch", capture)
    store = cases.make_store(tmp_path)
    status = cases.run(store, cases.FakeCompute(store, monkeypatch))
    assert not store.baseline_path.exists()
    assert status.baseline_outcome == "not_computed"
    rows = ledger.read_for(store).rows
    assert len({row.score for row in rows if row.result_file}) > 1  # 替身合分要分得開，傳錯人才考得出來
    for row in rows:
        if row.result_file:
            result = load_stage_one_result(store.candidate_path(row.trial_number))
            assert result.score.screening_total is not None and row.score is not None
            assert received[row.trial_number] == Scored(result.score.screening_total)
            assert row.score.hex() == result.score.screening_total.hex()


@pytest.mark.parametrize("kind,zone", [("unavailable", "unassessed"), ("different", "incomparable")])
def test_nonrankable_candidates_have_no_numeric_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str, zone: str) -> None:
    from aosr.search.sampler import Excluded, RankingZone, SamplerAdapter
    received: dict[int, Outcome] = {}
    tell = SamplerAdapter.tell_batch
    def capture(self: SamplerAdapter, outcomes: Mapping[int, Outcome]) -> None:
        received.update(outcomes)
        tell(self, outcomes)
    monkeypatch.setattr(SamplerAdapter, "tell_batch", capture)
    store = cases.make_store(tmp_path)
    compute = cases.FakeCompute(store, monkeypatch, unavailable=frozenset({1}) if kind == "unavailable" else frozenset(),
                                different=frozenset({1}) if kind == "different" else frozenset())
    cases.run(store, compute)
    row = next(row for row in ledger.read_for(store).rows if row.trial_number == 1)
    assert row.outcome == "excluded"
    assert row.reason == zone
    assert row.score is None
    assert received[row.trial_number] == Excluded(RankingZone(zone))


@pytest.mark.parametrize("damage,reason", [("source", "全向"), ("furniture", "家具"),
    ("walls", "六面"), ("seats", "前後左右上下"), ("purpose", "第一階段"), ("axis", "verification")])
def test_invalid_prerequisites_create_nothing(tmp_path: Path, damage: str, reason: str) -> None:
    item = stage.scheme()
    if damage == "source":
        item = item.model_copy(update={"source_model": "product_default"})
    elif damage == "furniture":
        from aosr.reporting.scheme import FurnitureSpec
        from tests.engine._furniture_cases import relative_item
        item = item.model_copy(update={"furniture": (FurnitureSpec.model_validate(relative_item()),)})
    elif damage == "walls":
        assert item.scene.impedance_pa_s_per_m_by_wall is not None
        walls = dict(item.scene.impedance_pa_s_per_m_by_wall)
        walls["floor"] *= 2
        item = item.model_copy(update={"scene": item.scene.model_copy(update={"impedance_pa_s_per_m_by_wall": walls})})
    elif damage == "seats":
        item = item.model_copy(update={"receiver_set": item.receiver_set.model_copy(update={"points": item.receiver_set.points[:-1]})})
    elif damage == "axis":
        from aosr.config.frequency_axis import LowFrequencyAxis
        item = item.model_copy(update={"scene": item.scene.model_copy(update={"low_frequency_axis": LowFrequencyAxis.VERIFICATION})})
    else:
        item = item.model_copy(update={"purpose": "unknown-purpose"})
    settings = SearchSettings.model_validate(settings_document() | {"purpose": item.purpose, "stage": "stage_one_fem_only"})
    with pytest.raises(ValueError, match=reason):
        SearchStore.create(tmp_path / "searches", project=item, settings=settings,
            identity=SearchIdentity("physics", "program", purpose_settings(item.purpose)),
            versions={"python": "test", "optuna": "test", "numpy": "test"})
    assert not (tmp_path / "searches").exists()


@pytest.mark.parametrize("command,label,extra", [("refine", "細算", ["--engine-commit", "test"]),
    ("auto", "自動外圈", ["--engine-commit", "test", "--modal-cache-dir", "cache"]),
    ("feedback", "回饋", []), ("select", "入選", ["--trial", "0", "--data-dir", "data"])])
def test_unsupported_commands_preserve_every_byte(tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                                command: str, label: str, extra: list[str]) -> None:
    from aosr.search.cli import main
    from aosr.search.run import SearchStatus
    store = cases.make_store(tmp_path)
    store.status_path.write_text(SearchStatus().model_dump_json())
    before = {p.relative_to(store.path): p.read_bytes() for p in store.path.rglob("*") if p.is_file()}
    code = main([command, str(store.path), *extra])
    assert code == 1
    assert f"這是第一階段搜尋資料夾，{label}還沒接上第一階段" in capsys.readouterr().err
    assert {p.relative_to(store.path): p.read_bytes() for p in store.path.rglob("*") if p.is_file()} == before


def _stable_rows(store: SearchStore) -> list[str]:
    return [row.model_dump_json() for row in sorted(ledger.read_for(store).rows, key=lambda row: row.trial_number)]


def test_completion_order_cannot_choose_comparison_or_change_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from uuid import UUID
    monkeypatch.setattr("uuid.uuid4", lambda: UUID(int=123))
    monkeypatch.setattr("aosr.search.timings._now", lambda: 0.)
    sequential = cases.make_store(tmp_path / "single", workers=1)
    parallel = cases.make_store(tmp_path / "parallel", workers=3)
    cases.run(sequential, cases.FakeCompute(sequential, monkeypatch, different=frozenset({2})))
    cases.run(parallel, cases.FakeCompute(parallel, monkeypatch, different=frozenset({2})))
    status = json.loads(parallel.status_path.read_bytes())
    assert status["comparison_trial"] == min(row.trial_number for row in ledger.read_for(parallel).rows if row.result_file)
    assert _stable_rows(sequential) == _stable_rows(parallel)
    assert sequential.status_path.read_bytes() == parallel.status_path.read_bytes()


@pytest.mark.parametrize("kill_after", [1, 4])
def test_killed_search_resumes_with_same_comparison_rows_and_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                 kill_after: int) -> None:
    from uuid import UUID
    from aosr.search.stage_one_run import resume_stage_one_search
    monkeypatch.setattr("uuid.uuid4", lambda: UUID(int=123))
    monkeypatch.setattr("aosr.search.timings._now", lambda: 0.)
    control = cases.make_store(tmp_path / "control", workers=3)
    interrupted = cases.make_store(tmp_path / "interrupted", workers=3)
    cases.run(control, cases.FakeCompute(control, monkeypatch))
    with pytest.raises(cases.Killed):
        cases.run(interrupted, cases.FakeCompute(interrupted, monkeypatch, kill_after=kill_after))
    resumed = resume_stage_one_search(interrupted, compute=cases.FakeCompute(interrupted, monkeypatch),
        probe=lambda: interrupted.identity, registry_path=config_path("quality_targets.toml"),
        run_date=stage.run_kwargs()["run_date"], engine_version=interrupted.identity.program_fingerprint)
    assert resumed.comparison_trial == json.loads(control.status_path.read_bytes())["comparison_trial"]
    assert _stable_rows(control) == _stable_rows(interrupted)
    assert control.status_path.read_bytes() == interrupted.status_path.read_bytes()


def test_killed_between_pin_save_and_first_row_resumes_identically(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 釘住編號已存、第一列還沒落帳就被砍：接續讀同一份結果再釘，帳本與狀態跟沒被砍的那一跑一樣。
    from uuid import UUID
    from aosr.search.stage_one_run import _StageOneRunner, resume_stage_one_search
    monkeypatch.setattr("uuid.uuid4", lambda: UUID(int=123))
    monkeypatch.setattr("aosr.search.timings._now", lambda: 0.)
    control = cases.make_store(tmp_path / "control", workers=3)
    interrupted = cases.make_store(tmp_path / "interrupted", workers=3)
    cases.run(control, cases.FakeCompute(control, monkeypatch))
    record = _StageOneRunner.record
    def killed(self: _StageOneRunner, *args: object, **kwargs: object) -> None:
        raise cases.Killed()
    monkeypatch.setattr(_StageOneRunner, "record", killed)
    with pytest.raises(cases.Killed):
        cases.run(interrupted, cases.FakeCompute(interrupted, monkeypatch))
    assert json.loads(interrupted.status_path.read_bytes())["comparison_trial"] is not None
    assert not ledger.read_for(interrupted).rows
    monkeypatch.setattr(_StageOneRunner, "record", record)
    resume_stage_one_search(interrupted, compute=cases.FakeCompute(interrupted, monkeypatch),
        probe=lambda: interrupted.identity, registry_path=config_path("quality_targets.toml"),
        run_date=stage.run_kwargs()["run_date"], engine_version=interrupted.identity.program_fingerprint)
    assert _stable_rows(control) == _stable_rows(interrupted)
    assert control.status_path.read_bytes() == interrupted.status_path.read_bytes()


@pytest.mark.parametrize("damage", ["missing", "identity", "missing-status-pin"])
def test_resume_never_substitutes_another_comparison_candidate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str) -> None:
    from aosr.search.stage_one_run import resume_stage_one_search
    from aosr.search.run import SearchStatus
    store = cases.make_store(tmp_path)
    with pytest.raises(cases.Killed):
        cases.run(store, cases.FakeCompute(store, monkeypatch, kill_after=4))
    status = SearchStatus.model_validate_json(store.status_path.read_bytes())
    assert status.comparison_trial is not None
    path = store.candidate_path(status.comparison_trial)
    if damage == "identity":
        document = json.loads(path.read_bytes())
        document["program_fingerprint"] = "calc-v1:" + "a" * 64
        path.write_text(json.dumps(document))
    else:
        path.unlink()
    if damage == "missing-status-pin":
        store.status_path.write_text(status.model_copy(update={"comparison_trial": None}).model_dump_json())
    compute = cases.FakeCompute(store, monkeypatch)
    resumed = resume_stage_one_search(store, compute=compute, probe=lambda: store.identity,
        registry_path=config_path("quality_targets.toml"), run_date=stage.run_kwargs()["run_date"],
        engine_version=store.identity.program_fingerprint)
    assert resumed.state == "interrupted"
    assert not compute.calls


def test_cli_start_resume_and_stop_use_stage_one_factory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.search import cli
    from aosr.search.run import SearchStatus
    input_store = cases.make_store(tmp_path / "input")
    monkeypatch.setattr(cli, "_identity", lambda purpose, capabilities: input_store.identity)
    project, settings, output = tmp_path / "project", tmp_path / "settings", tmp_path / "output"
    project.write_text(input_store.project.model_dump_json())
    settings.write_text(input_store.settings.model_dump_json())
    def factory(store: SearchStore, capabilities: Path, commit: str) -> cases.FakeCompute:
        return cases.FakeCompute(store, monkeypatch, kill_after=4)
    with pytest.raises(cases.Killed):
        cli.main(["start", "--project", str(project), "--settings", str(settings), "--root", str(output),
                  "--engine-commit", "test"], compute_factory=factory)
    folder, = output.iterdir()
    assert not cli.main(["resume", str(folder), "--engine-commit", "test"],
        compute_factory=lambda store, capabilities, commit: cases.FakeCompute(store, monkeypatch))
    assert SearchStatus.model_validate_json((folder / "status.json").read_bytes()).state == "budget_exhausted"
    assert not cli.main(["stop", str(folder)])
    assert (folder / "stop").exists()


@pytest.mark.parametrize("illegal", [0, 4])
def test_illegal_rows_are_recorded_before_and_after_pin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, illegal: int) -> None:
    # 試算 0 落在還沒釘比較身分的第一批、試算 4 落在已釘之後；兩種都要落帳一次、不送去算。
    from aosr.search.constraints import Reason, Violation, to_illegal
    from aosr.search.stage_one_run import _StageOneRunner
    prepare = _StageOneRunner._prepare
    def blocked(self: _StageOneRunner, proposals: Sequence[Proposal], saved: Sequence[ledger.LedgerRow]
                ) -> tuple[dict[int, Outcome], dict[int, CandidateJob], dict[int, dict[str, float]]]:
        outcomes, jobs, meters = prepare(self, proposals, saved)
        if jobs.pop(illegal, None) is not None:
            outcomes[illegal] = to_illegal((Violation(Reason.WALL_GAP, .1),))
        return outcomes, jobs, meters
    monkeypatch.setattr(_StageOneRunner, "_prepare", blocked)
    store = cases.make_store(tmp_path)
    compute = cases.FakeCompute(store, monkeypatch)
    status = cases.run(store, compute)
    rows = ledger.read_for(store).rows
    assert sorted(row.trial_number for row in rows) == list(range(store.settings.budget))
    row = next(row for row in rows if row.trial_number == illegal)
    assert row.outcome == "illegal" and row.result_file is None and illegal not in compute.calls
    assert status.comparison_trial == (1 if illegal == 0 else 0)
