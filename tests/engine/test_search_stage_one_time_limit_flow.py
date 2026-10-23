"""暫停先存快照、部分批落帳、命令列與只讀投影；計算替身不求解。"""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from aosr.config.paths import config_path
from aosr.search import ledger
from aosr.search.run import CandidateJob
from aosr.search.stage_one_worker import StageOneComputed
from aosr.search.store import SearchStore
from tests.engine import _search_stage_one_cases as search
from tests.engine import _stage_one_cases as cases


def pause_record(jobs: Sequence[CandidateJob]) -> dict[str, object]:
    return {"cap_s": 600., "phase": "candidates", "candidates": [
        {"trial_number": job.trial_number, "state": "交回" if job is jobs[0] else "超限",
         "slice_s": 100., "candidate_s": .25 if job is jobs[0] else 501.,
         "service_s": 100.25 if job is jobs[0] else 601.} for job in jobs],
        "groups": [{"trial_numbers": [job.trial_number for job in jobs], "slices": 2,
            "completed_slices": 2, "longest_slice_s": 100., "window_s": 110., "queue_s": 10.}]}


class PausingCompute:
    def __init__(self, store: SearchStore, monkeypatch: pytest.MonkeyPatch, *, pin_first: bool) -> None:
        self.fake = search.FakeCompute(store, monkeypatch)
        self.pin_first = pin_first
        self.batches: list[tuple[CandidateJob, ...]] = []
        self.yielded: list[StageOneComputed] = []

    def __call__(self, jobs: Sequence[CandidateJob], workers: int) -> Iterator[StageOneComputed]:
        from aosr.search.stage_one_time_limit import StageOneTimeLimit
        from aosr.search.stage_one_worker import CandidateTimeLimit
        first = not self.batches
        self.batches.append(tuple(jobs))
        targets = jobs if first and self.pin_first else jobs[:1]
        for result in self.fake(targets, workers):
            self.yielded.append(result)
            yield result
        if not (first and self.pin_first):
            raise CandidateTimeLimit(StageOneTimeLimit.model_validate(pause_record(jobs)))


@pytest.mark.parametrize("pin_first,expected_batch", [(False, 0), (True, 1)])
def test_partial_batch_saves_record_before_paused_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                       pin_first: bool, expected_batch: int) -> None:
    from aosr.search.stage_one_run import _StageOneRunner, start_stage_one_search, resume_stage_one_search
    from aosr.search.stage_one_time_limit import StageOneTimeLimit
    from aosr.search.run import SearchStatus, State
    from aosr.search.sampler import Outcome, SamplerAdapter
    from collections.abc import Mapping
    store = search.make_store(tmp_path, workers=2)
    compute = PausingCompute(store, monkeypatch, pin_first=pin_first)
    save = _StageOneRunner.save
    seen: list[StageOneTimeLimit] = []
    told: list[dict[int, object]] = []
    tell = SamplerAdapter.tell_batch
    def capture_tell(self: SamplerAdapter, outcomes: Mapping[int, Outcome]) -> None:
        told.append(dict(outcomes))
        tell(self, outcomes)
    def ordered_save(self: _StageOneRunner, *, state: State | None = None, message: str | None = None) -> SearchStatus:
        if state == "paused_time_limit":
            seen.append(StageOneTimeLimit.model_validate_json((store.path / "stage_one_time_limit.json").read_bytes()))
        return save(self, state=state, message=message)
    monkeypatch.setattr(_StageOneRunner, "save", ordered_save)
    monkeypatch.setattr(SamplerAdapter, "tell_batch", capture_tell)
    status = start_stage_one_search(store, compute=compute, probe=lambda: store.identity,
        registry_path=config_path("quality_targets.toml"), run_date=cases.run_kwargs()["run_date"],
        engine_version=store.identity.program_fingerprint)
    assert status.state == "paused_time_limit"
    paused_jobs = compute.batches[-1]
    assert "600" in status.message and f"試算 {paused_jobs[-1].trial_number}" in status.message
    assert "stage_one_time_limit.json" in status.message
    assert [record.batch for record in seen] == [expected_batch]
    assert seen[0].model_dump(mode="json", exclude={"batch", "schema_version"}) == pause_record(paused_jobs)
    rows = ledger.read_for(store).rows
    if pin_first:
        assert {row.trial_number for row in rows} == {result.job.trial_number for result in compute.yielded}
        by_number = {result.job.trial_number: result for result in compute.yielded}
        for row in rows:
            result = by_number[row.trial_number]
            assert row.seconds == result.seconds
            assert row.score is not None and result.result.score.screening_total is not None
            assert row.score.hex() == result.result.score.screening_total.hex()
        assert [set(batch) for batch in told] == [{job.trial_number for job in compute.batches[0]}]
    else:
        assert not rows and not told
    with pytest.raises(ValueError, match="paused_time_limit"):
        resume_stage_one_search(store, compute=compute, probe=lambda: store.identity,
            registry_path=config_path("quality_targets.toml"), run_date=cases.run_kwargs()["run_date"],
            engine_version=store.identity.program_fingerprint)


@pytest.mark.parametrize("command", ["start", "resume", "auto"])
def test_cli_paused_search_exit_four_without_refinement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                       command: str) -> None:
    from aosr.search import cli, stage_one_refine, stage_one_post_check
    from aosr.search.run import SearchStatus
    store = search.make_store(tmp_path, workers=2)
    monkeypatch.setattr(cli, "_identity", lambda *a: store.identity)
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("暫停不可開細算或入選後檢查")
    monkeypatch.setattr(stage_one_refine, "refine_stage_one_search", forbidden)
    monkeypatch.setattr(stage_one_post_check, "run_stage_one_post_check", forbidden)
    compute = PausingCompute(store, monkeypatch, pin_first=False)
    monkeypatch.setattr(cli, "_create", lambda *a: store)
    if command != "start":
        with pytest.raises(search.Killed):
            search.run(store, search.FakeCompute(store, monkeypatch, kill_after=0))
    extra = ["--modal-cache-dir", str(tmp_path / "cache")] if command == "auto" else []
    target = ["--project", str(tmp_path / "project"), "--settings", str(tmp_path / "settings"),
        "--root", str(tmp_path / "searches")] if command == "start" else [str(store.path)]
    code = cli.main([command, *target, "--engine-commit", "test", *extra], compute_factory=lambda *a: compute)
    assert code == 4
    status = SearchStatus.model_validate_json(store.status_path.read_bytes())
    assert status.state == "paused_time_limit"
    if command == "auto":
        assert status.outer.conclusion == "search_paused_time_limit"
        assert status.outer.message == "單題搜尋成本超過上限，搜尋暫停，未完成"


def test_report_snapshot_and_web_stopping_phase(tmp_path: Path) -> None:
    from aosr.search.run import SearchStatus
    from aosr.search.stage_one_time_limit import StageOneTimeLimit, save_time_limit
    from aosr.search.stage_one_report import build_stage_one_report, render_stage_one_text
    from aosr.gui.search_view import build_search_view
    store = search.make_store(tmp_path)
    ledger.create_for(store)
    store.status_path.write_text(SearchStatus(state="paused_time_limit", message="單題搜尋成本超限").model_dump_json())
    record = StageOneTimeLimit.model_validate({"cap_s": 600., "phase": "candidates", "batch": 2,
        "candidates": [
            {"trial_number": 5, "state": "超限", "slice_s": 100., "candidate_s": 501., "service_s": 601.},
            {"trial_number": 6, "state": "一起停下", "slice_s": 100., "candidate_s": 2., "service_s": 102.},
            {"trial_number": 7, "state": "沒開始", "slice_s": 100., "service_s": 100.}],
        "groups": [{"trial_numbers": [5, 6, 7], "slices": 2, "completed_slices": 2,
            "longest_slice_s": 100., "window_s": 110., "queue_s": 10.}]})
    before = render_stage_one_text(build_stage_one_report(store))
    save_time_limit(store, record)
    text = render_stage_one_text(build_stage_one_report(store))
    assert "單題搜尋成本上限：600.0 秒" in text
    assert "試算 5：超限；服務 601.0 秒（分片 100.0 秒／自己 501.0 秒）" in text
    assert "試算 6：一起停下" in text and "試算 7：沒開始" in text
    assert "分片組 5,6,7：排隊 10.0 秒" in text
    (store.path / "stage_one_time_limit.json").unlink()
    assert render_stage_one_text(build_stage_one_report(store)) == before
    view = build_search_view(store.path, server_physics=store.identity.physics_identity,
        server_program=store.identity.program_fingerprint)
    reasons = next(block for block in view.blocks if block.key == "reasons")
    assert "停止段落：搜尋因單題時間上限暫停" in reasons.lines


def test_broken_pause_record_only_breaks_its_own_section(tmp_path: Path) -> None:
    from aosr.search.run import SearchStatus
    from aosr.search.stage_one_report import build_stage_one_report, render_stage_one_text
    store = search.make_store(tmp_path)
    ledger.create_for(store)
    store.status_path.write_text(SearchStatus(state="paused_time_limit", message="單題搜尋成本超限").model_dump_json())
    before = render_stage_one_text(build_stage_one_report(store))
    (store.path / "stage_one_time_limit.json").write_text("{")
    text = render_stage_one_text(build_stage_one_report(store))
    lines, expected = text.splitlines(), before.splitlines()
    start = next(index for index, line in enumerate(lines) if line.startswith("單題搜尋成本上限紀錄讀不回："))
    # 錯誤原文可能跨好幾行；拿掉那一段之後，其餘逐行等於沒有紀錄時的報告。
    end = start + len(lines) - len(expected)
    assert end > start and lines[:start] + lines[end:] == expected


def test_empty_report_without_record_stays_byte_identical() -> None:
    from aosr.search.run import SearchStatus
    from aosr.search.stage_one_report import StageOneSearchReport, render_stage_one_text
    report = StageOneSearchReport("牆面假設", SearchStatus(), (), ())
    assert render_stage_one_text(report) == (
        "牆面假設\n未加入實際喇叭的低頻衰減，排名只適用於本次理想聲源基準；換播放系統後可能不同\n低頻拖尾不進自動排名，入選後要另外檢查\n"
        "停止原因：進行中；搜尋進行中\n外圈結論：未判定\n"
        "問過 0；算完 0；不合法 0（各原因：沒有）；不可估 0；不能同表 0\n"
        "名次｜試算編號｜離前牆／喇叭間距／聆聽距離（公尺）｜合分\n"
        "每支喇叭：主項／穩健性；最強峰／最深谷（頻率 Hz、高度 dB 對自己那條直線、寬度八度）\n"
        "不可排名候選（不給名次）\n300 Hz 交接敏感度（不改名次）\n"
        "範圍外：第一階段只算有限元素（20～300 Hz），沒有 300 Hz 交接，這一項不適用\n"
        "擺位穩定性（±2 公分，不改名次）\n"
        "範圍外：擺位穩定性是第二階段的檢查；第一階段的座位穩健性已在合分裡（主位前後左右上下六點）\n")


def test_second_stage_report_accepts_paused_state(tmp_path: Path) -> None:
    from aosr.search.run import SearchStatus
    from aosr.search.report import build_report, render_text
    from tests.engine._search_run_cases import make_store, RUN_DATE
    store, registry = make_store(tmp_path)
    ledger.create_for(store)
    store.status_path.write_text(SearchStatus(state="paused_time_limit").model_dump_json())
    assert "因單題時間上限暫停" in render_text(build_report(store, quality_targets_path=registry, run_date=RUN_DATE))


# 搜尋那一個帶設定檔的單題上限（#902：沒寫照老闆 10-22 的 600 秒），細算那一個不帶。
@pytest.mark.parametrize("configured", [None, 420.])
@pytest.mark.parametrize("command,expected_caps", [
    ("start", ["search"]), ("resume", ["search"]), ("refine", [None]), ("auto", ["search", None])])
def test_cli_constructs_distinct_search_and_refine_computes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str,
                                                          expected_caps: list[str | None], configured: float | None) -> None:
    from types import SimpleNamespace
    from aosr.search import cli, stage_one_worker, stage_one_refine, stage_one_outer, stage_one_run
    from aosr.search.run import SearchStatus
    from aosr.search.stage_one_worker import StageOneComputeFunction
    store = search.make_store(tmp_path)
    from aosr.search.settings import SearchSettings
    store = SearchStore.create(tmp_path / "configured", project=store.project,
        settings=SearchSettings.model_validate(store.settings.model_dump(mode="json") | {
            "stage_one_refine": {"top_n": 2, "far_control_min_front_distance_m": .8}}
            | ({} if configured is None else {"stage_one_candidate_cap_s": configured})),
        identity=store.identity, versions={"python": "test", "numpy": "test", "optuna": "test"})
    constructed: list[RecordingCompute] = []
    used: list[StageOneComputeFunction] = []
    class RecordingCompute:
        def __init__(self, *, search_cap_s: float | None, **kwargs: object) -> None:
            self.cap = search_cap_s
            constructed.append(self)
        def __call__(self, jobs: Sequence[CandidateJob], workers: int) -> Iterator[StageOneComputed]:
            pytest.fail("入口替身應接走計算")
            yield
    def search_entry(opened: SearchStore, *, compute: StageOneComputeFunction, **kwargs: object) -> SearchStatus:
        used.append(compute)
        return SearchStatus(state="paused_time_limit")
    def refine_entry(opened: SearchStore, *, compute: StageOneComputeFunction, **kwargs: object) -> SimpleNamespace:
        used.append(compute)
        return SimpleNamespace(state="interrupted")
    def auto_entry(opened: SearchStore, *, compute: StageOneComputeFunction,
                   refine_compute: StageOneComputeFunction, **kwargs: object) -> SearchStatus:
        used.extend((compute, refine_compute))
        return SearchStatus.model_validate({"outer": {"conclusion": "search_paused_time_limit"}})
    monkeypatch.setattr(cli, "_identity", lambda *a: store.identity)
    monkeypatch.setattr(cli, "_create", lambda *a: store)
    monkeypatch.setattr(stage_one_worker, "StageOneCompute", RecordingCompute)
    for name in ("start_stage_one_search", "resume_stage_one_search"):
        monkeypatch.setattr(stage_one_run, name, search_entry)
    monkeypatch.setattr(stage_one_refine, "refine_stage_one_search", refine_entry)
    monkeypatch.setattr(stage_one_outer, "auto_stage_one_search", auto_entry)
    target = ["--project", str(tmp_path / "project"), "--settings", str(tmp_path / "settings"),
        "--root", str(tmp_path / "searches")] if command == "start" else [str(store.path)]
    extra = ["--modal-cache-dir", str(tmp_path / "cache")] if command in ("refine", "auto") else []
    code = cli.main([command, *target, "--engine-commit", "test", *extra])
    assert code == (3 if command == "refine" else 4)
    search_cap = 600. if configured is None else configured
    assert [compute.cap for compute in constructed] == [search_cap if cap == "search" else None for cap in expected_caps]
    assert used == constructed


def test_injected_factory_called_once_and_shared_by_auto(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.search import cli, stage_one_outer
    from aosr.search.run import SearchStatus
    from aosr.search.stage_one_worker import StageOneComputeFunction
    store = search.make_store(tmp_path)
    fake = search.FakeCompute(store, monkeypatch)
    factory_calls: list[tuple[SearchStore, Path, str]] = []
    routed: list[StageOneComputeFunction] = []
    def factory(opened: SearchStore, capabilities: Path, commit: str) -> StageOneComputeFunction:
        factory_calls.append((opened, capabilities, commit))
        return fake
    def entry(opened: SearchStore, *, compute: StageOneComputeFunction,
              refine_compute: StageOneComputeFunction, **kwargs: object) -> SearchStatus:
        routed.extend((compute, refine_compute))
        return SearchStatus.model_validate({"outer": {"conclusion": "search_paused_time_limit"}})
    monkeypatch.setattr(stage_one_outer, "auto_stage_one_search", entry)
    capabilities = tmp_path / "capabilities"
    code = cli.main(["auto", str(store.path), "--engine-commit", "test", "--capabilities", str(capabilities),
        "--modal-cache-dir", str(tmp_path / "cache")], compute_factory=factory)
    assert code == 4
    assert [(opened.path, cap, commit) for opened, cap, commit in factory_calls] == [(store.path, capabilities, "test")]
    assert routed == [fake, fake]


def test_auto_routes_refine_to_separate_uncapped_compute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.search.stage_one_outer import auto_stage_one_search
    from tests.engine._stage_one_auto_cases import running
    store = running(tmp_path, monkeypatch)
    fake = search.FakeCompute(store, monkeypatch)
    searched: list[int | None] = []
    refined: list[int | None] = []
    def compute(jobs: Sequence[CandidateJob], workers: int) -> Iterator[StageOneComputed]:
        assert all(job.result_path.parent == store.candidate_path(0).parent for job in jobs)
        searched.extend(job.trial_number for job in jobs)
        yield from fake(jobs, workers)
    def refine(jobs: Sequence[CandidateJob], workers: int) -> Iterator[StageOneComputed]:
        assert all(job.result_path.parent == store.refine_dir for job in jobs)
        refined.extend(job.trial_number for job in jobs)
        yield from fake(jobs, workers)
    status = auto_stage_one_search(store, compute=compute, refine_compute=refine, probe=lambda: store.identity,
        registry_path=config_path("quality_targets.toml"), run_date=cases.run_kwargs()["run_date"],
        engine_version=store.identity.program_fingerprint)
    assert status.outer.conclusion == "complete" and searched and refined
