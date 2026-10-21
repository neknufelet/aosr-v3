"""細算保存、接續、單獨比較身分與命令列；全部用替身。"""
from collections.abc import Iterator, Sequence
from pathlib import Path
import json
import sys

import pytest

from aosr.config.frequency_axis import LowFrequencyAxis
from aosr.config.paths import config_path
from aosr.reporting.stage_one_result import load_stage_one_result, stage_one_comparison_identity
from aosr.search.run import IdentityProbe, CandidateJob, ComputeFailed, IdentityChanged, SearchStatus
from aosr.search.stage_one_worker import StageOneComputed, StageOneCompute, StageOneComputeFunction
from aosr.search.stage_one_refine_record import StageOneRefineStatus
from aosr.search.store import SearchIdentity, SearchStore, refine_scheme_id
from tests.engine import _search_stage_one_cases as search
from tests.engine import _stage_one_cases as stage
from tests.engine._stage_one_refine_cases import pool, snapshot

ROWS = [(8, 1., 5., .5), (2, 2., 8., .6), (1, 3., 4., .7), (5, 4., 7., .8)]


def refine(store: SearchStore, compute: StageOneComputeFunction, probe: IdentityProbe | None = None) -> StageOneRefineStatus:
    from aosr.search.stage_one_refine import refine_stage_one_search
    return refine_stage_one_search(store, compute=compute, probe=probe or (lambda: store.identity),
        registry_path=config_path("quality_targets.toml"), run_date=stage.run_kwargs()["run_date"],
        engine_version=store.identity.program_fingerprint)


def test_selection_saved_before_compute_and_top_only_rerank(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = pool(tmp_path, monkeypatch, ROWS)
    before = snapshot(store)
    fake = search.FakeCompute(store, monkeypatch)
    def compute(jobs: Sequence[CandidateJob], workers: int) -> Iterator[StageOneComputed]:
        doc = json.loads((store.path / "stage_one_refine.json").read_bytes())
        assert doc["state"] == "selected"
        assert {seat["trial_number"] for seat in doc["selection"]["candidates"]} == {n for n, _, _, _ in ROWS}
        for job in jobs:
            assert job.scheme.scene.low_frequency_axis == LowFrequencyAxis.VERIFICATION
            assert job.scheme.scheme_id == refine_scheme_id(store.search_id, job.trial_number)
            assert job.result_path.parent == store.refine_dir
        yield from fake(jobs, workers)
    status = refine(store, compute)
    assert status.state == "done"
    assert status.comparison_trial == min(n for n, _, _, _ in ROWS)
    expected = sorted((load_stage_one_result(store.refine_result_path(n)).score.screening_total, n) for n in (8, 2))
    assert [(s.score, s.trial_number) for s in sorted(status.results, key=lambda s: s.refine_rank or 99) if s.refine_rank] == expected
    assert all(s.refine_rank is None for s in status.results if s.trial_number in (1, 5))
    assert all(s.seconds == .25 for s in status.results)
    after = snapshot(store)
    assert {p: after[p] for p in before} == before
    saved = type(status).model_validate_json((store.path / "stage_one_refine.json").read_bytes())
    assert saved == status
    assert "入選後檢查還沒做" in status.message


def test_different_comparison_identity_excluded_from_rerank(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = pool(tmp_path, monkeypatch, ROWS)
    status = refine(store, search.FakeCompute(store, monkeypatch, different=frozenset({2})))
    bad = next(s for s in status.results if s.trial_number == 2)
    assert status.comparison_trial == 1
    assert bad.state == "incomparable" and bad.refine_rank is None
    assert "合分設定指紋" in bad.reason
    assert next(s for s in status.results if s.trial_number == 8).refine_rank == 1
    assert status.comparison_identity == stage_one_comparison_identity(load_stage_one_result(store.refine_result_path(1)))


def test_unassessed_result_has_no_refine_rank(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = pool(tmp_path, monkeypatch, ROWS)
    status = refine(store, search.FakeCompute(store, monkeypatch, unavailable=frozenset({8})))
    bad = next(s for s in status.results if s.trial_number == 8)
    assert bad.state == "unassessed" and bad.score is None and bad.refine_rank is None
    assert bad.reason


def test_interrupt_resume_reuses_checked_results_and_matches_uninterrupted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = pool(tmp_path, monkeypatch, ROWS)
    fake = search.FakeCompute(store, monkeypatch)
    def cut(jobs: Sequence[CandidateJob], workers: int) -> Iterator[StageOneComputed]:
        iterator = fake(jobs, workers)
        yield next(iterator)
        raise KeyboardInterrupt
    stopped = refine(store, cut)
    assert stopped.state == "interrupted"
    first = fake.calls[0]
    assert store.refine_result_path(first).is_file()
    resumed = search.FakeCompute(store, monkeypatch)
    status = refine(store, resumed)
    assert first not in resumed.calls
    assert set(fake.calls + resumed.calls) == {n for n, _, _, _ in ROWS}
    # 同一資料夾清掉自建細算結果，再一次跑完：摘要每一位相同（沒有時計時間戳）。
    for path in store.refine_dir.iterdir():
        path.unlink()
    (store.path / "stage_one_refine.json").unlink()
    complete = refine(store, search.FakeCompute(store, monkeypatch))
    assert status.model_dump_json() == complete.model_dump_json()


def test_changed_selection_rejected_without_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = pool(tmp_path, monkeypatch, ROWS)
    refine(store, search.FakeCompute(store, monkeypatch))
    lines = store.ledger_path.read_text().splitlines()
    doc = json.loads(lines[1]); doc["score"] += .1; lines[1] = json.dumps(doc)
    store.ledger_path.write_text("\n".join(lines) + "\n")
    before = snapshot(store)
    with pytest.raises(ValueError, match="挑席跟上次不同，搜尋帳本或設定被改過"):
        refine(store, search.FakeCompute(store, monkeypatch))
    assert snapshot(store) == before


@pytest.mark.parametrize("state", ["running", "interrupted", "failed", "user_stopped"])
def test_unfinished_search_rejected_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str) -> None:
    store = pool(tmp_path, monkeypatch, ROWS)
    store.status_path.write_text(SearchStatus.model_validate({"state": state}).model_dump_json())
    before = snapshot(store)
    with pytest.raises(ValueError, match="先把搜尋跑完"):
        refine(store, search.FakeCompute(store, monkeypatch))
    assert snapshot(store) == before


def test_missing_settings_cli_rejected_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                               capsys: pytest.CaptureFixture[str]) -> None:
    from aosr.search.cli import main
    store = pool(tmp_path, monkeypatch, ROWS, enabled=False)
    before = snapshot(store)
    with pytest.raises(ValueError, match="這個第一階段搜尋沒有設定細算（stage_one_refine），不能細算"):
        refine(store, search.FakeCompute(store, monkeypatch))
    code = main(["refine", str(store.path), "--engine-commit", "test"])
    assert code == 1
    assert "這個第一階段搜尋沒有設定細算（stage_one_refine），不能細算" in capsys.readouterr().err
    assert snapshot(store) == before


@pytest.mark.parametrize("error,state", [(KeyboardInterrupt(), "interrupted"), (IdentityChanged("changed"), "interrupted"),
    (ComputeFailed("bad compute"), "failed"), (RuntimeError("other"), "failed")])
def test_error_states_and_stale_stop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: BaseException, state: str) -> None:
    store = pool(tmp_path, monkeypatch, ROWS)
    store.refine_stop_path.touch()
    def failing(jobs: Sequence[CandidateJob], workers: int) -> Iterator[StageOneComputed]:
        raise error
        yield
    status = refine(store, failing)
    assert status.state == state
    assert "已刪除殘留的細算停止記號" in status.message
    assert not store.refine_stop_path.exists()
    assert str(error) in status.message


def test_stale_stop_runs_to_completion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = pool(tmp_path, monkeypatch, ROWS)
    store.refine_stop_path.touch()
    status = refine(store, search.FakeCompute(store, monkeypatch))
    assert status.state == "done" and "已刪除殘留的細算停止記號" in status.message
    assert not store.refine_stop_path.exists()


def test_probe_change_opens_no_compute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = pool(tmp_path, monkeypatch, ROWS)
    fake = search.FakeCompute(store, monkeypatch)
    changed = SearchIdentity(store.identity.physics_identity, "different-program", store.identity.purpose_settings)
    status = refine(store, fake, lambda: changed)
    assert status.state == "interrupted" and "program_fingerprint" in status.message
    assert not fake.calls and not store.refine_dir.exists()


def test_atomic_write_failure_keeps_original(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.search.stage_one_refine_record import save_stage_one_refine
    store = pool(tmp_path, monkeypatch, ROWS)
    status = refine(store, search.FakeCompute(store, monkeypatch))
    path = store.path / "stage_one_refine.json"
    before = path.read_bytes()
    import os
    original = os.fdopen
    class Broken:
        def __init__(self, descriptor: int) -> None:
            self.handle = original(descriptor, "w", encoding="utf-8")
        def __enter__(self) -> "Broken":
            return self
        def __exit__(self, *args: object) -> None:
            self.handle.close()
        def write(self, text: str) -> None:
            self.handle.write(text[:len(text) // 2])
            raise OSError("partial write")
    monkeypatch.setattr("aosr.search.stage_one_refine_record.os.fdopen", lambda descriptor, *a, **k: Broken(descriptor))
    with pytest.raises(OSError, match="partial write"):
        save_stage_one_refine(store, status)
    assert path.read_bytes() == before
    assert not list(store.path.glob(".stage_one_refine.json.*"))


def test_two_wave_subcommands_and_verification_schemes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.engine.test_search_stage_one_worker import SCRIPT
    store = pool(tmp_path, monkeypatch, ROWS)
    # 先以替身製造驗證軸結果供子行程複製，之後刪除測試自己建立的細算檔。
    prepared = refine(store, search.FakeCompute(store, monkeypatch))
    root = tmp_path / "commands"; root.mkdir()
    for entry in prepared.results:
        (root / f"template-{entry.trial_number}").write_bytes(store.refine_result_path(entry.trial_number).read_bytes())
    for path in store.refine_dir.iterdir():
        path.unlink()
    (store.path / "stage_one_refine.json").unlink()
    script = root / "child.py"; script.write_text(SCRIPT)
    worker = StageOneCompute(capabilities_path=root, engine_commit="test", search_id=store.search_id,
        fem_root=store.fem_path, identity=store.identity, runner=(sys.executable, str(script), "stage-one"),
        slice_runner=(sys.executable, str(script), "fem-slice"), poll_s=.01)
    status = refine(store, worker)
    assert status.state == "done"
    assert list(root.glob("fem-slice-*"))
    for entry in status.results:
        argv = json.loads((root / f"stage-one-trial-{entry.trial_number:06d}").read_bytes())
        assert "--fem-parts" in argv and "run" not in argv
        scheme_path = Path(argv[0])
        assert scheme_path.parent == store.refine_dir
        scheme = json.loads(scheme_path.read_bytes())
        assert scheme["scene"]["low_frequency_axis"] == "verification_linear_1hz"
        assert scheme["scheme_id"] == refine_scheme_id(store.search_id, entry.trial_number)


def test_cli_refine_dispatch_and_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.search import cli
    store = pool(tmp_path, monkeypatch, ROWS)
    monkeypatch.setattr(cli, "_identity", lambda *a: store.identity)
    assert not cli.main(["refine", str(store.path), "--engine-commit", "test"],
        compute_factory=lambda *a: search.FakeCompute(store, monkeypatch))
    assert json.loads((store.path / "stage_one_refine.json").read_bytes())["state"] == "done"


@pytest.mark.parametrize("error,expected", [(KeyboardInterrupt(), 3), (ComputeFailed("failure"), 1)])
def test_cli_error_exit_preserves_search_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                              error: BaseException, expected: int) -> None:
    from aosr.search import cli
    store = pool(tmp_path, monkeypatch, ROWS)
    before = snapshot(store)
    monkeypatch.setattr(cli, "_identity", lambda *a: store.identity)
    def compute(jobs: Sequence[CandidateJob], workers: int) -> Iterator[StageOneComputed]:
        raise error
        yield
    code = cli.main(["refine", str(store.path), "--engine-commit", "test"], compute_factory=lambda *a: compute)
    assert code == expected
    after = snapshot(store)
    assert {p: after[p] for p in before} == before


@pytest.mark.parametrize("damage", ["missing", "wrong_origin", "changed_identity", "search_axis", "bad_peak_index",
    "bad_peak_index_after_unmeasured"])
def test_resume_recomputes_bad_saved_verification(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str) -> None:
    store = pool(tmp_path, monkeypatch, ROWS)
    refine(store, search.FakeCompute(store, monkeypatch))
    target = store.refine_result_path(8)
    if damage == "missing":
        target.unlink()
    elif damage == "search_axis":
        # 搜尋軸那份改成細算代號：出處、代號、身分都對得上，只有軸與方案不同，接續必須重算、不准沿用。
        document = json.loads(store.candidate_path(8).read_bytes())
        renamed = refine_scheme_id(store.search_id, 8)
        document["scheme"]["scheme_id"] = renamed
        document["evaluation"]["candidate_id"] = renamed
        document["score"]["candidate_id"] = renamed
        target.write_text(json.dumps(document))
        assert load_stage_one_result(target).low_frequency_axis != LowFrequencyAxis.VERIFICATION
    elif damage in ("bad_peak_index", "bad_peak_index_after_unmeasured"):
        document = json.loads(target.read_bytes())
        speakers = document["evaluation"]["speakers"]
        if damage == "bad_peak_index":
            speakers[0]["payload"]["features"] = []
        else:  # 第一支量不到、第二支索引壞：不准因為先看到量不到就放過
            speakers[0]["payload"]["strongest_peak_index"] = None
            speakers[1]["payload"]["features"] = []
        target.write_text(json.dumps(document))
        load_stage_one_result(target)  # 模型讀得回；要靠細算自己的讀回核對抓到峰谷摘要壞了
    else:
        document = json.loads(target.read_bytes())
        if damage == "wrong_origin":
            document["origin"]["trial_number"] = 999
        else:
            document["program_fingerprint"] = "calc-v1:" + "a" * 64
        target.write_text(json.dumps(document))
    fake = search.FakeCompute(store, monkeypatch)
    status = refine(store, fake)
    assert status.state == "done"
    assert fake.calls == [8]


def test_rerank_follows_1hz_score_not_number_or_search_rank(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [(8, 1., 5., .5), (2, 2., 8., .6), (12, 3., 4.5, .7), (5, 4., 7., .8), (3, 5., 6., .4)]
    store = pool(tmp_path, monkeypatch, rows, top_n=4)
    status = refine(store, search.FakeCompute(store, monkeypatch))
    top = [8, 2, 12, 5]
    scores = {n: load_stage_one_result(store.refine_result_path(n)).score.screening_total for n in top}
    expected = sorted(top, key=lambda n: (scores[n], n))
    # 前提：替身真的分得出，而且答案跟「照編號」「照搜尋名次」都不同，排錯才會紅。
    assert len(set(scores.values())) == len(top)
    assert expected not in (sorted(top), top)
    ranked = sorted((s for s in status.results if s.refine_rank is not None), key=lambda s: s.refine_rank or 0)
    assert [s.trial_number for s in ranked] == expected


def test_killed_after_result_saved_lists_untimed_seat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.search.stage_one_refine_report import stage_one_refine_lines
    store = pool(tmp_path, monkeypatch, ROWS)
    fake = search.FakeCompute(store, monkeypatch)
    def killed(jobs: Sequence[CandidateJob], workers: int) -> Iterator[StageOneComputed]:
        next(fake(jobs, workers))  # 結果檔已存好，計算器還沒交回、摘要還沒記到秒數就被砍
        raise KeyboardInterrupt
        yield
    assert refine(store, killed).state == "interrupted"
    status = refine(store, search.FakeCompute(store, monkeypatch))
    assert status.state == "done"
    assert next(s for s in status.results if s.trial_number == fake.calls[0]).seconds is None
    text = "\n".join(stage_one_refine_lines(store))
    assert f"總秒數：{.25 * (len(ROWS) - 1):.6f} 秒" in text and "另有 1 份沒有計時紀錄" in text
