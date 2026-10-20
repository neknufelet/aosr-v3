"""真子行程替身驗兩波指令與結果核對；不求解。"""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import pytest

from aosr.reporting.result import ResultOrigin
from aosr.reporting.stage_one_result import run_stage_one
from aosr.search.run import CandidateJob, ComputeFailed, IdentityChanged
from aosr.search.stage_one_worker import StageOneCompute
from aosr.search.store import SearchIdentity, SearchStore
from aosr.search.worker import _Active
from tests.engine import _stage_one_cases as cases
from tests.engine._search_store_cases import purpose_settings

SCRIPT = '''
import json, sys
from pathlib import Path
phase = sys.argv[1]
args = sys.argv[2:]
root = Path(args[args.index('--capabilities') + 1])
out = Path(args[args.index('--out') + 1])
if (root / (phase + '-changed')).exists():
    sys.stderr.write('AOSR_IDENTITY_CHANGED\\n')
    sys.exit(3)
(root / (phase + '-' + out.stem)).write_text(json.dumps(args))
if phase == 'fem-slice':
    if (root / 'fail').exists():
        sys.exit(7)
    out.write_text('{}')
else:
    specific = root / ('template-' + args[args.index('--trial-number') + 1])
    out.write_bytes((specific if specific.exists() else root / 'template').read_bytes())
'''


def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str = "") -> tuple[StageOneCompute, CandidateJob]:
    from aosr.search.stage_one_worker import StageOneCompute
    item = cases.scheme()
    result = run_stage_one(item, shards=cases.shards(monkeypatch, item), origin=ResultOrigin(
        kind="search_candidate", search_id="test", trial_number=0), **cases.run_kwargs())
    if damage == "origin":
        result = result.model_copy(update={"origin": ResultOrigin(kind="run")})
    elif damage == "scheme":
        result = result.model_copy(update={"scheme": item.model_copy(update={"scheme_id": "other"}),
            "evaluation": result.evaluation.model_copy(update={"candidate_id": "other"}),
            "score": result.score.model_copy(update={"candidate_id": "other"})})
    elif damage in ("physics_identity", "program_fingerprint"):
        result = result.model_copy(update={damage: ("phys-v1:" if damage == "physics_identity" else "calc-v1:") + "a" * 64})
    (tmp_path / "template").write_text(result.model_dump_json())
    script = tmp_path / "child.py"
    script.write_text(SCRIPT)
    worker = StageOneCompute(capabilities_path=tmp_path, engine_commit="test", search_id="test", fem_root=tmp_path / "parts",
        identity=SearchIdentity(cases.run_kwargs()["physics_identity"], cases.run_kwargs()["program_fingerprint"], purpose_settings(item.purpose)),
        runner=(sys.executable, str(script), "stage-one"), slice_runner=(sys.executable, str(script), "fem-slice"))
    return worker, CandidateJob(0, item, tmp_path / "result.json")


def _second_job(tmp_path: Path, job: CandidateJob) -> CandidateJob:
    other = CandidateJob(1, job.scheme.model_copy(update={"scheme_id": "other"}), tmp_path / "other.json")
    document = json.loads((tmp_path / "template").read_bytes())
    document["scheme"]["scheme_id"] = "other"
    document["evaluation"]["candidate_id"] = "other"
    document["score"]["candidate_id"] = "other"
    document["origin"]["trial_number"] = 1
    (tmp_path / "template-1").write_text(json.dumps(document))
    return other


def test_two_waves_only_slice_then_stage_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    worker, job = setup(tmp_path, monkeypatch)
    other = _second_job(tmp_path, job)
    results = list(worker((job, other), 2))
    assert {r.job.trial_number for r in results} == {job.trial_number, other.trial_number}
    slices = list(tmp_path.glob("fem-slice-*"))
    assert slices
    argv = json.loads((tmp_path / "stage-one-result").read_text())
    assert argv[argv.index("--search-id") + 1] == "test"
    assert argv[argv.index("--trial-number") + 1] == "0"
    assert "run" not in argv
    parts = argv[argv.index("--fem-parts") + 1:]
    assert set(parts) == {json.loads(p.read_text())[json.loads(p.read_text()).index("--out") + 1] for p in slices}
    other_argv = json.loads((tmp_path / "stage-one-other").read_text())
    assert other_argv[other_argv.index("--fem-parts") + 1:] == parts
    for path in slices:
        arguments = json.loads(path.read_text())
        assert arguments[:arguments.index("--slice")] == [str(SearchStore.scheme_path_for(item.result_path)) for item in (job, other)]


def test_slice_failure_stops_before_second_wave(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    worker, job = setup(tmp_path, monkeypatch)
    (tmp_path / "fail").touch()
    with pytest.raises(ComputeFailed, match="分片"):
        list(worker((job,), 2))
    assert not list(tmp_path.glob("stage-one-*"))


@pytest.mark.parametrize("damage", ["origin", "scheme", "physics_identity", "program_fingerprint"])
def test_wrong_result_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str) -> None:
    worker, job = setup(tmp_path, monkeypatch, damage)
    error = IdentityChanged if damage in ("physics_identity", "program_fingerprint") else ComputeFailed
    with pytest.raises(error):
        list(worker((job,), 1))


@pytest.mark.parametrize("phase", ["fem-slice", "stage-one"])
def test_child_identity_exit_is_interruption(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str) -> None:
    worker, job = setup(tmp_path, monkeypatch)
    (tmp_path / (phase + "-changed")).touch()
    with pytest.raises(IdentityChanged):
        list(worker((job,), 1))


def test_candidate_seconds_stop_when_the_child_exits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 兩個完成通知都排進佇列之後才把時鐘撥快 100 秒；父行程這時才讀第二個，那 100 秒不准算進它的秒數。
    from aosr.search import stage_one_worker
    now = [0.]
    monkeypatch.setattr(stage_one_worker, "_clock", lambda: now[0])
    queued = threading.Semaphore(0)
    put = stage_one_worker._StampedQueue.put
    def counted(self: stage_one_worker._StampedQueue, item: _Active, block: bool = True, timeout: float | None = None) -> None:
        put(self, item, block, timeout)
        queued.release()
    monkeypatch.setattr(stage_one_worker._StampedQueue, "put", counted)
    worker, job = setup(tmp_path, monkeypatch)
    results = worker((job, _second_job(tmp_path, job)), 2)
    assert next(results).seconds == 0.
    assert queued.acquire(timeout=60) and queued.acquire(timeout=60)
    now[0] = 100.
    assert next(results).seconds == 0.
    assert not list(results)
