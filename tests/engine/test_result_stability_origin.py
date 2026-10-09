"""擺位移位出處的封閉名稱、舊文件逐位相容與真正子行程核出處。"""
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError

from aosr.reporting.result import ResultOrigin, SchemeResult
from aosr.search.placement_stability import SHIFT_NAMES
from aosr.search.run import ComputeFailed
from tests.engine.test_search_worker import setup_worker


@pytest.mark.parametrize("number", [None, 7])
def test_new_origin_accepts_exact_shift_names(number: int | None) -> None:
    from typing import get_args
    from aosr.reporting.result import PlacementShiftName
    assert set(get_args(PlacementShiftName)) == set(SHIFT_NAMES)
    for name in SHIFT_NAMES:
        document = {"kind": "search_placement_shift", "search_id": "search", "trial_number": number, "shift_name": name}
        assert ResultOrigin.model_validate(document).model_dump() == document
    for changes in ({"search_id": None}, {"search_id": " "}, {"shift_name": None},
                    {"shift_name": "unknown"}, {"trial_number": -1}):
        with pytest.raises(ValidationError):
            ResultOrigin.model_validate(document | changes)


@pytest.mark.parametrize("document", [
    {"kind": "run", "search_id": None, "trial_number": None},
    {"kind": "search_baseline", "search_id": "search", "trial_number": None},
    {"kind": "search_candidate", "search_id": "search", "trial_number": 7},
])
def test_old_origins_and_result_bytes_do_not_gain_fields(tmp_path: Path, document: dict[str, object]) -> None:
    import json
    from tests.engine._search_worker_cases import prepared_result

    origin = ResultOrigin.model_validate(document)
    assert origin.model_dump(mode="json") == document
    assert json.loads(origin.model_dump_json()) == document
    with pytest.raises(ValidationError):
        ResultOrigin.model_validate(document | {"shift_name": SHIFT_NAMES[0]})
    old = prepared_result(tmp_path).model_copy(update={"origin": origin}).model_dump_json()
    assert SchemeResult.model_validate_json(old).model_dump_json() == old


@pytest.mark.parametrize("number", [None, 7])
@pytest.mark.parametrize("shift_name", ["ear_up", "speakers_forward"])
def test_worker_shift_flag_and_origin_roundtrip(tmp_path: Path, number: int | None, shift_name: str) -> None:
    worker, jobs = setup_worker(tmp_path, {}, (number,))
    job = replace(jobs[0], shift_name=shift_name)
    computed, = worker((job,), 1)
    saved = SchemeResult.model_validate_json(job.result_path.read_bytes())
    assert computed.job == job
    assert saved.origin.model_dump() == {"kind": "search_placement_shift", "search_id": "search",
        "trial_number": number, "shift_name": shift_name}


@pytest.mark.parametrize("mismatch", ["wrong_kind", "wrong_shift", "candidate_origin", "wrong_origin", "wrong_trial"])
def test_worker_rejects_wrong_shift_origin(tmp_path: Path, mismatch: str) -> None:
    worker, jobs = setup_worker(tmp_path, {"7": {mismatch: True}}, (7,))
    with pytest.raises(ComputeFailed, match="結果讀回失敗"):
        list(worker((replace(jobs[0], shift_name="ear_up"),), 1))


def test_shift_child_keeps_folder_lock_after_parent_descriptor_closes(tmp_path: Path) -> None:
    import fcntl
    import os
    import time
    worker, jobs = setup_worker(tmp_path, {"7": {"sleep": 1.0}}, (7,))
    descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    worker.lock_fd = descriptor
    child = worker._start(replace(jobs[0], shift_name="ear_up"))
    os.close(descriptor)
    contender = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        deadline = time.monotonic() + 10
        while not (tmp_path / "event-7").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert (tmp_path / "event-7").is_file()
        with pytest.raises(BlockingIOError):
            fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        child.process.wait(timeout=10)
        assert worker._result(child).job.shift_name == "ear_up"
        fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        child.process.kill()
        child.process.wait(timeout=10)
        os.close(contender)
