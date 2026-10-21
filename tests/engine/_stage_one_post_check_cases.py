"""第一階段入選後檢查的明列診斷與串行子行程替身。"""
from pathlib import Path
import json
import sys

import pytest

from aosr.reporting.modal_diagnosis_model import ModalDiagnosis
from aosr.reporting.scheme import load_scheme
from aosr.search.store import SearchStore
from tests.engine import _modal_cases as modal, _search_stage_one_cases as search
from tests.engine._stage_one_refine_cases import pool
from tests.engine.test_search_stage_one_refine import ROWS, refine


def prepared(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SearchStore:
    store = pool(tmp_path, monkeypatch, ROWS)
    refine(store, search.FakeCompute(store, monkeypatch))
    return store


def diagnosis(store: SearchStore, number: int) -> ModalDiagnosis:
    scheme = load_scheme(store.scheme_path_for(store.refine_result_path(number)))
    doc = modal.sample(scheme)[0].model_dump(mode="json")
    for row, t60 in zip(doc["room_layer"]["modes"], (9., 1.6, 2.0, 8.), strict=True):
        row["t60_s"] = t60
    # 一根超線共振的大小為零，報告不得偽造有限相對分貝；最強那根（範圍外）隨試算編號變，各席相對值才不同。
    for pair in doc["placement_layer"]["pairs"]:
        pair["resonance_magnitude"] = [number + 5., 4., 0., 2.]
    return ModalDiagnosis.model_validate(doc)


def runner(root: Path, store: SearchStore, *, state: str = "diagnosed_not_scored", code: int = 0,
           stop_after: bool = False, lock_fd: int | None = None) -> tuple[str, ...]:
    root.mkdir(parents=True, exist_ok=True)
    for number, _, _, _ in ROWS:
        if not store.scheme_path_for(store.refine_result_path(number)).exists():
            continue
        value = diagnosis(store, number) if state == "diagnosed_not_scored" else ModalDiagnosis.model_validate({
            "state": state, **({"reason_text": "替身沒有計算"} if state in ("failed", "not_computed") else
                              {"reason_code": "unsupported_room"})})
        (root / f"{number}.json").write_text(value.model_dump_json())
    script = root / "child.py"
    script.write_text("import os,sys,json,shutil,time\nfrom pathlib import Path\n"
        f"root=Path({str(root)!r})\n"
        "active=root/'active'\nfd=os.open(active,os.O_CREAT|os.O_EXCL|os.O_WRONLY)\n"
        "scheme=Path(sys.argv[1])\nnumber=int(scheme.name.split('-')[1])\n"
        "with (root/'calls').open('a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n"
        "assert os.environ['OMP_NUM_THREADS']=='1'\nassert os.getsid(0)==os.getpid()\n"
        f"if {lock_fd!r} is not None: os.fstat({lock_fd!r})\n"
        "out=Path(sys.argv[sys.argv.index('--out')+1])\n"
        "shutil.copyfile(root/f'{number}.json',out)\ntime.sleep(.01)\n"
        f"if {stop_after!r}: Path({str(store.refine_stop_path)!r}).touch()\n"
        "os.close(fd)\nactive.unlink()\n"
        f"sys.stderr.write('替身錯誤原文'); sys.stderr.flush(); sys.exit({code})\n")
    return sys.executable, str(script)


def calls(root: Path) -> list[list[str]]:
    path = root / "calls"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
