"""搜尋子行程的小替身；合法結果沿用既有假物理產物。"""

from functools import lru_cache
from pathlib import Path

from aosr.reporting.result import SchemeResult
from tests.engine.test_scheme_pipeline import _control_result


@lru_cache
def _template() -> SchemeResult:
    return _control_result("wall-1")


def prepared_result(tmp_path: Path) -> SchemeResult:
    result = _template()
    path = tmp_path / "template"
    path.write_text(result.model_dump_json(), encoding="utf-8")
    return result


SCRIPT = '''
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("scheme")
parser.add_argument("--out")
parser.add_argument("--engine-commit")
parser.add_argument("--capabilities")
parser.add_argument("--search-id")
parser.add_argument("--trial-number", type=int)
args = parser.parse_args()
root = Path(args.capabilities)
number = str(args.trial_number)
options = json.loads((root / "options").read_text()).get(number, {})
out = Path(args.out)
event = root / ("event-" + number)


def publish(record):
    # 先寫暫存再換名：考卷輪詢事件檔時不會讀到清空或寫一半的檔。
    temporary = root / ("tmp-event-" + number)
    temporary.write_text(json.dumps(record))
    os.replace(temporary, event)


child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"]) if options.get("child") else None
record = {"start": time.monotonic(), "pid": os.getpid(), "child": child.pid if child else None,
          "env": dict(os.environ), "cwd": str(Path.cwd())}
publish(record)
time.sleep(options.get("sleep", 0))
sys.stderr.write(options.get("stderr", ""))
if options.get("exit"):
    sys.exit(options["exit"])
document = json.loads((root / "template").read_text())
scheme = json.loads(Path(args.scheme).read_text())
name = "wrong" if options.get("wrong_id") else scheme["scheme_id"]
document["scheme"]["scheme_id"] = name
def rename(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "candidate_id":
                value[key] = name
            else:
                rename(item)
    elif isinstance(value, list):
        for item in value:
            rename(item)
rename(document["candidate"])
document["origin"] = {"kind": "search_baseline" if args.trial_number is None else "search_candidate",
                      "search_id": "wrong" if options.get("wrong_origin") else args.search_id,
                      "trial_number": args.trial_number}
if options.get("wrong_trial"):
    document["origin"]["trial_number"] += 1
if options.get("wrong_kind"):
    document["origin"] = {"kind": "run"}
out.write_text(json.dumps(document))
record["end"] = time.monotonic()
publish(record)
'''


CLI_SCRIPT = '''
import json
import sys
from pathlib import Path
from aosr.search import cli
from aosr.search.run import CandidateJob
from aosr.search.worker import SubprocessCompute
from tests.engine._search_run_cases import FakeCompute

root = Path(sys.argv[1])
def factory(store, capabilities, commit):
    job = CandidateJob(None, store.project, root / "candidate-template")
    computed = next(FakeCompute(store)((job,), 1))
    document = json.loads((root / "template").read_text())
    document["candidate"] = computed.candidate.model_dump(mode="json")
    document["scheme"] = store.project.model_dump(mode="json")
    document["physics_identity"] = store.identity.physics_identity
    document["program_fingerprint"] = store.identity.program_fingerprint
    document["purpose_settings"] = store.identity.purpose_settings.model_dump(mode="json")
    (root / "template").write_text(json.dumps(document))
    return SubprocessCompute(capabilities_path=root, engine_commit=commit,
                             search_id=store.search_id, runner=(sys.executable, str(root / "runner")))

cli._interrupt_on_termination()
raise SystemExit(cli.main(sys.argv[2:], compute_factory=factory))
'''


ENTRY_SCRIPT = '''
import os
import runpy
import signal
import sys
from pathlib import Path
from aosr.search.store import SearchStore

def observe_entry(path):
    observed = []
    for termination in (signal.SIGTERM, signal.SIGHUP):
        try:
            os.kill(os.getpid(), termination)
        except KeyboardInterrupt:
            observed.append(termination.name)
    (path / "entry-signals").write_text("\\n".join(observed))
    raise RuntimeError("entrypoint observed")

SearchStore.open = observe_entry
sys.argv = ["aosr.search.cli", "stop", sys.argv[1]]
runpy.run_module("aosr.search.cli", run_name="__main__")
'''
