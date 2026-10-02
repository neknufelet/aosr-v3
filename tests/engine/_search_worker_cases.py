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
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"]) if options.get("child") else None
record = {"start": time.monotonic(), "pid": os.getpid(), "child": child.pid if child else None,
          "env": dict(os.environ), "cwd": str(Path.cwd())}
event.write_text(json.dumps(record))
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
event.write_text(json.dumps(record))
'''
