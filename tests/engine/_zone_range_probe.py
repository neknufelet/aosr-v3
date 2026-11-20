"""#1199 雙精度子行程：照 ``project.json`` 重建問題（工人開跑與接續都走的 ``run.load_problem``），記帳本標頭與換樣板那一側
拿到的可調範圍、x0 那一片起跑設計的界；``ledger`` 照重建的標頭寫一本只有標頭的帳；``resume`` 走真的接續入口
（``run.resume_search``），記它拒收的那一句（標頭跟帳本對不上時在跑任何一步之前就拒收）。開跑用
``_material_gradient_carryover_probe same``（真的 ``cli.main(["start", …])``，寫完 ``project.json`` 就停）。"""
from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path

from aosr.gradient import material_gradient_ledger as ledger
from aosr.gradient import run
from aosr.gradient.depth_fit import ZoneRange


def _ranges(value: Mapping[str, Mapping[str, ZoneRange]] | None) -> dict[str, dict[str, object]]:
    return {face: {family: given.model_dump(mode="json") for family, given in families.items()}
            for face, families in (value or {}).items()}


def problem(directory: Path) -> dict[str, object]:
    try:
        rebuilt = run.load_problem(run.read_project(directory))
    except ValueError as error:
        return {"error": str(error)}
    x0 = next(design for design in rebuilt.designs if design.surface_id == "x0")
    return {"header": _ranges(rebuilt.header.ranges), "templates": _ranges(rebuilt.templates.ranges),
            "x0": [list(x0.lower), list(x0.upper), list(x0.legal_thicknesses_m)]}


def write_header(directory: Path) -> dict[str, object]:
    rebuilt = run.load_problem(run.read_project(directory))
    ledger.write_ledger(directory / run.LEDGER_FILE, rebuilt.header, ())
    return {"written": True}


def resume(directory: Path) -> dict[str, object]:
    try:
        run.resume_search(directory, should_stop=lambda: True)
    except ValueError as error:
        return {"error": str(error)}
    return {"error": None}


def same_header(directory: Path) -> dict[str, object]:
    rebuilt = run.load_problem(run.read_project(directory))
    try:
        ledger.read_ledger(directory / run.LEDGER_FILE, expected_header=rebuilt.header)
    except ValueError as error:
        return {"error": str(error)}
    return {"error": None}


if __name__ == "__main__":
    mode, location = sys.argv[1:]
    modes = {"problem": problem, "ledger": write_header, "resume": resume, "same": same_header}
    folder = Path(location)
    (folder / "probe.json").write_text(json.dumps(modes[mode](folder), ensure_ascii=False), encoding="utf-8")
    raise SystemExit(0)
