"""#990 固定面真命令列：小房間、僅一面帶搜尋變數；開始與終點拆成獨立段落，命令各限 60 秒。"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from aosr.gradient import run
from aosr.gradient.endpoint_run import read_endpoint
from aosr.gradient.report import render_report
from aosr.reporting.scheme import load_scheme
from aosr.physics.report_surfaces import WholeWallZone
from tests.engine.test_material_gradient import _presets
from tests.engine._scheme_cache import shared_json
from tests.engine.test_fem_reduced import model as model, operators as operators
from tests.engine.test_material_gradient_cli import _command, _env, _start_args
from tests.engine.test_material_gradient_endpoint_cli import _inputs, _surrogate

FIXED = {"floor": "glass_wool_24k_50mm", "ceiling": "slotted_wood_typ9_absorber40_gap98"}


def _command_run(directory: Path, args: list[str]) -> None:
    process = subprocess.run(args, cwd=directory, env=_env(directory), capture_output=True, text=True, timeout=60)
    assert process.returncode == 0, process.stdout + process.stderr


def _start(directory: Path, request: pytest.FixtureRequest) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    _inputs(directory)
    _surrogate(directory, request)
    args = _start_args(directory, directory / "out", presets="none,none,glass_wool_24k_50mm,none,none,none")
    args.extend(("--mount", "on_wall"))
    for face, material in FIXED.items():
        args.extend(("--fixed-surface", f"{face}={material}"))
    _command_run(directory, args)
    folder, = (directory / "out").iterdir()
    return json.dumps(str(folder))


@pytest.fixture(scope="module")
def live_search(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> Path:
    """替身與開始只建一次，平行工人共用；終點另放一題。"""
    directory = (tmp_path_factory.mktemp("fixed-surface-cli") if worker_id == "master"
                 else tmp_path_factory.getbasetemp().parent / "fixed-surface-cli-shared")
    text = shared_json(tmp_path_factory, worker_id, "fixed-surface-cli-start", lambda: _start(directory, request))
    return Path(json.loads(text))


def test_real_start_and_resume_keep_fixed_materials(live_search: Path, tmp_path: Path) -> None:
    """抓 worker（子行程）讀回固定設定後跑錯材料，以及 resume（接續）遺失設定。"""
    live_search = shutil.copytree(live_search, tmp_path / "resume")
    project = run.read_project(live_search)
    result = run.parse_result((live_search / run.RESULT_FILE).read_bytes())
    assert project.fixed_surfaces == FIXED
    assert {s.surface_id: s.fixed_material_id for s in result.surfaces if s.fixed_material_id is not None} == FIXED
    assert all(not name.startswith(("floor.", "ceiling.")) for name in result.variable_names)
    before = (live_search / run.LEDGER_FILE).read_bytes()
    _command_run(live_search.parent.parent, _command("resume", str(live_search)))
    assert (live_search / run.LEDGER_FILE).read_bytes() == before
    assert run.parse_result((live_search / run.RESULT_FILE).read_bytes()) == result


def test_real_endpoint_exports_and_records_fixed_materials(live_search: Path, tmp_path: Path) -> None:
    """抓 endpoint（終點）重建配置用回原牆，或紀錄與人看的報告遺失固定面。"""
    live_search = shutil.copytree(live_search, tmp_path / "endpoint")
    _command_run(tmp_path, _command("endpoint", str(live_search)))
    record = read_endpoint(live_search)
    assert record.fixed_surfaces == FIXED
    zones = load_scheme(live_search / record.exported_scheme).scene.surface_zones_by_wall
    assert zones is not None
    for face, material_id in FIXED.items():
        raw = next(p.material for p in _presets() if p.id == material_id)
        expected = WholeWallZone.model_validate({"area": "whole_wall", "material": raw}).material
        assert zones[face][0].material == expected
    report = render_report(live_search)
    for face, material_id in FIXED.items():
        assert any(face in line and material_id in line and "固定" in line for line in report.splitlines())


@pytest.mark.parametrize("fixed", [{"floor": "glass_wool_96k_50mm", "ceiling": FIXED["ceiling"]},
                                  {"xL": FIXED["floor"], "ceiling": FIXED["ceiling"]}])
def test_real_resume_rejects_changed_fixed_settings(live_search: Path, tmp_path: Path, fixed: dict[str, str]) -> None:
    """抓 project（方案搜尋設定）改材料或改面後，子行程仍接上舊帳；拒收時兩本帳與結果不動。"""
    folder = shutil.copytree(live_search, tmp_path / "changed")
    project = run.read_project(folder).model_copy(update={"fixed_surfaces": fixed})
    run.write_json_atomic(folder / run.PROJECT_FILE, project)
    names = (run.LEDGER_FILE, run.TEMPLATES_FILE, run.RESULT_FILE)
    before = {name: (folder / name).read_bytes() for name in names}
    process = subprocess.run(_command("resume", str(folder)), cwd=tmp_path, env=_env(tmp_path),
                             capture_output=True, text=True, timeout=60)
    assert process.returncode == 1 and "標頭跟這一次對不上" in process.stderr
    assert {name: (folder / name).read_bytes() for name in names} == before
