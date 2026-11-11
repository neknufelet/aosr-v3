"""#1093 網頁開第三階段搜尋的真跑一次（本機合併前驗收，雲端不收）。

真的方案、真的替身（命令列考卷那一份小房間）、真的搜尋命令列與終點，經網頁的工作管理器從「設計設定」開跑；
只有開服務那一層換成本機行程（考卷不開真的 systemd 服務；真的服務另在票的工作區留一次實跑紀錄）。
"""
from __future__ import annotations

import shutil
import signal
import subprocess
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pytest

from aosr.config.capabilities import load_capabilities
from aosr.config.directivity_defaults import load_directivity_defaults
from aosr.config.material_gradient import SCHEME_MATERIAL, load_material_gradient_settings
from aosr.config.paths import config_path
from aosr.gradient import run
from aosr.gradient.boundary import named_materials
from aosr.gradient.endpoint_run import read_endpoint
from aosr.gui.gradient_jobs import GradientJobs
from aosr.gui.gradient_plan import GradientRequest
from aosr.gui.surrogate_slots import surrogate_path
from aosr.gui.zone_design import DesignSettings, ZoneDesign, write_design
from aosr.physics.fem_reduced_receipt import receipt_path
from aosr.physics.fem_reduced_store import manifest_path
from aosr.reporting.scheme import load_scheme
from tests.engine._cpu_pins import forward_cpu_pins
from tests.engine._six_search_cli_case import model as model, operators as operators
from tests.engine.test_material_gradient_cli import WAIT_S
from tests.engine.test_material_gradient_endpoint_cli import _inputs, _surrogate

NAME = "demo"
SEARCHED = "x0"
Record = dict[str, object]
RUNNING = ("checking", "searching", "stopping", "endpoint_running", "endpoint_stopping")


class LocalLauncher:
    """本機行程代替服務：新的行程群、停止只送給最外層（同 ``KillMode=mixed``）。"""

    def __init__(self, tmp: Path) -> None:
        self.tmp = tmp
        self.processes: dict[str, subprocess.Popen[bytes]] = {}

    def launch(self, unit: str, argv: Sequence[str], *, env: dict[str, str], cwd: Path) -> None:
        child = forward_cpu_pins(env)
        child["TMPDIR"] = str(self.tmp)
        self.processes[unit] = subprocess.Popen(list(argv), env=child, cwd=cwd, start_new_session=True,
                                                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                                stderr=subprocess.DEVNULL)

    def is_active(self, unit: str) -> bool:
        process = self.processes.get(unit)
        return process is not None and process.poll() is None

    def stop(self, unit: str) -> None:
        self.processes[unit].send_signal(signal.SIGTERM)


def _wait(jobs: GradientJobs, job_id: str) -> Record:
    deadline = time.monotonic() + WAIT_S * 2
    while True:
        view = jobs.get(job_id)
        if view["state"] not in RUNNING:
            return view
        assert time.monotonic() < deadline, view
        time.sleep(0.5)


def _logs(view: Record) -> str:
    return "\n".join(line for step in cast(list[Record], view["steps"]) for line in cast(list[str], step["log_tail"]))


@pytest.fixture(scope="module")
def searched(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> tuple[GradientJobs, Record, Path]:
    directory = tmp_path_factory.mktemp("gui-gradient")
    _inputs(directory)
    _surrogate(directory, request)
    data_dir = directory / "data"
    (data_dir / "schemes").mkdir(parents=True)
    (data_dir / "designs").mkdir()
    shutil.copyfile(directory / "scheme.json", data_dir / "schemes" / f"{NAME}.json")
    scheme = load_scheme(directory / "scheme.json")
    write_design(data_dir, DesignSettings(scheme_id=NAME, zones={SEARCHED: {"whole_wall": ZoneDesign(mode="search")}}),
                 scheme)
    source, slot = directory / "good" / "rom.npz", surrogate_path(data_dir, scheme)
    slot.parent.mkdir(parents=True)
    for make in (lambda path: path, manifest_path, receipt_path):
        shutil.copyfile(make(source), make(slot))
    jobs = GradientJobs(data_dir, launcher=LocalLauncher(directory), engine_commit="test",
                        capabilities=config_path("capabilities.toml"), scheme_lock=threading.RLock(),
                        settings_path=directory / "settings.toml")
    view = _wait(jobs, str(jobs.start(NAME, GradientRequest(device="cpu"))["job_id"]))
    assert view["state"] == "searched", _logs(view)
    return jobs, view, directory


def test_web_search_freezes_fixed_walls_and_searches_only_the_chosen_one(
        searched: tuple[GradientJobs, Record, Path]) -> None:
    """抓網頁轉的參數到了真的命令列被改掉：五面固定成方案自己的材料、只有交給搜尋那一面帶變數、貼在原牆上。"""
    _, view, _ = searched
    folder = Path(str(view["folder"]))
    project = run.read_project(folder)
    frozen = {wall: SCHEME_MATERIAL for wall in ("floor", "ceiling", "xL", "y0", "yL")}
    template = load_material_gradient_settings(config_path("material_gradient.toml")).family[0].template_preset
    assert project.fixed_surfaces == frozen and project.mount == "on_wall" and project.max_steps is None
    assert project.presets == (None, None, template, None, None, None)
    result = run.parse_result((folder / run.RESULT_FILE).read_bytes())
    assert {spec.surface_id: spec.fixed_material_id for spec in result.surfaces if spec.fixed_material_id} == frozen
    assert result.variable_names and all(name.startswith(f"{SEARCHED}.") for name in result.variable_names)
    assert cast(Record, view["progress"])["state"] == "finished"


def test_web_endpoint_exports_scheme_with_the_frozen_walls_unchanged(
        searched: tuple[GradientJobs, Record, Path]) -> None:
    """抓終點匯出的方案把固定的牆換掉：五面的材料要逐一等於方案原本那一面。"""
    jobs, view, directory = searched
    done = _wait(jobs, str(jobs.endpoint(str(view["job_id"]))["job_id"]))
    assert done["state"] == "done", _logs(done)
    folder = Path(str(done["folder"]))
    record = read_endpoint(folder)
    assert record.fixed_surfaces == {wall: SCHEME_MATERIAL for wall in ("floor", "ceiling", "xL", "y0", "yL")}
    capabilities = load_capabilities(config_path("capabilities.toml"))
    directivity = load_directivity_defaults(config_path("directivity_defaults.toml"))
    before = named_materials(run.scene_wall_materials(load_scheme(directory / "scheme.json"),
                                                      capabilities=capabilities, directivity=directivity))
    after = named_materials(run.scene_wall_materials(load_scheme(folder / record.exported_scheme),
                                                     capabilities=capabilities, directivity=directivity))
    assert set(before) == set(after) == {"floor", "ceiling", "x0", "xL", "y0", "yL"}
    for wall, material in before.items():
        if wall == SEARCHED:
            assert after[wall] != material
        else:
            assert after[wall] == material, wall
