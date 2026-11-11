"""#1093 網頁開第三階段梯度搜尋的後端：設計設定轉參數、替身格子、開跑／停止／接續／終點的接線、服務那一層、網址。

開服務那一層一律換成假的（只記下被叫了什麼、服務在不在由考卷控制），搜尋命令列不真的跑；真跑一次在
``slow_gui_gradient_jobs.py``（本機驗收）。包一層的執行入口（``gradient_exec``）用真的子行程考：離開碼、轉一次訊號。
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Awaitable, Callable, Sequence
from copy import deepcopy
from pathlib import Path
from typing import cast

import httpx
import pytest

from aosr.config.material_gradient import SCHEME_MATERIAL, load_material_gradient_settings
from aosr.config.frequency_axis import LowFrequencyAxis
from aosr.config.paths import config_path
from aosr.gradient import folder_files
from aosr.gradient.boundary import scheme_patch_keys, surface_ids
from aosr.gradient.cli import EXIT_STOPPED
from aosr.gui import gradient_exec
from aosr.gui.app import GuiSettings, create_app
from aosr.gui.gradient_jobs import (
    ActionRefused, GradientJobs, JobBusy, LaunchFailed, NeedsSurrogate, SystemdLauncher,
)
from aosr.gui.gradient_plan import GradientRequest, PlanRefused, plan_search, start_arguments
from aosr.gui.surrogate_slots import found_surrogate, slot_key, surrogate_path
from aosr.gui.zone_design import DesignSettings, ZoneDesign, write_design
from aosr.physics.fem_reduced_receipt import receipt_path
from aosr.physics.fem_reduced_store import manifest_path
from aosr.reporting.scheme import Scheme
from aosr.runtime import child_process_env
from tests.engine._cpu_pins import forward_cpu_pins
from tests.engine._furniture_cases import reference_document
from tests.engine._surface_zone_cases import NEW, rectangle_document

NAME = "zoned"
SETTINGS = load_material_gradient_settings(config_path("material_gradient.toml"))
TEMPLATE = SETTINGS.family[0].template_preset
Record = dict[str, object]


class FakeLauncher:
    """記下每一次開、查、停；``active`` 是考卷說「還在」的服務名。"""

    def __init__(self) -> None:
        self.launched: list[tuple[str, list[str], dict[str, str], Path]] = []
        self.stopped: list[str] = []
        self.active: set[str] = set()
        self.fail: str | None = None

    def launch(self, unit: str, argv: Sequence[str], *, env: dict[str, str], cwd: Path) -> None:
        if self.fail is not None:
            raise LaunchFailed(self.fail)
        self.launched.append((unit, list(argv), env, cwd))
        self.active.add(unit)

    def is_active(self, unit: str) -> bool:
        return unit in self.active

    def stop(self, unit: str) -> None:
        self.stopped.append(unit)


def _scheme_document(kind: str) -> dict[str, object]:
    """whole：舊格式六面固定阻抗；zoned：五面 2×2、地板整面；single：x0 一塊矩形蓋滿整面、其餘整面。"""
    if kind == "whole":
        return reference_document() | {"scheme_id": NAME}
    document = rectangle_document() | {"scheme_id": NAME}
    if kind == "single":
        scene = cast(dict[str, object], deepcopy(document["scene"]))
        zones = cast(dict[str, list[dict[str, object]]], scene[NEW])
        for wall in ("ceiling", "xL", "y0", "yL"):
            zones[wall] = [{"area": "whole_wall", "material": deepcopy(zones[wall][0]["material"])}]
        zones["x0"] = [{"area": "rectangle", "id": "all", "u_m": [0.0, 4.0], "v_m": [0.0, 3.0],
                        "material": deepcopy(zones["x0"][0]["material"])}]
        document = document | {"scene": scene}
    return document


def _scheme(kind: str = "zoned") -> Scheme:
    return Scheme.model_validate(_scheme_document(kind))


def _design(zones: dict[str, dict[str, str]]) -> DesignSettings:
    return DesignSettings(scheme_id=NAME, zones={wall: {zone: ZoneDesign.model_validate({"mode": mode})
                                                        for zone, mode in by_zone.items()}
                                                 for wall, by_zone in zones.items()})


SEARCH = {"x0": {"r0_c0": "search", "r1_c1": "search"}, "ceiling": {"r0_c1": "keep_original"}}


# ---- 設計設定 → 參數 ----

def test_zoned_design_maps_to_patch_order_template_presets_and_frozen_scheme_material() -> None:
    """抓面片序自己排、搜尋區沒拿到家族樣板、固定與保留原牆的區沒凍成方案自己的材料，或沒給步數也帶了 --max-steps。"""
    scheme = _scheme()
    plan = plan_search(scheme, _design(SEARCH), GradientRequest(), SETTINGS)
    surfaces = surface_ids(scheme_patch_keys(scheme))
    assert plan.surfaces == surfaces and len(surfaces) == 21
    searched = {"x0@r0_c0", "x0@r1_c1"}
    assert plan.presets == tuple(TEMPLATE if surface in searched else None for surface in surfaces)
    assert plan.fixed_surfaces == {surface: SCHEME_MATERIAL for surface in surfaces if surface not in searched}
    assert dict(zip(plan.surfaces, plan.modes, strict=True))["ceiling@r0_c1"] == "keep_original"
    assert plan.family == SETTINGS.family[0].id and plan.max_steps == SETTINGS.descent.max_steps == 60
    arguments = start_arguments(plan)
    assert "--max-steps" not in arguments and "--max-depth" not in arguments
    assert arguments[:6] == ["--presets", ",".join(TEMPLATE if s in searched else "none" for s in surfaces),
                             "--mount", "on_wall", "--device", "gpu"]
    pairs = [arguments[index + 1] for index, word in enumerate(arguments) if word == "--fixed-surface"]
    assert pairs == [f"{surface}={SCHEME_MATERIAL}" for surface in plan.fixed_surfaces]


@pytest.mark.parametrize(("kind", "design", "searched"), [
    ("whole", {"x0": {"whole_wall": "search"}}, "x0"),
    ("single", {"x0": {"all": "search"}}, "x0@all"),
])
def test_whole_wall_and_single_full_rectangle_keys(kind: str, design: dict[str, dict[str, str]], searched: str) -> None:
    """抓舊格式（沒有分區清單）整面一區的牆名、與一塊矩形蓋滿整面時的「牆@區」對錯。"""
    plan = plan_search(_scheme(kind), _design(design), GradientRequest(), SETTINGS)
    assert plan.surfaces == surface_ids(scheme_patch_keys(_scheme(kind)))
    assert dict(zip(plan.surfaces, plan.presets, strict=True))[searched] == TEMPLATE
    assert searched not in plan.fixed_surfaces and len(plan.fixed_surfaces) == len(plan.surfaces) - 1


def test_steps_depths_family_mount_and_device_pass_through() -> None:
    """抓自訂步數、深度上限（公尺照原值）、家族、安裝方式、裝置沒照實帶進命令列。"""
    family = SETTINGS.family[-1]
    request = GradientRequest(family=family.id, mount="rigid", device="cpu", max_steps=12,
                              max_depths={"x0": {"r0_c0": 0.1}})
    plan = plan_search(_scheme(), _design(SEARCH), request, SETTINGS)
    arguments = start_arguments(plan)
    assert plan.max_steps == 12 and arguments[-2:] == ["--max-steps", "12"]
    assert ["--max-depth", "x0@r0_c0=0.1"] == arguments[arguments.index("--max-depth"):arguments.index("--max-depth") + 2]
    assert {preset for preset in plan.presets if preset is not None} == {family.template_preset}
    assert arguments[2:6] == ["--mount", "rigid", "--device", "cpu"]


@pytest.mark.parametrize(("design", "request_fields", "message"), [
    ({}, {}, "沒有任何一區交給搜尋"),
    ({"x0": {"r0_c0": "keep_original"}}, {}, "沒有任何一區交給搜尋"),
    ({"x0": {"r9": "search"}}, {}, "對不上"),
    (SEARCH, {"family": "missing"}, "家族代號不認識"),
    (SEARCH, {"max_depths": {"x0": {"r0_c1": 0.1}}}, "不是交給搜尋的區"),
    (SEARCH, {"max_depths": {"x0": {"r9": 0.1}}}, "方案裡沒有"),
])
def test_plan_refusals_speak_plainly(design: dict[str, dict[str, str]], request_fields: Record, message: str) -> None:
    """抓一區都沒交給搜尋、設計設定對不上方案、家族不認識、深度上限給錯區時照樣開跑。"""
    with pytest.raises(PlanRefused, match=message):
        plan_search(_scheme(), _design(design), GradientRequest.model_validate(request_fields), SETTINGS)


@pytest.mark.parametrize("fields", [{"max_steps": 0}, {"max_steps": 1.5}, {"max_depths": {"x0": {"r0_c0": 0.0}}},
                                    {"max_depths": {"x0": {"r0_c0": math.inf}}}, {"device": "tpu"}, {"steps": 3}])
def test_request_rejects_bad_options(fields: Record) -> None:
    with pytest.raises(ValueError):
        GradientRequest.model_validate(fields)


# ---- 替身格子 ----

def _moved(document: dict[str, object], path: Sequence[str], value: object) -> Scheme:
    """複本裡沿路徑換一格（數字鍵是清單索引）。"""
    changed = deepcopy(document)
    node: object = changed
    for key in path[:-1]:
        node = cast(list[object], node)[int(key)] if isinstance(node, list) else cast(dict[str, object], node)[key]
    if isinstance(node, list):
        cast(list[object], node)[int(path[-1])] = value
    else:
        cast(dict[str, object], node)[path[-1]] = value
    return Scheme.model_validate(changed)


def test_slot_key_follows_geometry_points_air_axis_but_not_materials_or_purpose() -> None:
    """抓換了矩形座標、區代號、區序、喇叭或座位、空氣、低頻軸還拿同一格替身，或只換材料、用途就要重建替身。"""
    document = _scheme_document("zoned")
    base = slot_key(Scheme.model_validate(document))
    scene = cast(dict[str, object], document["scene"])
    zones = cast(dict[str, list[dict[str, object]]], scene[NEW])
    first = zones["x0"][0]
    other_material = {"kind": "constant_impedance", "impedance_pa_s_m": 4321.0}
    same = [_moved(document, ("scene", NEW, "x0", "0", "material"), other_material),
            Scheme.model_validate(document | {"purpose": "home_theater"})]
    assert all(slot_key(scheme) == base for scheme in same)
    swapped = deepcopy(document)
    swapped_zones = cast(dict[str, list[object]], cast(dict[str, object], swapped["scene"])[NEW])
    swapped_zones["x0"] = [swapped_zones["x0"][1], swapped_zones["x0"][0], *swapped_zones["x0"][2:]]
    speakers = cast(dict[str, dict[str, float]], document["speakers"])
    left = dict(speakers["left"]) | {"x": speakers["left"]["x"] + 0.01}
    different = [
        _moved(document, ("scene", NEW, "x0", "0", "id"), "renamed"),
        Scheme.model_validate(swapped),
        _moved(document, ("speakers", "left"), left),
        _moved(document, ("scene", "density_kg_m3"), cast(float, scene["density_kg_m3"]) + 0.01),
        _moved(document, ("scene", "sound_speed_m_s"), cast(float, scene["sound_speed_m_s"]) + 0.1),
        _moved(document, ("scene", "low_frequency_axis"), LowFrequencyAxis.SEARCH.value),
    ]
    assert first["id"] == "r0_c0"
    keys = [slot_key(scheme) for scheme in different]
    assert base not in keys and len(set(keys)) == len(keys)


def _place_surrogate(data_dir: Path, scheme: Scheme, *, parts: int = 3) -> Path:
    path = surrogate_path(data_dir, scheme)
    path.parent.mkdir(parents=True, exist_ok=True)
    for part in (path, manifest_path(path), receipt_path(path))[:parts]:
        part.write_bytes(b"fake")
    return path


def test_surrogate_needs_all_three_parts(tmp_path: Path) -> None:
    scheme = _scheme()
    _place_surrogate(tmp_path, scheme, parts=2)
    assert found_surrogate(tmp_path, scheme) is None
    _place_surrogate(tmp_path, scheme)
    assert found_surrogate(tmp_path, scheme) == surrogate_path(tmp_path, scheme)


# ---- 工作：開跑、停止、接續、終點 ----

def _data(tmp_path: Path, kind: str = "zoned", design: dict[str, dict[str, str]] | None = None,
          surrogate: bool = True) -> Path:
    data_dir = tmp_path / "data"
    (data_dir / "schemes").mkdir(parents=True)
    (data_dir / "designs").mkdir()
    scheme = _scheme(kind)
    (data_dir / "schemes" / f"{NAME}.json").write_text(scheme.model_dump_json(), encoding="utf-8")
    write_design(data_dir, _design(SEARCH if design is None else design), scheme)
    if surrogate:
        _place_surrogate(data_dir, scheme)
    return data_dir


def _jobs(data_dir: Path, launcher: FakeLauncher) -> GradientJobs:
    return GradientJobs(data_dir, launcher=launcher, engine_commit="c" * 40, capabilities=config_path("capabilities.toml"),
                        scheme_lock=threading.RLock())


def _folder(data_dir: Path, view: Record, status: Record | None = None) -> Path:
    """假裝搜尋命令列開好了自己的資料夾（還可以順便寫一份狀態檔）。"""
    folder = data_dir / "gradient" / str(view["job_id"]) / "root" / ("f" * 32)
    folder.mkdir(parents=True)
    (folder / folder_files.PROJECT_FILE).write_text("{}", encoding="utf-8")
    if status is not None:
        (folder / folder_files.STATUS_FILE).write_text(json.dumps(status), encoding="utf-8")
    return folder


def _finish(data_dir: Path, launcher: FakeLauncher, view: Record, code: int | None, *, error: str | None = None) -> None:
    steps = cast(list[Record], view["steps"])
    index = len(steps) - 1
    launcher.active.discard(str(steps[index]["unit"]))
    path = data_dir / "gradient" / str(view["job_id"]) / "steps" / f"{index}.exit.json"
    gradient_exec.write_exit(path, gradient_exec.StepExit(exit_code=code, error=error, finished_at=time.time()))


STATUS: Record = {"state": "running", "started_at": "2026-11-11T00:00:00+00:00", "updated_at": "2026-11-11T00:01:00+00:00",
                  "evaluations": 7, "rounds_done": 1, "last_value": 3.25, "stop_reason": None, "pid": 4242}


def test_start_without_surrogate_says_build_one_first_and_leaves_nothing(tmp_path: Path) -> None:
    """抓沒有替身也開跑、拿別份切法的替身，或留下半個工作資料夾。"""
    data_dir = _data(tmp_path, surrogate=False)
    _place_surrogate(data_dir, _scheme("single"))  # 別份切法的替身：不准拿來頂
    launcher = FakeLauncher()
    jobs = _jobs(data_dir, launcher)
    with pytest.raises(NeedsSurrogate, match="要先建替身") as refused:
        jobs.start(NAME, GradientRequest())
    assert refused.value.slot == surrogate_path(data_dir, _scheme()).parent
    assert launcher.launched == [] and list((data_dir / "gradient").iterdir()) == []
    assert jobs.preview(NAME, GradientRequest())["surrogate"] == {
        "found": False, "slot": str(refused.value.slot), "missing": ["替身模型", "替身說明檔（manifest）", "留出驗證收據"],
        "message": "要先建替身"}


def test_start_snapshots_inputs_and_launches_the_cli_through_the_wrapper(tmp_path: Path) -> None:
    """抓命令列指向網頁會再改的方案檔、沒走包一層、少了根目錄或替身、環境沒釘單緒、服務名重複。"""
    data_dir = _data(tmp_path)
    launcher = FakeLauncher()
    jobs = _jobs(data_dir, launcher)
    view = jobs.start(NAME, GradientRequest(max_steps=5))
    job_dir = data_dir / "gradient" / str(view["job_id"])
    assert view["state"] == "checking" and view["actions"] == {"stop": True, "resume": False, "endpoint": False}
    assert (job_dir / "scheme.json").read_bytes() == (data_dir / "schemes" / f"{NAME}.json").read_bytes()
    assert DesignSettings.model_validate_json((job_dir / "design.json").read_text()) == _design(SEARCH)
    (unit, argv, env, cwd), = launcher.launched
    assert unit == f"aosr-gradient-{view['job_id']}-0"
    wrapper = argv.index("--")
    assert argv[1:3] == ["-m", "aosr.gui.gradient_exec"] and argv[wrapper + 1:wrapper + 5] == [
        sys.executable, "-m", "aosr.gradient.cli", "start"]
    command = argv[wrapper + 5:]
    assert command[command.index("--scheme") + 1] == str(job_dir / "scheme.json")
    assert command[command.index("--surrogate") + 1] == str(surrogate_path(data_dir, _scheme()))
    assert command[command.index("--root") + 1] == str(job_dir / "root")
    assert command[command.index("--purpose") + 1] == _scheme().purpose
    assert command[command.index("--engine-commit") + 1] == "c" * 40
    assert command[-2:] == ["--max-steps", "5"]
    assert env["OMP_NUM_THREADS"] == env["MKL_NUM_THREADS"] == env["OPENBLAS_NUM_THREADS"] == "1"
    assert (cwd / "pyproject.toml").is_file()


def test_paths_handed_to_the_service_are_absolute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """抓相對路徑原樣交給服務：服務的工作目錄是 repo 根，相對的資料夾、設定檔、能力表在那裡指到別處（真服務實跑抓到的）。"""
    _data(tmp_path)
    (tmp_path / "settings.toml").write_text(config_path("material_gradient.toml").read_text(encoding="utf-8"))
    monkeypatch.chdir(tmp_path)
    launcher = FakeLauncher()
    jobs = GradientJobs(Path("data"), launcher=launcher, engine_commit="c" * 40,
                        capabilities=Path(os.path.relpath(config_path("capabilities.toml"), tmp_path)),
                        scheme_lock=threading.RLock(), settings_path=Path("settings.toml"))
    jobs.start(NAME, GradientRequest())
    _, argv, _, _ = launcher.launched[0]
    for flag in ("--scheme", "--surrogate", "--root", "--capabilities", "--settings"):
        assert Path(argv[argv.index(flag) + 1]).is_absolute(), flag
    assert Path(argv[3]).is_absolute() and Path(argv[4]).is_absolute()  # 離開碼檔與輸出檔


def test_one_job_at_a_time_on_the_whole_machine(tmp_path: Path) -> None:
    """抓第二跑（就算是別份方案）在第一跑還在算時也開得起來。"""
    data_dir = _data(tmp_path)
    launcher = FakeLauncher()
    jobs = _jobs(data_dir, launcher)
    first = jobs.start(NAME, GradientRequest())
    with pytest.raises(JobBusy) as busy:
        jobs.start(NAME, GradientRequest())
    assert busy.value.job_id == first["job_id"] and len(launcher.launched) == 1
    assert jobs.preview(NAME, GradientRequest())["busy"] == first["job_id"]


def test_stop_marker_then_resume_then_endpoint_then_done(tmp_path: Path) -> None:
    """整條：搜尋中有進度、停止放記號（不砍服務）、離開碼 3 是已停止、接續換新服務名、做完跑終點、終點做完。"""
    data_dir = _data(tmp_path)
    launcher = FakeLauncher()
    jobs = _jobs(data_dir, launcher)
    view = jobs.start(NAME, GradientRequest())
    folder = _folder(data_dir, view, STATUS)
    view = jobs.get(str(view["job_id"]))
    assert view["state"] == "searching" and view["folder"] == str(folder)
    assert cast(Record, view["progress"])["evaluations"] == 7 and view["max_steps"] == 60
    view = jobs.stop(str(view["job_id"]))
    assert (folder / folder_files.STOP_FILE).is_file() and launcher.stopped == []
    assert view["state"] == "stopping"
    _finish(data_dir, launcher, view, EXIT_STOPPED)
    view = jobs.get(str(view["job_id"]))
    assert view["state"] == "stopped" and view["actions"] == {"stop": False, "resume": True, "endpoint": False}
    with pytest.raises(ActionRefused, match="不能跑終點"):
        jobs.endpoint(str(view["job_id"]))
    view = jobs.resume(str(view["job_id"]))
    unit, argv, _, _ = launcher.launched[-1]
    assert unit.endswith("-1") and argv[-2:] == ["resume", str(folder)] and view["state"] == "searching"
    _finish(data_dir, launcher, view, 0)
    assert jobs.get(str(view["job_id"]))["state"] == "searched"
    view = jobs.endpoint(str(view["job_id"]))
    unit, argv, _, _ = launcher.launched[-1]
    assert unit.endswith("-2") and argv[-4:] == ["endpoint", str(folder), "--device", "gpu"]
    assert view["state"] == "endpoint_running"
    _finish(data_dir, launcher, view, 0)
    (folder / folder_files.ENDPOINT_FILE).write_text("{}", encoding="utf-8")
    view = jobs.get(str(view["job_id"]))
    assert view["state"] == "done" and not any(cast(dict[str, bool], view["actions"]).values())
    endpoint = cast(Record, view["endpoint"])
    assert endpoint["exported_scheme"] == str(folder / folder_files.EXPORTED_SCHEME_FILE)
    assert "不是「同一套切法、每區都不貼」" in str(endpoint["baseline_note"])


def test_stop_before_the_folder_exists_or_during_endpoint_stops_the_service(tmp_path: Path) -> None:
    """抓開跑前核對那一段（還沒有搜尋資料夾）按停止沒停到服務，或終點中只放了搜尋的停止記號。"""
    data_dir = _data(tmp_path)
    launcher = FakeLauncher()
    jobs = _jobs(data_dir, launcher)
    view = jobs.stop(str(jobs.start(NAME, GradientRequest())["job_id"]))
    assert launcher.stopped == [f"aosr-gradient-{view['job_id']}-0"] and view["state"] == "stopping"
    _finish(data_dir, launcher, view, EXIT_STOPPED)
    view = jobs.get(str(view["job_id"]))
    assert view["state"] == "stopped" and view["actions"] == {"stop": False, "resume": False, "endpoint": False}


@pytest.mark.parametrize(("stop_first", "state"), [(False, "interrupted"), (True, "stopped")])
def test_service_gone_without_exit_record(tmp_path: Path, stop_first: bool, state: str) -> None:
    """抓服務被強殺、機器重開（沒有離開碼檔）時一直顯示在算，或被當成做完。"""
    data_dir = _data(tmp_path)
    launcher = FakeLauncher()
    jobs = _jobs(data_dir, launcher)
    view = jobs.start(NAME, GradientRequest())
    _folder(data_dir, view)
    if stop_first:
        jobs.stop(str(view["job_id"]))
    launcher.active.clear()
    view = jobs.get(str(view["job_id"]))
    assert view["state"] == state and cast(dict[str, bool], view["actions"])["resume"]


def test_restarted_web_server_reattaches_from_files(tmp_path: Path) -> None:
    """抓狀態只記在記憶體：換一個新的工作管理器（網頁重開）還看得到在算、停得了。"""
    data_dir = _data(tmp_path)
    launcher = FakeLauncher()
    view = _jobs(data_dir, launcher).start(NAME, GradientRequest())
    folder = _folder(data_dir, view)
    again = _jobs(data_dir, launcher)
    listed = cast(list[Record], again.listing()["jobs"])
    assert [item["job_id"] for item in listed] == [view["job_id"]] and listed[0]["state"] == "searching"
    again.stop(str(view["job_id"]))
    assert (folder / folder_files.STOP_FILE).is_file()


def test_saved_scheme_changes_do_not_touch_the_running_snapshot(tmp_path: Path) -> None:
    data_dir = _data(tmp_path)
    jobs = _jobs(data_dir, FakeLauncher())
    view = jobs.start(NAME, GradientRequest())
    snapshot = data_dir / "gradient" / str(view["job_id"]) / "scheme.json"
    before = snapshot.read_bytes()
    (data_dir / "schemes" / f"{NAME}.json").write_text(_scheme("whole").model_dump_json(), encoding="utf-8")
    assert snapshot.read_bytes() == before


def test_launch_failure_is_recorded_as_a_failed_step(tmp_path: Path) -> None:
    data_dir = _data(tmp_path)
    launcher = FakeLauncher()
    launcher.fail = "systemd-run 開不了服務"
    jobs = _jobs(data_dir, launcher)
    with pytest.raises(LaunchFailed, match="開不了"):
        jobs.start(NAME, GradientRequest())
    view, = cast(list[Record], jobs.listing()["jobs"])
    step, = cast(list[Record], view["steps"])
    assert view["state"] == "failed" and step["exit_code"] is None and "開不了" in str(step["error"])
    assert jobs.active_job() is None


def test_failed_search_can_resume_and_running_job_refuses_follow_ups(tmp_path: Path) -> None:
    data_dir = _data(tmp_path)
    launcher = FakeLauncher()
    jobs = _jobs(data_dir, launcher)
    view = jobs.start(NAME, GradientRequest())
    _folder(data_dir, view)
    with pytest.raises(ActionRefused, match="不能接續"):
        jobs.resume(str(view["job_id"]))
    _finish(data_dir, launcher, view, 1)
    view = jobs.get(str(view["job_id"]))
    assert view["state"] == "failed" and cast(dict[str, bool], view["actions"])["resume"]
    with pytest.raises(ActionRefused, match="沒有在算"):
        jobs.stop(str(view["job_id"]))


# ---- 開服務那一層 ----

class _Runs:
    def __init__(self, *, code: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.commands: list[list[str]] = []
        self.answer = (code, stdout, stderr)

    def __call__(self, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        self.commands.append(list(command))
        code, stdout, stderr = self.answer
        return subprocess.CompletedProcess(list(command), code, stdout, stderr)


def test_systemd_launcher_command_line() -> None:
    """抓用了 --scope、沒有記憶體上限、OOM 會連坐、停止送給整組、服務沒自己卸掉、環境與工作目錄沒帶過去。"""
    runs = _Runs()
    launcher = SystemdLauncher(run=runs)
    launcher.launch("aosr-gradient-x-0", ["python", "-m", "aosr.gui.gradient_exec"], env={"B": "2", "A": "1"},
                    cwd=Path("/repo"))
    assert runs.commands == [["systemd-run", "--user", "--unit=aosr-gradient-x-0", "--collect", "--quiet",
                              "-p", "MemoryMax=32G", "-p", "OOMPolicy=continue", "-p", "KillMode=mixed",
                              "--working-directory=/repo", "--setenv=A=1", "--setenv=B=2",
                              "--", "python", "-m", "aosr.gui.gradient_exec"]]
    runs.answer = (0, "activating\n", "")
    assert launcher.is_active("u")
    runs.answer = (3, "inactive\n", "")
    assert not launcher.is_active("u")
    runs.answer = (0, "", "")
    launcher.stop("u")
    assert runs.commands[-1] == ["systemctl", "--user", "stop", "--no-block", "u"]
    runs.answer = (1, "", "Failed to connect to bus")
    with pytest.raises(LaunchFailed, match="Failed to connect to bus"):
        launcher.launch("u", ["x"], env={}, cwd=Path("/"))


# ---- 包一層的執行入口（真的子行程） ----

CHILD = """
import signal, sys, time, pathlib
count = pathlib.Path(sys.argv[1])
def handle(signum, frame):
    count.write_text(str(int(count.read_text() or 0) + 1))
    time.sleep(0.3)
    raise SystemExit(3)
signal.signal(signal.SIGTERM, handle)
count.write_text("0")
pathlib.Path(sys.argv[2]).write_text("ready")
print("child says hi", flush=True)
time.sleep(60)
"""


def _wrapper(tmp_path: Path, command: list[str]) -> subprocess.Popen[bytes]:
    env = forward_cpu_pins(child_process_env())
    return subprocess.Popen([sys.executable, "-m", "aosr.gui.gradient_exec", str(tmp_path / "exit.json"),
                             str(tmp_path / "step.log"), "--", *command], env=env, cwd=tmp_path)


def _exit(tmp_path: Path) -> gradient_exec.StepExit:
    return gradient_exec.StepExit.model_validate_json((tmp_path / "exit.json").read_text())


def test_wrapper_records_exit_code_and_output(tmp_path: Path) -> None:
    process = _wrapper(tmp_path, [sys.executable, "-c", "print('hello'); raise SystemExit(7)"])
    assert process.wait(timeout=60) == 0
    assert _exit(tmp_path).exit_code == 7 and "hello" in (tmp_path / "step.log").read_text()


def test_wrapper_forwards_sigterm_once_and_waits_for_the_child(tmp_path: Path) -> None:
    """抓包一層收到停止就自己先走（沒留離開碼）、沒轉給子行程，或轉了兩次。"""
    script = tmp_path / "child.py"
    script.write_text(CHILD, encoding="utf-8")
    process = _wrapper(tmp_path, [sys.executable, str(script), str(tmp_path / "count"), str(tmp_path / "ready")])
    deadline = time.monotonic() + 60
    while not (tmp_path / "ready").exists():
        assert time.monotonic() < deadline and process.poll() is None
        time.sleep(0.05)
    process.send_signal(signal.SIGTERM)
    time.sleep(0.1)
    process.send_signal(signal.SIGTERM)
    assert process.wait(timeout=60) == 0
    record = _exit(tmp_path)
    assert record.exit_code == 3 and record.signals == (signal.SIGTERM, signal.SIGTERM)
    assert (tmp_path / "count").read_text() == "1"


def test_wrapper_records_a_command_that_cannot_start(tmp_path: Path) -> None:
    process = _wrapper(tmp_path, [str(tmp_path / "missing-program")])
    assert process.wait(timeout=60) == 1
    record = _exit(tmp_path)
    assert record.exit_code is None and record.error is not None and "FileNotFoundError" in record.error


# ---- 網址與網頁行程 ----

Scenario = Callable[[httpx.AsyncClient], Awaitable[None]]


def _serve(data_dir: Path, launcher: FakeLauncher, scenario: Scenario) -> None:
    async def go() -> None:
        app = create_app(GuiSettings(engine_commit="a" * 40, data_dir=data_dir, gradient_launcher=launcher))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost") as client:
            await scenario(client)
    asyncio.run(go())


def test_routes_start_stop_and_report_errors(tmp_path: Path) -> None:
    """抓網址沒接上、錯誤碼混在一起（格式錯 400、沒替身 409 帶位置、在忙 409、沒這一跑 404）。"""
    data_dir = _data(tmp_path, surrogate=False)
    launcher = FakeLauncher()
    seen: list[Record] = []

    async def scenario(client: httpx.AsyncClient) -> None:
        refused = await client.post(f"/api/gradient/schemes/{NAME}/start", json={})
        assert refused.status_code == 409 and refused.json()["needs"] == "surrogate"
        _place_surrogate(data_dir, _scheme())
        bad = await client.post(f"/api/gradient/schemes/{NAME}/start", json={"max_steps": 0})
        assert bad.status_code == 400 and "格式不對" in bad.json()["error"]
        preview = await client.post(f"/api/gradient/schemes/{NAME}/plan", content=b"", headers={"content-type": "application/json"})
        assert preview.status_code == 200 and preview.json()["surrogate"]["found"] is True
        assert preview.json()["defaults"]["max_steps"] == 60
        started = await client.post(f"/api/gradient/schemes/{NAME}/start", json={"device": "cpu"})
        assert started.status_code == 200, started.text
        job_id = started.json()["job_id"]
        busy = await client.post(f"/api/gradient/schemes/{NAME}/start", json={})
        assert busy.status_code == 409 and busy.json()["busy"] == job_id
        stopped = await client.post(f"/api/gradient/jobs/{job_id}/stop", json={})
        assert stopped.status_code == 200 and stopped.json()["state"] == "stopping"
        listed = await client.get("/api/gradient/jobs")
        seen.extend(listed.json()["jobs"])
        assert (await client.get(f"/api/gradient/jobs/{'0' * 32}")).status_code == 404
        assert (await client.post(f"/api/gradient/jobs/{job_id}/explode", json={})).status_code == 404
        unknown = await client.post("/api/gradient/schemes/nope/plan", json={})
        assert unknown.status_code == 404

    _serve(data_dir, launcher, scenario)
    assert [job["job_id"] for job in seen] == [launcher.launched[0][0].split("-")[2]]


def test_web_process_never_imports_jax(tmp_path: Path) -> None:
    """抓網頁行程（含這一段的模組）把 JAX 拉進來：哪天誤叫一個陣列運算就會在網頁行程裡佔走顯卡記憶體。"""
    code = ("import sys; import aosr.gui.app, aosr.gui.gradient_jobs, aosr.gui.gradient_exec; "
            "raise SystemExit(1 if 'jax' in sys.modules else 0)")
    env = forward_cpu_pins(child_process_env())
    env["TMPDIR"] = str(tmp_path)
    done = subprocess.run([sys.executable, "-c", code], env=env, cwd=tmp_path, capture_output=True, text=True,
                          timeout=120)
    assert done.returncode == 0, done.stderr


def test_folder_files_are_the_ones_the_search_writes() -> None:
    """抓網頁讀的檔名或狀態格式跟搜尋本體寫的分了家。"""
    from aosr.gradient import endpoint_run, run

    assert run.GradientStatus is folder_files.GradientStatus
    assert (run.STATUS_FILE, run.STOP_FILE, run.PROJECT_FILE, endpoint_run.ENDPOINT_FILE,
            endpoint_run.EXPORTED_SCHEME_FILE) == (folder_files.STATUS_FILE, folder_files.STOP_FILE,
                                                   folder_files.PROJECT_FILE, folder_files.ENDPOINT_FILE,
                                                   folder_files.EXPORTED_SCHEME_FILE)
