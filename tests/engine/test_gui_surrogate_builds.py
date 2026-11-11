"""#1098 網頁替新切法建替身：真的建模命令列（小房間）經工作管理器開、留出驗證、准用閘門、換名進格子；
停止、驗證沒過、格子已有、忙碌互擋、進度與「推估」、網址。

真跑的幾題開本機行程代替服務（``_local_launcher``），建模命令列真的匯出完整模型、真的建、真的驗；
留出組數用考卷縮過的抽法設定（只少隨機組），產品預設的抽法另有一題真跑（停止那一題，跑到留出驗證中途才停）。
其餘題開服務那一層換成假的（``test_gui_gradient_jobs.FakeLauncher``），狀態與檔案由考卷寫。
"""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import threading
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import cast

import httpx
import pytest

from aosr.config.capabilities import load_capabilities
from aosr.config.directivity_defaults import load_directivity_defaults
from aosr.config.paths import config_path
from aosr.gradient import run
from aosr.gradient.surrogate_build import _rebuild_hint
from aosr.gradient.surrogate_files import (
    ESTIMATE_LABEL, MODEL_FILE, PROGRESS_FILE, RESULT_FILE, STAGE_DIR, TRAINING_FILE, RoundRecord, SurrogateProgress,
    SurrogateResult, estimate,
)
from aosr.gui import gradient_exec
from aosr.gui.app import GuiSettings, create_app
from aosr.gui.gradient_jobs import ActionRefused, GradientJobs, JobBusy, LaunchFailed, Launcher, NeedsSurrogate
from aosr.gui.gradient_plan import GradientRequest, PlanRefused
from aosr.gui.surrogate_builds import BUILD_UNIT_PREFIX, RETIRED_DIR, SurrogateBuilds, SurrogateRequest
from aosr.gui.surrogate_slots import SLOTS_DIR, SURROGATE_FILE, missing_parts, slot_dir, surrogate_path
from aosr.gui.zone_design import DesignSettings, ZoneDesign, write_design
from aosr.physics.fem_reduced_build import STOP_BUDGET, STOP_RESIDUAL_MET
from aosr.physics.fem_reduced_check import points_hex
from aosr.physics.fem_reduced_receipt import receipt_path
from aosr.physics.fem_reduced_store import operators_path
from aosr.reporting.fem_reduced_scheme import export_operators
from aosr.reporting.scheme import Scheme
from aosr.runtime import child_process_env
from tests.engine._cpu_pins import forward_cpu_pins
from tests.engine._local_launcher import LocalLauncher
from tests.engine._material_gradient_zones_case import scheme
from tests.engine.test_gui_gradient_jobs import FakeLauncher

NAME = "zoned"
Record = dict[str, object]
RUNNING = ("preparing", "building", "validating", "checking", "installing", "stopping")
WAIT_S = 300.0
QUICK = SurrogateRequest(workers=1, dense_threads=1)
SHRUNK_HOLDOUTS = 1


def _training_settings(directory: Path) -> Path:
    """考卷縮組數的抽法設定：留出隨機組只留一組（全部原牆、混搭兩組固定的照舊），其餘一字不改。"""
    text = config_path("surrogate_training.toml").read_text(encoding="utf-8")
    assert text.count("random_sets = 7") == 1
    path = directory / "surrogate_training.toml"
    path.write_text(text.replace("random_sets = 7", f"random_sets = {SHRUNK_HOLDOUTS}"), encoding="utf-8")
    return path


def _data(tmp_path: Path, case: Scheme | None = None, *, search: bool = True) -> Path:
    data_dir = tmp_path / "data"
    (data_dir / "schemes").mkdir(parents=True)
    (data_dir / "designs").mkdir()
    base = scheme() if case is None else case
    (data_dir / "schemes" / f"{NAME}.json").write_text(base.model_dump_json(), encoding="utf-8")
    zones = {"x0": {"right": ZoneDesign(mode="search")}} if search else {}
    write_design(data_dir, DesignSettings(scheme_id=NAME, zones=zones), base)
    return data_dir


def _managers(data_dir: Path, launcher: Launcher, training: Path | None) -> tuple[GradientJobs, SurrogateBuilds]:
    jobs = GradientJobs(data_dir, launcher=launcher, engine_commit="c" * 40, capabilities=config_path("capabilities.toml"),
                        scheme_lock=threading.RLock())
    return jobs, SurrogateBuilds(jobs, training_settings=training)


def _wait(builds: SurrogateBuilds, build_id: str, until: Callable[[Record], bool] | None = None) -> Record:
    deadline = time.monotonic() + WAIT_S
    while True:
        view = builds.get(build_id)
        if (until(view) if until is not None else view["state"] not in RUNNING):
            return view
        assert time.monotonic() < deadline, view
        time.sleep(0.05)


def _slot_npz(data_dir: Path) -> Path:
    return surrogate_path(data_dir, scheme())


# ---- 真跑：建好、放進格子、搜尋的閘門收 ----

@pytest.fixture(scope="module")
def installed(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, SurrogateBuilds, Record]:
    directory = tmp_path_factory.mktemp("surrogate-build")
    data_dir = _data(directory)
    _, builds = _managers(data_dir, LocalLauncher(directory), _training_settings(directory))
    view = _wait(builds, str(builds.start(NAME, QUICK)["build_id"]))
    assert view["state"] == "installed", view
    return data_dir, builds, view


def test_build_installs_a_surrogate_the_search_gate_accepts(installed: tuple[Path, SurrogateBuilds, Record]) -> None:
    """抓建好的替身開跑時被拒（網格、物理身分、座標、空氣、面片序、頻率軸任一格跟搜尋開跑那一套不同），
    或只是三件檔案在、其實沒核過：這裡用搜尋開跑同一組閘門函式、在考卷行程裡重算完整模型再核一次。"""
    data_dir, _, view = installed
    npz = _slot_npz(data_dir)
    assert missing_parts(npz) == [] and operators_path(npz).is_file() and (npz.parent / TRAINING_FILE).is_file()
    base = scheme()
    operators = export_operators(base, capabilities=load_capabilities(config_path("capabilities.toml")),
                                 directivity=load_directivity_defaults(config_path("directivity_defaults.toml")))
    points = run.SchemePoints(points_hex(operators.source_points), points_hex(operators.receiver_points))
    model = run.gated_surrogate(npz, mesh_hash=operators.mesh_hash, physics_identity=operators.physics_identity,
                                points=points)
    run.check_surrogate_matches_scheme(model, base, points)
    run.check_surrogate_axis_covers_scoring(model, base, run.search_objective_settings(base.purpose).stage_one)
    result = cast(Record, view["result"])
    assert result["verdict"] == "installed" and result["within_contract"] is True
    assert result["slot"] == str(npz.parent) and result["retired"] is None
    assert view["estimate"] is None and view["actions"] == {"stop": False}


def test_installed_build_records_progress_and_settings(installed: tuple[Path, SurrogateBuilds, Record]) -> None:
    """抓進度檔沒記到每一輪、預算或停止值不是產品設定那一份、行程數沒照請求、留出點數算錯。"""
    from aosr.config.fem_reduced import load_reduced_model_settings

    _, _, view = installed
    progress = SurrogateProgress.model_validate(view["progress"])
    reduced = load_reduced_model_settings(config_path("fem_reduced.toml"))
    assert progress.max_full_solves == reduced.max_full_solves and progress.stop_residual == reduced.stop_residual
    assert progress.workers == 1 and progress.dense_threads == 1 and progress.patches == 7
    assert progress.rounds and progress.rank == progress.rounds[-1].rank and progress.stop_reason
    assert progress.holdout_sets == 2 + SHRUNK_HOLDOUTS
    assert progress.holdout_sets is not None and progress.axis_points is not None
    assert progress.validation_total == progress.validation_done == progress.holdout_sets * progress.axis_points
    training = json.loads((_slot_npz(installed[0]).parent / TRAINING_FILE).read_text(encoding="utf-8"))
    assert len(training["holdout"]) == progress.holdout_sets and len(training["training"]) == progress.training_sets


def test_search_start_finds_the_built_surrogate(installed: tuple[Path, SurrogateBuilds, Record]) -> None:
    """抓建好放進格子之後，網頁開搜尋還是說要先建替身、或命令列指向別處。"""
    data_dir = installed[0]
    launcher = FakeLauncher()
    jobs, _ = _managers(data_dir, launcher, None)
    view = jobs.start(NAME, GradientRequest(device="cpu"))
    argv = launcher.launched[0][1]
    assert argv[argv.index("--surrogate") + 1] == str(_slot_npz(data_dir)) and view["state"] == "checking"


def test_existing_slot_needs_replace_and_replacing_retires_the_old_one(
        installed: tuple[Path, SurrogateBuilds, Record], tmp_path: Path) -> None:
    """抓格子已有替身還照樣重建蓋掉、或給了 replace 卻把舊的直接刪掉（舊的要整份搬去 retired）。"""
    data_dir, builds, first = installed
    with pytest.raises(ActionRefused, match="replace"):
        builds.start(NAME, QUICK)
    old = json.loads(receipt_path(_slot_npz(data_dir)).read_text(encoding="utf-8"))
    view = _wait(builds, str(builds.start(NAME, QUICK.model_copy(update={"replace": True}))["build_id"]))
    assert view["state"] == "installed", view
    result = cast(Record, view["result"])
    retired = Path(str(result["retired"]))
    assert retired.parent == data_dir / SLOTS_DIR / RETIRED_DIR and retired.name.startswith(str(first["slot_key"]))
    assert json.loads(receipt_path(retired / SURROGATE_FILE).read_text(encoding="utf-8")) == old
    assert missing_parts(_slot_npz(data_dir)) == []


# ---- 真跑：預算給太小、留出驗證不過 ----

def test_starved_build_fails_validation_and_never_enters_the_slot(tmp_path: Path) -> None:
    """抓留出驗證沒過也放進格子、或結果沒照實寫兩個差與兩把尺；搜尋開跑要照舊回「要先建替身」。"""
    data_dir = _data(tmp_path)
    _, builds = _managers(data_dir, LocalLauncher(tmp_path), _training_settings(tmp_path))
    view = _wait(builds, str(builds.start(NAME, QUICK.model_copy(update={"max_full_solves": 8}))["build_id"]))
    assert view["state"] == "not_validated", view
    result = SurrogateResult.model_validate(view["result"])
    assert result.within_contract is False and result.full_solves == 8 and result.stop_reason == STOP_BUDGET
    assert set(result.rulers) == {"frequency", "band"} and "沒有放進格子" in result.message
    assert "加大完整解預算" in result.message
    assert not slot_dir(data_dir, scheme()).exists()
    stage = tmp_path / "data" / SLOTS_DIR / "builds" / str(view["build_id"]) / "work" / STAGE_DIR
    assert receipt_path(stage / MODEL_FILE).is_file() and not operators_path(stage / MODEL_FILE).exists()
    jobs, _ = _managers(data_dir, FakeLauncher(), None)
    with pytest.raises(NeedsSurrogate):
        jobs.start(NAME, GradientRequest())


def test_rebuild_hint_says_whether_a_bigger_budget_can_help() -> None:
    """抓殘差先達標才停的也叫人加預算重建（固定種子，重建結果一模一樣，白等）。"""
    assert "加大完整解預算" in _rebuild_hint(STOP_BUDGET)
    residual = _rebuild_hint(STOP_RESIDUAL_MET)
    assert "加預算沒用" in residual and "一模一樣" in residual


# ---- 真跑：產品預設的抽法、留出驗證中途停 ----

def test_stopping_mid_validation_discards_the_build(tmp_path: Path) -> None:
    """抓停了還繼續、停完沒留離開碼 3、或建到一半的被放進格子；用產品預設的抽法（留出組數照設定檔）。"""
    data_dir = _data(tmp_path)
    _, builds = _managers(data_dir, LocalLauncher(tmp_path), None)
    build_id = str(builds.start(NAME, QUICK)["build_id"])
    running = _wait(builds, build_id, lambda view: view["state"] == "validating" or view["state"] not in RUNNING)
    assert running["state"] == "validating", running
    estimate_view = cast(Record, running["estimate"])
    assert estimate_view["label"] == ESTIMATE_LABEL
    stopped = builds.stop(build_id)
    assert stopped["state"] == "stopping"
    view = _wait(builds, build_id)
    assert view["state"] == "stopped" and view["exit_code"] == 3, view
    assert not slot_dir(data_dir, scheme()).exists() and view["result"] is None


# ---- 假的開服務層：忙碌、開建前核對、狀態、網址 ----

def _fake(tmp_path: Path, case: Scheme | None = None, *, search: bool = True
          ) -> tuple[Path, FakeLauncher, GradientJobs, SurrogateBuilds]:
    data_dir = _data(tmp_path, case, search=search)
    launcher = FakeLauncher()
    jobs, builds = _managers(data_dir, launcher, None)
    return data_dir, launcher, jobs, builds


def _finish(data_dir: Path, launcher: FakeLauncher, view: Record, code: int | None) -> None:
    launcher.active.discard(str(view["unit"]))
    path = data_dir / SLOTS_DIR / "builds" / str(view["build_id"]) / "steps" / "0.exit.json"
    gradient_exec.write_exit(path, gradient_exec.StepExit(exit_code=code, error=None, finished_at=time.time()))


def _work(data_dir: Path, view: Record) -> Path:
    work = data_dir / SLOTS_DIR / "builds" / str(view["build_id"]) / "work"
    work.mkdir(parents=True, exist_ok=True)
    return work


def _progress(**fields: object) -> SurrogateProgress:
    base: Record = {"phase": "building", "started_at": 1000.0, "phase_started_at": 1010.0, "updated_at": 1100.0,
                    "workers": 4, "patches": 7, "dofs": 3389, "axis_points": 94, "training_sets": 41, "holdout_sets": 9,
                    "max_full_solves": 600, "stop_residual": 0.001, "full_solves": 96, "rank": 267,
                    "snapshot_seconds": 48.0,
                    "rounds": [RoundRecord(round=0, full_solves=64, rank=200, max_residual=0.5, elapsed_s=40.0),
                               RoundRecord(round=1, full_solves=96, rank=267, max_residual=0.02, elapsed_s=70.0)]}
    return SurrogateProgress.model_validate(base | fields)


def test_command_line_snapshots_the_scheme_and_goes_through_the_wrapper(tmp_path: Path) -> None:
    """抓命令列指向網頁會再改的方案檔、沒走包一層、沒指到這份切法的格子、少了能力表，或服務名不在共用前綴底下
    （不在就躲過「整台一次一跑」）。"""
    data_dir, launcher, _, builds = _fake(tmp_path)
    view = builds.start(NAME, SurrogateRequest(max_full_solves=900, workers=2, dense_threads=3))
    unit, argv, env, _ = launcher.launched[0]
    assert unit.startswith(BUILD_UNIT_PREFIX) and unit == view["unit"]
    assert argv[1:3] == ["-m", "aosr.gui.gradient_exec"] and argv[argv.index("--") + 1:][1:3] == ["-m", "aosr.gradient.surrogate_build"]
    snapshot = Path(argv[argv.index("--scheme") + 1])
    assert snapshot.parent.name == view["build_id"] and snapshot.read_text(encoding="utf-8") == (
        data_dir / "schemes" / f"{NAME}.json").read_text(encoding="utf-8")
    assert argv[argv.index("--slot") + 1] == str(slot_dir(data_dir, scheme()))
    assert argv[argv.index("--capabilities") + 1] == str(config_path("capabilities.toml").resolve())
    for flag, value in (("--max-full-solves", "900"), ("--workers", "2"), ("--dense-threads", "3")):
        assert argv[argv.index(flag) + 1] == value
    assert "--replace" not in argv and "--training-settings" not in argv
    assert env["OPENBLAS_NUM_THREADS"] == "1" and view["state"] == "preparing"


def test_one_heavy_job_on_the_machine_both_ways(tmp_path: Path) -> None:
    """抓搜尋在跑時照樣開建模、或建模在跑時照樣開搜尋（整台一張顯卡、記憶體只夠一件）。"""
    data_dir, launcher, jobs, builds = _fake(tmp_path)
    launcher.active.add("aosr-gradient-0123456789abcdef0123456789abcdef-0")
    with pytest.raises(JobBusy):
        builds.start(NAME, QUICK)
    launcher.active.clear()
    view = builds.start(NAME, QUICK)
    slot = _slot_npz(data_dir)
    slot.parent.mkdir(parents=True)
    for path in (slot, slot.with_suffix(".manifest.json"), receipt_path(slot)):
        path.write_text("{}", encoding="utf-8")
    with pytest.raises(JobBusy) as busy:
        jobs.start(NAME, GradientRequest())
    assert busy.value.job_id == view["unit"]
    with pytest.raises(JobBusy):
        builds.start(NAME, QUICK.model_copy(update={"replace": True}))


@pytest.mark.parametrize(("search", "axis", "message"), [
    (False, True, "沒有任何一區交給搜尋"),
    (True, False, "低頻軸"),
])
def test_refuses_before_launching_what_the_search_would_refuse(tmp_path: Path, search: bool, axis: bool,
                                                               message: str) -> None:
    """抓開建前不擋搜尋開跑一定會拒的情況（白等幾個小時才知道不能用）。"""
    base = scheme()
    case = base if axis else base.model_copy(update={"scene": base.scene.model_copy(update={"low_frequency_axis": None})})
    _, launcher, _, builds = _fake(tmp_path, case, search=search)
    with pytest.raises(PlanRefused, match=message):
        builds.start(NAME, QUICK)
    assert launcher.launched == []


def test_states_from_files_and_service(tmp_path: Path) -> None:
    """抓階段字不照進度檔、停了沒說「停止中」、服務不見了沒分清停止與中斷、離開碼與判決對不起來。"""
    data_dir, launcher, _, builds = _fake(tmp_path)
    first = builds.start(NAME, QUICK)
    (_work(data_dir, first) / PROGRESS_FILE).write_text(_progress().model_dump_json(), encoding="utf-8")
    view = builds.get(str(first["build_id"]))
    assert view["state"] == "building" and cast(Record, view["estimate"])["label"] == ESTIMATE_LABEL
    assert builds.stop(str(first["build_id"]))["state"] == "stopping" and launcher.stopped == [first["unit"]]
    launcher.active.clear()
    assert builds.get(str(first["build_id"]))["state"] == "stopped"
    with pytest.raises(ActionRefused):
        builds.stop(str(first["build_id"]))
    second = builds.start(NAME, QUICK)
    launcher.active.clear()
    assert builds.get(str(second["build_id"]))["state"] == "interrupted"
    third = builds.start(NAME, QUICK)
    _finish(data_dir, launcher, third, 1)
    assert builds.get(str(third["build_id"]))["state"] == "failed"
    fourth = builds.start(NAME, QUICK)
    result = SurrogateResult(verdict="not_validated", message="沒過", stop_reason="預算停止、殘差目標未達", full_solves=8,
                             rank=32, build_seconds=1.0, within_contract=False, finished_at=time.time())
    (_work(data_dir, fourth) / RESULT_FILE).write_text(result.model_dump_json(), encoding="utf-8")
    _finish(data_dir, launcher, fourth, 4)
    assert builds.get(str(fourth["build_id"]))["state"] == "not_validated"
    fifth = builds.start(NAME, QUICK)
    (_work(data_dir, fifth) / RESULT_FILE).write_text(result.model_dump_json(), encoding="utf-8")
    _finish(data_dir, launcher, fifth, 0)
    assert builds.get(str(fifth["build_id"]))["state"] == "failed"  # 離開碼 0 卻沒有「放進格子」的判決：不當成好了
    listed = cast(list[Record], builds.listing()["builds"])
    assert [item["build_id"] for item in listed] == [fifth["build_id"], fourth["build_id"], third["build_id"],
                                                     second["build_id"], first["build_id"]]


def test_launch_failure_is_recorded(tmp_path: Path) -> None:
    data_dir, launcher, _, builds = _fake(tmp_path)
    launcher.fail = "沒有使用者層的 systemd"
    with pytest.raises(LaunchFailed):
        builds.start(NAME, QUICK)
    listed = cast(list[Record], builds.listing()["builds"])
    assert len(listed) == 1 and listed[0]["state"] == "failed" and listed[0]["error"] == "沒有使用者層的 systemd"
    assert not slot_dir(data_dir, scheme()).exists()


def test_preview_shows_slot_defaults_history_and_latest(tmp_path: Path) -> None:
    """抓開建前看不到格子在不在、預設預算與行程數、這台機器前幾次花了多久、這份切法最近一次建模。"""
    data_dir, launcher, _, builds = _fake(tmp_path)
    preview = builds.preview(NAME, SurrogateRequest())
    defaults = cast(Record, preview["defaults"])
    assert preview["found"] is False and preview["exists"] is False and preview["latest"] is None
    assert defaults == {"max_full_solves": 600, "stop_residual": 0.001, "workers": 4}
    assert preview["history"] == [] and ESTIMATE_LABEL in str(preview["estimate_note"])
    view = builds.start(NAME, QUICK)
    work = _work(data_dir, view)
    (work / PROGRESS_FILE).write_text(_progress(phase="installing").model_dump_json(), encoding="utf-8")
    result = SurrogateResult(verdict="installed", message="好了", stop_reason="抽到的候選殘差都在停止值內", full_solves=96,
                             rank=267, build_seconds=70.0, validation_seconds=30.0, within_contract=True,
                             finished_at=1200.0)
    (work / RESULT_FILE).write_text(result.model_dump_json(), encoding="utf-8")
    _finish(data_dir, launcher, view, 0)
    again = builds.preview(NAME, SurrogateRequest())
    assert cast(Record, again["latest"])["state"] == "installed"
    history = cast(list[Record], again["history"])
    assert len(history) == 1 and history[0]["total_seconds"] == 200.0 and history[0]["patches"] == 7


def test_estimate_is_labelled_and_extrapolates_measured_seconds() -> None:
    """抓推估沒標「推估」、建模段沒把留出驗證算進去、或留出驗證段不照已量的速度外推。"""
    preparing = estimate(_progress(phase="preparing", rounds=[], snapshot_seconds=None, full_solves=0), 1100.0)
    assert preparing["label"] == ESTIMATE_LABEL and preparing["remaining_s"] is None
    building = estimate(_progress(), 1100.0)
    # 最近一輪 30 秒、補 32 次；預算還剩 504 次＝16 輪；留出 9×94 點 × 每次 0.5 秒
    assert building["build_remaining_s"] == 16 * 30.0 and building["validation_remaining_s"] == 9 * 94 * 0.5
    assert building["remaining_s"] == 16 * 30.0 + 9 * 94 * 0.5 and building["elapsed_s"] == 100.0
    validating = estimate(_progress(phase="validating", phase_started_at=1000.0, validation_done=100,
                                    validation_total=846), 1050.0)
    assert validating["remaining_s"] == pytest.approx((846 - 100) * 0.5)
    assert estimate(_progress(phase="checking"), 1100.0)["remaining_s"] == 0.0


def _serve(data_dir: Path, launcher: FakeLauncher, scenario: Callable[[httpx.AsyncClient], Awaitable[None]]) -> None:
    async def go() -> None:
        app = create_app(GuiSettings(engine_commit="a" * 40, data_dir=data_dir, gradient_launcher=launcher))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost") as client:
            await scenario(client)
    asyncio.run(go())


def test_routes_plan_start_read_stop_and_errors(tmp_path: Path) -> None:
    """抓網址沒接上、錯誤碼混在一起（格式錯 400、在忙 409、格子已有 409、沒這一次 404、沒這份方案 404）。"""
    data_dir, launcher, _, _ = _fake(tmp_path)

    async def scenario(client: httpx.AsyncClient) -> None:
        plan = await client.post(f"/api/gradient/schemes/{NAME}/surrogate/plan", json={})
        assert plan.status_code == 200 and plan.json()["found"] is False
        bad = await client.post(f"/api/gradient/schemes/{NAME}/surrogate", json={"workers": 0})
        assert bad.status_code == 400 and "格式不對" in bad.json()["error"]
        started = await client.post(f"/api/gradient/schemes/{NAME}/surrogate", json={"workers": 1})
        assert started.status_code == 200, started.text
        build_id = started.json()["build_id"]
        busy = await client.post(f"/api/gradient/schemes/{NAME}/surrogate", json={})
        assert busy.status_code == 409 and busy.json()["busy"] == started.json()["unit"]
        read = await client.get(f"/api/gradient/surrogates/{build_id}")
        assert read.status_code == 200 and read.json()["state"] == "preparing"
        stopped = await client.post(f"/api/gradient/surrogates/{build_id}/stop", json={})
        assert stopped.status_code == 200 and stopped.json()["state"] == "stopping"
        listed = await client.get("/api/gradient/surrogates")
        assert [item["build_id"] for item in listed.json()["builds"]] == [build_id]
        assert (await client.get(f"/api/gradient/surrogates/{'0' * 32}")).status_code == 404
        assert (await client.post("/api/gradient/schemes/nope/surrogate/plan", json={})).status_code == 404
        launcher.active.clear()
        slot = _slot_npz(data_dir)
        slot.parent.mkdir(parents=True)
        slot.write_text("{}", encoding="utf-8")
        exists = await client.post(f"/api/gradient/schemes/{NAME}/surrogate", json={})
        assert exists.status_code == 409 and "replace" in exists.json()["error"]

    _serve(data_dir, launcher, scenario)


def test_web_process_never_imports_jax_with_the_build_modules(tmp_path: Path) -> None:
    """抓建替身那幾支把 JAX 拉進網頁行程（網頁只讀進度檔，不碰物理）。"""
    code = ("import sys; import aosr.gui.app, aosr.gui.surrogate_builds, aosr.gradient.surrogate_files; "
            "raise SystemExit(1 if 'jax' in sys.modules else 0)")
    env = forward_cpu_pins(child_process_env())
    env["TMPDIR"] = str(tmp_path)
    done = subprocess.run([sys.executable, "-c", code], env=env, cwd=tmp_path, capture_output=True, text=True,
                          timeout=120)
    assert done.returncode == 0, done.stderr
