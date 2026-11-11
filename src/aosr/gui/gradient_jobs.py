"""網頁開第三階段梯度搜尋的工作（#1093）：開跑、停止、接續、終點、讀進度。

一跑＝ ``data_dir/gradient/<工作代號>/`` 一個資料夾：
- ``job.json``：方案代號、請求、轉好的搜尋參數、替身路徑、每一步（開始／接續／終點）的服務名與命令列；
- ``scheme.json``、``design.json``：開跑那一刻方案與設計設定的快照。接續與終點要求方案檔逐位不變，命令列指向快照，
  不指向網頁之後還會再存的那一份；
- ``root/<搜尋代號>/``：搜尋命令列自己的資料夾（``project.json``、``status.json``、帳本、``result.json``、``endpoint.json``…）；
- ``steps/<n>.log``、``steps/<n>.exit.json``：每一步的輸出與離開碼（``gradient_exec`` 寫）。

每一步開一個使用者層的獨立服務（``systemd-run --user``，有記憶體上限；不是 ``--scope``，網頁重開、分頁被收都不連坐）。
網頁不記任何行程狀態在記憶體裡：狀態照「離開碼檔在不在、服務還在不在」現算，網頁重開也接得回來。
服務不見了又沒留下離開碼（被強殺、機器重開）照實講「中斷」，可以接續。
整台機器同時只准一跑（一張 12 GB 顯卡、46 GB 記憶體；搜尋最多約 16 GB、終點約 21 GB）。

終點那一份 ``endpoint.json`` 的基準是「方案存檔那一份材料」（固定的區已經算貼上），不是「同一套切法、每區都不貼」；
「改善多少」要用後者，由結果頁那一段算（#1092），這裡只交路徑、不把終點的改善數字當成改善多少講。
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from datetime import datetime
from pathlib import Path
from typing import Final, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from aosr.config.construction_presets import load_construction_presets
from aosr.config.material_gradient import MOUNT_ON_WALL, MaterialGradientSettings, load_material_gradient_settings
from aosr.config.paths import config_path
from aosr.gradient.cli import EXIT_STOPPED
from aosr.gradient.folder_files import ENDPOINT_FILE, EXPORTED_SCHEME_FILE, PROJECT_FILE, STATUS_FILE, STOP_FILE, GradientStatus
from aosr.gui.gradient_exec import StepExit, write_exit
from aosr.gui.gradient_plan import GradientRequest, PlanRefused, SearchPlan, plan_search, start_arguments
from aosr.gui.material_editor import USER_LIBRARY_NAME, plain_error, user_library_path
from aosr.gui.problem_text import checked_scheme
from aosr.gui.surrogate_slots import missing_parts, surrogate_path
from aosr.gui.zone_design import NAME, DesignSettings, read_design
from aosr.materials.library import FixedEntry, LibraryEntry, dump_user_library, load_user_library
from aosr.materials.search_library import MaterialsRefused, SearchMaterials, refuse_fixed_as_family, with_user_materials
from aosr.physics.fem_reduced_receipt import ReducedModelNotValidated
from aosr.reporting.scheme import Scheme
from aosr.runtime import child_process_env

JOBS_DIR: Final = "gradient"
JOB_ID = re.compile(r"[0-9a-f]{32}\Z")
JOB_VERSION: Final = "aosr.gui.gradient_job.v1"
UNIT_PREFIX: Final = "aosr-gradient-"
MEMORY_MAX: Final = "32G"
LOG_TAIL_LINES: Final = 8
REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNING_UNIT_STATES: Final = ("active", "activating", "deactivating", "reloading")
GONE_UNIT_STATES: Final = ("inactive", "failed", "unknown")
START_LOCK: Final = "aosr-gradient-start.lock"
# systemd 替每個使用者開的執行期資料夾（``XDG_RUNTIME_DIR`` 指的就是 /run/user/<使用者編號>）：跟 ``systemctl --user`` 同一個範圍。
USER_RUNTIME_ROOT: Final = Path("/run/user")
StepKind = Literal["start", "resume", "endpoint"]

STATE_TEXTS: Final[dict[str, str]] = {
    "checking": "開跑前核對中（匯出完整模型、核替身）",
    "searching": "搜尋中",
    "stopping": "停止中（下一次求值之前停）",
    "stopped": "已停止，可以接續",
    "interrupted": "服務不見了、沒留下離開碼（可能被強制結束或機器重開），可以接續",
    "failed": "搜尋失敗",
    "surrogate_refused": "替身核不過（例如程式換了物理身分、或替身跟這份切法對不上），要重建替身",
    "searched": "搜尋做完了，可以跑終點",
    "endpoint_running": "終點計算中（完整模型重算）",
    "endpoint_stopping": "終點停止中",
    "endpoint_stopped": "終點已停止，可以重跑終點",
    "endpoint_interrupted": "終點的服務不見了、沒留下離開碼，可以重跑終點",
    "endpoint_failed": "終點失敗，可以重跑終點",
    "done": "做完了：終點算好、方案已匯出",
}
ENDPOINT_BASELINE_TEXT: Final = ("終點檔裡的基準是方案存檔那一份材料（固定的區已經算貼上），不是「同一套切法、每區都不貼」；"
                                 "改善多少以結果頁算的為準")


class NeedsSurrogate(ValueError):
    """這份切法還沒有替身；``slot`` 是該放的位置（1-6c 建好放進去）。"""

    def __init__(self, message: str, slot: Path) -> None:
        super().__init__(message)
        self.slot = slot


class JobBusy(ValueError):
    def __init__(self, job_id: str) -> None:
        super().__init__(f"已經有一跑在算（{job_id}）：整台機器同時只跑一跑，等它做完或先停掉")
        self.job_id = job_id


class ActionRefused(ValueError):
    """這一跑現在的狀態不能做這件事。"""


class LaunchFailed(RuntimeError):
    """開不了服務；訊息是開服務那一層的原文。"""


class Launcher(Protocol):
    """開、查、停一個步驟的服務；網頁用 systemd，考卷換成本機行程或假的。"""

    def launch(self, unit: str, argv: Sequence[str], *, env: dict[str, str], cwd: Path) -> None: ...

    def is_active(self, unit: str) -> bool: ...

    def active_units(self) -> set[str]: ...

    def stop(self, unit: str) -> None: ...

    def starting(self) -> AbstractContextManager[object]:
        """查忙到開好服務整段要拿的鎖；凡是看得到同一批服務的網頁行程（不管開哪一個資料夾）都要搶同一把。"""
        ...


Runner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def _run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(command), capture_output=True, text=True, timeout=30, check=False)


class SystemdLauncher:
    """每一步一個使用者層服務：記憶體上限、OOM 不連坐、``KillMode=mixed``（停止只送給最外層，由它轉一次）。

    服務名每一步都不同（工作代號＋第幾步），``--collect`` 讓結束（含失敗）的服務自己卸掉，不會擋住下一步用名字。
    服務不繼承網頁的環境：環境變數逐格 ``--setenv`` 帶過去、工作目錄指定成 repo 根。
    """

    def __init__(self, *, memory_max: str = MEMORY_MAX, run: Runner = _run, lock_dir: Path | None = None) -> None:
        self.memory_max = memory_max
        self.run = run
        self.lock_dir = USER_RUNTIME_ROOT / str(os.getuid()) if lock_dir is None else lock_dir

    @contextmanager
    def starting(self) -> Iterator[None]:
        """檔案鎖放在這個使用者的執行期資料夾：整台這個使用者的每一個網頁行程都搶同一把，
        兩個不同資料夾的網頁同時查到「整台沒在跑」也不會各開一跑。資料夾不在（沒有使用者層的 systemd）就開不了。"""
        try:
            handle = (self.lock_dir / START_LOCK).open("a")
        except OSError as error:
            raise LaunchFailed(f"開跑鎖開不了（{self.lock_dir / START_LOCK}）：{error}") from error
        with handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def command(self, unit: str, argv: Sequence[str], *, env: dict[str, str], cwd: Path) -> list[str]:
        return ["systemd-run", "--user", f"--unit={unit}", "--collect", "--quiet",
                "-p", f"MemoryMax={self.memory_max}", "-p", "OOMPolicy=continue", "-p", "KillMode=mixed",
                f"--working-directory={cwd}", *(f"--setenv={name}={value}" for name, value in sorted(env.items())),
                "--", *argv]

    def launch(self, unit: str, argv: Sequence[str], *, env: dict[str, str], cwd: Path) -> None:
        done = self.run(self.command(unit, argv, env=env, cwd=cwd))
        if done.returncode != 0:
            raise LaunchFailed(f"systemd-run 開不了服務 {unit}（離開碼 {done.returncode}）：{done.stderr.strip()}")

    def is_active(self, unit: str) -> bool:
        """查不到（使用者服務匯流排連不上之類）就丟錯，不准當成「不在」——當成不在會放行第二跑。"""
        done = self.run(["systemctl", "--user", "is-active", unit])
        state = done.stdout.strip()
        if state in RUNNING_UNIT_STATES:
            return True
        if state in GONE_UNIT_STATES:
            return False
        raise LaunchFailed(f"查不到服務 {unit} 的狀態（離開碼 {done.returncode}）：{done.stderr.strip() or state}")

    def active_units(self) -> set[str]:
        """整台機器（這個使用者）還在的第三階段服務，不管是哪一個網頁資料夾開的。"""
        done = self.run(["systemctl", "--user", "list-units", "--plain", "--no-legend",
                         "--state=" + ",".join(RUNNING_UNIT_STATES), f"{UNIT_PREFIX}*"])
        if done.returncode != 0:
            raise LaunchFailed(f"查不到現在有哪些服務（離開碼 {done.returncode}）：{done.stderr.strip()}")
        return {line.split()[0].removesuffix(".service") for line in done.stdout.splitlines() if line.strip()}

    def stop(self, unit: str) -> None:
        done = self.run(["systemctl", "--user", "stop", "--no-block", unit])
        if done.returncode != 0:
            raise LaunchFailed(f"停不了服務 {unit}（離開碼 {done.returncode}）：{done.stderr.strip()}")


class JobStep(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    kind: StepKind
    unit: str = Field(min_length=1)
    launched_at: float
    argv: tuple[str, ...]
    stop_requested: bool = False


class GradientJob(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    schema_version: Literal["aosr.gui.gradient_job.v1"] = JOB_VERSION
    job_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    scheme_id: str
    created_at: float
    plan: SearchPlan
    surrogate: str
    steps: tuple[JobStep, ...] = ()


def write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def tail_lines(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()[-LOG_TAIL_LINES:]


class GradientJobs:
    """``scheme_lock`` 是網頁存方案、設計設定、封存共用的那一把：只在讀方案與設計設定做快照時拿；工作狀態另一把。"""

    def __init__(self, data_dir: Path, *, launcher: Launcher, engine_commit: str, capabilities: Path,
                 scheme_lock: AbstractContextManager[object], settings_path: Path | None = None,
                 python: str = sys.executable) -> None:
        # 服務的工作目錄是 repo 根，交給命令列的路徑一律先換成絕對路徑（相對路徑在服務裡會指到別處）。
        self.data_dir = data_dir.resolve()
        self.root = self.data_dir / JOBS_DIR
        self.root.mkdir(parents=True, exist_ok=True)
        self.launcher = launcher
        self.engine_commit = engine_commit
        self.capabilities = capabilities.resolve()
        self.scheme_lock = scheme_lock
        self.settings_path = None if settings_path is None else settings_path.resolve()
        self.python = python
        self.lock = threading.RLock()

    # ---- 讀輸入 ----

    def _materials(self) -> tuple[SearchMaterials, tuple[LibraryEntry, ...]]:
        """搜尋設定接上使用者材料庫的可最佳化材料（#1082 段 2-3a），與使用者材料本身；材料庫讀不進來照講原因、不開。"""
        settings = self.search_settings()
        try:
            entries = load_user_library(user_library_path(self.data_dir))
        except (OSError, ValueError, ValidationError) as error:
            raise PlanRefused(f"使用者材料庫（{USER_LIBRARY_NAME}）讀不進來，搜尋不開：{plain_error(error)}") from error
        try:
            return with_user_materials(settings, load_construction_presets(config_path("construction_presets.toml")),
                                       entries), entries
        except MaterialsRefused as error:
            raise PlanRefused(str(error)) from error

    def _plan(self, scheme: Scheme, design: DesignSettings | None, request: GradientRequest) -> tuple[SearchPlan, SearchMaterials,
                                                                                                         tuple[LibraryEntry, ...]]:
        materials, entries = self._materials()
        try:
            refuse_fixed_as_family((request.family,), entries)
        except MaterialsRefused as error:
            raise PlanRefused(str(error)) from error
        fixed_ids = frozenset((*(preset.id for preset in load_construction_presets(config_path("construction_presets.toml"))),
                               *(entry.id for entry in entries if isinstance(entry, FixedEntry))))
        return plan_search(scheme, design, request, materials.settings, fixed_ids), materials, entries

    def search_settings(self) -> MaterialGradientSettings:
        return load_material_gradient_settings(self.settings_path or config_path("material_gradient.toml"))

    def inputs(self, scheme_id: str) -> tuple[str, Scheme, DesignSettings | None]:
        if not NAME.fullmatch(scheme_id):
            raise ValueError("方案代號只收英數字、底線與連字號")
        with self.scheme_lock:
            path = self.data_dir / "schemes" / f"{scheme_id}.json"
            if not path.is_file():
                raise FileNotFoundError(f"沒有方案「{scheme_id}」")
            text = path.read_text(encoding="utf-8")
            design = read_design(self.data_dir, scheme_id)
        return text, checked_scheme(json.loads(text)), design

    def _surrogate(self, scheme: Scheme) -> Path:
        path = surrogate_path(self.data_dir, scheme)
        missing = missing_parts(path)
        if missing:
            raise NeedsSurrogate(f"要先建替身：這一份切法還沒有替身（缺{'、'.join(missing)}），"
                                 "不拿別份切法的替身頂替；建好的替身放在這一格", path.parent)
        return path

    # ---- 工作檔 ----

    def _dir(self, job_id: str) -> Path:
        if not JOB_ID.fullmatch(job_id):
            raise FileNotFoundError(job_id)
        return self.root / job_id

    def _job(self, job_id: str) -> GradientJob:
        path = self._dir(job_id) / "job.json"
        if not path.is_file():
            raise FileNotFoundError(f"沒有這一跑：{job_id}")
        return GradientJob.model_validate_json(path.read_text(encoding="utf-8"))

    def _save(self, job: GradientJob) -> None:
        write_text_atomic(self._dir(job.job_id) / "job.json", job.model_dump_json())

    def _exit_path(self, job_id: str, index: int) -> Path:
        return self._dir(job_id) / "steps" / f"{index}.exit.json"

    def _log_path(self, job_id: str, index: int) -> Path:
        return self._dir(job_id) / "steps" / f"{index}.log"

    def _exit(self, job_id: str, index: int) -> StepExit | None:
        path = self._exit_path(job_id, index)
        return StepExit.model_validate_json(path.read_text(encoding="utf-8")) if path.is_file() else None

    def folder(self, job: GradientJob) -> Path | None:
        """搜尋命令列在 ``root`` 底下開的那一個資料夾；開跑前核對那一段還沒有。"""
        root = self._dir(job.job_id) / "root"
        found = sorted(path for path in root.iterdir() if (path / PROJECT_FILE).is_file()) if root.is_dir() else []
        if len(found) > 1:
            raise RuntimeError(f"{root} 底下有 {len(found)} 個搜尋資料夾，一跑只該有一個")
        return found[0] if found else None

    # ---- 狀態 ----

    def _step_state(self, job: GradientJob, index: int) -> tuple[str, StepExit | None]:
        """running／finished／vanished：先看離開碼檔，再看服務；服務不在時再讀一次（剛好在兩次之間結束）。"""
        record = self._exit(job.job_id, index)
        if record is not None:
            return "finished", record
        if self.launcher.is_active(job.steps[index].unit):
            return "running", None
        record = self._exit(job.job_id, index)
        return ("finished", record) if record is not None else ("vanished", None)

    def _state(self, job: GradientJob, folder: Path | None) -> str:
        if not job.steps:
            return "failed"
        step = job.steps[-1]
        status, record = self._step_state(job, len(job.steps) - 1)
        endpoint = step.kind == "endpoint"
        if status == "running":
            if step.stop_requested:
                return "endpoint_stopping" if endpoint else "stopping"
            return "endpoint_running" if endpoint else ("searching" if folder is not None else "checking")
        if status == "vanished":
            if step.stop_requested:
                return "endpoint_stopped" if endpoint else "stopped"
            return "endpoint_interrupted" if endpoint else "interrupted"
        code = None if record is None else record.exit_code
        if code == 0:
            return "done" if endpoint else "searched"
        if code == EXIT_STOPPED:
            return "endpoint_stopped" if endpoint else "stopped"
        if endpoint:
            return "endpoint_failed"
        refused = f"失敗：{ReducedModelNotValidated.__name__}："  # 命令列印「梯度搜尋開始失敗：<例外名>：<原因>」
        log = self._log_path(job.job_id, len(job.steps) - 1)
        if step.kind == "start" and log.is_file() and refused in log.read_text(encoding="utf-8", errors="replace"):
            return "surrogate_refused"
        return "failed"

    def _actions(self, job: GradientJob, state: str, folder: Path | None) -> dict[str, bool]:
        running = state in ("checking", "searching", "stopping", "endpoint_running", "endpoint_stopping")
        return {"stop": running,
                "resume": folder is not None and state in ("stopped", "interrupted", "failed"),
                "endpoint": folder is not None and state in ("searched", "endpoint_stopped", "endpoint_interrupted",
                                                             "endpoint_failed")}

    def _progress(self, folder: Path | None) -> tuple[dict[str, object] | None, str | None]:
        if folder is None or not (folder / STATUS_FILE).is_file():
            return None, None
        try:
            return GradientStatus.model_validate_json((folder / STATUS_FILE).read_text(encoding="utf-8")).model_dump(
                mode="json"), None
        except (OSError, ValueError) as error:
            return None, f"狀態檔讀不懂：{type(error).__name__}"

    def view(self, job: GradientJob) -> dict[str, object]:
        folder = self.folder(job)
        state = self._state(job, folder)
        progress, note = self._progress(folder)
        steps = []
        for index, step in enumerate(job.steps):
            status, record = self._step_state(job, index)
            steps.append({"kind": step.kind, "unit": step.unit, "launched_at": step.launched_at, "state": status,
                          "stop_requested": step.stop_requested,
                          "exit_code": None if record is None else record.exit_code,
                          "error": None if record is None else record.error,
                          "finished_at": None if record is None else record.finished_at,
                          "log_tail": tail_lines(self._log_path(job.job_id, index))})
        endpoint = None
        if folder is not None and (folder / ENDPOINT_FILE).is_file():
            endpoint = {"file": str(folder / ENDPOINT_FILE), "exported_scheme": str(folder / EXPORTED_SCHEME_FILE),
                        "baseline_note": ENDPOINT_BASELINE_TEXT}
        return {"job_id": job.job_id, "scheme_id": job.scheme_id, "created_at": job.created_at,
                "state": state, "state_text": STATE_TEXTS[state], "actions": self._actions(job, state, folder),
                "plan": job.plan.model_dump(mode="json"), "max_steps": job.plan.max_steps,
                "folder": None if folder is None else str(folder), "progress": progress, "progress_note": note,
                "steps": steps, "endpoint": endpoint}

    def _jobs(self) -> list[GradientJob]:
        found = []
        for path in self.root.iterdir():
            if JOB_ID.fullmatch(path.name) and (path / "job.json").is_file():
                found.append(GradientJob.model_validate_json((path / "job.json").read_text(encoding="utf-8")))
        return sorted(found, key=lambda job: job.created_at, reverse=True)

    def active_job(self) -> str | None:
        """還在算的那一跑：先看這個資料夾的工作，再看整台機器有沒有別處開的第三階段服務。"""
        for job in self._jobs():
            if job.steps and self._step_state(job, len(job.steps) - 1)[0] == "running":
                return job.job_id
        units = sorted(self.launcher.active_units())
        return units[0] if units else None

    @contextmanager
    def launching(self) -> Iterator[None]:
        """查忙到開好服務整段拿開服務那一層給的鎖（:meth:`Launcher.starting`，整台共用）：同一個資料夾的兩個網頁行程
        （重開交接那一瞬間）、或兩個不同資料夾的網頁，都不會同時查到空閒而各開一跑；鎖放掉時服務已經在服務清單裡了。"""
        with self.lock, self.launcher.starting():
            yield

    # ---- 動作 ----

    def _launch(self, job: GradientJob, kind: StepKind, arguments: Sequence[str]) -> GradientJob:
        index = len(job.steps)
        argv = (self.python, "-m", "aosr.gui.gradient_exec", str(self._exit_path(job.job_id, index)),
                str(self._log_path(job.job_id, index)), "--", self.python, "-m", "aosr.gradient.cli", *arguments)
        step = JobStep(kind=kind, unit=f"{UNIT_PREFIX}{job.job_id}-{index}", launched_at=time.time(), argv=argv)
        job = job.model_copy(update={"steps": (*job.steps, step)})
        self._save(job)
        env = child_process_env(threads=1)
        try:
            self.launcher.launch(step.unit, argv, env=env, cwd=REPO_ROOT)
        except (LaunchFailed, OSError) as error:
            write_exit(self._exit_path(job.job_id, index), StepExit(exit_code=None, error=str(error),
                                                                   finished_at=time.time()))
            raise LaunchFailed(str(error)) from error
        return job

    def preview(self, scheme_id: str, request: GradientRequest) -> dict[str, object]:
        """不開跑：轉好的參數、替身在不在、預設值、現在有沒有一跑在算。"""
        _, scheme, design = self.inputs(scheme_id)
        plan, materials, _ = self._plan(scheme, design, request)
        settings = materials.settings
        path = surrogate_path(self.data_dir, scheme)
        missing = missing_parts(path)
        with self.lock:
            busy = self.active_job()
        return {"plan": plan.model_dump(mode="json"), "arguments": start_arguments(plan),
                "surrogate": {"found": not missing, "slot": str(path.parent), "missing": missing,
                              "message": None if not missing else "要先建替身"},
                "defaults": {"max_steps": settings.descent.max_steps, "family": settings.family[0].id,
                             "families": [family.id for family in settings.family], "mount": MOUNT_ON_WALL,
                             "device": GradientRequest().device},
                "busy": busy}

    def start(self, scheme_id: str, request: GradientRequest) -> dict[str, object]:
        text, scheme, design = self.inputs(scheme_id)
        plan, _, entries = self._plan(scheme, design, request)
        surrogate = self._surrogate(scheme)
        with self.launching():
            busy = self.active_job()
            if busy is not None:
                raise JobBusy(busy)
            job_id = uuid.uuid4().hex
            directory = self._dir(job_id)
            (directory / "root").mkdir(parents=True)
            (directory / "steps").mkdir()
            snapshot = directory / "scheme.json"
            write_text_atomic(snapshot, text)
            if design is not None:
                write_text_atomic(directory / "design.json", design.model_dump_json())
            arguments = ["start", "--scheme", str(snapshot), "--surrogate", str(surrogate), "--purpose", scheme.purpose,
                         "--root", str(directory / "root"), "--capabilities", str(self.capabilities),
                         "--engine-commit", self.engine_commit]
            if self.settings_path is not None:
                arguments.extend(("--settings", str(self.settings_path)))
            if entries:
                library = directory / USER_LIBRARY_NAME
                write_text_atomic(library, dump_user_library(entries))
                arguments.extend(("--materials", str(library)))
            job = GradientJob(job_id=job_id, scheme_id=scheme_id, created_at=time.time(), plan=plan,
                              surrogate=str(surrogate))
            self._save(job)
            return self.view(self._launch(job, "start", [*arguments, *start_arguments(plan)]))

    def get(self, job_id: str) -> dict[str, object]:
        with self.lock:
            return self.view(self._job(job_id))

    def listing(self) -> dict[str, object]:
        with self.lock:
            return {"jobs": [self.view(job) for job in self._jobs()]}

    def _marker_holds(self, step: JobStep, folder: Path) -> bool:
        """放停止記號停得到嗎：搜尋的步驟才看記號；接續那一步還要等工作行程已經起來（狀態檔是這一步之後寫的）——
        命令列接續開頭會先清掉舊記號，在那之前放的記號會被一起清掉。"""
        if step.kind == "start":
            return True
        if step.kind != "resume":
            return False
        try:
            status = GradientStatus.model_validate_json((folder / STATUS_FILE).read_text(encoding="utf-8"))
            return datetime.fromisoformat(status.started_at).timestamp() >= step.launched_at
        except (OSError, ValueError):
            return False

    def stop(self, job_id: str) -> dict[str, object]:
        """記號停得到（:meth:`_marker_holds`）就放停止記號（下一次求值之前停、回「已停止」）；其餘（開跑前核對、
        接續剛開、終點）停服務：最外層轉一次 SIGTERM，命令列轉成中斷、收好工作行程，一樣回「已停止」。"""
        with self.lock:
            job = self._job(job_id)
            if not job.steps or self._step_state(job, len(job.steps) - 1)[0] != "running":
                raise ActionRefused("這一跑沒有在算，不用停")
            step, folder = job.steps[-1], self.folder(job)
            if folder is not None and self._marker_holds(step, folder):
                (folder / STOP_FILE).touch()
            else:
                self.launcher.stop(step.unit)
            job = job.model_copy(update={"steps": (*job.steps[:-1], step.model_copy(update={"stop_requested": True}))})
            self._save(job)
            return self.view(job)

    def _follow_up(self, job_id: str, action: Literal["resume", "endpoint"]) -> dict[str, object]:
        with self.launching():
            job = self._job(job_id)
            folder = self.folder(job)
            state = self._state(job, folder)
            if not self._actions(job, state, folder)[action] or folder is None:
                word = "接續" if action == "resume" else "跑終點"
                raise ActionRefused(f"現在是「{STATE_TEXTS[state]}」，不能{word}")
            busy = self.active_job()
            if busy is not None:
                raise JobBusy(busy)
            if action == "resume":
                return self.view(self._launch(job, "resume", ["resume", str(folder)]))
            return self.view(self._launch(job, "endpoint", ["endpoint", str(folder), "--device", job.plan.request.device]))

    def resume(self, job_id: str) -> dict[str, object]:
        return self._follow_up(job_id, "resume")

    def endpoint(self, job_id: str) -> dict[str, object]:
        return self._follow_up(job_id, "endpoint")


# ---- 網址 ----

def _error(message: str, code: int, **extra: object) -> JSONResponse:
    return JSONResponse({"error": message, **extra}, status_code=code)


async def _request_body(request: Request) -> GradientRequest:
    raw = await request.body()
    return GradientRequest.model_validate(json.loads(raw) if raw.strip() else {})


def _answer(call: Callable[[], dict[str, object]]) -> dict[str, object] | JSONResponse:
    try:
        return call()
    except FileNotFoundError as error:
        return _error(str(error), 404)
    except NeedsSurrogate as error:
        return _error(str(error), 409, needs="surrogate", slot=str(error.slot))
    except JobBusy as error:
        return _error(str(error), 409, busy=error.job_id)
    except ActionRefused as error:
        return _error(str(error), 409)
    except LaunchFailed as error:
        return _error(str(error), 503)
    except (PlanRefused, ValueError) as error:
        return _error(str(error), 400)


class GradientHandlers:
    def __init__(self, jobs: GradientJobs) -> None:
        self.jobs = jobs

    async def _respond(self, call: Callable[[], dict[str, object]]) -> Response:
        answer = await run_in_threadpool(_answer, call)
        return answer if isinstance(answer, JSONResponse) else JSONResponse(answer)

    async def _with_body(self, request: Request,
                         call: Callable[[str, GradientRequest], dict[str, object]]) -> Response:
        try:
            body = await _request_body(request)
        except ValidationError as error:
            return _error("開跑選項格式不對：" + "；".join(str(item["msg"]) for item in error.errors()), 400)
        except ValueError:
            return _error("開跑選項不是 JSON", 400)
        name = request.path_params["name"]
        return await self._respond(lambda: call(name, body))

    async def preview(self, request: Request) -> Response:
        return await self._with_body(request, self.jobs.preview)

    async def start(self, request: Request) -> Response:
        return await self._with_body(request, self.jobs.start)

    async def listing(self, request: Request) -> Response:
        return await self._respond(self.jobs.listing)

    async def item(self, request: Request) -> Response:
        job_id = request.path_params["job_id"]
        return await self._respond(lambda: self.jobs.get(job_id))

    async def action(self, request: Request) -> Response:
        job_id, action = request.path_params["job_id"], request.path_params["action"]
        calls = {"stop": self.jobs.stop, "resume": self.jobs.resume, "endpoint": self.jobs.endpoint}
        if action not in calls:
            return _error(f"不認識的動作：{action}", 404)
        call = calls[action]
        return await self._respond(lambda: call(job_id))


def gradient_routes(jobs: GradientJobs) -> list[Route]:
    handlers = GradientHandlers(jobs)
    return [Route("/api/gradient/jobs", handlers.listing),
            Route("/api/gradient/jobs/{job_id}", handlers.item),
            Route("/api/gradient/jobs/{job_id}/{action}", handlers.action, methods=["POST"]),
            Route("/api/gradient/schemes/{name}/plan", handlers.preview, methods=["POST"]),
            Route("/api/gradient/schemes/{name}/start", handlers.start, methods=["POST"])]
