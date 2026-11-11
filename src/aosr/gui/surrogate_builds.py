"""網頁替新切法建替身（surrogate，縮減模型；#1098）：開建、讀進度與「要等多久」（推估）、停止。

一次建模＝ ``data_dir/surrogates/builds/<建模代號>/`` 一個資料夾：
- ``build.json``：方案代號、格子代號與路徑、請求、那一步的服務名與命令列；
- ``scheme.json``：開建那一刻方案的快照（命令列指向它，不指向網頁之後還會再存的那一份）；
- ``work/``：建模命令列（``aosr.gradient.surrogate_build``）的資料夾：``progress.json``、``result.json``、先建在這裡的 ``stage/``；
- ``steps/0.log``、``steps/0.exit.json``：輸出與離開碼（``gradient_exec`` 寫）。

建模跟搜尋共用開服務那一層（``GradientJobs.launcher``）、開跑鎖與服務名前綴：整台機器同時只跑一件第三階段的重計算，
建模中不能開搜尋、搜尋中不能開建模。建好、留出驗證兩把尺都過、搜尋開跑那一道准用閘門核過，命令列才把整份換名進格子
（``surrogate_slots.slot_dir``）；沒過就照實講、不放進格子，開跑照樣回「要先建替身」。停掉的建模不接續（要重來）。
格子裡已經有一份（例如被閘門拒收的舊替身）時，要請求裡說 ``replace``：新的核過才把舊的整份搬到 ``surrogates/retired/``。
"""
from __future__ import annotations

import json
import re
import sys
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from aosr.config.fem_reduced import load_reduced_model_settings
from aosr.config.paths import config_path
from aosr.config.surrogate_training import load_surrogate_training_settings
from aosr.gradient.surrogate_files import (
    EXIT_INSTALLED, EXIT_NOT_USABLE, EXIT_STOPPED, ESTIMATE_LABEL, PROGRESS_FILE, RESULT_FILE, SurrogateProgress,
    SurrogateResult, estimate,
)
from aosr.gui.gradient_exec import StepExit, write_exit
from aosr.gui.gradient_jobs import (
    REPO_ROOT, UNIT_PREFIX, ActionRefused, GradientJobs, JobBusy, LaunchFailed, tail_lines, write_text_atomic,
)
from aosr.gui.gradient_plan import GradientRequest, PlanRefused, plan_search
from aosr.gui.surrogate_slots import SLOTS_DIR, SURROGATE_FILE, missing_parts, slot_dir, slot_key
from aosr.runtime import child_process_env

BUILDS_DIR: Final = "builds"
RETIRED_DIR: Final = "retired"
BUILD_ID = re.compile(r"[0-9a-f]{32}\Z")
BUILD_VERSION: Final = "aosr.gui.surrogate_build.v1"
BUILD_UNIT_PREFIX: Final = f"{UNIT_PREFIX}surrogate-"
HISTORY_LIMIT: Final = 5

STATE_TEXTS: Final[dict[str, str]] = {
    "preparing": "建替身：匯出完整模型、算訓練與留出材料",
    "building": "建替身：完整解快照、補點、投影",
    "validating": "建替身：留出驗證（拿沒用來建的材料逐點完整解，跟替身比）",
    "checking": "建替身：用搜尋開跑那一道准用閘門核對",
    "installing": "建替身：放進這份切法的格子",
    "stopping": "停止中",
    "stopped": "已停止：建到一半的替身丟掉、不接續；要用得重新建",
    "interrupted": "服務不見了、沒留下離開碼（可能被強制結束或機器重開）；建到一半的不接續，要重新建",
    "failed": "建替身失敗（原因見輸出最後幾行）",
    "not_validated": "建好了但留出驗證沒過：這份替身不能用，沒有放進格子",
    "gate_refused": "建好、驗過，但搜尋開跑那一道准用閘門不收：沒有放進格子",
    "installed": "建好、留出驗證兩把尺都過、閘門核過，已放進這份切法的格子，可以開搜尋",
}
RUNNING_STATES: Final = ("preparing", "building", "validating", "checking", "installing", "stopping")


class SurrogateRequest(BaseModel):
    """開建時可以選的幾格；全部有預設。

    ``max_full_solves`` 是建模的完整解預算（不給照 ``fem_reduced.toml``）：停止規則不變（殘差達標或預算用完就停），
    建完一定跑留出驗證、兩把尺都過才放進格子。``workers`` 是完整解行程數（不給照 ``surrogate_training.toml``）、
    ``dense_threads`` 是稠密段的 BLAS 緒數（不給＝實體核數）。``replace``：格子裡已經有一份時，新的核過之後換掉它。
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    max_full_solves: int | None = Field(default=None, ge=1, strict=True)
    workers: int | None = Field(default=None, ge=1, strict=True)
    dense_threads: int | None = Field(default=None, ge=1, strict=True)
    replace: bool = False


class BuildStep(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    unit: str = Field(min_length=1)
    launched_at: float
    argv: tuple[str, ...]
    stop_requested: bool = False


class SurrogateBuild(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    schema_version: Literal["aosr.gui.surrogate_build.v1"] = BUILD_VERSION
    build_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    scheme_id: str
    slot_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    slot: str
    created_at: float
    request: SurrogateRequest
    step: BuildStep


def _read[T: BaseModel](path: Path, kind: type[T]) -> tuple[T | None, str | None]:
    if not path.is_file():
        return None, None
    try:
        return kind.model_validate_json(path.read_text(encoding="utf-8")), None
    except (OSError, ValueError) as error:
        return None, f"{path.name} 讀不懂：{type(error).__name__}"


class SurrogateBuilds:
    """跟 :class:`GradientJobs` 共用開服務那一層、開跑鎖、方案讀法與能力表；``training_settings`` 只給考卷縮組數用。"""

    def __init__(self, jobs: GradientJobs, *, python: str = sys.executable,
                 training_settings: Path | None = None) -> None:
        self.jobs = jobs
        self.data_dir = jobs.data_dir
        self.root = self.data_dir / SLOTS_DIR / BUILDS_DIR
        self.root.mkdir(parents=True, exist_ok=True)
        self.python = python
        self.training_settings = None if training_settings is None else training_settings.resolve()

    # ---- 檔 ----

    def _dir(self, build_id: str) -> Path:
        if not BUILD_ID.fullmatch(build_id):
            raise FileNotFoundError(build_id)
        return self.root / build_id

    def _build(self, build_id: str) -> SurrogateBuild:
        path = self._dir(build_id) / "build.json"
        if not path.is_file():
            raise FileNotFoundError(f"沒有這一次建模：{build_id}")
        return SurrogateBuild.model_validate_json(path.read_text(encoding="utf-8"))

    def _save(self, build: SurrogateBuild) -> None:
        write_text_atomic(self._dir(build.build_id) / "build.json", build.model_dump_json())

    def _exit_path(self, build_id: str) -> Path:
        return self._dir(build_id) / "steps" / "0.exit.json"

    def _log_path(self, build_id: str) -> Path:
        return self._dir(build_id) / "steps" / "0.log"

    def _builds(self) -> list[SurrogateBuild]:
        found = [SurrogateBuild.model_validate_json((path / "build.json").read_text(encoding="utf-8"))
                 for path in self.root.iterdir() if BUILD_ID.fullmatch(path.name) and (path / "build.json").is_file()]
        return sorted(found, key=lambda build: build.created_at, reverse=True)

    # ---- 狀態 ----

    def _exit(self, build: SurrogateBuild) -> StepExit | None:
        path = self._exit_path(build.build_id)
        return StepExit.model_validate_json(path.read_text(encoding="utf-8")) if path.is_file() else None

    def _status(self, build: SurrogateBuild) -> tuple[str, StepExit | None]:
        """running／finished／vanished：先看離開碼檔，再看服務；服務不在時再讀一次（剛好在兩次之間結束）。"""
        record = self._exit(build)
        if record is not None:
            return "finished", record
        if self.jobs.launcher.is_active(build.step.unit):
            return "running", None
        record = self._exit(build)
        return ("finished", record) if record is not None else ("vanished", None)

    def _state(self, build: SurrogateBuild, progress: SurrogateProgress | None,
               result: SurrogateResult | None) -> str:
        status, record = self._status(build)
        if status == "running":
            if build.step.stop_requested:
                return "stopping"
            return "preparing" if progress is None else progress.phase
        if status == "vanished":
            return "stopped" if build.step.stop_requested else "interrupted"
        code = None if record is None else record.exit_code
        if code == EXIT_INSTALLED and result is not None and result.verdict == "installed":
            return "installed"
        if code == EXIT_NOT_USABLE and result is not None and result.verdict != "installed":
            return result.verdict
        if code == EXIT_STOPPED or build.step.stop_requested:
            return "stopped"
        return "failed"

    def view(self, build: SurrogateBuild) -> dict[str, object]:
        work = self._dir(build.build_id) / "work"
        progress, progress_note = _read(work / PROGRESS_FILE, SurrogateProgress)
        result, result_note = _read(work / RESULT_FILE, SurrogateResult)
        state = self._state(build, progress, result)
        _, record = self._status(build)
        running = state in RUNNING_STATES
        return {"build_id": build.build_id, "scheme_id": build.scheme_id, "slot_key": build.slot_key, "slot": build.slot,
                "created_at": build.created_at, "request": build.request.model_dump(mode="json"),
                "state": state, "state_text": STATE_TEXTS[state], "actions": {"stop": running and state != "stopping"},
                "progress": None if progress is None else progress.model_dump(mode="json"),
                "estimate": estimate(progress, time.time()) if running and progress is not None else None,
                "result": None if result is None else result.model_dump(mode="json"),
                "notes": [note for note in (progress_note, result_note) if note is not None],
                "unit": build.step.unit, "exit_code": None if record is None else record.exit_code,
                "error": None if record is None else record.error, "log_tail": tail_lines(self._log_path(build.build_id))}

    def _history(self) -> list[dict[str, object]]:
        """這台機器做完的前幾次建模花了多久（做開建前的推估參考；面片與網格不同就不能直接套）。"""
        rows = []
        for build in self._builds():
            work = self._dir(build.build_id) / "work"
            progress, _ = _read(work / PROGRESS_FILE, SurrogateProgress)
            result, _ = _read(work / RESULT_FILE, SurrogateResult)
            if progress is None or result is None:
                continue
            rows.append({"build_id": build.build_id, "patches": progress.patches, "dofs": progress.dofs,
                         "axis_points": progress.axis_points, "holdout_sets": progress.holdout_sets,
                         "full_solves": result.full_solves, "rank": result.rank, "verdict": result.verdict,
                         "build_seconds": result.build_seconds, "validation_seconds": result.validation_seconds,
                         "total_seconds": result.finished_at - progress.started_at})
            if len(rows) == HISTORY_LIMIT:
                break
        return rows

    # ---- 動作 ----

    def _checked(self, scheme_id: str) -> tuple[str, Path, str]:
        """開建前網頁講得清楚的拒收（照搜尋開跑的同一套）：方案要有低頻軸、設計設定要轉得成搜尋參數；
        原牆能不能貼要算材料，留給命令列第一步判（網頁行程不載入物理）。"""
        text, scheme, design = self.jobs.inputs(scheme_id)
        if scheme.scene.low_frequency_axis is None:
            raise PlanRefused("方案場景沒有低頻軸（low_frequency_axis），搜尋開跑會拒收，不建替身")
        plan_search(scheme, design, GradientRequest(), self.jobs.search_settings())
        return text, slot_dir(self.data_dir, scheme), slot_key(scheme)

    def _latest(self, key: str) -> dict[str, object] | None:
        build = next((item for item in self._builds() if item.slot_key == key), None)
        return None if build is None else self.view(build)

    def preview(self, scheme_id: str, request: SurrogateRequest) -> dict[str, object]:
        """不開建：格子在哪、已經有沒有、這份切法最近一次建模、預設值、前幾次花了多久、現在忙不忙。"""
        _, slot, key = self._checked(scheme_id)
        sampling = load_surrogate_training_settings(self.training_settings or config_path("surrogate_training.toml"))
        reduced = load_reduced_model_settings(config_path("fem_reduced.toml"))
        with self.jobs.lock:
            busy = self.jobs.active_job()
            latest = self._latest(key)
            history = self._history()
        return {"slot": str(slot), "slot_key": key, "found": not missing_parts(slot / SURROGATE_FILE),
                "exists": slot.exists(), "latest": latest, "busy": busy, "request": request.model_dump(mode="json"),
                "defaults": {"max_full_solves": reduced.max_full_solves, "stop_residual": reduced.stop_residual,
                             "workers": sampling.compute.workers},
                "history": history,
                "estimate_note": (f"{ESTIMATE_LABEL}：開建前只有這台機器前幾次的實際花費可參考（面片、網格、頻率點數不同就不能直接套）；"
                                  "開建後照這一跑量到的速度推")}

    def start(self, scheme_id: str, request: SurrogateRequest) -> dict[str, object]:
        text, slot, key = self._checked(scheme_id)
        if slot.exists() and not request.replace:
            found = not missing_parts(slot / SURROGATE_FILE)
            raise ActionRefused("這份切法已經有替身了" + ("" if found else "（但缺件）") +
                                "：要換掉它（例如開跑說替身核不過）請求裡給 replace=true；新的建好、核過才換")
        with self.jobs.launching():
            busy = self.jobs.active_job()
            if busy is not None:
                raise JobBusy(busy)
            build_id = uuid.uuid4().hex
            directory = self._dir(build_id)
            (directory / "steps").mkdir(parents=True)
            snapshot = directory / "scheme.json"
            write_text_atomic(snapshot, text)
            arguments = ["--scheme", str(snapshot), "--work", str(directory / "work"), "--slot", str(slot),
                         "--capabilities", str(self.jobs.capabilities)]
            if request.replace:
                arguments.extend(("--replace", str(self.data_dir / SLOTS_DIR / RETIRED_DIR / f"{key}-{build_id}")))
            if request.max_full_solves is not None:
                arguments.extend(("--max-full-solves", str(request.max_full_solves)))
            if request.workers is not None:
                arguments.extend(("--workers", str(request.workers)))
            if request.dense_threads is not None:
                arguments.extend(("--dense-threads", str(request.dense_threads)))
            if self.training_settings is not None:
                arguments.extend(("--training-settings", str(self.training_settings)))
            argv = (self.python, "-m", "aosr.gui.gradient_exec", str(self._exit_path(build_id)), str(self._log_path(build_id)),
                    "--", self.python, "-m", "aosr.gradient.surrogate_build", *arguments)
            build = SurrogateBuild(build_id=build_id, scheme_id=scheme_id, slot_key=key, slot=str(slot),
                                   created_at=time.time(), request=request,
                                   step=BuildStep(unit=f"{BUILD_UNIT_PREFIX}{build_id}", launched_at=time.time(), argv=argv))
            self._save(build)
            try:
                self.jobs.launcher.launch(build.step.unit, argv, env=child_process_env(threads=1), cwd=REPO_ROOT)
            except (LaunchFailed, OSError) as error:
                write_exit(self._exit_path(build_id), StepExit(exit_code=None, error=str(error), finished_at=time.time()))
                raise LaunchFailed(str(error)) from error
            return self.view(build)

    def get(self, build_id: str) -> dict[str, object]:
        with self.jobs.lock:
            return self.view(self._build(build_id))

    def listing(self) -> dict[str, object]:
        with self.jobs.lock:
            return {"builds": [self.view(build) for build in self._builds()]}

    def stop(self, build_id: str) -> dict[str, object]:
        """停服務（最外層轉一次 SIGTERM，命令列收好工作行程、離開碼 3）；建到一半的不接續。"""
        with self.jobs.lock:
            build = self._build(build_id)
            if self._status(build)[0] != "running":
                raise ActionRefused("這一次建模沒有在算，不用停")
            self.jobs.launcher.stop(build.step.unit)
            build = build.model_copy(update={"step": build.step.model_copy(update={"stop_requested": True})})
            self._save(build)
            return self.view(build)


# ---- 網址 ----

def _error(message: str, code: int, **extra: object) -> JSONResponse:
    return JSONResponse({"error": message, **extra}, status_code=code)


def _answer(call: Callable[[], dict[str, object]]) -> dict[str, object] | JSONResponse:
    try:
        return call()
    except FileNotFoundError as error:
        return _error(str(error), 404)
    except JobBusy as error:
        return _error(str(error), 409, busy=error.job_id)
    except ActionRefused as error:
        return _error(str(error), 409)
    except LaunchFailed as error:
        return _error(str(error), 503)
    except (PlanRefused, ValueError) as error:
        return _error(str(error), 400)


class SurrogateHandlers:
    def __init__(self, builds: SurrogateBuilds) -> None:
        self.builds = builds

    async def _respond(self, call: Callable[[], dict[str, object]]) -> Response:
        answer = await run_in_threadpool(_answer, call)
        return answer if isinstance(answer, JSONResponse) else JSONResponse(answer)

    async def _with_body(self, request: Request,
                         call: Callable[[str, SurrogateRequest], dict[str, object]]) -> Response:
        raw = await request.body()
        try:
            body = SurrogateRequest.model_validate(json.loads(raw) if raw.strip() else {})
        except ValidationError as error:
            return _error("建替身選項格式不對：" + "；".join(str(item["msg"]) for item in error.errors()), 400)
        except ValueError:
            return _error("建替身選項不是 JSON", 400)
        name = request.path_params["name"]
        return await self._respond(lambda: call(name, body))

    async def preview(self, request: Request) -> Response:
        return await self._with_body(request, self.builds.preview)

    async def start(self, request: Request) -> Response:
        return await self._with_body(request, self.builds.start)

    async def listing(self, request: Request) -> Response:
        return await self._respond(self.builds.listing)

    async def item(self, request: Request) -> Response:
        build_id = request.path_params["build_id"]
        return await self._respond(lambda: self.builds.get(build_id))

    async def stop(self, request: Request) -> Response:
        build_id = request.path_params["build_id"]
        return await self._respond(lambda: self.builds.stop(build_id))


def surrogate_routes(builds: SurrogateBuilds) -> list[Route]:
    handlers = SurrogateHandlers(builds)
    return [Route("/api/gradient/surrogates", handlers.listing),
            Route("/api/gradient/surrogates/{build_id}", handlers.item),
            Route("/api/gradient/surrogates/{build_id}/stop", handlers.stop, methods=["POST"]),
            Route("/api/gradient/schemes/{name}/surrogate/plan", handlers.preview, methods=["POST"]),
            Route("/api/gradient/schemes/{name}/surrogate", handlers.start, methods=["POST"])]
