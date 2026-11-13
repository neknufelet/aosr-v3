"""網頁開第一、二階段擺位搜尋的工作（#700）：預覽設定、開跑、停止、接續、讀進度、放進結果清單。

一跑＝ ``data_dir/search-jobs/<工作代號>/`` 一個資料夾：
- ``job.json``：方案代號、階段、每一步（開跑／接續）的服務名與鏈檔；
- ``scheme.json``、``settings.json``：開跑那一刻方案的快照與補好的搜尋設定（命令列 ``start`` 吃的就是這兩份）；
- ``search.id``：命令列建好的搜尋資料夾代號（``start --id-file``）；搜尋資料夾本身開在 ``data_dir/searches/`` 底下，
  現成的搜尋進度頁（``/searches/<代號>``）就看得到；
- ``steps/<n>.chain.json``、``steps/<n>.log``、``steps/<n>.<步>.exit.json``、``steps/<n>.exit.json``：
  每一步的鏈檔、輸出、每一小步與整條的離開碼（``search_exec`` 寫）。

開跑＝一個使用者層獨立服務依序跑 ``start`` 與 ``auto``（細算、外圈結論、低頻診斷等附件）；接續＝只跑 ``auto``（它遇到
搜尋狀態還是「進行中」會自己接續）。網頁不記任何行程狀態在記憶體裡：照離開碼檔在不在、服務還在不在現算，網頁重開也接得回來。

停止接既有的停止記號（語意不改）：搜尋與細算兩個記號都放，批界才停；外圈已經有結論、正在補附件時記號停不到
（命令列 ``stop`` 的提示原文），或搜尋資料夾還沒建好，就停服務（最外層轉一次 SIGTERM，命令列轉成中斷、收好工作行程）。

整台機器同時只准一件重計算：跟第三階段搜尋、建替身共用同一把開跑鎖，服務清單互相看得到；命令列開的搜尋不是服務，
另讀 /proc/locks 看 ``data_dir/searches/`` 底下有沒有資料夾正被計算行程拿著。
"""
from __future__ import annotations

import json
import re
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Route

from aosr.config.paths import config_path
from aosr.gui.gradient_exec import StepExit, write_exit
from aosr.gui.gradient_jobs import (
    REPO_ROOT, SEARCH_UNIT_PREFIX, ActionRefused, LaunchFailed, Launcher, tail_lines, write_text_atomic,
)
from aosr.gui.material_editor import plain_error
from aosr.gui.search_exec import Chain, ChainStep, StepName
from aosr.gui.search_view import SEARCH_ID, held_search, search_line
from aosr.gui.wall_words import front_wall_guess
from aosr.gui.zone_design import NAME
from aosr.reporting.evaluation import purpose_settings
from aosr.reporting.scheme import Scheme, scheme_from_document
from aosr.reporting.validation import SchemeValidationError
from aosr.runtime import child_process_env
from aosr.search.labels import SEAT_LOCKED_NOTE, speaker_setup_text
from aosr.search.outer_status import snapshot_of
from aosr.search.run import SearchStatus
from aosr.search.select import select_refined
from aosr.search.settings import SearchSettings
from aosr.search.store import (
    REFINE_STOP_FILE, STOP_FILE, SearchStore, check_search_axis, checked_inputs, fill_settings_document,
)

JOBS_DIR: Final = "search-jobs"
PAGE: Final = Path(__file__).resolve().parent / "static" / "search_start.html"
SEARCHES_DIR: Final = "searches"
MODAL_CACHE_DIR: Final = "modal-cache"
JOB_ID = re.compile(r"[0-9a-f]{32}\Z")
UNIT_PREFIX: Final = SEARCH_UNIT_PREFIX
# 實量尖峰：第二階段 109 受控試跑 11.3 GB、第一階段第一批 8.8 GB（6 支工作行程）；上限約兩倍。
MEMORY_MAX: Final = "24G"
ID_FILE: Final = "search.id"
StepKind = Literal["start", "resume"]

STATE_TEXTS: Final[dict[str, str]] = {
    "starting": "開跑中（建搜尋資料夾、核方案與設定）",
    "searching": "搜尋中",
    "finishing": "細算與收尾中（細算、外圈結論、低頻診斷、交接敏感度、擺位穩定性）",
    "stopping": "停止中（這一批算完才停；收尾附件中按停會中斷附件）",
    "stopped": "已停止",
    "interrupted": "中斷了（服務不見了、被外面送了停止訊號，或命令列回報中斷）",
    "paused_time_limit": "因單題時間上限暫停（要調高上限接續，請叫助理）",
    "failed": "失敗（原因看下面的輸出）",
    "concluded": "收尾了，但沒有完成（原因看外圈結論）",
    "done": "做完了：細算完成",
}
# 外圈結論 → 這一頁的狀態；沒列的（細算用完上限、第一名是原方案…）都是「收尾了，但沒有完成」。
CONCLUSION_STATES: Final[dict[str, str]] = {
    "complete": "done", "user_stopped": "stopped", "search_failed": "failed", "refine_failed": "failed",
    "search_interrupted": "interrupted", "refine_interrupted": "interrupted", "search_paused_time_limit": "paused_time_limit",
}


class SettingsRefused(ValueError):
    """設定或方案過不了建搜尋資料夾前的檢查；訊息是哪一格、為什麼。"""


class SearchBusy(ValueError):
    def __init__(self, what: str) -> None:
        super().__init__(f"已經有一件重計算在跑（{what}）：整台機器同時只跑一件，等它做完或先停掉")
        self.what = what


class SearchJobStep(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    kind: StepKind
    unit: str = Field(min_length=1)
    launched_at: float
    chain: str
    stop_requested: bool = False


class SearchJob(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    schema_version: Literal["aosr.gui.search_job.v1"] = "aosr.gui.search_job.v1"
    job_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    scheme_id: str
    created_at: float
    stage_one: bool
    steps: tuple[SearchJobStep, ...] = ()


def _plain(error: Exception) -> str:
    if isinstance(error, SchemeValidationError):
        return "；".join(str(problem) for problem in error.problems)
    return plain_error(error)


def scheme_facts(scheme: Scheme) -> dict[str, object]:
    """表單旁邊要給人看的方案事實：房間、主位、喇叭擺法（改擺法回方案輸入頁）、有沒有家具。"""
    room = scheme.scene.room_m
    heights = {point.z for point in scheme.speakers.values()}
    try:
        scheme.scene.require_whole_walls()
        zoned = None
    except ValueError:
        zoned = ("這份方案的牆面有矩形分區（第三階段用的切法）；第一、二階段搜尋只收整面牆的方案，"
                 "預覽會照實拒收。請改用整面牆的那一份方案")
    return {"scheme_id": scheme.scheme_id, "purpose": scheme.purpose,
            "speaker_height_m": heights.pop() if len(heights) == 1 else None,
            "room_m": {"Lx": room.Lx, "Ly": room.Ly, "Lz": room.Lz},
            "primary_m": list(scheme.receiver_set.primary.position_m),
            "speaker_setup": None if scheme.speaker_setup is None else speaker_setup_text(scheme.speaker_setup),
            "has_speaker_setup": scheme.speaker_setup is not None,
            "furniture": len(scheme.furniture or ()),
            "front_wall_guess": front_wall_guess(scheme), "seat_locked_note": SEAT_LOCKED_NOTE, "zoned": zoned}


class SearchJobs:
    """``scheme_lock`` 是網頁存方案共用的那一把：只在讀方案做快照時拿；工作狀態另一把。"""

    def __init__(self, data_dir: Path, *, launcher: Launcher, engine_commit: str, capabilities: Path,
                 scheme_lock: AbstractContextManager[object], others: Callable[[], str | None] = lambda: None,
                 python: str = sys.executable) -> None:
        self.data_dir = data_dir.resolve()
        self.root = self.data_dir / JOBS_DIR
        self.root.mkdir(parents=True, exist_ok=True)
        self.searches = self.data_dir / SEARCHES_DIR
        self.modal_cache = self.data_dir / MODAL_CACHE_DIR
        self.launcher = launcher
        self.engine_commit = engine_commit
        self.capabilities = capabilities.resolve()
        self.scheme_lock = scheme_lock
        # 同一個網頁裡別的重計算（第三階段搜尋、建替身）有沒有在跑；它們的服務也在共用的服務清單裡，這裡只多看自己的工作檔。
        self.others = others
        self.python = python
        self.lock = threading.RLock()

    # ---- 讀輸入 ----

    def inputs(self, scheme_id: str) -> tuple[str, Scheme]:
        if not NAME.fullmatch(scheme_id):
            raise ValueError("方案代號只收英數字、底線與連字號")
        with self.scheme_lock:
            path = self.data_dir / "schemes" / f"{scheme_id}.json"
            if not path.is_file():
                raise FileNotFoundError(f"沒有方案「{scheme_id}」")
            text = path.read_text(encoding="utf-8")
        try:
            return text, scheme_from_document(json.loads(text))
        except (ValueError, ValidationError) as error:
            raise SettingsRefused(f"方案讀不進來：{_plain(error)}") from error

    def settings(self, scheme: Scheme, body: object) -> SearchSettings:
        """表單 → 搜尋設定：用途照方案；可從方案推出的三格照命令列同一支補；再過建資料夾前的全部檢查（不寫檔）。"""
        if not isinstance(body, dict):
            raise SettingsRefused("搜尋設定要是一個 JSON 物件")
        document = fill_settings_document(scheme, {**body, "purpose": scheme.purpose})
        try:
            settings = SearchSettings.model_validate(document)
            project, settings = checked_inputs(scheme, settings)
            purpose_settings(config_path("quality_targets.toml"), project.purpose)
            check_search_axis(project)
        except (ValueError, ValidationError) as error:
            raise SettingsRefused(_plain(error)) from error
        return settings

    def previous(self) -> dict[str, object]:
        """``data_dir/searches/`` 底下每一場的設定（新的在前）：表單「從上一場帶入」用；讀不懂的那場略過、照講。"""
        found: list[tuple[float, dict[str, object]]] = []
        skipped: list[str] = []
        if self.searches.is_dir():
            for path in self.searches.iterdir():
                if not (SEARCH_ID.fullmatch(path.name) and path.is_dir() and not path.is_symlink()):
                    continue
                try:
                    store = SearchStore.open(path)
                    created = (path / "project.json").stat().st_mtime
                except (OSError, ValueError) as error:
                    skipped.append(f"{path.name}：{_plain(error)}")
                    continue
                settings = store.settings.model_dump(mode="json")
                settings.pop("purpose", None)
                found.append((created, {"search_id": path.name, "scheme_id": store.project.scheme_id,
                                        "created_at": created, "stage_one": store.settings.stage is not None,
                                        "settings": settings}))
        return {"searches": [item for _, item in sorted(found, key=lambda pair: pair[0], reverse=True)],
                "skipped": skipped}

    # ---- 工作檔 ----

    def _dir(self, job_id: str) -> Path:
        if not JOB_ID.fullmatch(job_id):
            raise FileNotFoundError(job_id)
        return self.root / job_id

    def _job(self, job_id: str) -> SearchJob:
        path = self._dir(job_id) / "job.json"
        if not path.is_file():
            raise FileNotFoundError(f"沒有這一跑：{job_id}")
        return SearchJob.model_validate_json(path.read_text(encoding="utf-8"))

    def _save(self, job: SearchJob) -> None:
        write_text_atomic(self._dir(job.job_id) / "job.json", job.model_dump_json())

    def _steps(self, job_id: str) -> Path:
        return self._dir(job_id) / "steps"

    @staticmethod
    def _record(path: Path) -> StepExit | None:
        return StepExit.model_validate_json(path.read_text(encoding="utf-8")) if path.is_file() else None

    def folder(self, job: SearchJob) -> Path | None:
        """命令列建好的搜尋資料夾（照代號檔）；還沒建好、或代號檔指的不是合法的資料夾就是 None。"""
        try:
            name = (self._dir(job.job_id) / ID_FILE).read_text(encoding="utf-8").strip()
        except (OSError, ValueError):
            return None
        path = self.searches / name
        return path if SEARCH_ID.fullmatch(name) and path.is_dir() and not path.is_symlink() else None

    def _jobs(self) -> list[SearchJob]:
        found = []
        for path in self.root.iterdir():
            if JOB_ID.fullmatch(path.name) and (path / "job.json").is_file():
                found.append(SearchJob.model_validate_json((path / "job.json").read_text(encoding="utf-8")))
        return sorted(found, key=lambda job: job.created_at, reverse=True)

    # ---- 狀態 ----

    def _step_state(self, job: SearchJob, index: int) -> tuple[str, StepExit | None]:
        """running／finished／vanished：先看整條的離開碼檔，再看服務；服務不在時再讀一次（剛好在兩次之間結束）。"""
        done = self._steps(job.job_id) / f"{index}.exit.json"
        record = self._record(done)
        if record is not None:
            return "finished", record
        if self.launcher.is_active(job.steps[index].unit):
            return "running", None
        record = self._record(done)
        return ("finished", record) if record is not None else ("vanished", None)

    def _status(self, folder: Path | None) -> SearchStatus | None:
        if folder is None:
            return None
        try:
            return SearchStatus.model_validate_json((folder / "status.json").read_bytes())
        except (OSError, ValueError):
            return None

    @staticmethod
    def _conclusion(status: SearchStatus | None) -> str | None:
        """外圈結論，而且是對著現在這份狀態下的（快照對得上；命令列 stop 的提示也這樣判）；舊的不算。"""
        if status is None or status.outer.conclusion is None or status.outer.snapshot != snapshot_of(status):
            return None
        return status.outer.conclusion

    def _state(self, job: SearchJob, folder: Path | None) -> str:
        if not job.steps:
            return "failed"
        index = len(job.steps) - 1
        step = job.steps[index]
        status, record = self._step_state(job, index)
        auto_ran = (self._steps(job.job_id) / f"{index}.auto.exit.json").is_file()
        if status == "running":
            if step.stop_requested:
                return "stopping"
            if folder is None:
                return "starting"
            current = self._status(folder)
            if current is not None and current.state == "running":
                return "searching"  # auto 遇到進行中的搜尋先接完（接續那一步、回饋開的下一輪）才細算（#1136）
            start_done = (self._steps(job.job_id) / f"{index}.start.exit.json").is_file()
            return "finishing" if step.kind == "resume" or start_done else "searching"
        if status == "vanished":
            return "stopped" if step.stop_requested else "interrupted"
        code = None if record is None else record.exit_code
        if step.stop_requested:
            return "stopped"
        if record is not None and record.signals:
            return "interrupted"  # 不是網頁按停：外面停了服務、登出或關機
        conclusion = self._conclusion(self._status(folder))
        if auto_ran and conclusion is not None:
            return CONCLUSION_STATES.get(conclusion, "concluded")
        if code == 0:
            return "failed" if auto_ran else "stopped"  # start 正常結束卻沒接 auto：有人放了停止記號
        if code == 4:
            return "paused_time_limit"
        if code == 3:
            return "interrupted"
        return "failed"

    def _actions(self, job: SearchJob, state: str, folder: Path | None, status: SearchStatus | None) -> dict[str, bool]:
        """接續：外圈還沒結論、搜尋與細算都沒判失敗或中斷、也沒放停止記號——auto 接得下去才給。
        放進結果清單：外圈有結論、細算有第一名就給（收尾附件被停或被殺，搜尋結果照樣拿得出來）；第一階段還沒接入選。"""
        running = state in ("starting", "searching", "finishing", "stopping")
        conclusion = self._conclusion(status)
        resumable = (folder is not None and status is not None and conclusion is None
                     and status.state not in ("failed", "interrupted", "user_stopped", "paused_time_limit")
                     and status.refine.state not in ("failed", "interrupted")
                     and not (folder / STOP_FILE).exists() and not (folder / REFINE_STOP_FILE).exists())
        return {"stop": running,
                "resume": resumable and state in ("interrupted", "failed"),
                "select": (not running and not job.stage_one and status is not None and conclusion is not None
                           and status.refine.best is not None)}

    def view(self, job: SearchJob) -> dict[str, object]:
        folder = self.folder(job)
        state = self._state(job, folder)
        status = self._status(folder)
        steps = []
        for index, step in enumerate(job.steps):
            kind, record = self._step_state(job, index)
            steps.append({"kind": step.kind, "unit": step.unit, "launched_at": step.launched_at, "state": kind,
                          "stop_requested": step.stop_requested,
                          "exit_code": None if record is None else record.exit_code,
                          "error": None if record is None else record.error,
                          "finished_at": None if record is None else record.finished_at,
                          "log_tail": tail_lines(self._steps(job.job_id) / f"{index}.log")})
        summary = None if folder is None else search_line(folder)
        best = None if status is None else status.refine.best
        return {"job_id": job.job_id, "scheme_id": job.scheme_id, "created_at": job.created_at,
                "stage_one": job.stage_one, "state": state, "state_text": STATE_TEXTS[state],
                "actions": self._actions(job, state, folder, status),
                "search_id": None if folder is None else folder.name,
                "search_url": None if folder is None else f"/searches/{folder.name}",
                "summary": summary,
                "outer": None if status is None else status.outer.message,
                "refine_best": best, "steps": steps}

    def _held_search(self) -> str | None:
        """``data_dir/searches/`` 底下正被計算行程拿著的搜尋資料夾（命令列開的也算）；判不出就丟錯，不當成沒有。"""
        try:
            return held_search(self.searches)
        except ValueError as error:
            raise LaunchFailed(str(error)) from error

    def active(self) -> str | None:
        """還在算的那一件：這個資料夾的搜尋工作、整台機器的重計算服務（第三階段、建替身、別處開的搜尋）、
        網頁裡別的重計算、命令列開著的搜尋。"""
        for job in self._jobs():
            if job.steps and self._step_state(job, len(job.steps) - 1)[0] == "running":
                return job.job_id
        units = sorted(self.launcher.active_units())
        if units:
            return units[0]
        return self.others() or self._held_search()

    @contextmanager
    def launching(self) -> Iterator[None]:
        with self.lock, self.launcher.starting():
            yield

    # ---- 動作 ----

    def _launch(self, job: SearchJob, kind: StepKind, chain_steps: tuple[tuple[StepName, list[str], bool], ...],
                folder: Path | None) -> SearchJob:
        index = len(job.steps)
        steps_dir = self._steps(job.job_id)
        chain = Chain(log=str(steps_dir / f"{index}.log"), done=str(steps_dir / f"{index}.exit.json"),
                      steps=tuple(ChainStep(name=name, argv=(self.python, "-m", "aosr.search.cli", *arguments),
                                            record=str(steps_dir / f"{index}.{name}.exit.json"), append_folder=append)
                                  for name, arguments, append in chain_steps),
                      folder=None if folder is None else str(folder), search_root=str(self.searches),
                      id_file=str(self._dir(job.job_id) / ID_FILE))
        chain_path = steps_dir / f"{index}.chain.json"
        write_text_atomic(chain_path, chain.model_dump_json())
        argv = (self.python, "-m", "aosr.gui.search_exec", str(chain_path))
        step = SearchJobStep(kind=kind, unit=f"{UNIT_PREFIX}{job.job_id}-{index}", launched_at=time.time(),
                             chain=str(chain_path))
        job = job.model_copy(update={"steps": (*job.steps, step)})
        self._save(job)
        try:
            self.launcher.launch(step.unit, argv, env=child_process_env(threads=1), cwd=REPO_ROOT)
        except (LaunchFailed, OSError) as error:
            write_exit(steps_dir / f"{index}.exit.json", StepExit(exit_code=None, error=str(error), finished_at=time.time()))
            raise LaunchFailed(str(error)) from error
        return job

    def _auto_arguments(self) -> list[str]:
        return ["auto", "--modal-cache-dir", str(self.modal_cache), "--capabilities", str(self.capabilities),
                "--engine-commit", self.engine_commit]

    def preview(self, scheme_id: str, body: object) -> dict[str, object]:
        """不開跑：補好、驗過的設定，方案事實，現在整台忙不忙。"""
        _, scheme = self.inputs(scheme_id)
        facts = scheme_facts(scheme)
        settings = self.settings(scheme, body)
        with self.lock:
            busy = self.active()
        return {"settings": settings.model_dump(mode="json"), "facts": facts, "busy": busy}

    def facts(self, scheme_id: str) -> dict[str, object]:
        return scheme_facts(self.inputs(scheme_id)[1])

    def start(self, scheme_id: str, body: object) -> dict[str, object]:
        text, scheme = self.inputs(scheme_id)
        settings = self.settings(scheme, body)
        with self.launching():
            busy = self.active()
            if busy is not None:
                raise SearchBusy(busy)
            job_id = uuid.uuid4().hex
            directory = self._dir(job_id)
            (directory / "steps").mkdir(parents=True)
            write_text_atomic(directory / "scheme.json", text)
            write_text_atomic(directory / "settings.json", settings.model_dump_json(indent=1))
            self.searches.mkdir(parents=True, exist_ok=True)
            self.modal_cache.mkdir(parents=True, exist_ok=True)
            start = ["start", "--project", str(directory / "scheme.json"), "--settings", str(directory / "settings.json"),
                     "--root", str(self.searches), "--id-file", str(directory / ID_FILE),
                     "--capabilities", str(self.capabilities), "--engine-commit", self.engine_commit]
            job = SearchJob(job_id=job_id, scheme_id=scheme_id, created_at=time.time(), stage_one=settings.stage is not None)
            self._save(job)
            return self.view(self._launch(job, "start", (("start", start, False), ("auto", self._auto_arguments(), True)),
                                          None))

    def get(self, job_id: str) -> dict[str, object]:
        with self.lock:
            return self.view(self._job(job_id))

    def listing(self) -> dict[str, object]:
        """清單＋整台忙不忙；查不到服務清單或鎖清單時清單照給，``busy_problem`` 照講原因、網頁照樣不准開。"""
        with self.lock:
            jobs = [self.view(job) for job in self._jobs()]
            try:
                busy, problem = self.active(), None
            except LaunchFailed as error:
                busy, problem = None, f"查不到整台機器現在忙不忙，先不開新的一跑：{error}"
            return {"jobs": jobs, "busy": busy, "busy_problem": problem}

    def stop(self, job_id: str) -> dict[str, object]:
        """外圈還沒結論就放搜尋與細算兩個停止記號（批界才停）；資料夾還沒建好、或外圈已有結論（正在補附件）就停服務。"""
        with self.lock:
            job = self._job(job_id)
            if not job.steps or self._step_state(job, len(job.steps) - 1)[0] != "running":
                raise ActionRefused("這一跑沒有在算，不用停")
            step, folder = job.steps[-1], self.folder(job)
            if not self._stop_by_marker(folder):
                self.launcher.stop(step.unit)
            job = job.model_copy(update={"steps": (*job.steps[:-1], step.model_copy(update={"stop_requested": True}))})
            self._save(job)
            return self.view(job)

    def _stop_by_marker(self, folder: Path | None) -> bool:
        """外圈還沒結論就放兩個記號；放完再讀一次狀態——這中間剛好下了結論（開始補附件，記號停不到）就回假、改停服務。
        資料夾還沒建好、狀態讀不到、記號放不下去，一樣回假。"""
        status = self._status(folder)
        if folder is None or status is None or self._conclusion(status) is not None:
            return False
        try:
            (folder / STOP_FILE).touch()
            (folder / REFINE_STOP_FILE).touch()
        except OSError:
            return False
        return self._conclusion(self._status(folder)) is None

    def resume(self, job_id: str) -> dict[str, object]:
        with self.launching():
            job = self._job(job_id)
            folder = self.folder(job)
            state = self._state(job, folder)
            if not self._actions(job, state, folder, self._status(folder))["resume"] or folder is None:
                raise ActionRefused(f"現在是「{STATE_TEXTS[state]}」，不能接續")
            busy = self.active()
            if busy is not None:
                raise SearchBusy(busy)
            return self.view(self._launch(job, "resume", (("auto", [*self._auto_arguments(), str(folder)], False),), folder))

    def select(self, job_id: str) -> dict[str, object]:
        """細算第一名放進結果清單（命令列 ``select`` 同一支；同一個已放過就回原來那一份）。"""
        with self.lock:
            job = self._job(job_id)
            folder = self.folder(job)
            status = self._status(folder)
            state = self._state(job, folder)
            if not self._actions(job, state, folder, status)["select"] or folder is None or status is None:
                raise ActionRefused(f"現在是「{STATE_TEXTS[state]}」，沒有可以放進結果清單的細算第一名")
            best = status.refine.best
            outcome = select_refined(SearchStore.open(folder), None if best == "baseline" else best, self.data_dir)
            return {"run_id": outcome.run_id, "url": f"/results/{outcome.run_id}", "is_new": outcome.is_new,
                    "candidate": "原方案" if best == "baseline" else f"{best} 號"}


# ---- 網址 ----

def _error(message: str, code: int, **extra: object) -> JSONResponse:
    return JSONResponse({"error": message, **extra}, status_code=code)


def _answer(call: Callable[[], dict[str, object]]) -> dict[str, object] | JSONResponse:
    try:
        return call()
    except FileNotFoundError as error:
        return _error(str(error), 404)
    except SearchBusy as error:
        return _error(str(error), 409, busy=error.what)
    except ActionRefused as error:
        return _error(str(error), 409)
    except LaunchFailed as error:
        return _error(str(error), 503)
    except ValueError as error:
        return _error(str(error), 400)


class SearchJobHandlers:
    def __init__(self, jobs: SearchJobs) -> None:
        self.jobs = jobs

    async def _respond(self, call: Callable[[], dict[str, object]]) -> Response:
        answer = await run_in_threadpool(_answer, call)
        return answer if isinstance(answer, JSONResponse) else JSONResponse(answer)

    async def _with_body(self, request: Request, call: Callable[[str, object], dict[str, object]]) -> Response:
        raw = await request.body()
        try:
            body: object = json.loads(raw) if raw.strip() else {}
        except ValueError:
            return _error("搜尋設定不是 JSON", 400)
        name = request.path_params["name"]
        return await self._respond(lambda: call(name, body))

    async def preview(self, request: Request) -> Response:
        return await self._with_body(request, self.jobs.preview)

    async def start(self, request: Request) -> Response:
        return await self._with_body(request, self.jobs.start)

    async def facts(self, request: Request) -> Response:
        name = request.path_params["name"]
        return await self._respond(lambda: self.jobs.facts(name))

    async def listing(self, request: Request) -> Response:
        return await self._respond(self.jobs.listing)

    async def previous(self, request: Request) -> Response:
        return await self._respond(self.jobs.previous)

    async def item(self, request: Request) -> Response:
        job_id = request.path_params["job_id"]
        return await self._respond(lambda: self.jobs.get(job_id))

    async def action(self, request: Request) -> Response:
        job_id, action = request.path_params["job_id"], request.path_params["action"]
        calls = {"stop": self.jobs.stop, "resume": self.jobs.resume, "select": self.jobs.select}
        if action not in calls:
            return _error(f"不認識的動作：{action}", 404)
        call = calls[action]
        return await self._respond(lambda: call(job_id))


async def _page(request: Request) -> Response:
    return FileResponse(PAGE, media_type="text/html")


def search_job_routes(jobs: SearchJobs) -> list[Route]:
    handlers = SearchJobHandlers(jobs)
    return [Route("/search-start", _page), Route("/api/search-jobs", handlers.listing),
            Route("/api/search-jobs/previous", handlers.previous),
            Route("/api/search-jobs/schemes/{name}", handlers.facts),
            Route("/api/search-jobs/schemes/{name}/preview", handlers.preview, methods=["POST"]),
            Route("/api/search-jobs/schemes/{name}/start", handlers.start, methods=["POST"]),
            Route("/api/search-jobs/{job_id}", handlers.item),
            Route("/api/search-jobs/{job_id}/{action}", handlers.action, methods=["POST"])]
