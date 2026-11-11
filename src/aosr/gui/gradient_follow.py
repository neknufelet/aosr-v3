"""網頁第三階段搜尋那一頁（#1106，#1082 段 1-6b）：頁面、進度細節、把終點匯出的方案存成新方案並開一般計算。

開跑、停止、接續、終點、建替身的後端在 ``gradient_jobs``（#1093）與 ``surrogate_builds``（#1098）；這一支只補畫面要的三件：

- 進度細節：往下走那一段走了幾步（帳本 ``ledger.jsonl`` 裡 ``descent`` 的列數減掉起點那一列；照原始文字數，不載入物理）、
  每一步輸出裡「注意：」開頭的行（例如使用者材料空腔不調；命令列只印在輸出，不寫進狀態檔）、最後一步失敗時含「失敗：」
  的原文（背景開跑被拒的原因，例如物理拒算、材料庫讀不進來），不只最後幾行；
- 接結果頁：終點做完後，把匯出的方案（``scheme-endpoint.json``）照網頁存方案的同一套檢查存成新方案
  ``<原方案>-search-<這一跑代號前 8 碼>``，開跑那一刻的設計設定跟著帶過去（區代號不變；有指原方案就照指），再開一般計算；
  結果頁（#1092）的逐區表與「改善多少」照它自己的規則算，這裡不算改善；
- 查開過沒有：同一跑只開一次，記在這一跑資料夾的 ``adopted.json``，再按就回同一筆；開了計算、紀錄還沒落盤就中斷的，
  重按時找這份新方案已經開過的那一筆接回來（網頁伺服器是單一行程，``JobManager`` 也這樣假設，不防兩個行程同時按）。

結果頁沒指原方案時是拆「貼在原牆上」的加貼層找回原牆（``zone_result.derive_baseline``）；選了「貼在硬底上」的那一跑
交給搜尋的區拆不回去、會被當成沒動，細節裡照實警告（``baseline_warning``），不替使用者猜原牆。
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Route

from aosr.config.material_gradient import MOUNT_RIGID
from aosr.gradient.folder_files import LEDGER_FILE
from aosr.gui.gradient_jobs import JOB_ID, ActionRefused, GradientJobs
from aosr.gui.gradient_plan import PlanRefused
from aosr.gui.jobs import JobManager
from aosr.gui.zone_design import DesignSettings, write_design
from aosr.reporting.scheme import Scheme

PAGE: Final = Path(__file__).resolve().parent / "static" / "gradient.html"
ADOPTED_FILE: Final = "adopted.json"
DESCENT_PHASE: Final = "descent"
NOTE_PREFIX: Final = "注意："
FAILURE_MARK: Final = "失敗："
LINE_LIMIT: Final = 6
NAME_ROOM: Final = 180  # 方案代號上限 200 字，留給「-search-」與 8 碼
ADOPTED_WORD: Final = "search"
RIGID_BASELINE_WARNING: Final = (
    "這一跑選了「貼在硬底上」，開跑時的設計設定又沒指原方案：結果頁找原牆是拆「貼在原牆上」的加貼層，"
    "這一跑交給搜尋的區拆不回去、會被當成沒動，改善算不進來。要算改善，存成方案後在那份方案的設計設定指原方案。")


class Adoption(BaseModel):
    """這一跑的終點方案存成哪一份方案、開了哪一筆一般計算。"""

    model_config = ConfigDict(frozen=True, extra="forbid")
    job_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    scheme_id: str
    run_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    created_at: float


def adopted_name(scheme_id: str, job_id: str) -> str:
    return f"{scheme_id[:NAME_ROOM]}-{ADOPTED_WORD}-{job_id[:8]}"


def descent_steps(folder: Path) -> int | None:
    """往下走那一段已走幾步：帳本裡 ``descent`` 列數減起點那一列（60 步走滿是 61 列）。沒有帳本回 None。

    只照原始文字數列（網頁行程不載入物理）；讀不懂的列不算、也不拒收——帳本壞了由搜尋本體報錯。"""
    path = folder / LEDGER_FILE
    if not path.is_file():
        return None
    rows = 0
    with path.open(encoding="utf-8", errors="replace") as stream:
        next(stream, None)  # 第一列是標頭
        for line in stream:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and row.get("phase") == DESCENT_PHASE:
                rows += 1
    return max(0, rows - 1)


def _lines(path: Path, keep: Callable[[str], bool]) -> list[str]:
    if not path.is_file():
        return []
    found = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if keep(line)]
    return list(dict.fromkeys(found))[-LINE_LIMIT:]


@dataclass(frozen=True)
class GradientFollow:
    """``problems``：網頁存方案用的同一套檢查（回空的就是過了），由網頁伺服器交進來。"""

    gradient: GradientJobs
    jobs: JobManager
    problems: Callable[[object], Sequence[str]]

    def _job_dir(self, job_id: str) -> Path:
        if not JOB_ID.fullmatch(job_id):
            raise FileNotFoundError(f"沒有這一跑：{job_id}")
        return self.gradient.root / job_id

    def _adoption(self, job_id: str) -> Adoption | None:
        path = self._job_dir(job_id) / ADOPTED_FILE
        return Adoption.model_validate_json(path.read_text(encoding="utf-8")) if path.is_file() else None

    def _adoption_view(self, adoption: Adoption | None) -> dict[str, object] | None:
        if adoption is None:
            return None
        try:
            run = self.jobs.get(adoption.run_id)
            status, error = run.get("status"), run.get("error")
        except (OSError, ValueError, KeyError) as exc:
            status, error = "unknown", f"讀不到這一筆計算的狀態：{exc}"
        return {"scheme_id": adoption.scheme_id, "run_id": adoption.run_id, "created_at": adoption.created_at,
                "run_status": status, "run_error": error, "result_url": f"/results/{adoption.run_id}"}

    def _baseline_warning(self, job_id: str, view: dict[str, object]) -> str | None:
        plan = view["plan"]
        request = plan.get("request") if isinstance(plan, dict) else None
        if not isinstance(request, dict) or request.get("mount") != MOUNT_RIGID:
            return None
        snapshot = self._job_dir(job_id) / "design.json"
        if snapshot.is_file() and DesignSettings.model_validate_json(
                snapshot.read_text(encoding="utf-8")).original_scheme_id is not None:
            return None
        return RIGID_BASELINE_WARNING

    def detail(self, job_id: str) -> dict[str, object]:
        view = self.gradient.get(job_id)
        folder = view["folder"]
        steps = view["steps"]
        assert isinstance(steps, list)
        logs = [self._job_dir(job_id) / "steps" / f"{index}.log" for index in range(len(steps))]
        notes: list[str] = []
        for log in logs:
            notes.extend(_lines(log, lambda line: line.lstrip().startswith(NOTE_PREFIX)))
        failures: list[str] = []
        if steps and isinstance(steps[-1], dict) and steps[-1].get("state") == "finished" and steps[-1].get("exit_code") != 0:
            failures = _lines(logs[-1], lambda line: FAILURE_MARK in line)
        return {"job_id": job_id, "state": view["state"], "max_steps": view["max_steps"],
                "descent_steps": None if folder is None else descent_steps(Path(str(folder))),
                "notes": list(dict.fromkeys(notes)), "failures": failures,
                "baseline_warning": self._baseline_warning(job_id, view),
                "adopted": self._adoption_view(self._adoption(job_id))}

    def _started_run(self, name: str) -> str | None:
        """這份新方案已經開過的一般計算（最晚開的那一筆）；沒有回 None。

        方案名字帶這一跑代號前 8 碼，有它的一般計算只會是這裡開的（或使用者之後從首頁再開的同一份方案）：
        開了計算、``adopted.json`` 還沒落盤就中斷，重按時接回這一筆，不再開第二筆。"""
        latest: tuple[float, str] | None = None
        for state_path in (self.jobs.data_dir / "runs").glob("*.json"):
            try:
                state = self.jobs.read_state(state_path.stem)
                started = float(str(state.get("started_at")))
            except (OSError, ValueError, TypeError):
                continue
            run_id = state.get("run_id")
            if state.get("scheme_id") == name and isinstance(run_id, str) and (latest is None or started > latest[0]):
                latest = (started, run_id)
        return None if latest is None else latest[1]

    def adopt(self, job_id: str) -> dict[str, object]:
        """終點做完的這一跑：匯出的方案存成新方案、帶設計設定、開一般計算；開過就回同一筆。"""
        with self.gradient.scheme_lock:
            done = self._adoption(job_id)
            if done is not None:
                return self.detail(job_id)
            view = self.gradient.get(job_id)
            endpoint = view["endpoint"]
            if view["state"] != "done" or not isinstance(endpoint, dict):
                raise ActionRefused(f"現在是「{view['state_text']}」：終點做完、方案匯出之後才能拿去算結果頁")
            scheme_id = str(view["scheme_id"])
            name = adopted_name(scheme_id, job_id)
            document = json.loads(Path(str(endpoint["exported_scheme"])).read_text(encoding="utf-8"))
            if not isinstance(document, dict):
                raise PlanRefused("終點匯出的方案不是一份方案文件")
            document["scheme_id"] = name
            problems = self.problems(document)
            if problems:
                raise PlanRefused("終點匯出的方案過不了網頁的方案檢查：" + "；".join(problems))
            scheme = Scheme.model_validate(document)
            path = self.gradient.data_dir / "schemes" / f"{name}.json"
            if path.exists():
                if Scheme.model_validate_json(path.read_text(encoding="utf-8")) != scheme:
                    raise ActionRefused(f"已經有一份叫「{name}」的方案、內容不同；先把它改名或刪掉")
            else:
                _write_atomic(path, scheme.model_dump_json())
            snapshot = self._job_dir(job_id) / "design.json"
            if snapshot.is_file():
                design = DesignSettings.model_validate_json(snapshot.read_text(encoding="utf-8"))
                write_design(self.gradient.data_dir, design.model_copy(update={"scheme_id": name}), scheme)
            run_id = self._started_run(name)
            if run_id is None:
                run_id = str(self.jobs.start(path)["run_id"])
            adoption = Adoption(job_id=job_id, scheme_id=name, run_id=run_id, created_at=time.time())
            _write_atomic(self._job_dir(job_id) / ADOPTED_FILE, adoption.model_dump_json())
            return self.detail(job_id)


def _write_atomic(path: Path, text: str) -> None:
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _answer(call: Callable[[], dict[str, object]]) -> dict[str, object] | JSONResponse:
    try:
        return call()
    except FileNotFoundError as error:
        return JSONResponse({"error": str(error)}, status_code=404)
    except ActionRefused as error:
        return JSONResponse({"error": str(error)}, status_code=409)
    except (PlanRefused, ValidationError, ValueError) as error:
        return JSONResponse({"error": str(error)}, status_code=400)


def gradient_follow_routes(follow: GradientFollow) -> list[Route]:
    async def page(request: Request) -> Response:
        return FileResponse(PAGE, media_type="text/html")

    async def respond(call: Callable[[], dict[str, object]]) -> Response:
        answer = await run_in_threadpool(_answer, call)
        return answer if isinstance(answer, JSONResponse) else JSONResponse(answer, headers={"Cache-Control": "no-store"})

    async def detail(request: Request) -> Response:
        job_id = request.path_params["job_id"]
        return await respond(lambda: follow.detail(job_id))

    async def adopt(request: Request) -> Response:
        job_id = request.path_params["job_id"]
        return await respond(lambda: follow.adopt(job_id))

    return [Route("/gradient", page), Route("/api/gradient/runs/{job_id}/detail", detail),
            Route("/api/gradient/runs/{job_id}/adopt", adopt, methods=["POST"])]
