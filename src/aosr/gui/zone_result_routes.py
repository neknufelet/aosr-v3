"""結果頁分區附表的兩個入口（#1092）：查（GET）與開算原牆基準（POST）。

查的時候絕不開算：109 那種房一場要半小時，只有按鈕（POST）才開。開算前先核三件事：設計結果是現在這一版物理
算的（不然基準跟它比不得）、基準跟設計真的有一區不同、同一份基準沒有正在算或已經算好。開算用的就是查的那一次
推出來的基準；鎖裡再核一次最新一筆基準計算還是查的時候看到的那一筆，中間有人開了新的一份就不再開。基準方案過不過
得了檢查交給計算行程自己判，過不了會記成「上一次沒算完」並附上原因。

設計設定檔在、卻讀不出來時照實拒收：拿不準它有沒有指原方案，不能改用拆加貼層的基準。

找基準結果只看網頁資料夾裡每一筆計算的狀態紀錄（小檔）裡方案代號等於基準代號的那幾筆，取最新一筆；算好的還要讀回
核對：方案逐欄等於這一次推出來的基準、物理身分等於設計那一份，才拿來比。

讀檔分兩層：狀態判斷只照格式讀結果檔（方案、物理身分、計算時間；不重評）；真的要比的那一刻，兩份才走跟結果頁
同一條讀回（重驗輸入、重評、判物理），被拒收就照講。109 那種結果一份重評要好幾秒，不能每次查都做。
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from aosr.config.capabilities import CapabilityTable
from aosr.config.directivity_defaults import DirectivityDefaults
from aosr.config.paths import config_path
from aosr.config.quality_targets import load_quality_targets
from aosr.gui.jobs import JobManager
from aosr.gui.modal_jobs import scheme_snapshot
from aosr.gui.zone_design import DesignSettings, read_design
from aosr.gui.zone_result import (
    PLACE_NOTE, Baseline, BaselineRefused, Improvement, OriginalWalls, ZoneRow, derive_baseline, improvement, is_zoned,
    original_walls, zone_rows)
from aosr.reporting.compare import comparison_problems
from aosr.reporting.evaluation import LoadedResult
from aosr.reporting.result import SchemeResult
from aosr.reporting.scheme import Scheme

RUN_ID_LENGTH = 32
State = Literal["same", "refused", "design_stale", "none", "running", "failed",
                "baseline_stale", "baseline_rejected", "cannot_compare", "done"]
BASELINE_LINE = ("原牆基準：同一套切法（每一區的位置與代號都不動，低頻網格才一樣）、每一區換回原牆，"
                 "跟設計用同一把尺比。每一區怎麼換回原牆寫在表的最後一欄。")
UNWRAP_NOTE = ("設計設定沒有指原方案：有加貼層的區拆掉加貼層當原牆；沒有加貼層的區當成沒動。"
               "如果有一區其實是整個換掉材料，要在設計設定指原方案，那一區的改善才算得進來。")


@dataclass(frozen=True)
class ZoneResultDeps:
    """入口要用的東西都由網頁伺服器交進來；這一支不自己讀設定、不自己開伺服器。"""

    data_dir: Path
    jobs: JobManager
    capabilities: CapabilityTable
    directivity: DirectivityDefaults
    physics_identity: str
    server_stale: Callable[[], bool]
    stale_response: Callable[[], JSONResponse]
    load_result: Callable[[Path, Path], LoadedResult]


class ZoneResultView(BaseModel):
    """結果頁分區附表整份；``zoned`` 是假時其餘都是空的、頁面整節不出。"""

    model_config = ConfigDict(frozen=True, extra="forbid")
    zoned: bool
    rows: tuple[ZoneRow, ...] = ()
    notes: tuple[str, ...] = ()
    state: State | None = None
    state_text: str = ""
    can_start: bool = False
    running: bool = False
    baseline_run_id: str | None = None
    baseline_result_url: str | None = None
    improvement: Improvement | None = None


@dataclass(frozen=True)
class Found:
    """找到的最新一筆基準計算：狀態原文；算好時加上結果檔路徑與照格式讀到的結果（讀不出就是 None 與原因）。"""

    state: dict[str, object]
    path: Path | None = None
    result: SchemeResult | None = None
    rejected: str = ""


def _settings(deps: ZoneResultDeps, scheme: Scheme) -> DesignSettings | None:
    """設計設定；沒有這一份回 None；檔在卻讀不出來就拒收（拿不準有沒有指原方案）。"""
    try:
        return read_design(deps.data_dir, scheme.scheme_id)
    except (ValueError, OSError) as exc:
        first = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        raise BaselineRefused(f"設計設定讀不出來（{first}），拿不準有沒有指原方案；到設計設定重存一次") from exc


def _original(deps: ZoneResultDeps, settings: DesignSettings | None) -> OriginalWalls | None:
    """設計設定指的原方案；沒指回 None；指了卻讀不到、或過不了檢查就拒收。"""
    if settings is None or settings.original_scheme_id is None:
        return None
    path = deps.data_dir / "schemes" / f"{settings.original_scheme_id}.json"
    try:
        scheme = Scheme.model_validate_json(path.read_bytes())
    except (ValueError, OSError) as exc:
        raise BaselineRefused(f"設計設定指的原方案「{settings.original_scheme_id}」讀不到或不在了") from exc
    return original_walls(scheme, capabilities=deps.capabilities, directivity=deps.directivity)


def _latest_run(deps: ZoneResultDeps, scheme_id: str) -> dict[str, object] | None:
    """計算紀錄裡方案代號是這一份基準的最新一筆；算中的先走一次查狀態（真的結束才收屍）。"""
    newest: tuple[float, str] | None = None
    for path in (deps.jobs.data_dir / "runs").glob("*.json"):
        if len(path.stem) != RUN_ID_LENGTH:
            continue
        try:
            state = deps.jobs.read_state(path.stem)
        except (OSError, ValueError):
            continue
        if state.get("scheme_id") != scheme_id:
            continue
        started = float(str(state.get("started_at", 0)))
        if newest is None or started > newest[0]:
            newest = (started, path.stem)
    return None if newest is None else deps.jobs.get(newest[1])


def _find(deps: ZoneResultDeps, baseline: Baseline) -> Found | None:
    state = _latest_run(deps, baseline.scheme.scheme_id)
    if state is None or state.get("status") != "done":
        return None if state is None else Found(state)
    path = deps.data_dir / "results" / f"{state['run_id']}.json"
    try:
        result = SchemeResult.model_validate_json(path.read_bytes())
    except (ValueError, OSError) as exc:
        return Found(state, path, rejected=str(exc))
    if result.scheme != baseline.scheme:
        return Found(state, path, rejected="算好的基準方案跟這一次推出來的不一樣")
    return Found(state, path, result)


def _status(deps: ZoneResultDeps, design: SchemeResult, baseline: Baseline) -> tuple[State, str, Found | None]:
    """基準現在在哪一步；只讀不寫。"""
    if not baseline.changed:
        return "same", "每一區都跟原牆基準一樣，沒有東西可比。", None
    if design.physics_identity != deps.physics_identity:
        return ("design_stale", "這一份設計結果是舊版物理算的；原牆基準要跟它同一版才比得。"
                "先按上面「用現在的程式重算」，到新的結果頁再算原牆基準。", None)
    found = _find(deps, baseline)
    if found is None:
        minutes = max(1, round(design.timings.total_s / 60)) if design.timings.total_s else None
        wait = f"估計跟設計這一次差不多久（這一次跑了約 {minutes} 分鐘）。" if minutes else ""
        return "none", f"原牆基準還沒算。{wait}", None
    status = found.state.get("status")
    if status == "running":
        return "running", str(found.state.get("display_text", "原牆基準計算中")), found
    if status != "done":
        lines = found.state.get("stderr_tail")
        tail = "；".join(str(line) for line in lines) if isinstance(lines, list) else ""
        return "failed", f"原牆基準上一次沒算完（{status}）。{tail}", found
    if found.result is None:
        return "baseline_rejected", f"原牆基準的結果讀不出來：{found.rejected}", found
    if found.result.physics_identity != design.physics_identity:
        return "baseline_stale", "原牆基準是舊版物理算的，要重算才比得。", found
    return "done", "", found


def _compare(deps: ZoneResultDeps, view: ZoneResultView, design_path: Path, baseline_path: Path) -> ZoneResultView:
    """兩份走結果頁同一條讀回再比；讀回被拒收、或比較檢查不過，照講原因、不出數字。"""
    targets = config_path("quality_targets.toml")
    try:
        design = deps.load_result(design_path, targets).result
    except (ValueError, ValidationError, json.JSONDecodeError, OSError) as exc:
        return view.model_copy(update={"state": "cannot_compare", "state_text": f"設計結果讀回時被拒收，不比：{exc}"})
    try:
        baseline = deps.load_result(baseline_path, targets).result
    except (ValueError, ValidationError, json.JSONDecodeError, OSError) as exc:
        return view.model_copy(update={"state": "baseline_rejected", "can_start": True,
                                       "state_text": f"原牆基準的結果讀回時被拒收，不比：{exc}"})
    problems = comparison_problems((design, baseline))
    if problems:
        return view.model_copy(update={"state": "cannot_compare", "state_text": "比不得：" + "；".join(problems)})
    registry = load_quality_targets(targets)
    return view.model_copy(update={"improvement": improvement(design, baseline, registry)})


def build_zone_view(deps: ZoneResultDeps, design_path: Path, design: SchemeResult) -> ZoneResultView:
    """整份附表；任何一步拒收都照講原因，逐區表照樣出（表本身不靠基準算好）。"""
    return _build(deps, design_path, design)[0]


def _build(deps: ZoneResultDeps, design_path: Path, design: SchemeResult) -> tuple[ZoneResultView, Baseline | None]:
    """附表與它用的那一份基準（推不出來是 None）；開算只用這裡推出來的那一份。"""
    scheme = design.scheme
    if not is_zoned(scheme):
        return ZoneResultView(zoned=False), None
    settings: DesignSettings | None = None
    try:
        settings = _settings(deps, scheme)
        original = _original(deps, settings)
        baseline = derive_baseline(scheme, original)
    except BaselineRefused as exc:
        rows = zone_rows(derive_baseline(scheme), settings, scheme, baseline_known=False)
        return ZoneResultView(zoned=True, rows=rows, notes=(PLACE_NOTE,), state="refused",
                              state_text=f"原牆基準推不出來：{exc}"), None
    notes = [PLACE_NOTE, BASELINE_LINE]
    if original is None:
        notes.append(UNWRAP_NOTE)
    state, text, found = _status(deps, design, baseline)
    view = ZoneResultView(zoned=True, rows=zone_rows(baseline, settings, scheme), notes=tuple(notes), state=state,
                          state_text=text, running=state == "running",
                          can_start=state in ("none", "failed", "baseline_stale", "baseline_rejected"),
                          baseline_run_id=str(found.state["run_id"]) if found is not None else None,
                          baseline_result_url=(f"/results/{found.state['run_id']}"
                                               if found is not None and found.result is not None else None))
    if state != "done" or found is None or found.path is None:
        return view, baseline
    return _compare(deps, view, design_path, found.path), baseline


def lock_refusal(view: ZoneResultView, latest: dict[str, object] | None) -> dict[str, object] | None:
    """鎖裡最後一關：最新一筆基準計算要還是鎖外查的時候看到的那一筆（沒有就還是沒有），才准開。

    查完到拿到鎖之間有人開了新的一份（算中或已經算好），都不再開；原本那一筆壞了要重算照樣准。
    """
    if latest is not None and latest.get("status") == "running":
        return {"error": "原牆基準已經在算了", "state": "running"}
    if (None if latest is None else latest.get("run_id")) != view.baseline_run_id:
        return {"error": "原牆基準剛剛有人開了新的一份，重新整理這一頁就看得到",
                "state": None if latest is None else latest.get("status")}
    return None


class ZoneResultHandlers:
    def __init__(self, deps: ZoneResultDeps) -> None:
        self.deps = deps

    def _design(self, run_id: str) -> tuple[Path, Scheme]:
        if len(run_id) != RUN_ID_LENGTH or any(char not in "0123456789abcdef" for char in run_id):
            raise ValueError("計算代號無效")
        path = self.deps.data_dir / "results" / f"{run_id}.json"
        if not path.is_file():
            raise FileNotFoundError(f"沒有這一份結果：{run_id}")
        return path, scheme_snapshot(path)

    def _view(self, run_id: str) -> tuple[ZoneResultView, Baseline | None]:
        path, scheme = self._design(run_id)
        if not is_zoned(scheme):
            return ZoneResultView(zoned=False), None
        return _build(self.deps, path, SchemeResult.model_validate_json(path.read_bytes()))

    async def item(self, request: Request) -> Response:
        if self.deps.server_stale():
            return self.deps.stale_response()
        try:
            view, _ = await run_in_threadpool(self._view, request.path_params["run_id"])
            return JSONResponse(view.model_dump(mode="json"), headers={"Cache-Control": "no-store"})
        except FileNotFoundError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        except (ValueError, ValidationError, json.JSONDecodeError, OSError) as exc:
            return JSONResponse({"error": f"分區附表讀不出來：{exc}"}, status_code=400)

    def _start(self, run_id: str) -> tuple[int, dict[str, object]]:
        """重的檢查（讀回設計結果）在鎖外做；鎖裡只重查最新一筆基準計算再開：兩個人同時按，只開一份。"""
        view, baseline = self._view(run_id)
        if not view.can_start or baseline is None:
            return 409, {"error": view.state_text or "這一份不用算原牆基準", "state": view.state}
        with self.deps.jobs._lock:
            refusal = lock_refusal(view, _latest_run(self.deps, baseline.scheme.scheme_id))
            if refusal is not None:
                return 409, refusal
            return 200, self.deps.jobs.start_snapshot(
                baseline.scheme.model_dump_json(), baseline.scheme.scheme_id,
                job_fields={"display_label": f"原牆基準（{baseline.scheme.scheme_id}）", "zone_baseline_for": run_id})

    async def start(self, request: Request) -> Response:
        if self.deps.server_stale():
            return self.deps.stale_response()
        try:
            code, body = await run_in_threadpool(self._start, request.path_params["run_id"])
            return JSONResponse(body, status_code=code)
        except FileNotFoundError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        except (ValueError, ValidationError, json.JSONDecodeError, OSError, RuntimeError) as exc:
            return JSONResponse({"error": f"原牆基準沒有開始：{exc}"}, status_code=400)


def zone_result_routes(deps: ZoneResultDeps) -> list[Route]:
    handlers = ZoneResultHandlers(deps)
    return [Route("/api/results/{run_id}/zones", handlers.item),
            Route("/api/results/{run_id}/zones/baseline", handlers.start, methods=["POST"])]
