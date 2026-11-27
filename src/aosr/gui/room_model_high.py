"""網頁「自訂房間」那一頁的高音計算：起算、查工作兩條路由（票 #1309 的網頁那一段）。

吃跟低頻同一份驗過的快照（:meth:`aosr.gui.room_model.RoomModels.snapshot_text`），交給
:mod:`aosr.reporting.room_shell_high_run` 子行程；工作管理員跟低頻那一支同一套，各自一個資料夾。
同一份模型、同一組吸音只算一次：還在算或算好了就回同一個工作。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from aosr.gui.jobs import JobManager
from aosr.gui.room_model import RoomModels

# 參考時間：雲端容器量的測試房（32 個平面、1168 格、兩支喇叭、131 個頻點）約 51 秒，只供畫面提示。
REFERENCE_SECONDS = 60
REFERENCE_LABEL = "測試房 70 m³"
_RUN_ID = re.compile(r"[0-9a-f]{32}")


class RoomModelHigh:
    def __init__(self, models: RoomModels, data_dir: Path, runner: tuple[str, ...], engine_commit: str,
                 capabilities: Path) -> None:
        self.models = models
        self.started: dict[str, dict[str, object]] = {}
        self.manager = JobManager(data_dir / "room-model-high-jobs", runner, engine_commit, capabilities,
                                  reference_seconds=REFERENCE_SECONDS, reference_label=REFERENCE_LABEL,
                                  result_url_template="/api/room-model-high-jobs/{run_id}")

    def start(self, model_id: str, document: object) -> dict[str, object]:
        text = self.models.snapshot_text(model_id, document)
        earlier = self.started.get(text)
        if earlier is not None:
            try:
                if self.manager.get(str(earlier["run_id"]))["status"] in ("running", "done"):
                    return earlier
            except (ValueError, FileNotFoundError, KeyError):
                pass
        job = self.manager.start_snapshot(text, model_id[:12], job_fields={"running_label": "計算高音中"})
        self.started[text] = job
        return job

    def job(self, run_id: str) -> dict[str, object]:
        if _RUN_ID.fullmatch(run_id) is None:
            raise ValueError("工作代號無效")
        state = self.manager.get(run_id)
        response: dict[str, object] = {"status": state["status"], "text": state["display_text"]}
        if state["status"] == "done":
            response["result"] = json.loads(Path(str(state["result_path"])).read_text(encoding="utf-8"))
        elif state["status"] != "running":
            tail = state.get("stderr_tail")
            response["error"] = "\n".join(str(line) for line in tail) if isinstance(tail, list) else ""
        return response


def room_model_high_routes(high: RoomModelHigh) -> list[Route]:
    async def start(request: Request) -> Response:
        try:
            return JSONResponse(high.start(request.path_params["model_id"], await request.json()))
        except FileNotFoundError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    async def job(request: Request) -> Response:
        try:
            return JSONResponse(high.job(request.path_params["run_id"]))
        except (ValueError, FileNotFoundError, KeyError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)

    return [Route("/api/room-models/{model_id}/high-runs", start, methods=["POST"]),
            Route("/api/room-model-high-jobs/{run_id}", job)]
