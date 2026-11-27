"""網頁「自訂房間」那一頁（票 #1315）：上傳 Rhino 檔 → 檢查、圖層 → 每層選吸音 → 背景算低頻。

伺服器這一層只讀檔、跑匯入檢查、畫模型圖（都在一秒內）；切網格與求解交給
:mod:`aosr.reporting.room_shell_run` 子行程，用跟模態診斷同一個工作管理員起、查、停。

上傳走 JSON（整站 POST 只收 JSON）：檔案內容放 base64。同一份檔用內容雜湊當代號，重傳不重存。
每個圖層從材料庫（內建＋使用者）選一筆，或選硬牆；起算時把選到的那一筆材料原樣封進快照，
子行程照同一份材料算垂直入射阻抗（:func:`aosr.physics.wall_surfaces.normal_impedance`）。
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from pydantic import ValidationError
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Route

from aosr.config.paths import config_path
from aosr.config.room_shell_import import RoomShellImportSettings, load_room_shell_import_settings
from aosr.geometry.rhino_room import read_rhino_room
from aosr.geometry.room_shell import RoomShell, check_room_shell
from aosr.gui.jobs import JobManager
from aosr.gui.material_editor import user_library_path
from aosr.gui.material_surfaces import GROUP_SHORT_WORDS, placement
from aosr.materials.library import LibraryEntry, builtin_entries, full_library
from aosr.reporting.room_shell_picture import cutaway_svg, layer_color

# 「硬牆」的選項代號：材料代號一定是小寫英文字母開頭，這個代號不會跟任何一筆材料撞到。
HARD_KEY = "-"
# 參考時間：florian-coder 上量的測試房（300 Hz 網格、94 點、單緒 PARDISO 333 秒＋模態 70 秒），只供畫面提示。
REFERENCE_SECONDS = 420
REFERENCE_LABEL = "測試房 70 m³ 在 florian-coder 上"
_MODEL_ID = re.compile(r"[0-9a-f]{64}")
_RUN_ID = re.compile(r"[0-9a-f]{32}")
_BODY_ALLOWANCE = 64 * 1024


def _layer_areas(room: RoomShell) -> dict[str, float]:
    """每個圖層的面積，直接從三角形算：檢查沒過（例如座位朝向不對）時也有，畫面才看得到圖。"""
    areas: dict[str, float] = {}
    for t in (*room.shell, *(t for solid in room.solids for t in solid.triangles)):
        cross = np.cross(np.subtract(t.b, t.a), np.subtract(t.c, t.a))
        areas[t.layer] = areas.get(t.layer, 0.0) + float(np.linalg.norm(cross)) / 2.0
    return areas


def _preview(room: RoomShell, names: list[str]) -> str:
    """原模型的剖視圖：外殼面朝房內、造型面朝外，三角形自己的法線就朝空氣。"""
    triangles = [*room.shell, *(t for solid in room.solids for t in solid.triangles)]
    if not triangles:
        return ""
    corners = np.array([(t.a, t.b, t.c) for t in triangles], dtype=np.float64)
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    return cutaway_svg(corners, normals, [names.index(t.layer) for t in triangles])


class RoomModels:
    def __init__(self, data_dir: Path, runner: tuple[str, ...], engine_commit: str, capabilities: Path) -> None:
        self.models = data_dir / "room-models"
        self.library_path = user_library_path(data_dir)
        self.models.mkdir(parents=True, exist_ok=True)
        self.settings: RoomShellImportSettings = load_room_shell_import_settings(config_path("room_shell_import.toml"))
        # 上傳那包 JSON 的上限：檔案上限換成 base64 的長度，再留一點給檔名與括號。超過就不往下讀。
        self.body_limit = (self.settings.max_file_bytes + 2) // 3 * 4 + _BODY_ALLOWANCE
        # 同一份模型、同一組吸音只算一次：還在算或算好了就回同一個工作，重按不重算。
        self.started: dict[str, dict[str, object]] = {}
        self.manager = JobManager(data_dir / "room-model-jobs", runner, engine_commit, capabilities,
                                  reference_seconds=REFERENCE_SECONDS, reference_label=REFERENCE_LABEL,
                                  result_url_template="/api/room-model-jobs/{run_id}")

    def library(self) -> tuple[LibraryEntry, ...]:
        """內建＋使用者材料；使用者檔讀不進來就只列內建（材料庫那一頁會講原因）。"""
        try:
            return full_library(self.library_path)
        except (ValueError, ValidationError, OSError):
            return builtin_entries()

    def choices(self) -> list[dict[str, str]]:
        """下拉選單：硬牆在最前，材料照「能用在哪」分組（組的順序同材料庫那一頁），組內照材料庫順序。"""
        order = list(GROUP_SHORT_WORDS)
        entries = sorted(self.library(), key=lambda entry: order.index(placement(entry).group))
        return [{"key": HARD_KEY, "label": "硬牆", "group": ""},
                *({"key": entry.id, "label": entry.label, "group": GROUP_SHORT_WORDS[placement(entry).group]}
                  for entry in entries)]

    def upload(self, document: object) -> dict[str, object]:
        if not isinstance(document, dict) or not isinstance(document.get("data_base64"), str):
            raise ValueError("要帶 data_base64（檔案內容）")
        try:
            content = base64.b64decode(document["data_base64"], validate=True)
        except binascii.Error as exc:
            raise ValueError("data_base64 不是有效的 base64") from exc
        if len(content) > self.settings.max_file_bytes:
            raise ValueError(f"檔案 {len(content) / 1e6:.1f} MB，超過上限 {self.settings.max_file_bytes / 1e6:.0f} MB")
        model_id = hashlib.sha256(content).hexdigest()
        folder = self.models / model_id
        new_folder = not folder.exists()
        folder.mkdir(exist_ok=True)
        path = folder / "model.3dm"
        fresh = not path.exists()
        if fresh:
            path.write_bytes(content)
        try:
            room = read_rhino_room(path, self.settings.max_file_bytes)
        except ValueError:
            if fresh:  # 讀不了的檔不留在磁碟上；資料夾只在這次新建的才拿掉
                path.unlink()
                if new_folder:
                    folder.rmdir()
            raise
        report = check_room_shell(room, self.settings)
        area = _layer_areas(room)
        names = sorted(area)
        summary = {"id": model_id, "ok": report.ok, "layers": names}
        (folder / "summary.json").write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
        return {**summary,
                "problems": [{"text": p.text, "blocking": p.blocking} for p in report.problems],
                "layers": [{"name": name, "area_m2": round(area[name], 2), "color": layer_color(i)}
                           for i, name in enumerate(names)],
                "speakers": len(room.speakers), "choices": self.choices(),
                "preview_svg": _preview(room, names)}

    def start(self, model_id: str, document: object) -> dict[str, object]:
        if _MODEL_ID.fullmatch(model_id) is None:
            raise ValueError("模型代號無效")
        summary_path = self.models / model_id / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError("沒有這份模型，請重新上傳")
        summary: object = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(summary, dict) or summary.get("ok") is not True:
            raise ValueError("這份模型沒過匯入檢查，不能算")
        layers = summary["layers"]
        chosen = document.get("materials") if isinstance(document, dict) else None
        if not isinstance(chosen, dict) or not isinstance(layers, list) or set(chosen) != set(layers):
            raise ValueError("每個圖層都要選一種材料")
        entries = {entry.id: entry for entry in self.library()}
        if any(not isinstance(value, str) or (value != HARD_KEY and value not in entries) for value in chosen.values()):
            raise ValueError("材料選項無效：每個圖層給一個材料代號（字串），或硬牆")
        snapshot = {"model_path": str(self.models / model_id / "model.3dm"),
                    "material_by_layer": {name: None if chosen[name] == HARD_KEY
                                          else entries[chosen[name]].model_dump(mode="json") for name in layers}}
        text = json.dumps(snapshot, ensure_ascii=False, sort_keys=True)
        earlier = self.started.get(text)
        if earlier is not None:
            try:
                if self.manager.get(str(earlier["run_id"]))["status"] in ("running", "done"):
                    return earlier
            except (ValueError, FileNotFoundError, KeyError):
                pass
        job = self.manager.start_snapshot(text, model_id[:12], job_fields={"running_label": "計算低頻中"})
        self.started[text] = job
        return job

    def job(self, run_id: str) -> dict[str, object]:
        if _RUN_ID.fullmatch(run_id) is None:
            raise ValueError("工作代號無效")
        state = self.manager.get(run_id)
        response: dict[str, object] = {"status": state["status"], "text": state["display_text"]}
        if state["status"] == "done":
            result_path = Path(str(state["result_path"]))
            response["result"] = json.loads(result_path.read_text(encoding="utf-8"))
        elif state["status"] != "running":
            tail = state.get("stderr_tail")
            response["error"] = "\n".join(str(line) for line in tail) if isinstance(tail, list) else ""
        return response


def room_model_routes(models: RoomModels, static: Path) -> list[Route]:
    async def page(request: Request) -> Response:
        return FileResponse(static / "room_model.html", media_type="text/html")

    async def bounded_json(request: Request) -> object:
        """邊讀邊數：宣告的或實際讀到的超過上限就停，不把一整包讀進記憶體再判。"""
        declared = request.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > models.body_limit:
            raise ValueError(f"上傳太大，檔案上限 {models.settings.max_file_bytes / 1e6:.0f} MB")
        chunks: list[bytes] = []
        size = 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > models.body_limit:
                raise ValueError(f"上傳太大，檔案上限 {models.settings.max_file_bytes / 1e6:.0f} MB")
            chunks.append(chunk)
        try:
            document: object = json.loads(b"".join(chunks))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError("上傳內容不是 JSON") from exc
        return document

    async def upload(request: Request) -> Response:
        try:
            return JSONResponse(models.upload(await bounded_json(request)))
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    async def start(request: Request) -> Response:
        try:
            return JSONResponse(models.start(request.path_params["model_id"], await request.json()))
        except FileNotFoundError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    async def job(request: Request) -> Response:
        try:
            return JSONResponse(models.job(request.path_params["run_id"]))
        except (ValueError, FileNotFoundError, KeyError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)

    return [Route("/room-model", page), Route("/api/room-models", upload, methods=["POST"]),
            Route("/api/room-models/{model_id}/runs", start, methods=["POST"]),
            Route("/api/room-model-jobs/{run_id}", job)]
