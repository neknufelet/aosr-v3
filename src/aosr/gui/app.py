"""本機單人網頁：輸入、驗證、存檔、2D 圖與獨立計算。"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Route

from aosr.config.capabilities import CapabilityTable, load_capabilities
from aosr.config.directivity_defaults import DirectivityDefaults, load_directivity_defaults
from aosr.config.paths import config_path
from aosr.geometry.shoebox import Point
from aosr.gui.jobs import JobManager
from aosr.physics.report_source import default_source_model
from aosr.reporting.display import impedance_multiple
from aosr.reporting.scheme import Scheme
from aosr.reporting.validation import SchemeValidationError, validate_scheme, validated_scheme


STATIC = Path(__file__).parent / "static"
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")
COMMIT = re.compile(r"[0-9a-f]{40}\Z")


def repo_root() -> Path:
    """從本套件往上找到本專案 pyproject.toml。"""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("找不到 repo 根目錄")


@dataclass(frozen=True)
class GuiSettings:
    engine_commit: str
    data_dir: Path = Path.home() / "room-acoustic-data"
    runner: tuple[str, ...] | None = None


def _scheme_path(data_dir: Path, name: str) -> Path:
    if not SAFE_ID.fullmatch(name) or len(name) > 200:
        raise ValueError("方案代號只收英數字、底線與連字號")
    return data_dir / "schemes" / f"{name}.json"


def _read_scheme(path: Path) -> Scheme:
    loaded: object = json.loads(path.read_text(encoding="utf-8"))
    return validated_scheme(loaded)


def _require_scheme_id(path: Path, document: object) -> None:
    if isinstance(document, dict) and document.get("scheme_id") != path.stem:
        raise ValueError("內文 scheme_id 必須等於路徑方案代號")


def _is_json_media_type(value: str) -> bool:
    return value.split(";", 1)[0].strip().lower() == "application/json"


def _bad(exc: Exception, code: int = 400) -> JSONResponse:
    if isinstance(exc, SchemeValidationError):
        return JSONResponse({"problems": [vars(item) for item in exc.problems]}, status_code=422)
    return JSONResponse({"error": str(exc)}, status_code=code)


def _plan(scheme: Scheme, directivity: DirectivityDefaults) -> dict[str, object]:
    primary = Point(*scheme.receiver_set.primary.position_m)
    roles = {channel.speaker_id: channel.role for channel in scheme.channel_group.channels}
    aim = (default_source_model(primary, directivity).model_dump(mode="json")["aim_m"]
           if scheme.source_model == "product_default" else None)
    room = scheme.scene.room_m
    return {
        "room": {"Lx": room.Lx, "Ly": room.Ly, "Lz": room.Lz},
        "speakers": [{"id": speaker_id, "role": roles[speaker_id],
                      "role_label": f"聲道 {roles[speaker_id]}",
                      "point": {"x": point.x, "y": point.y, "z": point.z}, "aim": aim}
                     for speaker_id, point in scheme.speakers.items()],
        "receivers": [{"id": receiver.receiver_id, "role": receiver.role.value,
                       "role_label": {"primary": "主位", "surrounding": "周圍點",
                                      "other_seat": "其他座位"}[receiver.role.value],
                       "point": dict(zip(("x", "y", "z"), receiver.position_m, strict=True))}
                      for receiver in scheme.receiver_set.points],
    }


class GuiHandlers:
    """路由共用已載設定和資料目錄；單人本機使用。"""

    def __init__(self, settings: GuiSettings) -> None:
        if not COMMIT.fullmatch(settings.engine_commit):
            raise ValueError("engine_commit 必須是 40 位十六進位提交 id")
        self.data_dir = settings.data_dir.expanduser().resolve()
        if self.data_dir.is_relative_to(repo_root()):
            raise ValueError("data_dir 不准在 repo 內")
        for name in ("schemes", "runs", "results"):
            child = self.data_dir / name
            child.mkdir(parents=True, exist_ok=True)
            if child.resolve().is_relative_to(repo_root()):
                raise ValueError(f"{name} 不准在 repo 內")
        capabilities_path = config_path("capabilities.toml")
        self.capabilities: CapabilityTable = load_capabilities(capabilities_path)
        self.directivity = load_directivity_defaults(config_path("directivity_defaults.toml"))
        runner = settings.runner or (sys.executable, "-m", "aosr.reporting.scheme_cli", "run")
        self.jobs = JobManager(self.data_dir, runner, settings.engine_commit, capabilities_path)

    async def index(self, request: Request) -> Response:
        return FileResponse(STATIC / "index.html", media_type="text/html")

    async def asset(self, request: Request) -> Response:
        name = request.path_params["name"]
        if name not in {"app.js", "style.css"}:
            return _bad(ValueError("沒有這個靜態檔"), 404)
        return FileResponse(STATIC / name)

    async def example(self, request: Request) -> Response:
        loaded: object = json.loads((repo_root() / "blueprint" /
                                     "scheme_reference_room.json").read_text(encoding="utf-8"))
        scheme = Scheme.model_validate(loaded)
        return JSONResponse({"scheme": scheme.model_dump(mode="json"),
                             "rho_c": scheme.scene.density_kg_m3 * scheme.scene.sound_speed_m_s,
                             "rho_c_label": f"ρc：{scheme.scene.density_kg_m3 * scheme.scene.sound_speed_m_s:.1f} 帕·秒／公尺",
                             "feature_match_note": "沿用考卷基線，未查證"})

    async def validate(self, request: Request) -> Response:
        document = cast(object, await request.json())
        problems = validate_scheme(document, capabilities=self.capabilities,
                                   directivity=self.directivity)
        multiples: dict[str, float | None] = {}
        try:
            scene = Scheme.model_validate(document).scene
            multiples = {wall: impedance_multiple(value, scene.density_kg_m3,
                                                    scene.sound_speed_m_s)
                         for wall, value in scene.impedance_pa_s_per_m_by_wall.items()}
        except ValidationError:
            pass
        labels = {wall: f"約 ρc 的 {multiple:.2f} 倍" if multiple is not None else ""
                  for wall, multiple in multiples.items()}
        return JSONResponse({"problems": [vars(item) for item in problems],
                             "impedance_multiples": multiples,
                             "impedance_labels": labels})

    async def schemes(self, request: Request) -> Response:
        names = sorted(path.stem for path in (self.data_dir / "schemes").glob("*.json")
                       if SAFE_ID.fullmatch(path.stem))
        return JSONResponse({"schemes": names})

    async def scheme_item(self, request: Request) -> Response:
        try:
            path = _scheme_path(self.data_dir, request.path_params["name"])
            if request.method == "GET":
                return JSONResponse({"scheme": _read_scheme(path).model_dump(mode="json")})
            document = cast(object, await request.json())
            _require_scheme_id(path, document)
            problems = validate_scheme(document, capabilities=self.capabilities,
                                       directivity=self.directivity)
            if problems:
                return JSONResponse({"problems": [vars(item) for item in problems]},
                                    status_code=422)
            scheme = Scheme.model_validate(document)
            temporary = path.with_suffix(".tmp")
            try:
                temporary.write_text(scheme.model_dump_json(), encoding="utf-8")
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
            return JSONResponse({"scheme_id": path.stem, "message": "方案已儲存"})
        except (ValueError, FileNotFoundError, OSError) as exc:
            return _bad(exc, 404 if isinstance(exc, FileNotFoundError) else 400)

    async def plan(self, request: Request) -> Response:
        try:
            path = _scheme_path(self.data_dir, request.path_params["name"])
            return JSONResponse(_plan(_read_scheme(path), self.directivity))
        except (ValueError, FileNotFoundError, OSError) as exc:
            return _bad(exc, 404 if isinstance(exc, FileNotFoundError) else 400)

    async def runs(self, request: Request) -> Response:
        body = cast(object, await request.json())
        if not isinstance(body, dict) or not isinstance(body.get("scheme_id"), str):
            return _bad(ValueError("需要 scheme_id"))
        try:
            path = _scheme_path(self.data_dir, body["scheme_id"])
            scheme = _read_scheme(path)
            problems = validate_scheme(scheme, capabilities=self.capabilities,
                                       directivity=self.directivity)
            if problems:
                return JSONResponse({"problems": [vars(item) for item in problems]},
                                    status_code=422)
            return JSONResponse(self.jobs.start(path))
        except (ValueError, FileNotFoundError, OSError) as exc:
            return _bad(exc, 404 if isinstance(exc, FileNotFoundError) else 400)

    async def run_item(self, request: Request) -> Response:
        run_id = request.path_params["run_id"]
        if not re.fullmatch(r"[0-9a-f]{32}", run_id):
            return _bad(ValueError("計算代號無效"))
        try:
            state = (await run_in_threadpool(self.jobs.stop, run_id) if request.method == "POST"
                     else self.jobs.get(run_id))
            return JSONResponse(state)
        except (ValueError, KeyError, OSError) as exc:
            return _bad(exc, 404 if isinstance(exc, FileNotFoundError) else 400)


def create_app(settings: GuiSettings) -> Starlette:
    """建立本機入口；不啟動網路伺服器。"""
    handlers = GuiHandlers(settings)
    app = Starlette(routes=[
        Route("/", handlers.index), Route("/static/{name}", handlers.asset),
        Route("/api/example", handlers.example),
        Route("/api/validate", handlers.validate, methods=["POST"]),
        Route("/api/schemes", handlers.schemes),
        Route("/api/schemes/{name}", handlers.scheme_item, methods=["GET", "PUT"]),
        Route("/api/plan/{name}", handlers.plan),
        Route("/api/runs", handlers.runs, methods=["POST"]),
        Route("/api/runs/{run_id}", handlers.run_item),
        Route("/api/runs/{run_id}/stop", handlers.run_item, methods=["POST"]),
    ])
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])

    async def json_requests(request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.method in {"POST", "PUT"} and not _is_json_media_type(request.headers.get("content-type", "")):
            return _bad(ValueError("只收 application/json"), 415)
        response = await call_next(request)
        if response.status_code == 400 and response.headers.get("content-type", "").startswith("text/plain"):
            return _bad(ValueError("Host 不受信任"))
        return response

    app.add_middleware(BaseHTTPMiddleware, dispatch=json_requests)

    async def unexpected_error(request: Request, exc: Exception) -> Response:
        if isinstance(exc, HTTPException):
            return _bad(ValueError(str(exc.detail)), exc.status_code)
        if isinstance(exc, json.JSONDecodeError | UnicodeDecodeError):
            return _bad(ValueError("內文不是有效 JSON"))
        return _bad(exc, 500)

    app.add_exception_handler(HTTPException, unexpected_error)
    # 內文不是 JSON 是使用者輸入的錯：單獨登記，在路由那一層就換成 400 JSON，不冒到最外層當 500。
    app.add_exception_handler(json.JSONDecodeError, unexpected_error)
    app.add_exception_handler(UnicodeDecodeError, unexpected_error)
    app.add_exception_handler(Exception, unexpected_error)
    return app
