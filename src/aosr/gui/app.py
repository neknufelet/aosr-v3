"""本機單人網頁：輸入、驗證、存檔、2D 圖與獨立計算。"""
from __future__ import annotations

import ipaddress
import json
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import date
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
from aosr.config.directivity_defaults import load_directivity_defaults
from aosr.config.paths import config_path
from aosr.gui.jobs import JobManager, ResultMoveConflict, ResultStatus
from aosr.gui.modal_jobs import ModalJobs, scheme_snapshot
from aosr.gui.compare_view import (
    CompareView, build_compare_view, compare_run_notices, curves_csv, summary_csv)
from aosr.gui.capability_view import capability_lists
from aosr.gui.search_view import build_search_view, list_searches, search_path
from aosr.gui.search_best import BestCache, build_best_view
from aosr.gui.labels import label_tables
from aosr.gui.input_setup import INPUT_SHAPE_ERROR, InputEdit, InputShapeError, edit_input, input_defaults, input_preview
from aosr.gui.result_list import ResultList, ResultSummary
from aosr.gui.plan_view import plan_for
from aosr.gui.problem_text import SchemeProblemsError, checked_scheme, plain_problems, plan_problems
from aosr.reporting.display import impedance_multiple
from aosr.reporting.compare import comparison_problems
from aosr.reporting.calculation_fingerprint import calculation_fingerprint, short_fingerprint
from aosr.reporting.scheme import Scheme
from aosr.reporting.evaluation import LoadedResult, ResultStanding, load_result
from aosr.reporting.physics_identity import physics_identity
from aosr.reporting.result import RESULT_SCHEMA_VERSION, SchemeResult
from aosr.gui.result_view import ResultView, build_result_view
from aosr.config.quality_targets import load_quality_targets
from aosr.reporting.validation import SchemeProblem, SchemeValidationError, validate_scheme


STATIC = Path(__file__).parent / "static"
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")
COMMIT = re.compile(r"[0-9a-f]{40}\Z")
RUN_ID = re.compile(r"[0-9a-f]{32}\Z")
SERVER_UPDATED = "程式已更新，網頁伺服器要重開才看得了結果（請助理重開）"
# 結果檔的格式版本；拒收時比這個舊的才叫「舊格式」。
RESULT_VERSION = re.compile(r"aosr\.scheme_result\.v([0-9]+)\Z")
OLD_FORMAT_TEXT = "這份是舊格式的結果（程式更新前算的），要用現在的程式重算才看得到"
OLD_FORMAT_SIDE_TEXT = "{side} 那份是舊格式的結果（程式更新前算的），要用現在的程式重算才能比較"
STANDING_TEXT = {
    ResultStanding.CURRENT: "物理與評分設定都跟現在相同",
    ResultStanding.RERANKED: "評分的權重或設定改過：已用存下的指標重新排名，數字跟存檔時相同，不用重算",
    ResultStanding.REMEASURED: "評分的量法或設定改過：已用存下的物理結果重新量過再排名，不用重算物理",
    ResultStanding.NEEDS_PHYSICS: "物理計算的程式或設定改過：畫面上是舊的物理結果配現在的評分，要重算物理（約 6 分鐘）才能跟現在算的結果比較",
}
STATIC_NAMES = {"app.js", "results.js", "compare.js", "plan.js",
                "style.css", "home.css", "result.css", "compare.css",
                "searches.html", "searches.js", "searches.css", "search_best.js", "capabilities.js", "modal.js"}
LOCAL_HOSTS = ("127.0.0.1", "localhost")
# 另外准許的網址主機名：只收小寫的主機名（機器短名、點分全名或 IPv4 位址，例如 Tailscale 給這台的名字），
# 不收萬用字元、埠號或大寫——TrustedHost（只認登記網址的把關）遇到 * 就等於不把關。
HOST_NAME = re.compile(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)*\Z")
# 伺服器可以聽的位址：本機，或 Tailscale（私人網路）發給這台的位址（100.64.0.0/10）。不收 0.0.0.0 與區網位址。
# 聽在 Tailscale 位址也擋不住區網與容器：Linux 會把任何網卡收到、寄給本機任一位址的封包照收
# （弱主機模型），這台的防火牆也沒開。所以聽 Tailscale 位址時，另外只接受下面兩個網段來的連線。
LOOPBACK = ipaddress.IPv4Address("127.0.0.1")
TAILSCALE_NET = ipaddress.IPv4Network("100.64.0.0/10")
# 扣掉 100.115.92.0/23：Tailscale 不會把這段發給裝置，它自己的防火牆規則對這段也不擋（留給 ChromeOS 虛擬機），
# 區網裡收得到閘道流量的人可以拿這段當來源、完成連線。
TAILNET_CLIENTS = (ipaddress.IPv4Network("127.0.0.1/32"),
                   *TAILSCALE_NET.address_exclude(ipaddress.IPv4Network("100.115.92.0/23")))


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
    modal_runner: tuple[str, ...] | None = None
    # 另外准許的網址主機名：從自己其他 Tailscale 裝置直接連進來時，瀏覽器帶的是這台的 Tailscale 名字。
    # 預設空的＝只認本機。
    extra_hosts: tuple[str, ...] = ()
    # 連線來源只准這幾個網段；空的＝不看來源（只聽 127.0.0.1 時外面本來就連不進來）。
    client_networks: tuple[ipaddress.IPv4Network, ...] = ()
    startup_fingerprint: str | None = None
    startup_physics_identity: str | None = None


def client_allowed(host: str | None, networks: tuple[ipaddress.IPv4Network, ...]) -> bool:
    """連線來源位址是否落在准許的網段；讀不出位址一律不准。"""
    try:
        address = ipaddress.IPv4Address(host or "")
    except ValueError:
        return False
    return any(address in network for network in networks)


def listen_address(value: str) -> str:
    """伺服器要聽的位址：只准本機或 Tailscale 位址，其他一律拒絕。"""
    try:
        address = ipaddress.IPv4Address(value)
    except ValueError as exc:
        raise ValueError(f"聽的位址要是 IPv4：{value!r}") from exc
    if address != LOOPBACK and address not in TAILSCALE_NET:
        raise ValueError(f"只准聽 127.0.0.1 或 Tailscale 位址（100.64.0.0/10），不准 {value}")
    return str(address)


def _allowed_hosts(settings: GuiSettings) -> list[str]:
    for host in settings.extra_hosts:
        if not HOST_NAME.fullmatch(host):
            raise ValueError(f"另外准許的網址只收小寫主機名，不收萬用字元或埠號：{host!r}")
    return [*LOCAL_HOSTS, *settings.extra_hosts]


def _scheme_path(data_dir: Path, name: str) -> Path:
    if not SAFE_ID.fullmatch(name) or len(name) > 200:
        raise ValueError("方案代號只收英數字、底線與連字號")
    return data_dir / "schemes" / f"{name}.json"


def _read_scheme(path: Path) -> Scheme:
    loaded: object = json.loads(path.read_text(encoding="utf-8"))
    return checked_scheme(loaded)


def _same_saved_scheme(path: Path, scheme: Scheme) -> bool:
    """存檔前跟磁碟上那份按值比；沒有檔、舊格式或壞掉的舊檔都當成不同，不拿舊檔的問題擋新存。"""
    try:
        return path.is_file() and _read_scheme(path) == scheme
    except (ValueError, OSError):
        return False


def _name_taken(name: str, busy: str | None) -> str:
    # 開舊方案會用存檔內容蓋掉表單上剛改的；那一份又改不得時，叫他開舊方案就是死路，直接叫他換名字。
    if busy:
        return (f"已經有叫「{name}」的方案，而且它{busy}，不能再改；要保留表單上現在的設定，"
                "請在「新名字」這一格填一個新名字，再按「另存新名字」")
    return (f"已經有叫「{name}」的方案；要保留表單上現在的設定，請在「新名字」這一格填一個新名字，再按「另存新名字」。"
            "要改原本那一份，先用「開舊方案」打開再改（表單上現在改的不會帶過去）")


def _require_scheme_id(path: Path, document: object) -> None:
    if isinstance(document, dict) and document.get("scheme_id") != path.stem:
        raise ValueError("內文 scheme_id 必須等於路徑方案代號")


def _is_json_media_type(value: str) -> bool:
    return value.split(";", 1)[0].strip().lower() == "application/json"


def _problems_response(problems: tuple[SchemeProblem, ...], document: object) -> JSONResponse:
    """方案檢查不過：每條問題寫表單上的中文欄名與白話，同一句只一條（原路徑另外留著）。"""
    return JSONResponse({"problems": plain_problems(problems, document)}, status_code=422)


def _bad(exc: Exception, code: int = 400) -> JSONResponse:
    if isinstance(exc, SchemeValidationError):
        document = exc.document if isinstance(exc, SchemeProblemsError) else None
        return _problems_response(exc.problems, document)
    message = f"{type(exc).__name__}: {exc}" if code == 500 else str(exc)
    return JSONResponse({"error": message}, status_code=code)


def _compare_export_response(view: CompareView, a_id: str, b_id: str, kind: str) -> Response:
    content = curves_csv(view) if kind == "curves" else summary_csv(view)
    return Response(content, media_type="text/csv; charset=utf-8", headers={
        "Content-Disposition":
            f'attachment; filename="compare-{a_id[:8]}-{b_id[:8]}-{kind}.csv"'})


def _compare_json_response(compare: CompareView, plans: object, scale_room: dict[str, float],
                           results: dict[str, SchemeResult], current: str) -> JSONResponse:
    return JSONResponse({**compare.model_dump(mode="json", exclude_none=True),
                         "plans": plans, "plan_scale_room": scale_room,
                         "outdated_schemes": _outdated_schemes(results, current)})


def _rejection_reason(exc: ValueError | ValidationError | json.JSONDecodeError | OSError) -> str:
    if not isinstance(exc, ValidationError):
        return str(exc)
    fields = [".".join(map(str, issue["loc"])) or "結果檔" for issue in exc.errors()]
    lead = "結果檔格式是舊版" if "schema_version" in fields else "結果檔欄位不符合現行格式"
    return f"{lead}（欄位 {'、'.join(fields)}）"


def _older_result_format(path: Path) -> bool:
    """結果檔頭的格式版本比現在舊才算舊格式；讀不出來、不是結果檔或版本不比現在舊的都不算，不多說。"""
    current = RESULT_VERSION.fullmatch(RESULT_SCHEMA_VERSION)
    try:
        with path.open(encoding="utf-8") as handle:
            document: object = json.load(handle)
    except (ValueError, OSError):
        return False
    version = document.get("schema_version") if isinstance(document, dict) else None
    found = RESULT_VERSION.fullmatch(version) if isinstance(version, str) else None
    return current is not None and found is not None and int(found[1]) < int(current[1])


def _rejected_response(exc: ValueError | ValidationError | json.JSONDecodeError | OSError,
                       rerun_url: str, *, side: str | None = None,
                       old_format: bool = False,
                       run_fields: dict[str, object] | None = None) -> JSONResponse:
    """結果被拒收：原因、重算網址；比較頁多帶哪一邊。舊格式另給機器看的種類與一句白話。"""
    body: dict[str, object] = {"rejected": True, "reason": _rejection_reason(exc),
                               "rerun_url": rerun_url}
    if side is not None:
        body["side"] = side
    if old_format:
        body["reason_kind"] = "old_format"
        body["reason_text"] = (OLD_FORMAT_TEXT if side is None
                               else OLD_FORMAT_SIDE_TEXT.format(side=side.upper()))
    if run_fields:
        body.update(run_fields)
    return JSONResponse(body, status_code=409)


def _problem_response(problems: tuple[str, ...], results: dict[str, SchemeResult],
                      current: str, rerun_urls: dict[str, str],
                      run_notices: tuple[str, ...] = ()) -> JSONResponse:
    outdated = [side for side, result in results.items()
                if result.physics_identity != current]
    updated = tuple(result.model_copy(update={"physics_identity": current})
                    if side in outdated else result for side, result in results.items())
    useful_urls = ({side: rerun_urls[side] for side in outdated}
                   if not comparison_problems(updated) else {})
    return JSONResponse({"problems": problems, "rerun_urls": useful_urls,
                         "outdated_sides": outdated,
                         **({"run_notices": run_notices} if run_notices else {}),
                         "outdated_schemes": _outdated_schemes(results, current)}, status_code=409)


def _outdated_schemes(results: dict[str, SchemeResult], current: str) -> dict[str, str]:
    return {side: result.scheme.scheme_id for side, result in results.items()
            if result.physics_identity != current}


def _plan_scale_room(results: dict[str, SchemeResult]) -> dict[str, float]:
    return {axis: max(getattr(result.scheme.scene.room_m, axis) for result in results.values())
            for axis in ("Lx", "Ly", "Lz")}


def _result_paths(data_dir: Path) -> list[Path]:
    return sorted((path for path in (data_dir / "results").glob("*.json")
                   if RUN_ID.fullmatch(path.stem)),
                  key=lambda item: item.stat().st_mtime_ns, reverse=True)


class GuiHandlers:
    """路由共用已載設定和資料目錄；單人本機使用。"""

    def __init__(self, settings: GuiSettings) -> None:
        if not COMMIT.fullmatch(settings.engine_commit):
            raise ValueError("engine_commit 必須是 40 位十六進位提交 id")
        self.data_dir = settings.data_dir.expanduser().resolve()
        if self.data_dir.is_relative_to(repo_root()):
            raise ValueError("data_dir 不准在 repo 內")
        for name in ("schemes", "runs", "results", "searches", "modal-cache", "modal-jobs"):
            child = self.data_dir / name
            child.mkdir(parents=True, exist_ok=True)
            if child.resolve().is_relative_to(repo_root()):
                raise ValueError(f"{name} 不准在 repo 內")
        capabilities_path = config_path("capabilities.toml")
        self.capabilities: CapabilityTable = load_capabilities(capabilities_path)
        self.directivity = load_directivity_defaults(config_path("directivity_defaults.toml"))
        self.best_cache = BestCache()
        runner = settings.runner or (sys.executable, "-m", "aosr.reporting.scheme_cli", "run")
        self.jobs = JobManager(self.data_dir, runner, settings.engine_commit, capabilities_path)
        self.archive_jobs = JobManager(self.data_dir / "archive", runner,
                                       settings.engine_commit, capabilities_path)
        # 兩區共用同一把鎖：讀清單、查狀態與搬動互斥，封存區只讀自己的計算紀錄。
        self.archive_jobs._lock = self.jobs._lock
        modal_runner = settings.modal_runner or (sys.executable, "-m", "aosr.reporting.scheme_cli", "modal")
        self.modal_jobs = ModalJobs(self.data_dir, modal_runner, settings.engine_commit, capabilities_path)
        self.startup_fingerprint = (settings.startup_fingerprint or
                                    calculation_fingerprint(capabilities_path=capabilities_path))
        self.startup_physics_identity = (settings.startup_physics_identity or
                                         physics_identity(capabilities=self.capabilities,
                                                          directivity=self.directivity))
        self.result_list = ResultList(self.startup_physics_identity,
                                      config_path("quality_targets.toml"))
        self.capabilities_path = capabilities_path

    def _server_stale(self) -> bool:
        return calculation_fingerprint(capabilities_path=self.capabilities_path) != self.startup_fingerprint

    def _updated_response(self) -> JSONResponse:
        return JSONResponse({"server_notice": SERVER_UPDATED, "error": SERVER_UPDATED},
                            status_code=409)

    async def index(self, request: Request) -> Response:
        return FileResponse(STATIC / "index.html", media_type="text/html")

    async def asset(self, request: Request) -> Response:
        name = request.path_params["name"]
        if name not in STATIC_NAMES:
            return _bad(ValueError("沒有這個靜態檔"), 404)
        return FileResponse(STATIC / name)

    async def vendor(self, request: Request) -> Response:
        name = request.path_params["name"]
        if name not in {"uPlot.iife.min.js", "uPlot.min.css"}:
            return _bad(ValueError("沒有這個靜態檔"), 404)
        return FileResponse(STATIC / "vendor" / "uplot" / name)

    async def searches_page(self, request: Request) -> Response:
        if "search_id" in request.path_params:
            try:
                search_path(self.data_dir / "searches", request.path_params["search_id"])
            except (ValueError, OSError) as exc:
                return _bad(exc, 404)
        return FileResponse(STATIC / "searches.html", media_type="text/html")

    async def searches(self, request: Request) -> Response:
        data = await run_in_threadpool(list_searches, self.data_dir / "searches")
        return JSONResponse(data, headers={"Cache-Control": "no-store"})

    async def search_item(self, request: Request) -> Response:
        try:
            path = search_path(self.data_dir / "searches", request.path_params["search_id"])
        except (ValueError, OSError) as exc:
            return _bad(exc, 404)
        view = await run_in_threadpool(build_search_view, path,
                                      server_physics=self.startup_physics_identity,
                                      server_program=self.startup_fingerprint)
        return JSONResponse(view.model_dump(mode="json"), headers={"Cache-Control": "no-store"})

    async def search_best(self, request: Request) -> Response:
        try:
            path = search_path(self.data_dir / "searches", request.path_params["search_id"])
        except (ValueError, OSError) as exc:
            return _bad(exc, 404)
        try:
            data = await run_in_threadpool(build_best_view, path, which=request.query_params.get("which"),
                                           cache=self.best_cache, directivity=self.directivity)
        except ValueError as exc:
            return _bad(exc)
        return JSONResponse(data, headers={"Cache-Control": "no-store"})

    async def result_page(self, request: Request) -> Response:
        if not RUN_ID.fullmatch(request.path_params["run_id"]):
            return _bad(ValueError("計算代號無效"))
        return FileResponse(STATIC / "results.html", media_type="text/html")

    async def compare_page(self, request: Request) -> Response:
        return FileResponse(STATIC / "compare.html", media_type="text/html")

    async def labels(self, request: Request) -> Response:
        return JSONResponse(label_tables())

    async def input_defaults(self, request: Request) -> Response:
        return JSONResponse(input_defaults())

    async def input_edit(self, request: Request) -> Response:
        document: object = await request.json()
        try:
            data = InputEdit.model_validate(document)
            return JSONResponse({"scheme": edit_input(data)})
        except ValidationError:
            return _bad(ValueError("填值請求格式不完整，請先填好房間、喇叭與座位"))
        except InputShapeError as exc:
            return _bad(exc)

    async def input_preview(self, request: Request) -> Response:
        document: object = await request.json()
        if not isinstance(document, dict):
            return _bad(ValueError("需要一份方案"))
        try:
            return JSONResponse(input_preview(document, self.capabilities, self.directivity))
        except InputShapeError as exc:
            return _bad(exc)
        except ValidationError:
            return _bad(ValueError(INPUT_SHAPE_ERROR))

    async def example(self, request: Request) -> Response:
        loaded: object = json.loads((repo_root() / "blueprint" /
                                     "scheme_reference_room.json").read_text(encoding="utf-8"))
        scheme = Scheme.model_validate(loaded)
        # 峰谷配對容差：同種的兩個峰（或兩個谷）中心頻率相差不超過它就算同一個，
        # 左右聲道之間（src/aosr/scoring/channel_matching.py::_matched_features）與
        # 座位之間（src/aosr/scoring/listening_area.py::_matched_pairs）都用它；
        # 值照範例方案填、未查證為產品預設（docs/decisions/gui-first-local-2d.md 表單那一段）。
        tolerance = scheme.channel_group.feature_match_tolerance_hz
        return JSONResponse({"scheme": scheme.model_dump(mode="json"),
                             "rho_c": scheme.scene.density_kg_m3 * scheme.scene.sound_speed_m_s,
                             "rho_c_label": f"ρc：{scheme.scene.density_kg_m3 * scheme.scene.sound_speed_m_s:.1f} 帕·秒／公尺",
                             "feature_match_note": (
                                 f"峰谷配對容差 {tolerance:g} Hz：比較左右聲道、比較主位與周圍點時，"
                                 f"兩個峰（或兩個谷）中心頻率相差 {tolerance:g} Hz 以內就算同一個；"
                                 "這個值是照範例方案填的，還沒查證適不適合當產品預設")})

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
        return JSONResponse({"problems": plain_problems(problems, document),
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
                return _problems_response(problems, document)
            scheme = Scheme.model_validate(document)
            save_as = request.headers.get("if-none-match") == "*"
            if _same_saved_scheme(path, scheme):
                return JSONResponse({"scheme_id": path.stem, "message": "方案沒有變動"})
            busy = self._scheme_in_use(path.stem)
            # 另存撞到名字：方案檔在，或檔不在但這個名字已經有正常完成的結果（完成或沒有計算紀錄）／正在算，
            # 都叫他換名字。失敗、停止的產物不凍名字。
            if save_as and (path.exists() or busy):
                return _bad(ValueError(_name_taken(path.stem, busy)), 409)
            if busy:
                return _bad(ValueError(f"「{path.stem}」{busy}；改過的設定請用「另存新名字」存成新方案，{path.stem} 才留得住當比較基準"), 409)
            descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp")
            temporary = Path(name)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                    stream.write(scheme.model_dump_json())
                if save_as:
                    try:
                        path.hardlink_to(temporary)
                    except FileExistsError:
                        if _same_saved_scheme(path, scheme):
                            return JSONResponse({"scheme_id": path.stem, "message": "方案沒有變動"})
                        return _bad(ValueError(_name_taken(path.stem, self._scheme_in_use(path.stem))), 409)
                else:
                    temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
            return JSONResponse({"scheme_id": path.stem, "message": "方案已儲存"})
        except (ValueError, FileNotFoundError, OSError) as exc:
            return _bad(exc, 404 if isinstance(exc, FileNotFoundError) else 400)

    def _scheme_in_use(self, name: str) -> str | None:
        """這個代號正在算或已有正常完成的結果就凍結；失敗與停止產物留作診斷。"""
        running = self.jobs.list_recent()["running"]
        if isinstance(running, list) and any(
                isinstance(item, dict) and item.get("scheme_id") == name for item in running):
            return "正在計算"
        # #755（老闆選 A）：封存後放開名字，只看清單上（沒封存）正常完成的結果。
        if any(item.scheme_id == name and item.run_status in {"done", "none"}
               for item in self._result_summaries(False)):
            return "已經有算好的結果"
        return None

    async def plan(self, request: Request) -> Response:
        try:
            if request.method == "POST":
                document = cast(object, await request.json())
                problems = validate_scheme(document, capabilities=self.capabilities,
                                           directivity=self.directivity)
                if problems:
                    data: dict[str, object] = {"problems": plain_problems(problems, document)}
                    try:
                        plan = plan_for(Scheme.model_validate(document), self.directivity, mark_blockers=True)
                        data.update(plan=plan, problems=plan_problems(problems, document, plan))
                    except ValueError:
                        pass
                    return JSONResponse(data, status_code=422)
                return JSONResponse({**plan_for(Scheme.model_validate(document), self.directivity),
                                     "message": "檢查通過"})
            path = _scheme_path(self.data_dir, request.path_params["name"])
            return JSONResponse(plan_for(_read_scheme(path), self.directivity, mark_blockers=True))
        except (ValueError, FileNotFoundError, OSError) as exc:
            return _bad(exc, 404 if isinstance(exc, FileNotFoundError) else 400)

    async def runs(self, request: Request) -> Response:
        if request.method == "GET":
            return JSONResponse(self.jobs.list_recent())
        body = cast(object, await request.json())
        if not isinstance(body, dict) or not isinstance(body.get("scheme_id"), str):
            return _bad(ValueError("需要 scheme_id"))
        try:
            path = _scheme_path(self.data_dir, body["scheme_id"])
            scheme = _read_scheme(path)
            problems = validate_scheme(scheme, capabilities=self.capabilities,
                                       directivity=self.directivity)
            if problems:
                return _problems_response(problems, scheme)
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

    def _result_path(self, run_id: str) -> Path:
        if not RUN_ID.fullmatch(run_id):
            raise ValueError("計算代號無效")
        return self.data_dir / "results" / f"{run_id}.json"

    async def results(self, request: Request) -> Response:
        found = self._result_summaries()
        data: dict[str, object] = {"results": [item.model_dump() for item in found]}
        if self._server_stale():
            data["server_notice"] = SERVER_UPDATED
        return JSONResponse(data)

    def _saved_scheme(self, name: str) -> dict[str, object] | None:
        """現在存著的同名方案（照方案模型整理）；沒有檔、代號不合法、讀不出都當成沒有，不標。"""
        try:
            return _read_scheme(_scheme_path(self.data_dir, name)).model_dump(mode="json")
        except (ValueError, OSError):
            return None

    def _result_summaries(self, archived: bool = False) -> list[ResultSummary]:
        manager = self.archive_jobs if archived else self.jobs
        with self.jobs._lock:
            found = self.result_list.list(_result_paths(manager.data_dir), manager.result_status,
                                          self._saved_scheme)
            return ([item.model_copy(update={"result_url": ""}) for item in found]
                    if archived else found)

    async def archive(self, request: Request) -> Response:
        return JSONResponse({"results": [item.model_dump() for item in self._result_summaries(True)]})

    async def archive_result(self, request: Request) -> Response:
        return self._move_result(request, restore=False)

    async def restore_result(self, request: Request) -> Response:
        return self._move_result(request, restore=True)

    def _move_result(self, request: Request, *, restore: bool) -> Response:
        run_id = request.path_params["run_id"]
        if not RUN_ID.fullmatch(run_id):
            return _bad(ValueError("計算代號無效"))
        try:
            with self.jobs._lock:
                item = next((row for row in self._result_summaries(restore)
                             if row.run_id == run_id), None)
                if restore:
                    self.jobs.restore_result(run_id)
                else:
                    self.jobs.archive_result(run_id)
                label = f"「{item.scheme_id}」" if item else "這一筆結果"
                if item and item.finished_text != "讀不出":
                    label += f"（{item.finished_text} 算完）"
                message = (f"已搬回{label}；在結果清單可以查看" if restore else
                           f"已封存{label}；在下面「已封存」可以搬回")
                return JSONResponse({"message": message})
        except ResultMoveConflict as exc:
            return _bad(exc, 409)
        except FileNotFoundError as exc:
            return _bad(exc, 404)
        except OSError as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    async def result_item(self, request: Request) -> Response:
        run_id = request.path_params["run_id"]
        if not RUN_ID.fullmatch(run_id):
            return _bad(ValueError("計算代號無效"))
        if self._server_stale():
            return self._updated_response()
        path = self._result_path(run_id)
        run_fields: dict[str, object] = dict(self.jobs.result_status(run_id).notice_fields())
        try:
            if not path.is_file():
                raise FileNotFoundError(run_id)
            targets = config_path("quality_targets.toml")
            start = time.perf_counter()
            loaded_result = await run_in_threadpool(self._load_result, path, targets)
            result = loaded_result.result
            loaded = time.perf_counter()
            view = await run_in_threadpool(build_result_view, result,
                                           quality_targets_path=targets)
            built = time.perf_counter()
            data = view.model_dump(mode="json")
            data["plan"] = plan_for(result.scheme, self.directivity)
            data.update(capability_lists(self.capabilities))
            data.update(run_fields)
            data["fingerprint_text"] = short_fingerprint(result.physics_identity)
            data["standing"] = loaded_result.standing.value
            data["standing_text"] = STANDING_TEXT[loaded_result.standing]
            if loaded_result.standing is ResultStanding.NEEDS_PHYSICS:
                data["rerun_url"] = f"/api/results/{run_id}/rerun"
            response = JSONResponse(data)
            encoded = time.perf_counter()
            response.headers["Server-Timing"] = (
                f"load;dur={(loaded - start) * 1000:.2f}, "
                f"view;dur={(built - loaded) * 1000:.2f}, "
                f"json;dur={(encoded - built) * 1000:.2f}")
            return response
        except (ValueError, ValidationError, json.JSONDecodeError) as exc:
            return _rejected_response(exc, f"/api/results/{run_id}/rerun",
                                      old_format=await run_in_threadpool(_older_result_format, path),
                                      run_fields=run_fields)
        except (FileNotFoundError, OSError) as exc:
            return _bad(exc, 404)

    async def result_modal(self, request: Request) -> Response:
        if self._server_stale():
            return self._updated_response()
        try:
            path = self._result_path(request.path_params["run_id"])
            scheme = await run_in_threadpool(scheme_snapshot, path)
            body = await run_in_threadpool(self.modal_jobs.lookup, scheme,
                result_id=path.stem, calculate=request.method == "POST", job_id=request.query_params.get("job_id"))
            return JSONResponse(body, headers={"Cache-Control": "no-store"})
        except (ValueError, OSError) as exc:
            return _bad(exc, 404 if isinstance(exc, FileNotFoundError) else 400)

    async def modal_job(self, request: Request) -> Response:
        run_id = request.path_params["run_id"]
        if not RUN_ID.fullmatch(run_id):
            return _bad(ValueError("模態工作代號無效"))
        try:
            manager = self.modal_jobs.manager
            state = (await run_in_threadpool(manager.stop, run_id) if request.method == "POST"
                     else manager.get(run_id))
            return JSONResponse(state)
        except (ValueError, OSError) as exc:
            return _bad(exc, 404 if isinstance(exc, FileNotFoundError) else 400)

    def _compare_statuses(self, a_id: str, b_id: str
                          ) -> tuple[dict[str, ResultStatus], tuple[str, ...], dict[str, object]]:
        """比較兩邊的計算狀態；成功、不能比、被拒收三條路共用同一組警語。"""
        statuses = {side: self.jobs.result_status(run_id)
                    for side, run_id in (("a", a_id), ("b", b_id))}
        notices = compare_run_notices(statuses["a"], statuses["b"])
        return statuses, notices, ({"run_notices": notices} if notices else {})

    def _load_result(self, path: Path, targets: Path) -> LoadedResult:
        return load_result(path, capabilities=self.capabilities, directivity=self.directivity,
                           quality_targets_path=targets, physics_identity=self.startup_physics_identity)

    async def _compare_load(self, paths: dict[str, Path], targets: Path,
                            rerun_urls: dict[str, str], run_fields: dict[str, object]
                            ) -> dict[str, SchemeResult] | Response:
        results: dict[str, SchemeResult] = {}
        for side, path in paths.items():
            try:
                loaded = await run_in_threadpool(self._load_result, path, targets)
                results[side] = loaded.result
            except (ValueError, ValidationError, json.JSONDecodeError) as exc:
                return _rejected_response(exc, rerun_urls[side], side=side, old_format=(
                    await run_in_threadpool(_older_result_format, path)), run_fields=run_fields)
            except OSError as exc:
                # 跟結果頁一樣：讀不動檔回 404，不是結果本身被拒收，重算也解不了。
                return _bad(ValueError(f"{side.upper()} 的結果檔讀不動：{exc}"), 404)
        return results

    async def _compare_views(self, results: dict[str, SchemeResult], targets: Path,
                             rerun_urls: dict[str, str], run_fields: dict[str, object]
                             ) -> dict[str, ResultView] | Response:
        views: dict[str, ResultView] = {}
        for side, result in results.items():
            try:
                views[side] = await run_in_threadpool(build_result_view, result,
                                                      quality_targets_path=targets)
            except (ValueError, ValidationError) as exc:
                return _rejected_response(exc, rerun_urls[side], side=side, run_fields=run_fields)
        return {side: view.model_copy(update=capability_lists(self.capabilities))
                for side, view in views.items()}

    async def compare_item(self, request: Request) -> Response:
        a_id, b_id = request.path_params["a"], request.path_params["b"]
        if (kind := request.path_params.get("kind")) not in (None, "curves", "summary"):
            return _bad(ValueError("匯出種類找不到"), 404)
        if not RUN_ID.fullmatch(a_id) or not RUN_ID.fullmatch(b_id):
            return _bad(ValueError("計算代號無效"))
        if self._server_stale():
            return self._updated_response()
        if a_id == b_id:
            return _bad(ValueError("A 和 B 是同一份結果"), 409)
        rerun_urls = {side: f"/api/results/{run_id}/rerun"
                      for side, run_id in (("a", a_id), ("b", b_id))}
        paths = {side: self._result_path(run_id)
                 for side, run_id in (("a", a_id), ("b", b_id))}
        statuses, run_notices, run_fields = self._compare_statuses(a_id, b_id)
        for side, path in paths.items():
            if not path.is_file():
                return _bad(ValueError(f"{side.upper()} 的結果檔找不到"), 404)
        targets = config_path("quality_targets.toml")
        start = time.perf_counter()
        results = await self._compare_load(paths, targets, rerun_urls, run_fields)
        if isinstance(results, Response):
            return results
        loaded = time.perf_counter()
        a_result, b_result = results["a"], results["b"]
        problems = comparison_problems((a_result, b_result))
        if problems:
            return _problem_response(problems, results, self.startup_physics_identity, rerun_urls,
                                     run_notices)
        views = await self._compare_views(results, targets, rerun_urls, run_fields)
        if isinstance(views, Response):
            return views
        built = time.perf_counter()
        compare = await run_in_threadpool(
            build_compare_view, a_run_id=a_id, a=a_result, view_a=views["a"],
            b_run_id=b_id, b=b_result, view_b=views["b"],
            quality_targets=load_quality_targets(targets), run_date=date.today(),
            a_status=statuses["a"], b_status=statuses["b"])
        compared = time.perf_counter()
        if kind is not None:
            return _compare_export_response(compare, a_id, b_id, kind)
        # 比較文字資料除了頻響 dB 陣列不給 null：沿用結果頁模型的可空欄位直接不輸出；
        # 圖面照 /api/plan 的契約，全向點源的 aim 仍是 null。
        plans = {side: plan_for(result.scheme, self.directivity)
                 for side, result in results.items()}
        scale_room = _plan_scale_room(results)
        response = _compare_json_response(compare, plans, scale_room, results,
                                          self.startup_physics_identity)
        encoded = time.perf_counter()
        response.headers["Server-Timing"] = (
            f"load;dur={(loaded - start) * 1000:.2f}, "
            f"view;dur={(built - loaded) * 1000:.2f}, "
            f"compare;dur={(compared - built) * 1000:.2f}, "
            f"json;dur={(encoded - compared) * 1000:.2f}")
        return response

    async def rerun_result(self, request: Request) -> Response:
        run_id = request.path_params["run_id"]
        if not RUN_ID.fullmatch(run_id):
            return _bad(ValueError("計算代號無效"))
        if self._server_stale():
            return self._updated_response()
        try:
            result_path = self._result_path(run_id)
            if not result_path.is_file():
                raise FileNotFoundError(run_id)
            document: object = json.loads(result_path.read_text(encoding="utf-8"))
            if not isinstance(document, dict) or "scheme" not in document:
                raise ValueError("這份結果沒有方案快照")
            scheme = checked_scheme(document["scheme"])
            problems = validate_scheme(scheme, capabilities=self.capabilities,
                                       directivity=self.directivity)
            if problems:
                return _problems_response(problems, scheme)
            # 代號只拿來顯示：過白名單才照原樣記，不合格就記「代號無效」（路徑一律不用它）。
            label = (scheme.scheme_id if SAFE_ID.fullmatch(scheme.scheme_id) and len(scheme.scheme_id) <= 200
                     else "（代號無效）")
            return JSONResponse(self.jobs.start_snapshot(scheme.model_dump_json(), label))
        except (ValueError, FileNotFoundError, OSError) as exc:
            return _bad(exc, 404 if isinstance(exc, FileNotFoundError) else 400)


def create_app(settings: GuiSettings) -> Starlette:
    """建立本機入口；不啟動網路伺服器。"""
    allowed_hosts = _allowed_hosts(settings)
    handlers = GuiHandlers(settings)
    app = Starlette(routes=[
        Route("/", handlers.index), Route("/static/{name}", handlers.asset),
        Route("/static/vendor/uplot/{name}", handlers.vendor),
        Route("/results/{run_id}", handlers.result_page),
        Route("/compare/{a}/{b}", handlers.compare_page),
        Route("/searches", handlers.searches_page),
        Route("/searches/{search_id}", handlers.searches_page),
        Route("/api/searches", handlers.searches),
        Route("/api/searches/{search_id}", handlers.search_item),
        Route("/api/searches/{search_id}/best", handlers.search_best),
        Route("/api/example", handlers.example),
        Route("/api/labels", handlers.labels),
        Route("/api/input-defaults", handlers.input_defaults),
        Route("/api/input-edit", handlers.input_edit, methods=["POST"]),
        Route("/api/input-preview", handlers.input_preview, methods=["POST"]),
        Route("/api/validate", handlers.validate, methods=["POST"]),
        Route("/api/schemes", handlers.schemes),
        Route("/api/schemes/{name}", handlers.scheme_item, methods=["GET", "PUT"]),
        Route("/api/plan", handlers.plan, methods=["POST"]),
        Route("/api/plan/{name}", handlers.plan),
        Route("/api/runs", handlers.runs, methods=["GET", "POST"]),
        Route("/api/runs/{run_id}", handlers.run_item),
        Route("/api/runs/{run_id}/stop", handlers.run_item, methods=["POST"]),
        Route("/api/results", handlers.results),
        Route("/api/archive", handlers.archive),
        Route("/api/results/{run_id}/archive", handlers.archive_result, methods=["POST"]),
        Route("/api/archive/{run_id}/restore", handlers.restore_result, methods=["POST"]),
        Route("/api/compare/{a}/{b}", handlers.compare_item),
        Route("/api/compare/{a}/{b}/export/{kind}", handlers.compare_item),
        Route("/api/results/{run_id}", handlers.result_item),
        Route("/api/results/{run_id}/modal", handlers.result_modal, methods=["GET", "POST"]),
        Route("/api/modal-jobs/{run_id}", handlers.modal_job),
        Route("/api/modal-jobs/{run_id}/stop", handlers.modal_job, methods=["POST"]),
        Route("/api/results/{run_id}/rerun", handlers.rerun_result, methods=["POST"]),
    ])
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)

    async def json_requests(request: Request, call_next: RequestResponseEndpoint) -> Response:
        if settings.client_networks and not client_allowed(
                request.client.host if request.client else None, settings.client_networks):
            return _bad(ValueError("只接受這台本機與 Tailscale 裝置的連線"), 403)
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
