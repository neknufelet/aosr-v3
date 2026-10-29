"""牆面材料梯度搜尋命令列（#561 辛一、辛三）：``python -m aosr.gradient.cli start|resume|stop|endpoint|report``。

- ``start --scheme <方案檔> --surrogate <替身 npz> --presets <六個 id，逗號分隔，不貼的面寫 none> --purpose <用途>
  --root <資料夾根> [--settings <material_gradient.toml>] [--capabilities <能力表>] [--engine-commit <提交>]``：
  先核輸入（preset 在家族裡、用途認得、方案匯出完整模型拿網格雜湊、物理身分與喇叭座位座標、替身過准用閘門而且
  對得上這幾格、替身座位與喇叭的名字對得上方案），都過才建 ``<root>/<32 位十六進位>/``（名字由輸入算，同一組輸入同一個資料夾），
  寫 ``project.json``，在雙精度子行程把整條鏈跑完。拒收不建資料夾。
- ``resume <資料夾>``：拿鎖、重算現在的物理身分跟 ``project.json`` 比（不同就拒收），再逐格核輸入（方案檔雜湊、
  方案座標、替身過閘門、manifest 與收據摘要、設定指紋；子行程重建目標前會再核一次），清掉留下的 ``stop`` 記號，
  子行程照帳本接著跑（標頭對不上拒收；前段沒寫完重跑前段）。
- ``stop <資料夾>``：建 ``stop`` 記號，不拿鎖，回 0；子行程在下一次求值之前停。
- ``endpoint <資料夾>``：搜尋跑完之後拿鎖、起雙精度子行程做終點（``endpoint_run``）：同一道輸入閘門、完整模型重算
  ``result.json`` 那一點、方案原本六面的完整模型基準、匯出完整方案檔 ``scheme-endpoint.json``、寫 ``endpoint.json``；
  不動 ``status.json``。沒有 ``result.json`` 就失敗。
- ``report --folder <資料夾>``：只讀，印人看的文字；有對得上帳本的 ``endpoint.json`` 就印終點一節，對不上印過期，
  沒有就最後一行說終點還沒做。

``--presets`` 寫 ``none`` 的面是「不貼、保留方案原本的牆」（老闆 10-28：不貼算一個選項、用現在的六面當起點）：
搜尋、終點、匯出都用方案場景那一面原本的材料，不是剛性。

離開碼跟擺位搜尋命令列同一套：0 完成、1 失敗（含替身准用閘門拒用，錯誤原文寫標準錯誤）、2 參數錯、3 中斷或停止記號。
資料夾鎖：資料夾本身的描述子 ``flock``，子行程經 ``pass_fds`` 繼承；父行程只關自己那一份、不解鎖。
``start`` 在暫存名的資料夾裡先拿鎖、寫好 ``project.json`` 再換成正式名，所以看得到的資料夾一出現就有鎖：
另一個 ``start`` 撞到同名資料夾時，鎖被拿著就報「還有計算在跑」，沒被拿著就報「已存在、用 resume」，兩種都回 1。
父行程收到 SIGTERM、SIGHUP（只在 ``__main__`` 轉成中斷）就對子行程群送 SIGTERM，等它寫完 interrupted 再回 3。
建子行程那一刻把 SIGINT、SIGTERM、SIGHUP 暫時擋住（``pthread_sigmask``），建好、``process`` 拿到手才放開：
中斷不會落在「子行程已經起了、父行程還不知道它」的那一段，所以不留孤兒（審查 B2）；子行程繼承了擋住的遮罩，
子行程入口裝好轉換之後自己放開。``TERMINATE_GRACE_S`` 是送 SIGTERM 之後等子行程寫完 interrupted、自己結束的寬限
秒數，過了才 SIGKILL；它是收行程的耐心，不是精度門檻。

資料夾根由 ``--root`` 給，不放進網頁讀的 ``searches/``：網頁會把 32 位十六進位資料夾當擺位搜尋去讀，
這裡沒有那邊的檔，會報讀取錯誤。

記憶體與執行緒（下面是派工單轉述的示範房實量，這一支沒有重量）：示範房替身（秩 1632、188 點軸）一次值＋梯度約
13 秒（BLAS 單緒）；一支行程載一份替身算值＋梯度，尖峰 RSS 約 9～16 GB。程式不包記憶體上限，呼叫端自己包
``systemd-run --scope -p MemoryMax=… -p OOMPolicy=continue``；子行程一律 ``threads=1``（OpenMP、MKL、OpenBLAS 單緒）。
``start`` 另外在父行程匯出一次完整模型拿網格雜湊（示範房要幾十秒），匯出完就放掉。
"""
from __future__ import annotations

import argparse
import fcntl
import os
import signal
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator, Set
from contextlib import contextmanager
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING, Final

from aosr import runtime
from aosr.config.paths import config_path
from aosr.geometry.shoebox import Wall

if TYPE_CHECKING:
    from aosr.gradient.run import GradientProject

EXIT_FAILED: Final = 1
EXIT_STOPPED: Final = 3
KEEP_WALL: Final = "none"
SEARCHES_DIR: Final = "searches"
TERMINATE_GRACE_S: Final = 60.0
_REPO_ROOT: Final = Path(__file__).resolve().parents[3]
FORWARDED_SIGNALS: Final[Set[signal.Signals]] = frozenset((signal.SIGINT, signal.SIGTERM, signal.SIGHUP))


class FolderBusy(Exception):
    """拿不到資料夾鎖（還有計算在跑），或資料夾打不開；什麼都沒動。"""


def _presets(text: str) -> tuple[str | None, ...]:
    """``a,b,none,...`` → 六個 preset id（``none`` 是不貼、保留方案原本的牆）；個數不是牆數就是參數錯（離開碼 2）。"""
    items = tuple(item.strip() for item in text.split(","))
    walls = [wall.wall_name() for wall in Wall.all()]
    if len(items) != len(walls) or any(not item for item in items):
        raise argparse.ArgumentTypeError(
            f"--presets 要照 {','.join(walls)} 給 {len(walls)} 個非空的 preset id（不貼的面寫 {KEEP_WALL}），收到 {len(items)} 個")
    return tuple(None if item == KEEP_WALL else item for item in items)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aosr.gradient.cli")
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("start", help="開始牆面材料梯度搜尋")
    start.add_argument("--scheme", type=Path, required=True, help="方案檔（座位組要一主位加六周圍點）")
    start.add_argument("--surrogate", type=Path, required=True, help="驗過的替身 npz（旁邊要有 manifest 與驗證收據）")
    start.add_argument("--presets", type=_presets, required=True, help="六面 preset id，照 Wall.all()，不貼（保留方案原本的牆）寫 none")
    start.add_argument("--purpose", required=True, help="用途（評分設定）")
    start.add_argument("--root", type=Path, required=True, help="資料夾根（不要放在網頁讀的 searches/ 底下）")
    start.add_argument("--settings", type=Path, default=config_path("material_gradient.toml"))
    start.add_argument("--capabilities", type=Path, default=config_path("capabilities.toml"))
    start.add_argument("--engine-commit", default=None)
    resume = commands.add_parser("resume", help="接續")
    resume.add_argument("folder", type=Path)
    stop = commands.add_parser("stop", help="建立停止記號")
    stop.add_argument("folder", type=Path)
    endpoint = commands.add_parser("endpoint", help="終點：完整模型重算、原牆基準、匯出方案檔")
    endpoint.add_argument("folder", type=Path)
    report = commands.add_parser("report", help="只讀報告")
    report.add_argument("--folder", type=Path, required=True)
    return parser


@contextmanager
def _locked(folder: Path) -> Iterator[int]:
    """對資料夾描述子拿非阻塞的獨佔鎖；結束只關父邊描述子，不解鎖（子行程繼承的那一份還拿著）。"""
    try:
        descriptor = os.open(folder, os.O_RDONLY | os.O_DIRECTORY)
    except OSError as error:
        raise FolderBusy(f"打不開資料夾 {folder}，沒有動任何檔：{error}") from error
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise FolderBusy(f"資料夾 {folder} 還有計算在跑（鎖被拿著），等它結束再試") from error
        except OSError as error:
            raise FolderBusy(f"拿不到資料夾 {folder} 的鎖，沒有動任何檔：{error}") from error
        yield descriptor
    finally:
        os.close(descriptor)


def _terminate(process: subprocess.Popen[bytes]) -> None:
    """對子行程群送 SIGTERM（它會轉成中斷、寫 interrupted），等不到就 SIGKILL。"""
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=TERMINATE_GRACE_S)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    except ProcessLookupError:
        process.wait()


@contextmanager
def keep_signal_handlers() -> Iterator[None]:
    """區塊結束時照 Python 記的處置把 SIGINT、SIGTERM、SIGHUP 重裝一次。

    匯出完整模型會叫網格程式庫（gmsh），它收尾時在 C 層把訊號處置改回預設，Python 這邊卻還以為轉換裝著：
    之後父行程一收到 SIGTERM 就直接死、子行程變孤兒（修補一的端到端考卷量到的）。終點子行程匯出完整模型也包這一層
    （不然匯出後 SIGTERM 不再轉成中斷）。
    """
    saved = {number: signal.getsignal(number) for number in FORWARDED_SIGNALS}
    try:
        yield
    finally:
        for number, handler in saved.items():
            if handler is not None:
                signal.signal(number, handler)


def _spawn_and_wait(folder: Path, mode: str, lock_fd: int) -> int:
    """起雙精度子行程（單緒、新的行程群、繼承鎖），等它結束、把離開碼傳出來。"""
    env = runtime.child_process_env(threads=1)
    runtime.configure_jax(platforms="cpu", enable_x64=True, env=env)
    process: subprocess.Popen[bytes] | None = None
    try:
        previous = signal.pthread_sigmask(signal.SIG_BLOCK, FORWARDED_SIGNALS)
        try:
            process = subprocess.Popen([sys.executable, "-m", "aosr.gradient.worker", str(folder), mode], env=env,
                                       start_new_session=True, pass_fds=(lock_fd,), cwd=_REPO_ROOT,
                                       stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous)
        code = process.wait()
    except KeyboardInterrupt:
        if process is not None:
            _terminate(process)
        return EXIT_STOPPED
    if code < 0:
        sys.stderr.write(f"梯度子行程被訊號 {-code} 結束（例如記憶體上限）；狀態檔停在它最後寫的那一份\n")
        return EXIT_FAILED
    return code


def _start(args: argparse.Namespace) -> int:
    from aosr.config.capabilities import load_capabilities
    from aosr.config.directivity_defaults import load_directivity_defaults
    from aosr.config.material_gradient import load_material_gradient_settings
    from aosr.config.quality_targets import load_quality_targets
    from aosr.gradient import run
    from aosr.physics.fem_reduced_check import points_hex
    from aosr.physics.fem_reduced_receipt import receipt_path
    from aosr.reporting.fem_reduced_scheme import export_operators
    from aosr.reporting.scheme import load_scheme
    from aosr.scoring.stage_one_score import stage_one_score_settings

    scheme_path, surrogate, settings_path = args.scheme.resolve(), args.surrogate.resolve(), args.settings.resolve()
    if SEARCHES_DIR in args.root.resolve().parts:
        raise ValueError(f"--root 不能在 {SEARCHES_DIR}/ 底下：網頁會把裡面的資料夾當擺位搜尋讀、報讀取錯誤")
    scheme = load_scheme(scheme_path)
    if scheme.scene.low_frequency_axis is None:
        raise ValueError("方案場景沒有 low_frequency_axis（低頻軸）：搜尋跑完之後終點量大峰要它，開始前就拒收")
    settings = load_material_gradient_settings(settings_path)
    capabilities = load_capabilities(args.capabilities)
    directivity = load_directivity_defaults(config_path("directivity_defaults.toml"))
    run.designs_for(args.presets, settings,
                    kept=run.scene_wall_materials(scheme, capabilities=capabilities, directivity=directivity))
    score = stage_one_score_settings(load_quality_targets(config_path("quality_targets.toml")), args.purpose)
    with keep_signal_handlers():
        operators = export_operators(scheme, capabilities=capabilities, directivity=directivity)
    physics, mesh = operators.physics_identity, operators.mesh_hash
    points = run.SchemePoints(points_hex(operators.source_points), points_hex(operators.receiver_points))
    del operators
    model = run.gated_surrogate(surrogate, mesh_hash=mesh, physics_identity=physics, points=points)
    run.check_surrogate_matches_scheme(model, scheme, points)
    run.check_surrogate_axis_covers_scoring(model, scheme, score)
    del model
    hashes = {"scheme_sha256": run.file_sha256(scheme_path), "surrogate_manifest_sha256": run.manifest_sha256(surrogate)}
    identity = run.gradient_id(**hashes, presets=args.presets, purpose=args.purpose,
                               settings_fingerprint=settings.fingerprint, physics_identity=physics)
    project = run.GradientProject(
        format_version=run.PROJECT_VERSION, gradient_id=identity, scheme_path=str(scheme_path), **hashes,
        surrogate_path=str(surrogate), receipt_path=str(receipt_path(surrogate)),
        receipt_sha256=run.file_sha256(receipt_path(surrogate)), presets=args.presets, purpose=args.purpose,
        settings_path=str(settings_path), settings_fingerprint=settings.fingerprint,
        capabilities_path=str(args.capabilities.resolve()), physics_identity=physics, mesh_hash=mesh,
        source_points=points.sources, receiver_points=points.receivers, engine_commit=args.engine_commit, started_at=run.now_text())
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    return _create_and_run(root, project)


def _create_and_run(root: Path, project: GradientProject) -> int:
    """暫存名資料夾裡先拿鎖、寫 ``project.json``，再換成正式名、起子行程。同名資料夾已在就拒收（不碰它）。"""
    from aosr.gradient import run

    folder = root / project.gradient_id
    if folder.exists():
        return _existing(folder)
    staging = Path(tempfile.mkdtemp(prefix=f".{project.gradient_id}-", dir=root))
    with _locked(staging) as descriptor:
        try:
            run.write_json_atomic(staging / run.PROJECT_FILE, project)
            staging.rename(folder)
        except BaseException:
            (staging / run.PROJECT_FILE).unlink(missing_ok=True)
            staging.rmdir()
            if folder.exists():
                return _existing(folder)
            raise
        return _spawn_and_wait(folder, "start", descriptor)


def _existing(folder: Path) -> int:
    with _locked(folder):
        pass
    sys.stderr.write(f"資料夾 {folder} 已存在：同一組輸入已經開始過，要接續用 resume\n")
    return EXIT_FAILED


def _resume(args: argparse.Namespace) -> int:
    from aosr.config.capabilities import load_capabilities
    from aosr.config.directivity_defaults import load_directivity_defaults
    from aosr.gradient import run
    from aosr.reporting.physics_identity import physics_identity

    folder = args.folder.resolve()
    project = run.read_project(folder)
    with _locked(folder) as descriptor:
        current = physics_identity(capabilities=load_capabilities(project.capabilities_path),
                                   directivity=load_directivity_defaults(config_path("directivity_defaults.toml")))
        if current != project.physics_identity:
            sys.stderr.write(f"物理身分跟開始時不同（現在 {current}，開始時 {project.physics_identity}），"
                             "替身是舊程式建的，不接續\n")
            return EXIT_FAILED
        run.check_inputs(project)
        (folder / run.STOP_FILE).unlink(missing_ok=True)
        return _spawn_and_wait(folder, "resume", descriptor)


def _stop(args: argparse.Namespace) -> int:
    from aosr.gradient import run

    folder = args.folder.resolve()
    run.read_project(folder)
    (folder / run.STOP_FILE).touch()
    sys.stdout.write(f"已建停止記號 {folder / run.STOP_FILE}；子行程在下一次求值之前停\n")
    return 0


def _endpoint(args: argparse.Namespace) -> int:
    """拿資料夾鎖、起終點子行程（同一條 ``_spawn_and_wait``）；輸入閘門與結果檔核對在子行程裡做。"""
    from aosr.gradient import run

    folder = args.folder.resolve()
    run.read_project(folder)
    with _locked(folder) as descriptor:
        return _spawn_and_wait(folder, "endpoint", descriptor)


def _report(args: argparse.Namespace) -> int:
    from aosr.gradient.report import render_report

    sys.stdout.write(render_report(args.folder.resolve()))
    return 0


_COMMANDS: Final[dict[str, tuple[str, Callable[[argparse.Namespace], int]]]] = {
    "start": ("開始", _start), "resume": ("接續", _resume), "stop": ("停止", _stop), "endpoint": ("終點", _endpoint),
    "report": ("報告", _report)}


def main(argv: list[str] | None = None) -> int:
    """參數錯由 argparse 以離開碼 2 結束；其餘錯誤原文寫標準錯誤、回 1；中斷回 3。"""
    args = _parser().parse_args(argv)
    runtime.configure_calculation_jax()
    label, command = _COMMANDS[args.command]
    try:
        return command(args)
    except FolderBusy as error:
        sys.stderr.write(f"{error}\n")
        return EXIT_FAILED
    except KeyboardInterrupt:
        sys.stderr.write(f"{label}被中斷\n")
        return EXIT_STOPPED
    except Exception as error:
        sys.stderr.write(f"梯度搜尋{label}失敗：{type(error).__name__}：{error}\n")
        return EXIT_FAILED


def _interrupt(signum: int, frame: FrameType | None) -> None:
    raise KeyboardInterrupt


if __name__ == "__main__":
    for termination in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(termination, _interrupt)
    raise SystemExit(main())
