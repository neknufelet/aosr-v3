"""牆面材料梯度搜尋命令列（#561 辛一）：``python -m aosr.gradient.cli start|resume|stop|report``。

- ``start --scheme <方案檔> --surrogate <替身 npz> --presets <六個 id，逗號分隔，剛性面寫 none> --purpose <用途>
  --root <資料夾根> [--settings <material_gradient.toml>] [--capabilities <能力表>] [--engine-commit <提交>]``：
  先核輸入（preset 在家族裡、用途認得、方案匯出完整模型拿網格雜湊與物理身分、替身過准用閘門而且對得上兩格、
  替身座位與喇叭對得上方案），都過才建 ``<root>/<32 位十六進位>/``（名字由輸入算，同一組輸入同一個資料夾），
  寫 ``project.json``，在雙精度子行程把整條鏈跑完。拒收不建資料夾。
- ``resume <資料夾>``：拿鎖、重算現在的物理身分跟 ``project.json`` 比（不同就拒收），清掉留下的 ``stop`` 記號，
  子行程照帳本接著跑（標頭對不上拒收；前段沒寫完重跑前段）。
- ``stop <資料夾>``：建 ``stop`` 記號，不拿鎖，回 0；子行程在下一次求值之前停。
- ``report --folder <資料夾>``：只讀，印人看的文字，最後一行固定是終點完整模型重算未做（辛二）。

離開碼跟擺位搜尋命令列同一套：0 完成、1 失敗（含替身准用閘門拒用，錯誤原文寫標準錯誤）、2 參數錯、3 中斷或停止記號。
資料夾鎖：資料夾本身的描述子 ``flock``，子行程經 ``pass_fds`` 繼承；父行程只關自己那一份、不解鎖。
``start`` 在暫存名的資料夾裡先拿鎖、寫好 ``project.json`` 再換成正式名，所以看得到的資料夾一出現就有鎖：
另一個 ``start`` 撞到同名資料夾時，鎖被拿著就報「還有計算在跑」，沒被拿著就報「已存在、用 resume」，兩種都回 1。
父行程收到 SIGTERM、SIGHUP（只在 ``__main__`` 轉成中斷）就對子行程群送 SIGTERM，等它寫完 interrupted 再回 3。

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
from collections.abc import Callable, Iterator
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
RIGID: Final = "none"
SEARCHES_DIR: Final = "searches"
TERMINATE_GRACE_S: Final = 60.0
_REPO_ROOT: Final = Path(__file__).resolve().parents[3]


class FolderBusy(Exception):
    """拿不到資料夾鎖（還有計算在跑），或資料夾打不開；什麼都沒動。"""


def _presets(text: str) -> tuple[str | None, ...]:
    """``a,b,none,...`` → 六個 preset id（``none`` 是剛性面）；個數不是牆數就是參數錯（離開碼 2）。"""
    items = tuple(item.strip() for item in text.split(","))
    walls = [wall.wall_name() for wall in Wall.all()]
    if len(items) != len(walls) or any(not item for item in items):
        raise argparse.ArgumentTypeError(
            f"--presets 要照 {','.join(walls)} 給 {len(walls)} 個非空的 preset id（剛性面寫 {RIGID}），收到 {len(items)} 個")
    return tuple(None if item == RIGID else item for item in items)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aosr.gradient.cli")
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("start", help="開始牆面材料梯度搜尋")
    start.add_argument("--scheme", type=Path, required=True, help="方案檔（座位組要一主位加六周圍點）")
    start.add_argument("--surrogate", type=Path, required=True, help="驗過的替身 npz（旁邊要有 manifest 與驗證收據）")
    start.add_argument("--presets", type=_presets, required=True, help="六面 preset id，照 Wall.all()，剛性面寫 none")
    start.add_argument("--purpose", required=True, help="用途（評分設定）")
    start.add_argument("--root", type=Path, required=True, help="資料夾根（不要放在網頁讀的 searches/ 底下）")
    start.add_argument("--settings", type=Path, default=config_path("material_gradient.toml"))
    start.add_argument("--capabilities", type=Path, default=config_path("capabilities.toml"))
    start.add_argument("--engine-commit", default=None)
    resume = commands.add_parser("resume", help="接續")
    resume.add_argument("folder", type=Path)
    stop = commands.add_parser("stop", help="建立停止記號")
    stop.add_argument("folder", type=Path)
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


def _spawn_and_wait(folder: Path, mode: str, lock_fd: int) -> int:
    """起雙精度子行程（單緒、新的行程群、繼承鎖），等它結束、把離開碼傳出來。"""
    env = runtime.child_process_env(threads=1)
    runtime.configure_jax(platforms="cpu", enable_x64=True, env=env)
    process = subprocess.Popen([sys.executable, "-m", "aosr.gradient.worker", str(folder), mode], env=env,
                               start_new_session=True, pass_fds=(lock_fd,), cwd=_REPO_ROOT,
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    try:
        code = process.wait()
    except KeyboardInterrupt:
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
    from aosr.physics.fem_reduced_receipt import load_validated_reduced, receipt_path
    from aosr.reporting.fem_reduced_scheme import export_operators
    from aosr.reporting.scheme import load_scheme
    from aosr.scoring.stage_one_score import stage_one_score_settings

    scheme_path, surrogate, settings_path = args.scheme.resolve(), args.surrogate.resolve(), args.settings.resolve()
    if SEARCHES_DIR in args.root.resolve().parts:
        raise ValueError(f"--root 不能在 {SEARCHES_DIR}/ 底下：網頁會把裡面的資料夾當擺位搜尋讀、報讀取錯誤")
    scheme = load_scheme(scheme_path)
    settings = load_material_gradient_settings(settings_path)
    run.designs_for(args.presets, settings)
    stage_one_score_settings(load_quality_targets(config_path("quality_targets.toml")), args.purpose)
    operators = export_operators(scheme, capabilities=load_capabilities(args.capabilities),
                                 directivity=load_directivity_defaults(config_path("directivity_defaults.toml")))
    physics, mesh = operators.physics_identity, operators.mesh_hash
    del operators
    model = load_validated_reduced(surrogate, rulers=run.holdout_rulers(), expected_mesh_hash=mesh,
                                   expected_physics_identity=physics)
    run.check_surrogate_matches_scheme(model, scheme)
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
        engine_commit=args.engine_commit, started_at=run.now_text())
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
        (folder / run.STOP_FILE).unlink(missing_ok=True)
        return _spawn_and_wait(folder, "resume", descriptor)


def _stop(args: argparse.Namespace) -> int:
    from aosr.gradient import run

    folder = args.folder.resolve()
    run.read_project(folder)
    (folder / run.STOP_FILE).touch()
    sys.stdout.write(f"已建停止記號 {folder / run.STOP_FILE}；子行程在下一次求值之前停\n")
    return 0


def _report(args: argparse.Namespace) -> int:
    from aosr.gradient.report import render_report

    sys.stdout.write(render_report(args.folder.resolve()))
    return 0


_COMMANDS: Final[dict[str, tuple[str, Callable[[argparse.Namespace], int]]]] = {
    "start": ("開始", _start), "resume": ("接續", _resume), "stop": ("停止", _stop), "report": ("報告", _report)}


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
