"""牆面材料梯度搜尋命令列（#561 辛一、辛三、庚三）：``python -m aosr.gradient.cli start|resume|stop|endpoint|report``。

- ``start --scheme <方案檔> --surrogate <替身 npz> --presets <六個 id，逗號分隔，不貼的面寫 none> --purpose <用途>
  --root <資料夾根> [--settings <material_gradient.toml>] [--capabilities <能力表>] [--engine-commit <提交>]``：
  先核輸入（preset 在家族裡、用途認得、方案匯出完整模型拿網格雜湊、物理身分與喇叭座位座標、替身過准用閘門而且
  對得上這幾格、替身座位與喇叭的名字對得上方案），都過才建 ``<root>/<32 位十六進位>/``（名字由輸入算，同一組輸入同一個資料夾），
  寫 ``project.json``，在雙精度子行程把整條鏈跑完。拒收不建資料夾。
- ``resume <資料夾>``：拿鎖、重算現在的物理身分跟 ``project.json`` 比（不同就拒收），再逐格核輸入（方案檔雜湊、
  方案座標、替身過閘門、manifest 與收據摘要、設定指紋、梯度目標設定指紋——#561 壬一加大峰代理之前開始的資料夾沒有這一格，
  一律拒收；子行程重建目標前會再核一次），清掉留下的 ``stop`` 記號，
  子行程照帳本接著跑（標頭對不上拒收；前段沒寫完重跑前段）；庚二帳本已收尾、還沒有 ``templates.jsonl`` 的資料夾
  （庚三之前跑完的）接續就開換樣板段，跑完把 ``result.json`` 覆寫成 v3。介面跟庚三之前一樣。
- ``stop <資料夾>``：建 ``stop`` 記號，不拿鎖，回 0；子行程在下一次求值之前停。
- ``endpoint <資料夾>``：搜尋跑完之後拿鎖、起雙精度子行程做終點（``endpoint_run``）：同一道輸入閘門、完整模型重算
  ``result.json`` 那一點、方案原本六面的完整模型基準、匯出完整方案檔 ``scheme-endpoint.json``、寫 ``endpoint.json``；
  不動 ``status.json``。沒有 ``result.json`` 就失敗。
- ``report --folder <資料夾>``：只讀，印人看的文字；有對得上帳本的 ``endpoint.json`` 就印終點一節，對不上印過期，
  沒有就最後一行說終點還沒做。

``--presets`` 寫 ``none`` 的面是「不貼、保留方案原本的牆」（老闆 10-28：不貼算一個選項、用現在的六面當起點）：
搜尋、終點、匯出都用方案場景那一面原本的材料，不是剛性。
``--mount``（#990，安裝方式，每次搜尋一個值，預設 ``rigid``）：``on_wall`` 是「貼在原牆上」——貼樣板的面新增層背後接
方案場景那一面原本的牆（不是剛背），換樣板沿用；記進 ``project.json``，非剛背時資料夾名也不同。只有 ``start`` 會開新搜尋，
``resume``／``endpoint`` 照 ``project.json`` 記的安裝方式。

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
import hashlib
import json
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
from aosr.physics.fem_full_batch import checked_workers, checked_device_workers
from aosr.config.paths import config_path
from aosr.geometry.shoebox import Wall
from aosr.config.material_gradient import MOUNT_RIGID, MOUNTS, fixed_target_id
from aosr.materials.search_library import (
    SNAPSHOT_FILE, MaterialsRefused, PresetMaterial, SearchMaterials, cavity_notes, fixed_candidates, read_user_entries,
    refuse_fixed_as_family, snapshot_bytes, user_fixed_entries, with_user_materials,
)

if TYPE_CHECKING:
    from aosr.gradient.gpu import Device
    from aosr.gradient.run import GradientProject
    from aosr.gradient.material_gradient import SurfaceDesign
    from aosr.config.material_gradient import MaterialGradientSettings
    from aosr.physics.fem_reduced import FullOperators, ReducedModel
    from aosr.physics.fem_reduced_check import SurrogateSpotCheck
    from aosr.reporting.scheme import Scheme
    from aosr.physics.fem_reduced_boundary import FrequencyBoundaryImpedances
    from aosr.gradient.fixed_candidates import FixedCandidateCheck
    from aosr.materials.library import LibraryEntry

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
    """``a,b,none,...`` → 面片 preset；至少六片，確切片數載入方案後核對，none 保留該片材料。"""
    items = tuple(item.strip() for item in text.split(","))
    walls = [wall.wall_name() for wall in Wall.all()]
    if len(items) < len(walls) or any(not item for item in items):
        raise argparse.ArgumentTypeError(
            f"--presets 要照 {','.join(walls)} 給 {len(walls)} 個非空的 preset id（不貼的面寫 {KEEP_WALL}），收到 {len(items)} 個")
    return tuple(None if item == KEEP_WALL else item for item in items)


def _workers(text: str) -> int:
    try:
        return checked_workers(int(text))
    except ValueError as error:
        raise argparse.ArgumentTypeError("--workers 必須是正整數") from error


def _frequency_batch(text: str) -> int:
    """命令列頻點批量：只收正整數。"""
    try:
        value = int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("frequency_batch 必須是正整數") from error
    if value <= 0:
        raise argparse.ArgumentTypeError("frequency_batch 必須是正整數")
    return value


def _max_steps(text: str) -> int:
    """命令列往下走步數上限（#1073）：只收正整數。"""
    try:
        value = int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("--max-steps 必須是正整數") from error
    if value <= 0:
        raise argparse.ArgumentTypeError("--max-steps 必須是正整數")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aosr.gradient.cli")
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("start", help="開始牆面材料梯度搜尋")
    start.add_argument("--scheme", type=Path, required=True, help="方案檔（座位組要一主位加六周圍點）")
    start.add_argument("--surrogate", type=Path, required=True, help="驗過的替身 npz（旁邊要有 manifest 與驗證收據）")
    start.add_argument("--presets", type=_presets, required=True, help="面片 preset id，照 boundary_patches 順序；整面照 Wall.all()，不貼寫 none")
    start.add_argument("--purpose", required=True, help="用途（評分設定）")
    start.add_argument("--root", type=Path, required=True, help="資料夾根（不要放在網頁讀的 searches/ 底下）")
    start.add_argument("--settings", type=Path, default=config_path("material_gradient.toml"))
    start.add_argument("--device", choices=("cpu", "gpu"), default="cpu", help="搜尋裝置：cpu 雙精度；gpu 單精度鏈")
    start.add_argument("--frequency-batch", type=_frequency_batch, default=None, metavar="N",
                       help="替身每批求解的頻點數（正整數）；反傳逐批重算")
    start.add_argument("--max-steps", type=_max_steps, default=None, metavar="N",
                       help="往下走最多幾步（正整數），例如 30、60、100；不給照 --settings 那份的 [descent] max_steps；"
                            "指定的值存進搜尋資料夾，接著跑、終點、報告照用")
    start.add_argument("--max-depth", action="append", default=[], metavar="面=公尺",
                       help="每面總深度上限，可重複；整面或面片，例如 x0=0.08、x0@u0v0=0.08")
    start.add_argument("--capabilities", type=Path, default=config_path("capabilities.toml"))
    start.add_argument("--engine-commit", default=None)
    start.add_argument("--mount", choices=MOUNTS, default=MOUNT_RIGID,
                       help="安裝方式：rigid 樣板照原樣（剛背）；on_wall 貼在原牆上（新增層背後接方案原本那面牆）")
    start.add_argument("--fixed-surface", action="append", default=[], metavar="面=材料id",
                       help="固定指定面的材料，不參與優化；可重複給不同面，材料 id 取現有材料庫，"
                            "寫 scheme-material 就是用方案檔那一面自己的材料")
    start.add_argument("--materials", type=Path, default=None, metavar="材料庫.json",
                       help="使用者材料庫（網頁資料夾的 materials.json 的複本）：可最佳化材料一筆一個家族接在設定檔的家族後面，"
                            "存一份快照進搜尋資料夾；不給就只用設定檔的家族")
    start.add_argument("--allowed", action="append", default=[], metavar="面=目標,目標",
                       help="每面（或面片）准用的換樣板目標，可重複：家族代號、fixed:<材料代號>（整塊換成固定材料）、none；"
                            "沒給的面照舊換全部家族與 none；起跑那一面的目標一定要在清單裡")
    start.add_argument("--workers", type=_workers, default=1,
                       help="准用清單有固定材料時，開跑前替身留出驗證的完整解工人數；預設 1")
    resume = commands.add_parser("resume", help="接續")
    resume.add_argument("folder", type=Path)
    stop = commands.add_parser("stop", help="建立停止記號")
    stop.add_argument("folder", type=Path)
    endpoint = commands.add_parser("endpoint", help="終點：完整模型重算、原牆基準、匯出方案檔")
    endpoint.add_argument("folder", type=Path)
    endpoint.add_argument("--device", choices=("cpu", "gpu"), default="cpu", help="完整解裝置；預設 cpu")
    endpoint.add_argument("--workers", type=_workers, default=1, help="完整解工人數；預設 1（單行程）")
    report = commands.add_parser("report", help="只讀報告")
    report.add_argument("--folder", type=Path, required=True)
    stamp = commands.add_parser("stamp-surrogate", help="用原身分與完整算子替已驗過的替身補指紋")
    stamp.add_argument("--scheme", type=Path, required=True)
    stamp.add_argument("--surrogate", type=Path, required=True)
    stamp.add_argument("--capabilities", type=Path, default=config_path("capabilities.toml"))
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


def _spawn_and_wait(folder: Path, mode: str, lock_fd: int, *, workers: int = 1, device: Device = "cpu") -> int:
    """起雙精度子行程（單緒、新的行程群、繼承鎖），等它結束、把離開碼傳出來。"""
    env = runtime.child_process_env(threads=1)
    runtime.configure_jax(platforms="cuda,cpu" if device == "gpu" and mode != "endpoint" else "cpu",
                          enable_x64=True, env=env)
    command = [sys.executable, "-m", "aosr.gradient.worker", str(folder), mode]
    if mode == "endpoint" and device == "gpu":
        checked_device_workers(device, workers)
        command.extend(("--device", device))
    if workers != 1:
        command.extend(("--workers", str(checked_workers(workers))))
    process: subprocess.Popen[bytes] | None = None
    try:
        previous = signal.pthread_sigmask(signal.SIG_BLOCK, FORWARDED_SIGNALS)
        try:
            process = subprocess.Popen(command, env=env,
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
    from aosr.gradient import run
    from aosr.gradient.fixed_surfaces import fixed_targets_of, parse_allowed_targets, parse_fixed_surfaces, parse_max_depths
    from aosr.physics import fem_reduced, fem_reduced_check, fem_reduced_receipt
    from aosr.gradient import boundary
    from aosr.reporting.fem_reduced_scheme import export_operators

    fixed, caps, allowed = (parse_fixed_surfaces(args.fixed_surface), parse_max_depths(args.max_depth),
                            parse_allowed_targets(args.allowed))
    scheme_path, surrogate, settings_path = args.scheme.resolve(), args.surrogate.resolve(), args.settings.resolve()
    scheme = _start_scheme(args.root, scheme_path)
    materials, settings, snapshot, materials_sha = _start_materials(args, settings_path, allowed)
    capabilities, directivity = load_capabilities(args.capabilities), load_directivity_defaults(config_path("directivity_defaults.toml"))
    kept = run.scene_wall_materials(scheme, capabilities=capabilities, directivity=directivity)
    run.require_supported_original_walls(kept.values())
    designs = run.designs_for(args.presets, settings, kept=kept, mount=args.mount, fixed_surfaces=fixed, max_depths=caps, table=materials.presets)
    expanded = _checked_allowed(allowed, designs, settings, fixed, caps, materials.presets)
    objective = run.search_objective_settings(args.purpose)
    with keep_signal_handlers():
        operators = export_operators(scheme, capabilities=capabilities, directivity=directivity)
    physics, mesh = operators.physics_identity, operators.mesh_hash
    full_operators = operators if fixed_targets_of(expanded) else None
    points = run.SchemePoints(fem_reduced_check.points_hex(operators.source_points), fem_reduced_check.points_hex(operators.receiver_points))
    manifest_identity = _surrogate_identity(surrogate, physics)
    fingerprint = fem_reduced.operators_digest(operators) if manifest_identity != physics else None
    if fingerprint is None:
        del operators
        model = run.gated_surrogate(surrogate, mesh_hash=mesh, physics_identity=physics, points=points)
    else:
        model = run.gated_surrogate(surrogate, mesh_hash=mesh, physics_identity=physics, points=points,
                                    current_operators_digest=fingerprint)
    run.check_surrogate_matches_scheme(model, scheme, points)
    run.check_surrogate_axis_covers_scoring(model, scheme, objective.stage_one)
    _check_start_materials(materials, model, scheme)
    checks = _fixed_candidate_checks(model, full_operators, designs, expanded, materials.presets, scheme, args.workers)
    del full_operators
    spot = None
    if fingerprint is not None:
        spot = _carryover_spot_check(model, operators, designs, scheme)
        del operators
    del model
    hashes = {"scheme_sha256": run.file_sha256(scheme_path), "surrogate_manifest_sha256": run.manifest_sha256(surrogate)}
    identity = run.gradient_id(**hashes, presets=args.presets, purpose=args.purpose,
                               settings_fingerprint=settings.fingerprint, physics_identity=physics,
                               objective_settings_fingerprint=objective.fingerprint, mount=args.mount, fixed_surfaces=fixed,
                               max_depths=caps, patch_keys=boundary.scheme_patch_keys(scheme), frequency_batch=args.frequency_batch, device=args.device, materials_sha256=materials_sha,
                               allowed_targets=allowed)
    root = args.root.resolve()
    project = run.GradientProject(
        format_version=run.PROJECT_VERSION, gradient_id=identity, scheme_path=str(scheme_path),
        scheme_sha256=hashes["scheme_sha256"], surrogate_manifest_sha256=hashes["surrogate_manifest_sha256"],
        surrogate_path=str(surrogate), receipt_path=str(fem_reduced_receipt.receipt_path(surrogate)),
        receipt_sha256=run.file_sha256(fem_reduced_receipt.receipt_path(surrogate)), presets=args.presets, purpose=args.purpose,
        settings_path=str(settings_path), settings_fingerprint=settings.fingerprint,
        capabilities_path=str(args.capabilities.resolve()), physics_identity=physics, mesh_hash=mesh,
        source_points=points.sources, receiver_points=points.receivers, engine_commit=args.engine_commit, started_at=run.now_text(),
        objective_settings_fingerprint=objective.fingerprint, mount=args.mount, fixed_surfaces=fixed, max_depths=caps,
        frequency_batch=args.frequency_batch, max_steps=args.max_steps, device=args.device,
        surrogate_physics_identity=str(manifest_identity) if fingerprint is not None else None,
        operators_digest=fingerprint, surrogate_spot_check=spot, materials_sha256=materials_sha,
        materials_path=None if snapshot is None else str(root / identity / SNAPSHOT_FILE),
        allowed_targets=allowed, fixed_candidate_checks=checks)
    root.mkdir(parents=True, exist_ok=True)
    return _create_and_run(root, project) if snapshot is None else _create_and_run(root, project, snapshot)


def _check_start_materials(materials: SearchMaterials, model: ReducedModel, scheme: Scheme) -> None:
    """接進來的使用者家族在這一場的頻點與空氣下逐一建得起、算得出（:func:`aosr.gradient.run.check_user_families`）；
    空腔不調的那幾筆照實印一行。沒有使用者家族什麼都不做（不碰替身）。"""
    if not materials.entries:
        return
    import numpy as np

    from aosr.gradient.run import check_user_families

    check_user_families(materials, tuple(float(value) for value in np.asarray(model.frequencies_hz)),
                        density_kg_m3=scheme.scene.density_kg_m3, sound_speed_m_s=scheme.scene.sound_speed_m_s)
    for note in cavity_notes(materials):
        sys.stderr.write(f"注意：{note}\n")


def _start_scheme(root: Path, scheme_path: Path) -> Scheme:
    """開跑前兩條：資料夾根不在網頁讀的 searches/ 底下；方案場景要有低頻軸（終點量大峰要它）。"""
    from aosr.reporting.scheme import load_scheme

    if SEARCHES_DIR in root.resolve().parts:
        raise ValueError(f"--root 不能在 {SEARCHES_DIR}/ 底下：網頁會把裡面的資料夾當擺位搜尋讀、報讀取錯誤")
    scheme = load_scheme(scheme_path)
    if scheme.scene.low_frequency_axis is None:
        raise ValueError("方案場景沒有 low_frequency_axis（低頻軸）：搜尋跑完之後終點量大峰要它，開始前就拒收")
    return scheme


def _checked_allowed(allowed: dict[str, tuple[str, ...]], designs: tuple[SurfaceDesign, ...],
                     settings: MaterialGradientSettings, fixed: dict[str, str],
                     caps: dict[str, float], table: tuple[PresetMaterial, ...]) -> dict[str, tuple[str, ...]]:
    """准用清單對起跑配置核一次（:func:`~aosr.gradient.fixed_surfaces.check_allowed_against_start`），回展開到面片的清單。
    起跑配置任何一面直接寫固定材料都拒收（起點要是家族樣板或 none）；六類評分收不了的固定材料（裸實測表）在匯出完整模型
    與關卡之前就拒收（:func:`~aosr.gradient.fixed_candidates.refuse_unscored_fixed_candidates`），不花完整解的錢。"""
    from aosr.config.material_gradient import is_fixed_target
    from aosr.gradient.fixed_candidates import refuse_unscored_fixed_candidates
    from aosr.gradient.fixed_surfaces import check_allowed_against_start, expanded_allowed_targets

    surfaces = tuple((design.surface_id, design.preset_id) for design in designs)
    if any(is_fixed_target(preset) for _, preset in surfaces):
        raise ValueError("--presets 不能直接寫固定材料（fixed:…）：起點要是家族樣板或 none，固定材料從 --allowed 進來")
    check_allowed_against_start(allowed, surfaces, settings, fixed_surfaces=fixed, max_depths=caps)
    expanded = expanded_allowed_targets(allowed, [name for name, _ in surfaces])
    refuse_unscored_fixed_candidates(designs, expanded, table)
    return expanded


def _fixed_candidate_checks(model: ReducedModel, operators: FullOperators | None, designs: tuple[SurfaceDesign, ...],
                            allowed: dict[str, tuple[str, ...]], table: tuple[PresetMaterial, ...], scheme: Scheme,
                            workers: int) -> tuple[FixedCandidateCheck, ...]:
    """准用清單有固定材料：表外補值照實印、每一個各做一組替身留出驗證（沒過就拒收）；沒有就什麼都不做。"""
    from aosr.gradient.fixed_candidates import check_fixed_candidates, table_range_notes
    from aosr.gradient.fixed_surfaces import fixed_targets_of

    targets = fixed_targets_of(allowed)
    if not targets:
        return ()
    assert operators is not None
    for note in table_range_notes(targets, table, tuple(float(value) for value in model.frequencies_hz)):
        sys.stderr.write(f"注意：{note}\n")
    checks = check_fixed_candidates(model, operators, designs, allowed, table, density_kg_m3=scheme.scene.density_kg_m3,
                                    sound_speed_m_s=scheme.scene.sound_speed_m_s, workers=workers)
    for check in checks:
        sys.stderr.write(f"固定材料 {check.target} 替身留出驗證通過：逐頻最大差 {check.frequency_db:g} dB、帶能量最大差 "
                         f"{check.band_db:g} dB（完整解 {check.full_solves} 次、{check.seconds:.1f} 秒）\n")
    return checks


def _start_materials(args: argparse.Namespace, settings_path: Path,
                     allowed: dict[str, tuple[str, ...]]) -> tuple[SearchMaterials, MaterialGradientSettings,
                                                                    bytes | None, str | None]:
    """這一場的材料、生效設定（步數上限已套）、材料快照內文與整份 SHA-256。

    ``--materials`` 給了：讀使用者材料庫（檔案不在、讀不進來都照講原因拒收）、起跑配置不准填固定材料、可最佳化材料接成家族；
    沒給、或材料庫裡沒有可最佳化材料：設定檔與產品構造預設原樣、沒有快照（資料夾名與 ``project.json`` 逐位照舊）。"""
    from aosr.config.material_gradient import load_material_gradient_settings, with_max_steps
    from aosr.gradient.fixed_surfaces import fixed_targets_of
    from aosr.gradient.run import presets_table

    base = load_material_gradient_settings(settings_path)
    entries: tuple[LibraryEntry, ...] = ()
    if args.materials is None:
        materials = SearchMaterials(base, presets_table(), ())
    else:
        path = args.materials.resolve()
        if not path.is_file():
            raise MaterialsRefused(f"--materials 指的使用者材料庫 {path} 不存在，搜尋不開（沒有使用者材料就不帶這個旗標）")
        entries = read_user_entries(path)
        refuse_fixed_as_family(args.presets, entries)
        materials = with_user_materials(base, presets_table(), entries)
    settings = materials.settings if args.max_steps is None else with_max_steps(materials.settings, args.max_steps)
    targets = fixed_targets_of(allowed)
    user_fixed = user_fixed_entries([fixed_target_id(target) for target in targets], entries)
    if targets:
        materials = materials._replace(presets=(*materials.presets, *fixed_candidates(targets, materials.presets, user_fixed)))
    if not materials.entries and not user_fixed:
        return materials, settings, None, None
    content = snapshot_bytes(materials.entries, user_fixed)
    return materials, settings, content, hashlib.sha256(content).hexdigest()


def _surrogate_identity(surrogate: Path, current: str) -> str:
    """只預讀分支；manifest 的可信度仍由完整准用閘門核對。"""
    from aosr.physics.fem_reduced_store import manifest_path

    try:
        document = json.loads(manifest_path(surrogate).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return current  # 預讀失敗仍走原閘門，由它保留既有拒收錯誤與原因。
    return str(document.get("physics_identity", current)) if isinstance(document, dict) else current


def _carryover_spot_check(model: ReducedModel, operators: FullOperators, designs: tuple[SurfaceDesign, ...],
                           scheme: Scheme) -> SurrogateSpotCheck:
    """用下降真正的起跑材料做抽查，尺由登記簿讀；不套終點離散規格與匯出空腔裁切。"""
    from aosr.gradient.run import holdout_rulers
    from aosr.physics.fem_reduced import even_indices
    from aosr.physics.fem_reduced_check import spot_check_reduced

    axis = tuple(model.frequencies_hz[index] for index in even_indices(8, len(model.frequencies_hz)))
    impedances = _starting_impedances(designs, axis, scheme)
    ruler = holdout_rulers().frequency
    spot = spot_check_reduced(model, operators, impedances, axis, ruler_name=ruler.name, frequency_db=ruler.value)
    if not spot.passed:
        raise ValueError(f"替身沿用抽查不通過：最大差 {spot.max_db} dB（尺 {ruler.name}={ruler.value} dB），要重建")
    sys.stderr.write(f"替身身分換過、完整模型逐位相同、抽查 {len(axis)} 點最大差 {spot.max_db:g} dB"
                     f"（尺 {ruler.name}={ruler.value:g} dB），沿用\n")
    return spot


def _starting_impedances(designs: tuple[SurfaceDesign, ...], frequencies_hz: tuple[float, ...],
                         scheme: Scheme) -> FrequencyBoundaryImpedances:
    """保留原牆、固定材料與樣板全用搜尋同一條材料鏈；θ 先投影到盒界與總深度內（:func:`~aosr.gradient.fixed_candidates.design_impedances`）。"""
    from aosr.gradient.fixed_candidates import design_impedances

    return design_impedances(designs, frequencies_hz, density_kg_m3=scheme.scene.density_kg_m3,
                             sound_speed_m_s=scheme.scene.sound_speed_m_s)


def _create_and_run(root: Path, project: GradientProject, snapshot: bytes | None = None) -> int:
    """暫存名資料夾裡先拿鎖、寫 ``project.json``（有材料快照先寫 ``materials.json``），再換成正式名、起子行程。
    同名資料夾已在就拒收（不碰它）。"""
    from aosr.gradient import run

    folder = root / project.gradient_id
    if folder.exists():
        return _existing(folder)
    staging = Path(tempfile.mkdtemp(prefix=f".{project.gradient_id}-", dir=root))
    with _locked(staging) as descriptor:
        try:
            if snapshot is not None:
                (staging / SNAPSHOT_FILE).write_bytes(snapshot)
            run.write_json_atomic(staging / run.PROJECT_FILE, project)
            staging.rename(folder)
        except BaseException:
            (staging / run.PROJECT_FILE).unlink(missing_ok=True)
            (staging / SNAPSHOT_FILE).unlink(missing_ok=True)
            staging.rmdir()
            if folder.exists():
                return _existing(folder)
            raise
        return _spawn_and_wait(folder, "start", descriptor, device=project.device)


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
        return _spawn_and_wait(folder, "resume", descriptor, device=project.device)


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

    checked_device_workers(args.device, args.workers)
    folder = args.folder.resolve()
    run.read_project(folder)
    with _locked(folder) as descriptor:
        if args.device == "gpu":
            return _spawn_and_wait(folder, "endpoint", descriptor, workers=args.workers, device=args.device)
        if args.workers == 1:
            return _spawn_and_wait(folder, "endpoint", descriptor)
        return _spawn_and_wait(folder, "endpoint", descriptor, workers=args.workers)


def _report(args: argparse.Namespace) -> int:
    from aosr.gradient.run import project_presets, read_project
    from aosr.gradient.report import render_report

    project = read_project(args.folder.resolve())
    project_presets(project)  # 材料快照被改過就拒收（SHA-256 對不上丟 ValueError），不印半份報告
    if (batch := project.frequency_batch) is not None:
        sys.stdout.write(f"替身頻點分批：每批 {batch} 點（反傳逐批重算）\n")
    sys.stdout.write(_max_steps_line(project))
    sys.stdout.write(render_report(args.folder.resolve()))
    return 0


def _max_steps_line(project: GradientProject) -> str:
    """報告開頭那一行往下走步數上限（#1073）：開跑時指定的照記的印；沒指定的照設定檔，讀不到或指紋不同照實說。"""
    from aosr.gradient.run import project_settings

    if project.max_steps is not None:
        return f"往下走步數上限：{project.max_steps} 步（開跑時指定）\n"
    try:
        settings = project_settings(project)
    except (OSError, ValueError) as error:
        return f"往下走步數上限：設定讀不到（{type(error).__name__}）\n"
    if settings.fingerprint != project.settings_fingerprint:
        return "往下走步數上限：設定指紋不同，無法確認這一跑的上限\n"
    return f"往下走步數上限：{settings.descent.max_steps} 步（設定檔）\n"


def _stamp_surrogate(args: argparse.Namespace) -> int:
    from aosr.config.capabilities import load_capabilities
    from aosr.config.directivity_defaults import load_directivity_defaults
    from aosr.gradient.run import holdout_rulers
    from aosr.physics.fem_reduced_store import save_operators_stamp
    from aosr.reporting.fem_reduced_scheme import export_operators
    from aosr.reporting.scheme import load_scheme

    with keep_signal_handlers():
        operators = export_operators(
            load_scheme(args.scheme), capabilities=load_capabilities(args.capabilities),
            directivity=load_directivity_defaults(config_path("directivity_defaults.toml")))
    target = save_operators_stamp(args.surrogate, operators, rulers=holdout_rulers())
    sys.stdout.write(f"替身完整模型指紋：{target}\n")
    return 0


_COMMANDS: Final[dict[str, tuple[str, Callable[[argparse.Namespace], int]]]] = {
    "start": ("開始", _start), "resume": ("接續", _resume), "stop": ("停止", _stop), "endpoint": ("終點", _endpoint),
    "report": ("報告", _report), "stamp-surrogate": ("補指紋", _stamp_surrogate)}


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
