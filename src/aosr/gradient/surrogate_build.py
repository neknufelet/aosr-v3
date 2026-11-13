"""替一份切法建替身（surrogate，縮減模型）的命令列（#1098）：網頁在獨立服務裡叫它，人也可以直接叫。

``python -m aosr.gradient.surrogate_build --scheme <方案檔> --work <建模資料夾> --slot <格子資料夾> --capabilities <能力表>
[--settings <搜尋設定>] [--materials <使用者材料庫>] [--allowed 面=目標,… …]``

照搜尋開跑那一段（``aosr.gradient.cli`` 的 ``start``）的同一組取法，免得建出來的替身開跑時被拒：
完整模型算子照 ``export_operators`` 的預設網格、頻率軸照方案場景的低頻軸、原牆照 ``run.scene_wall_materials``；
家族也跟開跑同一份（#1082 段 2-3c）：``--settings`` 的家族在前、``--materials`` 的可最佳化材料一筆一個接在後，
接進來的先在這一場的頻點與空氣下逐一建得起、算得出（``run.check_user_families``，比匯出完整模型便宜，排前面）。
``--allowed``（跟開跑同一個寫法）列的固定材料候選也一起訓練與留出（:mod:`aosr.gradient.surrogate_training`）。
有產品設定檔以外的家族或固定材料候選時，訓練說明記下訓練過的家族、範圍與候選（:mod:`aosr.gradient.surrogate_coverage`），
進格子前用開跑同一支 ``check_covers`` 核一次。
步驟：開跑前核對（方案有低頻軸、原牆能貼）→ 匯出完整模型 → 訓練與留出材料（:mod:`aosr.gradient.surrogate_training`）
→ 建模（停止規則照 ``fem_reduced.toml``：殘差達標或完整解預算用完。``--stop-residual``、``--max-full-solves`` 只換這一次建模的
停止值與預算，設定檔預設不動；換了照實寫進建模帳，建模帳在 manifest 與輸入雜湊裡，驗證收據對的就是它，等於進了替身身分）
→ 一定跑留出驗證、寫驗證收據 → 兩把尺都過才補完整模型指紋 → 用搜尋開跑那一道准用閘門核
（``run.gated_surrogate``、``check_surrogate_matches_scheme``、``check_surrogate_axis_covers_scoring``）→ 整個資料夾換名進格子。
准用清單有固定材料候選時（#1082 段 2-8），建模之後、留出驗證之前先用開跑那一組起跑配置（``--presets``、``--mount``、
``--fixed-surface``、``--max-depth``，跟開跑同一個寫法；網頁照預覽那一份搜尋計畫交）跑開跑同一道固定材料核對
（:func:`aosr.gradient.fixed_candidates.judge_candidates`，全部候選都核）；有沒過的就補訓練（:meth:`TrainingSets.extension_sets`：
候選放在准用它的片、其他片圍著起跑配置變；核對那一組本身剔除、留著當留出），接著同一份快照往下補點
（:meth:`~aosr.physics.fem_reduced_build.ReducedModelBuilder.extend`），完整解合計不超過這一次建模的預算；每補一批重核一次
（核對配置的完整解只解一次，之後只重算替身那一側）。全部過了才往下驗、裝，並在格子裡留核對紀錄
（:mod:`aosr.gradient.surrogate_fixed_checks`，網頁預覽與開跑拿它對這一次的配置）；到預算還沒過、或補的樣本殘差都已在
停止值內還是不夠準，就照實講、不放進格子。尺、停止殘差、預設預算一律不動。
驗證或閘門沒過：照實寫 ``result.json``、留在建模資料夾、不放進格子，離開碼 4。被停（SIGTERM／SIGINT）離開碼 3，不接續；
開始換名進格子之後才到的停止只記下、不打斷（見 :class:`_StopGate`）。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING, Final, cast

from aosr import runtime
from aosr.gradient.surrogate_files import (
    EXIT_FAILED, EXIT_INSTALLED, EXIT_NOT_USABLE, EXIT_STOPPED, MODEL_FILE, PROGRESS_FILE, RESULT_FILE, STAGE_DIR,
    TRAINING_FILE, Phase, RoundRecord, SurrogateProgress, SurrogateResult, Verdict,
)

if TYPE_CHECKING:
    from aosr.config.capabilities import CapabilityTable
    from aosr.config.directivity_defaults import DirectivityDefaults
    from aosr.config.fem_reduced import ReducedModelSettings
    from aosr.config.material_gradient import Mount
    from aosr.gradient.fixed_candidates import CandidateVerdict
    from aosr.gradient.surrogate_training import MaterialSet, TrainingSets
    from aosr.materials.search_library import PresetMaterial
    from aosr.physics.fem_reduced import ReducedModel
    from aosr.physics.fem_reduced_build import ReducedModelBuilder
    from aosr.gradient.surrogate_coverage import FixedTarget
    from aosr.materials.search_library import SearchMaterials
    from aosr.config.surrogate_training import SurrogateTrainingSettings
    from aosr.physics.fem_reduced import FullOperators
    from aosr.physics.fem_reduced_boundary import FrequencyBoundaryImpedances
    from aosr.physics.fem_reduced_build import BuildProgress
    from aosr.reporting.scheme import Scheme

SIGNALS: Final = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
# 補訓練最多加幾批新的組（一批補到池裡殘差到停止值才加下一批；預算先用完就先停）。
MAX_EXTENSION_BATCHES: Final = 8
# 補訓練每補幾輪就重核一次開跑那一道固定材料核對（2-8 實量：part2b 池裡殘差卡在 0.0015 降不到停止值，
# 候選卻在第 4 輪就過了；一路補到停止值或上限才核會白解到上限）。
EXTENSION_ROUNDS_PER_CHECK: Final = 1


class _Progress:
    """``progress.json`` 的寫手：每次換一份、同資料夾暫存檔換名（讀的人只看得到完整的舊檔或新檔）。"""

    def __init__(self, path: Path, first: SurrogateProgress) -> None:
        self.path = path
        self.current = first
        self._write()

    def _write(self) -> None:
        from aosr.gradient.run import write_json_atomic
        write_json_atomic(self.path, self.current)

    def update(self, **fields: object) -> None:
        self.current = self.current.model_copy(update={**fields, "updated_at": time.time()})
        self._write()

    def phase(self, phase: Phase, **fields: object) -> None:
        now = time.time()
        self.update(phase=phase, phase_started_at=now, **fields)

    def build_round(self, report: BuildProgress) -> None:
        rounds = self.current.rounds
        if report.latest is not None:
            rounds = (*rounds, RoundRecord(round=report.latest.round, full_solves=report.full_solves, rank=report.rank,
                                           max_residual=report.latest.max_residual, elapsed_s=report.elapsed_s))
        self.update(full_solves=report.full_solves, rank=report.rank, rounds=rounds,
                    snapshot_seconds=report.seconds.get("snapshot"))

    def validation(self, done: int, total: int) -> None:
        self.update(validation_done=done, validation_total=total)


def _presets(text: str) -> tuple[str | None, ...]:
    from aosr.gradient.cli import _presets as parse

    return parse(text)


def _parser() -> argparse.ArgumentParser:
    from aosr.config.material_gradient import MOUNT_RIGID, MOUNTS

    parser = argparse.ArgumentParser(prog="python -m aosr.gradient.surrogate_build", description=__doc__.splitlines()[0])
    parser.add_argument("--scheme", type=Path, required=True, help="方案檔（網頁交的是開跑那一刻的快照）")
    parser.add_argument("--work", type=Path, required=True, help="建模資料夾（進度、結果、先建在這裡的 stage/）")
    parser.add_argument("--slot", type=Path, required=True, help="這份切法的替身格子（建好、核過才換名進來）")
    parser.add_argument("--capabilities", type=Path, required=True, help="能力表（跟搜尋開跑同一份）")
    parser.add_argument("--settings", type=Path, default=None,
                        help="搜尋設定（跟開跑的 --settings 同一份）；不給照產品的 material_gradient.toml")
    parser.add_argument("--materials", type=Path, default=None, metavar="材料庫.json",
                        help="使用者材料庫（跟開跑的 --materials 同一份）：可最佳化材料一筆一個家族一起訓練")
    parser.add_argument("--allowed", action="append", default=[], metavar="面=目標,目標",
                        help="每區准用清單（跟開跑的 --allowed 同一個寫法，可重複）：列到的固定材料候選一起訓練")
    parser.add_argument("--presets", type=_presets, default=None, metavar="a,b,none,…",
                        help="開跑那一組起跑樣板（跟開跑的 --presets 同一個寫法）；准用清單有固定材料時必給，裝進格子前照它核")
    parser.add_argument("--mount", choices=MOUNTS, default=MOUNT_RIGID, help="開跑的裝法（跟開跑的 --mount 同一個預設）")
    parser.add_argument("--fixed-surface", action="append", default=[], metavar="面=材料id",
                        help="開跑的固定面（跟開跑的 --fixed-surface 同一個寫法，可重複）")
    parser.add_argument("--max-depth", action="append", default=[], metavar="面=公尺",
                        help="開跑的深度上限（跟開跑的 --max-depth 同一個寫法，可重複）")
    parser.add_argument("--replace", type=Path, default=None,
                        help="格子裡已經有一份（例如閘門拒收的舊替身）：新的核過之後把舊的整個搬到這裡，再放新的")
    parser.add_argument("--max-full-solves", type=int, default=None,
                        help="這一次建模的完整解預算；不給照 fem_reduced.toml（停止規則不變：殘差達標或預算用完就停）")
    parser.add_argument("--stop-residual", type=float, default=None,
                        help="這一次建模的停止殘差；不給照 fem_reduced.toml（只在這一次生效，照實記進建模帳）")
    parser.add_argument("--workers", type=int, default=None, help="完整解行程數；不給照 surrogate_training.toml")
    parser.add_argument("--dense-threads", type=int, default=None, help="稠密段 BLAS 緒數；不給＝實體核數")
    return parser


def _result(work: Path, result: SurrogateResult) -> None:
    from aosr.gradient.run import write_json_atomic
    write_json_atomic(work / RESULT_FILE, result)


def _rebuild_hint(stop_reason: str) -> str:
    """驗證沒過時重建有沒有用：預算用完才停的，加大預算可能會過；殘差先達標才停的，照同樣設定重建結果一模一樣
    （訓練材料與補點都是固定種子），加預算沒用。"""
    from aosr.physics.fem_reduced_build import STOP_BUDGET, STOP_ROUND_LIMIT

    if stop_reason == STOP_ROUND_LIMIT:
        return ("建模停止原因：補訓練補到開跑那一道固定材料核對過了就停（候選池的殘差還沒到停止值）——"
                "照同樣設定重建會停在同一輪、結果一模一樣（固定種子），加預算沒用；要過得給這一次建模更嚴的停止殘差（stop_residual）")
    if stop_reason == STOP_BUDGET:
        return f"建模停止原因：{stop_reason}——加大完整解預算（max_full_solves）重建可能會過"
    return (f"建模停止原因：{stop_reason}——照同樣設定重建結果一模一樣（固定種子），加預算沒用；"
            "要過得給這一次建模更嚴的停止殘差（stop_residual）")


class _StopGate:
    """停止訊號（SIGTERM／SIGINT／SIGHUP）的處理函式：平常收到就打斷（離開碼 3）；``holding`` 打開之後收到只記下、不打斷，
    而且打開了就不再關——放進格子那一刻起這一次建模就是建好了，晚到的停止不准把它翻成「被停」，也不准把退舊、入新、
    寫判決三步打斷在中間（格子空掉或判決跟格子對不上）。用 Python 層的旗標、不用 ``pthread_sigmask``：完整解行程池、
    BLAS 都開緒，作業系統可能把訊號交給沒擋住的緒；Python 的訊號處理函式則一律在主緒跑，旗標擋得全。"""

    def __init__(self) -> None:
        self.holding = False
        self.deferred: list[int] = []

    def __call__(self, signum: int, frame: FrameType | None) -> None:
        if self.holding:
            self.deferred.append(signum)
            return
        raise KeyboardInterrupt


STOPS: Final = _StopGate()


def _install(stage: Path, slot: Path, replace: Path | None) -> str | None:
    """整個資料夾換名進格子；格子已有東西時只在給了 ``--replace`` 才先把舊的搬走（同一顆磁碟的換名，不複製）。
    新的換不進去（例如跨磁碟換名失敗）就把舊的搬回格子再報原本的錯：格子不會因為一次失敗的安裝變空。
    連搬回都失敗時報一個錯、把兩次失敗與舊的現在在哪裡一起講出來（命令列只印最外層那一個錯，不印例外鏈）。"""
    retired = None
    if slot.exists():
        if replace is None:
            raise FileExistsError(f"替身格子 {slot} 已經有東西；要換掉舊的得給 --replace（舊的搬去哪）")
        replace.parent.mkdir(parents=True, exist_ok=True)
        slot.rename(replace)
        retired = replace
    try:
        slot.parent.mkdir(parents=True, exist_ok=True)
        stage.rename(slot)
    except BaseException as error:
        if retired is not None:
            try:
                retired.rename(slot)
            except OSError as again:
                raise OSError(f"新的換不進格子 {slot}（{type(error).__name__}：{error}），舊的也搬不回去"
                              f"（{type(again).__name__}：{again}）：舊的留在 {retired}，要手動搬回 {slot}") from error
        raise
    return None if retired is None else str(retired)


def _commit(gate: _StopGate, stage: Path, slot: Path, replace: Path | None,
            record: Callable[[str | None], None]) -> None:
    """放進格子＋寫 installed 判決是一段不被停打斷的提交：先關住停止訊號（見 :class:`_StopGate`），再換名、再寫判決。"""
    gate.holding = True
    record(_install(stage, slot, replace))


@dataclass
class _Run:
    """一次建模的輸入與寫手：命令列參數解好、設定讀好、進度檔開好。"""

    work: Path
    stage: Path
    slot: Path
    replace: Path | None
    scheme: Scheme
    sampling: SurrogateTrainingSettings
    settings: ReducedModelSettings
    materials: SearchMaterials
    fixed: tuple[FixedTarget, ...]
    workers: int
    dense_threads: int | None
    progress: _Progress
    start: _Start | None = None


@dataclass(frozen=True)
class _Start:
    """開跑那一組起跑配置（命令列參數原樣解好）與核對用的構造清單（搜尋的構造清單＋固定材料候選），以及建模一開頭量的
    算核對的程式與環境身分（#1142：寫核對紀錄時再量一次，兩次不同＝建模途中程式換過，紀錄不給開跑沿用）。"""

    presets: tuple[str | None, ...]
    mount: Mount
    fixed_surfaces: dict[str, str]
    max_depths: dict[str, float]
    allowed: dict[str, tuple[str, ...]]
    table: tuple[PresetMaterial, ...]
    code_identity: str


@dataclass
class _Prepared:
    """匯出好的完整模型、頻率軸、訓練與留出材料（組與阻抗），以及固定材料核對要用的配置（沒有候選就是空的）。"""

    operators: FullOperators
    axis: tuple[float, ...]
    sets: TrainingSets
    training_sets: list[MaterialSet]
    holdout_sets: list[MaterialSet]
    training: list[FrequencyBoundaryImpedances]
    holdout: list[FrequencyBoundaryImpedances]
    configurations: dict[str, FrequencyBoundaryImpedances] = field(default_factory=dict)
    search: tuple[str, ...] = ()
    expanded: dict[str, tuple[str, ...]] = field(default_factory=dict)


def _open(args: argparse.Namespace) -> _Run:
    from aosr.config.fem_reduced import load_reduced_model_settings
    from aosr.config.paths import config_path
    from aosr.config.material_gradient import load_material_gradient_settings
    from aosr.config.surrogate_training import load_surrogate_training_settings
    from aosr.config.material_gradient import fixed_target_id
    from aosr.gradient.boundary import scheme_patch_keys, surface_ids
    from aosr.gradient.cli import user_entries
    from aosr.gradient.fixed_candidates import check_code_identity
    from aosr.gradient.fixed_surfaces import (
        expanded_allowed_targets, fixed_targets_of, parse_allowed_targets, parse_fixed_surfaces, parse_max_depths,
    )
    from aosr.gradient.run import presets_table
    from aosr.gradient.surrogate_coverage import fixed_targets
    from aosr.materials.search_library import fixed_candidates, user_fixed_entries, with_user_materials
    from aosr.reporting.scheme import load_scheme

    started = time.time()
    work, slot = args.work.resolve(), args.slot.resolve()
    replace = None if args.replace is None else args.replace.resolve()
    stage = work / STAGE_DIR
    if stage.exists() or (work / RESULT_FILE).exists():
        raise FileExistsError(f"建模資料夾 {work} 已經用過（有 {STAGE_DIR}/ 或 {RESULT_FILE}）；一次建模一個新資料夾")
    if slot.exists() and replace is None:
        raise FileExistsError(f"替身格子 {slot} 已經有東西；要換掉舊的得給 --replace")
    work.mkdir(parents=True, exist_ok=True)
    sampling = load_surrogate_training_settings(config_path("surrogate_training.toml"))
    settings = load_reduced_model_settings(config_path("fem_reduced.toml"))
    overrides = {name: value for name, value in (("max_full_solves", args.max_full_solves),
                                                 ("stop_residual", args.stop_residual)) if value is not None}
    if overrides:
        settings = type(settings).model_validate({**settings.model_dump(), **overrides})
    workers = sampling.compute.workers if args.workers is None else args.workers
    settings_path = config_path("material_gradient.toml") if args.settings is None else args.settings.resolve()
    entries = () if args.materials is None else user_entries(args.materials)
    materials = with_user_materials(load_material_gradient_settings(settings_path), presets_table(), entries)
    scheme = load_scheme(args.scheme.resolve())
    allowed = expanded_allowed_targets(parse_allowed_targets(args.allowed), surface_ids(scheme_patch_keys(scheme)))
    targets = fixed_targets_of(allowed)
    candidates = fixed_candidates(targets, materials.presets,
                                  user_fixed_entries([fixed_target_id(target) for target in targets], entries))
    fixed = fixed_targets(allowed, candidates)
    start = None
    if fixed:
        if args.presets is None:
            raise ValueError("准用清單有固定材料時要給開跑那一組起跑配置（--presets，跟開跑同一個寫法）："
                             "裝進格子之前要照它核固定材料")
        # 程式身分在建模之前量（#1142）：數字是這一個行程開頭載入的程式算的，不是建完那一刻磁碟上的（網頁起的建替身
        # 跑主 repo 那份程式，建模途中主線可能拉了新提交）；算不出來也在長計算之前就停。
        start = _Start(presets=args.presets, mount=args.mount, fixed_surfaces=parse_fixed_surfaces(args.fixed_surface),
                       max_depths=parse_max_depths(args.max_depth), allowed=parse_allowed_targets(args.allowed),
                       table=(*materials.presets, *candidates), code_identity=check_code_identity())
    progress = _Progress(work / PROGRESS_FILE, SurrogateProgress(
        phase="preparing", started_at=started, phase_started_at=started, updated_at=started, workers=workers,
        dense_threads=args.dense_threads, patches=len(scheme_patch_keys(scheme)),
        max_full_solves=settings.max_full_solves, stop_residual=settings.stop_residual))
    return _Run(work, stage, slot, replace, scheme, sampling, settings, materials, fixed, workers, args.dense_threads,
                progress, start)


def _prepare(job: _Run, capabilities_path: Path) -> _Prepared:
    """開跑前核對、匯出完整模型、算訓練與留出材料的阻抗，訓練說明寫進 stage/；有固定材料候選時再照開跑那一組起跑配置
    算每個候選的核對配置（:func:`aosr.gradient.cli.start_designs`、:func:`~aosr.gradient.fixed_candidates.candidate_configurations`，
    跟開跑同兩支）。"""
    from aosr.config.capabilities import load_capabilities
    from aosr.config.construction_environment import construction_environment
    from aosr.config.directivity_defaults import load_directivity_defaults
    from aosr.config.frequency_axis import low_frequency_axis_frequencies
    from aosr.config.paths import config_path
    from aosr.gradient import run
    from aosr.gradient.cli import keep_signal_handlers
    from aosr.gradient.endpoint import checked_air
    from aosr.gradient.surrogate_training import TrainingSets
    from aosr.reporting.fem_reduced_scheme import export_operators

    scheme = job.scheme
    if scheme.scene.low_frequency_axis is None:
        raise ValueError("方案場景沒有 low_frequency_axis（低頻軸）：搜尋開跑會拒收，不建替身")
    capabilities = load_capabilities(capabilities_path.resolve())
    directivity = load_directivity_defaults(config_path("directivity_defaults.toml"))
    saved = run.scene_wall_materials(scheme, capabilities=capabilities, directivity=directivity)
    axis = low_frequency_axis_frequencies(scheme.scene.low_frequency_axis)[0]
    materials = job.materials
    run.check_user_families(materials, axis, density_kg_m3=scheme.scene.density_kg_m3,
                            sound_speed_m_s=scheme.scene.sound_speed_m_s)
    configurations, search, expanded = _configurations(job, axis, capabilities, directivity)
    with keep_signal_handlers():
        operators = export_operators(scheme, capabilities=capabilities, directivity=directivity)
    air, ranges = construction_environment(density_kg_m3=scheme.scene.density_kg_m3,
                                           sound_speed_m_s=scheme.scene.sound_speed_m_s)
    checked_air(air, density_kg_m3=operators.density_kg_m3, sound_speed_m_s=operators.sound_speed_m_s, source="完整模型算子")
    users = tuple(entry.id for entry in materials.entries)
    sets = TrainingSets(saved, materials.settings, job.sampling, air=air, ranges=ranges, frequencies_hz=axis,
                        presets=materials.presets if users else None, user_families=users, fixed=job.fixed)
    training_sets, holdout_sets = sets.training_sets(), sets.holdout_sets()
    training = [sets.impedances(item) for item in training_sets]
    holdout = [sets.impedances(item) for item in holdout_sets]
    job.stage.mkdir(parents=True)
    _write_training(job, sets, training_sets, holdout_sets)
    return _Prepared(operators, axis, sets, training_sets, holdout_sets, training, holdout, configurations, search, expanded)


def _write_training(job: _Run, sets: TrainingSets, training_sets: Sequence[MaterialSet],
                    holdout_sets: Sequence[MaterialSet]) -> None:
    """訓練說明寫進 stage/（補訓練之後同資料夾暫存檔換名整份換一份，補的那幾組記在訓練那一節後面）。"""
    from aosr.gradient.surrogate_coverage import coverage_for

    document = sets.document(training_sets, holdout_sets, coverage_for(job.materials, job.fixed))
    path = job.stage / TRAINING_FILE
    text = json.dumps(document, ensure_ascii=False, allow_nan=False) + "\n"
    if not path.exists():
        with path.open("x", encoding="utf-8") as handle:
            handle.write(text)
        return
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _configurations(job: _Run, axis: Sequence[float], capabilities: CapabilityTable, directivity: DirectivityDefaults
                    ) -> tuple[dict[str, FrequencyBoundaryImpedances], tuple[str, ...], dict[str, tuple[str, ...]]]:
    """每個固定材料候選在開跑那一組起跑配置的核對配置、開跑時交給搜尋的片、展開到面片的准用清單；沒有候選全是空的。
    起跑配置開跑會拒收的（准用清單跟起跑樣板不合、深度放不下），這裡也先拒收。"""
    from aosr.gradient.cli import start_designs
    from aosr.gradient.fixed_candidates import candidate_configurations
    from aosr.gradient.fixed_surfaces import expanded_fixed_surfaces

    start = job.start
    if start is None:
        return {}, (), {}
    scheme = job.scheme
    designs, expanded = start_designs(scheme, job.materials.settings, start.table, start.presets, mount=start.mount,
                                      fixed=start.fixed_surfaces, caps=start.max_depths, allowed=start.allowed,
                                      capabilities=capabilities, directivity=directivity)
    names = [design.surface_id for design in designs]
    frozen = expanded_fixed_surfaces(start.fixed_surfaces, names) if start.fixed_surfaces else {}
    configurations = candidate_configurations(designs, expanded, start.table, axis,
                                              density_kg_m3=scheme.scene.density_kg_m3,
                                              sound_speed_m_s=scheme.scene.sound_speed_m_s)
    return configurations, tuple(name for name in names if name not in frozen), expanded


def same_configuration(left: FrequencyBoundaryImpedances, right: FrequencyBoundaryImpedances) -> bool:
    """兩組材料是不是同一組（每片、每個頻點的阻抗相對差都在 1e-9 內；剛性對剛性）：補訓練的組跟核對那一組相同就剔除——
    核對那一組要留著當留出，不准進訓練。訓練鏈與搜尋鏈算同一份材料可能差在最後幾位，所以不比逐位。"""
    import numpy as np

    first_rows, second_rows = cast(Mapping[object, object], left), cast(Mapping[object, object], right)
    if set(first_rows) != set(second_rows):
        return False
    for key in first_rows:
        a, b = first_rows[key], second_rows[key]
        if a is None or b is None:
            if a is not b:
                return False
            continue
        first, second = np.asarray(a, dtype=np.complex128), np.asarray(b, dtype=np.complex128)
        if first.shape != second.shape or not np.all(np.abs(first - second) <= 1e-9 * np.abs(second)):
            return False
    return True


@dataclass
class _Extension:
    """補訓練的結果：補完的替身、每個候選最後一次判決、補進去的組、停在哪裡。"""

    model: ReducedModel
    verdicts: dict[str, CandidateVerdict]
    added: list[MaterialSet]
    batches: int
    stalled: bool


def _record_check(job: _Run, check: int, model: ReducedModel, verdicts: Mapping[str, CandidateVerdict]) -> None:
    """這一次核的每個候選一列寫進 ``fixed_check_rounds.jsonl``、同一句印進建模日誌（#1128：只存最後一次，中間幾輪
    過不過、差多少事後查不到）。不改判決。"""
    from aosr.gradient.surrogate_files import FIXED_ROUNDS_FILE, FixedCheckRound, fixed_round_lines

    def finite(value: float) -> float | None:
        return value if math.isfinite(value) else None

    try:
        rows = [FixedCheckRound(check=check, full_solves=model.ledger.full_solves, rank=model.ledger.rank, target=target,
                                frequency_db=finite(verdict.frequency_db), band_db=finite(verdict.band_db),
                                worst_hz=None if verdict.passed or not math.isfinite(verdict.worst_hz) else verdict.worst_hz,
                                passed=verdict.passed, seconds=max(0.0, verdict.seconds))
                for target, verdict in sorted(verdicts.items())]
        with (job.work / FIXED_ROUNDS_FILE).open("a", encoding="utf-8") as stream:
            for row in rows:
                stream.write(row.model_dump_json() + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    except (OSError, ValueError) as error:
        # 紀錄只給人看：寫不進去照講一句、建模照走，不讓它改掉判決之後的路（審查必修）。
        sys.stderr.write(f"固定材料核對第 {check} 次的紀錄沒寫進去（判決照常）：{error}\n")
        return
    for line in fixed_round_lines(rows):
        sys.stderr.write(line + "\n")


def _fixed_checked(job: _Run, builder: ReducedModelBuilder, model: ReducedModel, prepared: _Prepared) -> _Extension:
    """開跑那一道固定材料核對（全部候選）；有沒過的就補訓練，每補一輪（一次補點＋重算基底）就重核一次、全過就停：
    候選池是補進來的那幾批，池裡的殘差還沒到停止值就同一池接著補；到了停止值（或池用完）才再加一批新的組。
    停在：全過、預算用完、加了一批新的組卻一點都沒補進（``stalled``：補的樣本殘差都已在停止值內，再補同類也沒用）、
    或已加滿 :data:`MAX_EXTENSION_BATCHES` 批。核對那一組本身不進訓練。"""
    from aosr.gradient.fixed_candidates import judge_candidates
    from aosr.physics.fem_reduced_build import STOP_ROUND_LIMIT

    axis = prepared.axis
    total = len(prepared.configurations) * len(axis)
    job.progress.phase("fixed_checking", validation_done=0, validation_total=total)
    verdicts = judge_candidates(model, prepared.operators, prepared.configurations, prepared.expanded, workers=job.workers,
                                progress=job.progress.validation)
    job.progress.validation(total, total)
    checks = 1
    _record_check(job, checks, model, verdicts)
    added: list[MaterialSet] = []
    batch, stalled, continuing = 0, False, False
    assert job.start is not None
    while (failing := {target for target, verdict in verdicts.items() if not verdict.passed}) \
            and builder.full_solves < job.settings.max_full_solves:
        fresh: list[tuple[MaterialSet, FrequencyBoundaryImpedances]] = []
        if not continuing:
            if batch >= MAX_EXTENSION_BATCHES:
                break
            for item in job.fixed:
                if item.target not in failing:
                    continue
                for material_set in prepared.sets.extension_sets(item, search=prepared.search, mount=job.start.mount,
                                                                 batch=batch):
                    impedances = prepared.sets.impedances(material_set)
                    if not any(same_configuration(impedances, held) for held in prepared.configurations.values()):
                        fresh.append((material_set, impedances))
            batch += 1
            if not fresh:
                continue
        before = builder.full_solves
        job.progress.phase("building", stop_reason=None)
        model = builder.extend([impedances for _, impedances in fresh], max_rounds=EXTENSION_ROUNDS_PER_CHECK)
        added.extend(material_set for material_set, _ in fresh)
        if builder.full_solves == before:
            # 這一池補不進點：新加的一批一點都沒補進就是補不動；同一池接著補卻補不進，就換一批新的組。
            if fresh:
                stalled = True
                break
            continuing = False
            continue
        job.progress.phase("fixed_checking", validation_done=0, validation_total=total)
        verdicts = judge_candidates(model, prepared.operators, prepared.configurations, prepared.expanded,
                                    workers=job.workers, previous=verdicts, progress=job.progress.validation)
        checks += 1
        _record_check(job, checks, model, verdicts)
        continuing = builder.stop_reason == STOP_ROUND_LIMIT
    return _Extension(model, verdicts, added, batch, stalled)


def _extension_failure(job: _Run, extension: _Extension) -> str:
    """補訓練完還有沒過的候選：白話講哪一個、在哪個頻率差多少、補到哪裡停、接下來能怎麼辦（尺與停止殘差不動）。"""
    from aosr.gradient.fixed_candidates import inaccurate_text

    words = "；".join(inaccurate_text(verdict, trained=True) for verdict in extension.verdicts.values() if not verdict.passed)
    solves, cap = extension.model.ledger.full_solves, job.settings.max_full_solves
    if extension.stalled:
        where = (f"補訓練了 {len(extension.added)} 組（完整解合計 {solves} 次），補的樣本殘差都已在停止值 "
                 f"{job.settings.stop_residual:g} 內，再補同類樣本也沒用")
        hint = "可以在建替身那一格把停止殘差調小再重建試試（兩把尺不動）"
    elif solves >= cap:
        where = f"補訓練到完整解上限 {cap} 次還不夠準"
        hint = "加大完整解上限（max_full_solves）重建可能會過"
    else:
        where = (f"補訓練了 {extension.batches} 批（每次建模最多 {MAX_EXTENSION_BATCHES} 批，完整解合計 {solves} 次、"
                 f"上限 {cap} 次還沒用完）還不夠準")
        hint = ("預算不是卡住的原因；同一組配置與設定原樣重建會抽到同一批樣本、結果一樣，"
                "要換結果得在建替身那一格把停止殘差調小（兩把尺不動）")
    return (f"開跑前那一道固定材料核對沒過：{words}。{where}，這份替身沒有放進格子（裝進去開跑也會被拒）。{hint}")


def _train_fixed(job: _Run, builder: ReducedModelBuilder, model: ReducedModel, prepared: _Prepared) -> _Extension | None:
    """有固定材料候選才做：核、不夠準就補訓練（:func:`_fixed_checked`）；補了就把補的組寫進訓練說明。"""
    if not prepared.configurations:
        return None
    extension = _fixed_checked(job, builder, model, prepared)
    if extension.added:
        _write_training(job, prepared.sets, [*prepared.training_sets, *extension.added], prepared.holdout_sets)
        job.progress.update(training_sets=len(prepared.training) + len(extension.added))
    return extension


def _refuse_fixed(job: _Run, extension: _Extension) -> None:
    """補訓練完還有候選沒過：照實寫判決（數字抄差最多的那一個候選的報告），不放進格子。"""
    from aosr.gradient.run import holdout_rulers

    ledger = extension.model.ledger
    worst = max((verdict for verdict in extension.verdicts.values() if not verdict.passed),
                key=lambda verdict: verdict.frequency_db)
    rulers = holdout_rulers()
    _result(job.work, SurrogateResult(
        verdict="not_validated", message=_extension_failure(job, extension), stop_reason=ledger.stop_reason,
        full_solves=ledger.full_solves, rank=ledger.rank, build_seconds=ledger.seconds["total"],
        frequency_db_by_seat=dict(worst.report.frequency_db_by_seat), band_db_by_seat=dict(worst.report.band_db_by_seat),
        rulers={"frequency": rulers.frequency.value, "band": rulers.band.value}, within_contract=False,
        finished_at=time.time()))


def _fixed_record(job: _Run, npz: Path, verdicts: Mapping[str, CandidateVerdict], prepared: _Prepared,
                  frequencies_hz: Sequence[float]) -> None:
    """核對紀錄寫進 stage/（跟替身一起換名進格子）：哪一組起跑配置、哪一份替身、每個候選的數字；v2（#1142）再記開跑沿用
    要比的身分：頻率軸（裝進去那份替身自己的）、完整模型指紋、算核對的程式身分（建模一開頭量的那一個）、每個候選准用的片、
    材料身分摘要、完整解次數、核對配置的阻抗摘要。寫之前再量一次程式身分：跟開頭不同（建模途中程式或環境換過，數字是
    哪一份程式算的說不準）就寫成舊格式 v1——網頁預覽照讀，開跑那一道照舊重核。"""
    from aosr.gradient.fixed_candidates import candidate_material_digest, check_code_identity, holdout_ruler_values
    from aosr.physics.fem_reduced import operators_digest
    from aosr.gradient.surrogate_fixed_checks import (
        FIXED_CHECKS_FILE, FIXED_CHECKS_VERSION_V2, FixedCheckRow, FixedChecksRecord, axis_hex, configuration_document,
        configuration_sha256, impedances_sha256, manifest_digest, write_record,
    )
    from aosr.physics.fem_reduced_store import manifest_path

    start = job.start
    assert start is not None
    configuration = configuration_document(presets=start.presets, mount=start.mount, fixed_surfaces=start.fixed_surfaces,
                                           max_depths=start.max_depths, allowed=start.allowed,
                                           scene_json=job.scheme.scene.model_dump_json())
    frequency_ruler, band_ruler = holdout_ruler_values()
    materials = {item.target: item.coverage().material_sha256 for item in job.fixed}
    record = FixedChecksRecord(
        schema_version=FIXED_CHECKS_VERSION_V2,
        configuration_sha256=configuration_sha256(configuration), configuration=configuration,
        manifest_sha256=manifest_digest(manifest_path(npz)), frequencies_hz=axis_hex(frequencies_hz),
        operators_digest=operators_digest(prepared.operators), code_identity=start.code_identity,
        checks=tuple(FixedCheckRow(target=target, material_sha256=materials[target], frequency_db=verdict.frequency_db,
                                   band_db=verdict.band_db, frequency_ruler=frequency_ruler, band_ruler=band_ruler,
                                   passed=verdict.passed, surfaces=verdict.surfaces,
                                   material_digest=candidate_material_digest(target, start.table),
                                   full_solves=verdict.report.full_solves,
                                   impedances_sha256=impedances_sha256(prepared.configurations[target]))
                     for target, verdict in sorted(verdicts.items())))
    if check_code_identity() != start.code_identity:
        sys.stderr.write("注意：建替身途中算核對的程式或環境換過（例如主線拉了新提交），固定材料核對紀錄寫成舊格式："
                         "網頁預覽照舊，開跑那一道會照舊重核、不沿用這次的數字\n")
        record = record.without_reuse_identity()
    write_record(job.stage / FIXED_CHECKS_FILE, record)


def _built(job: _Run, prepared: _Prepared) -> tuple[ReducedModel, _Extension | None] | None:
    """建模；有固定材料候選就接著核、不夠準就補訓練（:func:`_train_fixed`）。補完還有候選沒過就寫判決、回 None（不裝）；
    否則回替身與補訓練紀錄（沒有候選是 None）。快照與基底在這裡放掉。"""
    from aosr.physics.fem_reduced_build import ReducedModelBuilder

    operators, axis = prepared.operators, prepared.axis
    job.progress.phase("building", dofs=int(operators.p2.basis.N), axis_points=len(axis), training_sets=len(prepared.training),
                       holdout_sets=len(prepared.holdout))
    builder = ReducedModelBuilder(operators, axis, prepared.training, settings=job.settings, workers=job.workers,
                                  dense_threads=job.dense_threads, progress=job.progress.build_round)
    model = builder.build()
    extension = _train_fixed(job, builder, model, prepared)
    builder.release()
    if extension is None:
        return model, None
    if any(not verdict.passed for verdict in extension.verdicts.values()):
        _refuse_fixed(job, extension)
        return None
    return extension.model, extension


def build(args: argparse.Namespace) -> int:
    from aosr.gradient import run
    from aosr.physics.fem_reduced_check import points_hex, validate_reduced, within_reduced_contract
    from aosr.physics.fem_reduced_receipt import save_validation_receipt
    from aosr.gradient.surrogate_coverage import check_covers
    from aosr.physics.fem_reduced_store import save_operators_stamp, save_reduced

    job = _open(args)
    prepared = _prepare(job, args.capabilities)
    operators, axis, holdout = prepared.operators, prepared.axis, prepared.holdout
    built = _built(job, prepared)
    if built is None:
        return EXIT_NOT_USABLE
    model, extension = built
    ledger = model.ledger
    npz = job.stage / MODEL_FILE
    save_reduced(model, npz)
    job.progress.phase("validating", stop_reason=ledger.stop_reason, build_seconds=ledger.seconds["total"],
                       full_solves=ledger.full_solves, rank=ledger.rank, validation_done=0,
                       validation_total=len(holdout) * len(axis))
    rulers = run.holdout_rulers()
    clock = time.perf_counter()
    report = validate_reduced(model, operators, holdout, workers=job.workers, dense_threads=job.dense_threads,
                              progress=job.progress.validation)
    seconds = time.perf_counter() - clock
    save_validation_receipt(npz, report, rulers=rulers, seconds=seconds)
    within = within_reduced_contract(report, frequency_db=rulers.frequency.value, band_db=rulers.band.value)
    extended = ("" if extension is None or not extension.added else
                f"；開跑前那一道固定材料核對先沒過，補訓練 {len(extension.added)} 組之後過了（完整解合計 {ledger.full_solves} 次）")

    def verdict(kind: Verdict, message: str, *, slot_text: str | None = None, retired: str | None = None) -> None:
        _result(job.work, SurrogateResult(
            verdict=kind, message=message, stop_reason=ledger.stop_reason, full_solves=ledger.full_solves,
            rank=ledger.rank, build_seconds=ledger.seconds["total"], validation_seconds=seconds,
            frequency_db_by_seat=dict(report.frequency_db_by_seat), band_db_by_seat=dict(report.band_db_by_seat),
            rulers={"frequency": rulers.frequency.value, "band": rulers.band.value}, within_contract=within,
            slot=slot_text, retired=retired, finished_at=time.time()))

    if not within:
        verdict("not_validated", f"留出驗證沒過（逐頻最大差 {max(report.frequency_db_by_seat.values()):.4g} dB，尺 "
                f"{rulers.frequency.value:g} dB；帶能量最大差 {max(report.band_db_by_seat.values()):.4g} dB，尺 "
                f"{rulers.band.value:g} dB）：這份替身不能用，沒有放進格子。{_rebuild_hint(ledger.stop_reason)}")
        return EXIT_NOT_USABLE
    job.progress.phase("checking")
    save_operators_stamp(npz, operators, rulers=rulers)
    points = run.SchemePoints(points_hex(operators.source_points), points_hex(operators.receiver_points))
    del model
    try:
        gated = run.gated_surrogate(npz, mesh_hash=operators.mesh_hash, physics_identity=operators.physics_identity,
                                    points=points)
        run.check_surrogate_matches_scheme(gated, job.scheme, points)
        run.check_surrogate_axis_covers_scoring(gated, job.scheme,
                                                run.search_objective_settings(job.scheme.purpose).stage_one)
        check_covers(job.stage / TRAINING_FILE, job.materials, job.fixed)
    except ValueError as error:
        verdict("gate_refused", f"搜尋開跑那一道准用閘門不收（{type(error).__name__}：{error}），沒有放進格子")
        return EXIT_NOT_USABLE
    if extension is not None:
        _fixed_record(job, npz, extension.verdicts, prepared, gated.frequencies_hz)
    job.progress.phase("installing")
    _commit(STOPS, job.stage, job.slot, job.replace, lambda retired: verdict(
        "installed", "建好、留出驗證兩把尺都過、搜尋的准用閘門核過，已放進這份切法的格子" + extended,
        slot_text=str(job.slot), retired=retired))
    return EXIT_INSTALLED


def main(argv: Sequence[str] | None = None) -> int:
    """參數錯由 argparse 以離開碼 2 結束；放進格子 0；留出驗證或閘門不過 4；被停 3；其餘錯誤原文寫標準錯誤、回 1。"""
    args = _parser().parse_args(None if argv is None else list(argv))
    runtime.configure_calculation_jax()
    try:
        return build(args)
    except KeyboardInterrupt:
        sys.stderr.write("建替身被停止（不接續，要重來）\n")
        return EXIT_STOPPED
    except Exception as error:
        sys.stderr.write(f"建替身失敗：{type(error).__name__}：{error}\n")
        return EXIT_FAILED


if __name__ == "__main__":
    for number in SIGNALS:
        signal.signal(number, STOPS)
    raise SystemExit(main())
