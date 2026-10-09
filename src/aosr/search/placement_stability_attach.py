"""搜尋最後的擺位附件；整批真算、細算身分評分、每點原子記錄，不改兩本帳。"""
from __future__ import annotations

import hashlib
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from uuid import uuid4

from aosr.config.capabilities import load_capabilities
from aosr.config.directivity_defaults import load_directivity_defaults
from aosr.config.paths import config_path
from aosr.config.precision_contracts import default_precision_contracts_path, furniture_contact_rel
from aosr.config.quality_targets import QualityTargets, load_quality_targets
from aosr.reporting.result import ResultOrigin, SchemeResult
from aosr.reporting.scheme import Scheme, load_scheme
from aosr.scoring.ranking import ComparisonIdentity
from aosr.search import crossover_record
from aosr.search.outer_status import attachment_skip_reason, snapshot_of
from aosr.search.placement_stability import Outcome, PointOutcome, report_arithmetic, select_finalists
from aosr.search.placement_stability_events import baseline_boundary_distances, furniture_events
from aosr.search.placement_stability_geometry import check_shift, generate_shifts
from aosr.search.placement_stability_record import (
    BATCH_ABORTED, DIRECTORY, IDENTITY_SKIP, STOPPED_REASON, FinalistBoundaries, IdentityStamp,
    PointRecord, StabilitySummary, attachment_path, is_stale, read_summary, scheme_hash, summary_path, write_summary,
)
from aosr.search.refine import RefineLedger, RefineRow
from aosr.search.refine_run import _Refiner, header_for
from aosr.search.report_comparison import read_refinement_rows
from aosr.search.run import CandidateJob, Compute, ComputedCandidate, IdentityChanged, IdentityProbe, SearchStatus, _check_computed, _check_identity
from aosr.search.sampler import Excluded, RankingZone, Scored
from aosr.search.scoring import screening_outcome
from aosr.search.store import SearchIdentity, SearchStore
from aosr.search.timings import WallClock

ComputeFactory = Callable[[Path], Compute]
OWNED_POINT = re.compile(r"point-[0-9a-f]{32}(?:-scheme)?\.(?:json|stderr)\Z")
OWNED_FEM = re.compile(r"fem-[0-9a-f]{32}\Z")
# scheme_cli 的 NamedTemporaryFile 使用預設前綴 tmp、空後綴與八位隨機名。
OWNED_TEMP = re.compile(r"(?:summary-write-.+\.tmp|tmp[a-z0-9_]{8})\Z")


def _previous(folder: Path) -> StabilitySummary | None:
    try:
        return read_summary(folder)
    except (OSError, ValueError):
        return None


def _empty(store: SearchStore, status: SearchStatus) -> StabilitySummary:
    return StabilitySummary(search_id=store.search_id, conclusion=status.outer.conclusion,
                            snapshot=snapshot_of(status), identity=IdentityStamp.of(store.identity))


def _crossover(store: SearchStore, status: SearchStatus) -> tuple[crossover_record.CrossoverSummary | str, str]:
    try:
        summary = crossover_record.read_summary(store.path)
    except (OSError, ValueError) as error:
        reason = str(error).replace("\r", " ").replace("\n", "；")
        try:
            stamp = hashlib.sha256(crossover_record.summary_path(store.path).read_bytes()).hexdigest()
        except OSError:
            stamp = f"unreadable:{reason}"
        return f"交接摘要讀不到：{reason}", stamp
    if summary is None:
        return "交接摘要讀不到：沒有摘要", "missing"
    stamp = hashlib.sha256(summary.model_dump_json().encode()).hexdigest()
    try:
        stale = crossover_record.is_stale(summary, crossover_record.fresh_summary(store, status))
    except (OSError, ValueError) as error:
        reason = str(error).replace("\r", " ").replace("\n", "；")
        return f"交接摘要無法判舊：{reason}", stamp
    if stale:
        return "交接摘要舊了", stamp
    if summary.state in ("failed", "stopped", "skipped"):
        return f"交接摘要不能用：{summary.reason_text or summary.state}", stamp
    return summary, stamp


def fresh_summary(store: SearchStore, status: SearchStatus) -> StabilitySummary:
    """唯讀判舊只讀帳與交接摘要，不讀完整結果、不重評。"""
    rows = read_refinement_rows(store)
    stamps = crossover_record.row_stamps(rows)
    crossing, stamp = _crossover(store, status)
    return _empty(store, status).model_copy(update={"rows": stamps,
        "rows_fingerprint": crossover_record.rows_fingerprint(stamps), "crossover_stamp": stamp,
        "selection": select_finalists(rows, crossing)})


def _skip_summary(store: SearchStore, status: SearchStatus) -> StabilitySummary:
    """跳過不計算，能讀到的帳與交接戳記仍各自留下。"""
    summary = _empty(store, status)
    try:
        if store.refine_ledger_path.is_file():
            rows = read_refinement_rows(store)
            stamps = crossover_record.row_stamps(rows)
            summary = summary.model_copy(update={"rows": stamps, "rows_fingerprint": crossover_record.rows_fingerprint(stamps)})
    except (OSError, ValueError):
        pass
    try:
        crossing = crossover_record.read_summary(store.path)
        if crossing is not None:
            summary = summary.model_copy(update={"crossover_stamp": hashlib.sha256(crossing.model_dump_json().encode()).hexdigest()})
    except (OSError, ValueError):
        pass
    return summary


def _kept_points(summary: StabilitySummary, previous: StabilitySummary | None) -> StabilitySummary:
    """磁碟缺的點從上一代帶下去；恢復時仍由 _reuse 逐點重新查結果。"""
    keys = {(p.trial_number, p.shift_name) for p in summary.points}
    points = summary.points + tuple(p for p in previous.points if (p.trial_number, p.shift_name) not in keys) if previous else summary.points
    return summary.model_copy(update={"points": points, "arithmetic": None, "boundaries": (),
        "computed_points": sum(p.result_file is not None and p.outcome is not None for p in points),
        "total_points": sum(p.scheme_hash is not None for p in points)})


def _read_refined(store: SearchStore, number: int | None) -> tuple[Scheme, SchemeResult]:
    path = store.refine_result_path(number)
    scheme = load_scheme(store.scheme_path_for(path))
    result = SchemeResult.model_validate_json(path.read_bytes())
    _check_identity(SearchIdentity(result.physics_identity, result.program_fingerprint, result.purpose_settings),
                    store.identity, "細算結果")
    expected = ResultOrigin(kind="search_baseline" if number is None else "search_candidate",
                            search_id=store.search_id, trial_number=number)
    if result.origin != expected or result.scheme != scheme:
        raise ValueError("細算結果與出處或細算方案不同")
    return scheme, result


def _pinned(store: SearchStore, status: SearchStatus, rows: tuple[RefineRow, ...],
            registry: QualityTargets, run_date: date) -> tuple[ComparisonIdentity, ...]:
    """沿細算帳次序，以細算本身的 read_saved／pin 重建釘身分，包含被淘汰的錨點。"""
    # 計算不會在讀取與 pin 路徑被使用。
    header, _ = RefineLedger.read(store.refine_ledger_path)
    if header != header_for(store):
        raise ValueError("細算帳身分與搜尋快照不同")
    runner = _Refiner(store, lambda jobs, workers: iter(()), lambda: store.identity, registry,
                      run_date, store.identity.program_fingerprint, status, WallClock())
    runner.check_baseline()
    for row in rows:
        candidate = runner.read_saved(row.trial_number)
        if candidate is None:
            raise ValueError("細算釘身分的結果讀不回")
        scheme, _ = _read_refined(store, row.trial_number)
        runner.pin(candidate, scheme)
        if runner.pinned is not None:
            return runner.pinned
    raise ValueError("細算沒有定得出的比較身分")


def _read_shift(store: SearchStore, point: PointRecord) -> SchemeResult:
    if point.result_file is None:
        raise ValueError("移位結果尚未保存")
    result = SchemeResult.model_validate_json(attachment_path(store.path, point.result_file).read_bytes())
    expected = ResultOrigin.model_validate(dict(kind="search_placement_shift", search_id=store.search_id,
        trial_number=point.trial_number, shift_name=point.shift_name))
    _check_identity(SearchIdentity(result.physics_identity, result.program_fingerprint, result.purpose_settings),
                    store.identity, "移位結果")
    if result.origin != expected or scheme_hash(result.scheme) != point.scheme_hash:
        raise ValueError("移位結果出處或方案內容不同")
    return result


def _reuse(store: SearchStore, previous: StabilitySummary | None, point: PointRecord) -> tuple[PointRecord, SchemeResult] | None:
    if previous is None or previous.identity != IdentityStamp.of(store.identity):
        return None
    key = point.trial_number, point.shift_name, point.scheme_hash
    for saved in previous.points:
        if (saved.trial_number, saved.shift_name, saved.scheme_hash) != key or saved.outcome is None:
            continue
        try:
            return saved, _read_shift(store, saved)
        except (OSError, ValueError, IdentityChanged):
            pass
    return None


def _referenced(summary: StabilitySummary | None) -> set[str]:
    paths: set[str] = set()
    if summary is not None:
        for point in summary.points:
            if point.result_file is not None:
                result = Path(point.result_file)
                paths.update((str(result), str(SearchStore.scheme_path_for(result)), str(SearchStore.stderr_path_for(result))))
            if point.fem_root is not None:
                paths.add(point.fem_root)
    return paths


def _finish(folder: Path, summary: StabilitySummary, previous: StabilitySummary | None) -> None:
    """完成才清本附件自有檔，保留新摘要與前一代所指結果、方案、錯誤與分片。"""
    write_summary(folder, summary)
    root = summary_path(folder).parent
    if not summary.completed or root.is_symlink():
        return
    kept = _referenced(summary) | _referenced(previous)
    for path in root.iterdir():
        if path.is_symlink() or str(path.relative_to(folder)) in kept:
            continue
        if (OWNED_POINT.fullmatch(path.name) or OWNED_TEMP.fullmatch(path.name)) and path.is_file():
            path.unlink(missing_ok=True)
        elif OWNED_FEM.fullmatch(path.name) and path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)


@dataclass
class _Attacher:
    store: SearchStore
    status: SearchStatus
    summary: StabilitySummary
    previous: StabilitySummary | None
    registry: QualityTargets
    run_date: date
    pinned: tuple[ComparisonIdentity, ...]
    contact_rel: float
    points: list[PointRecord] = field(default_factory=list)
    jobs: list[CandidateJob] = field(default_factory=list)
    baselines: dict[int | None, SchemeResult] = field(default_factory=dict)

    def save(self) -> None:
        self.summary = self.summary.model_copy(update={"points": tuple(self.points),
            "computed_points": sum(p.result_file is not None and p.outcome is not None for p in self.points)})
        write_summary(self.store.path, self.summary)

    def scored(self, point: PointRecord, result: SchemeResult) -> PointRecord:
        outcome, _ = screening_outcome(result.candidate, result.scheme, registry=self.registry,
            run_date=self.run_date, engine_version=self.store.identity.program_fingerprint, pinned=self.pinned)
        labels: dict[RankingZone, Outcome] = {RankingZone.ELIMINATED: "excluded",
            RankingZone.UNASSESSED: "not_evaluated", RankingZone.INCOMPARABLE: "not_comparable"}
        label: Outcome = "scored" if isinstance(outcome, Scored) else labels[outcome.zone] if isinstance(outcome, Excluded) else "not_evaluated"
        events = furniture_events(self.baselines[point.trial_number], result)
        return point.model_copy(update={"outcome": label, "total_cost": outcome.value if isinstance(outcome, Scored) else None,
            "model_discontinuity": bool(events), "furniture_events": events, "reason_text": ""})

    def prepare(self, fem_root: Path, capabilities_path: Path) -> None:
        capabilities = load_capabilities(capabilities_path)
        directivity = load_directivity_defaults(config_path("directivity_defaults.toml"))
        for finalist in self.summary.selection.finalists:
            scheme, base = _read_refined(self.store, finalist.trial_number)
            self.baselines[finalist.trial_number] = base
            for shift in generate_shifts(self.store.project, self.store.settings, scheme):
                legal = check_shift(self.store.project, shift, contact_rel=self.contact_rel,
                                    capabilities=capabilities, directivity=directivity)
                point = PointRecord.model_validate(dict(trial_number=finalist.trial_number, shift_name=shift.name,
                    outcome=None if legal.outcome == "ready" else legal.outcome, violations=legal.violations,
                    problems=legal.problems, out_of_spec=bool(legal.out_of_spec), spec_violations=legal.out_of_spec,
                    outside_search=bool(legal.outside_search), outside_search_quantities=legal.outside_search,
                    search_range_not_checked=legal.search_range_not_checked))
                if legal.outcome == "ready":
                    point = self.prepare_job(point, shift.scheme, fem_root)
                self.points.append(point)
        self.summary = self.summary.model_copy(update={"total_points": sum(p.scheme_hash is not None for p in self.points)})
        self.save()

    def prepare_job(self, point: PointRecord, scheme: Scheme, fem_root: Path) -> PointRecord:
        point = point.model_copy(update={"scheme_hash": scheme_hash(scheme)})
        reused = _reuse(self.store, self.previous, point)
        if reused is not None:
            saved, result = reused
            return self.scored(point.model_copy(update={"result_file": saved.result_file, "fem_root": saved.fem_root}), result)
        root = summary_path(self.store.path).parent
        path = root / f"point-{uuid4().hex}.json"
        self.jobs.append(CandidateJob(point.trial_number, scheme, path, point.shift_name))
        # 派出前先存計算目的地，失敗點仍只有原因、沒有假分数。
        return point.model_copy(update={"result_file": str(path.relative_to(self.store.path)),
                                       "fem_root": str(fem_root.relative_to(self.store.path))})

    def compute(self, factory: ComputeFactory, fem_root: Path) -> None:
        if not self.jobs:
            return
        expected = {(j.trial_number, j.shift_name): j for j in self.jobs}
        seen: set[tuple[int | None, str | None]] = set()
        iterator = factory(fem_root)(tuple(self.jobs), self.store.settings.max_workers)
        try:
            for computed in iterator:
                key = computed.job.trial_number, computed.job.shift_name
                if key not in expected or key in seen:
                    raise ValueError("計算交回未要求或重複的移位點")
                _check_computed(computed, expected[key], self.store.identity)
                self.receive(computed)
                seen.add(key)
            if seen != expected.keys():
                raise ValueError("計算沒有交回整批移位點")
        finally:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()

    def receive(self, computed: ComputedCandidate) -> None:
        index = next(i for i, p in enumerate(self.points)
                     if (p.trial_number, p.shift_name) == (computed.job.trial_number, computed.job.shift_name))
        result = _read_shift(self.store, self.points[index])
        if result.candidate != computed.candidate or result.scheme != computed.job.scheme:
            raise ValueError("保存的移位結果與計算交回的內容不同")
        self.points[index] = self.scored(self.points[index], result)
        self.save()

    def finish(self) -> StabilitySummary:
        outcomes: dict[int | None, dict[str, PointOutcome]] = {f.trial_number: {p.shift_name: p.as_outcome() for p in self.points if p.trial_number == f.trial_number}
                    for f in self.summary.selection.finalists}
        arithmetic = report_arithmetic(self.summary.selection.finalists, outcomes)
        boundaries = tuple(FinalistBoundaries(trial_number=f.trial_number, distances=baseline_boundary_distances(
            self.baselines[f.trial_number], purpose=self.registry.purpose(self.store.project.purpose), contact_rel=self.contact_rel))
            for f in self.summary.selection.finalists)
        summary = self.summary.model_copy(update={"state": "done", "completed": True,
            "arithmetic": arithmetic, "boundaries": boundaries})
        _finish(self.store.path, summary, self.previous)
        return summary


def record_stability_error(store: SearchStore, status: SearchStatus, error: BaseException, *,
                           previous: StabilitySummary | None = None) -> None:
    """只收這一段；已交回的點保持，未交回的點記同批中止，人手再跑只補缺的。"""
    summary = _previous(store.path) or _empty(store, status)
    if summary.state == "done":
        try:
            if not is_stale(summary, fresh_summary(store, status)):
                return
        except (OSError, ValueError):
            pass
    summary = _kept_points(summary, previous)
    stopped = isinstance(error, KeyboardInterrupt)
    points = tuple(p.model_copy(update={"reason_text": BATCH_ABORTED}) if p.outcome is None else p for p in summary.points)
    write_summary(store.path, summary.model_copy(update={"state": "stopped" if stopped else "failed",
        "completed": False, "reason_text": STOPPED_REASON if stopped else str(error), "points": points}))


def attach_stability(store: SearchStore, *, status: SearchStatus, quality_targets_path: Path,
                     run_date: date, probe: IdentityProbe, compute_factory: ComputeFactory,
                     capabilities_path: Path | None = None) -> StabilitySummary:
    """由收尾持鎖呼叫；計算工廠在跳過、身分與整份重用之後才建立。"""
    previous = _previous(store.path)
    reason = attachment_skip_reason(status.outer.conclusion)
    observed = None if reason else probe()
    if reason or observed != store.identity:
        summary = _kept_points(_skip_summary(store, status), previous).model_copy(update={
            "state": "skipped", "completed": True, "reason_text": reason or IDENTITY_SKIP,
            "observed_identity": IdentityStamp.of(observed) if observed is not None else None})
        _finish(store.path, summary, previous)
        return summary
    try:
        summary = fresh_summary(store, status)
        if previous is not None and previous.state == "done" and not is_stale(previous, summary):
            return previous
        write_summary(store.path, summary)
        if not summary.selection.finalists:
            summary = _kept_points(_skip_summary(store, status).model_copy(update={"selection": summary.selection}), previous)
            summary = summary.model_copy(update={"state": "skipped", "completed": True,
                "reason_text": "沒有可排名的入圍方案"})
            _finish(store.path, summary, previous)
            return summary
        registry = load_quality_targets(quality_targets_path)
        runner = _Attacher(store, status, summary, previous, registry, run_date,
            _pinned(store, status, read_refinement_rows(store), registry, run_date),
            furniture_contact_rel(default_precision_contracts_path()))
        fem_root = summary_path(store.path).parent / f"fem-{uuid4().hex}"
        runner.prepare(fem_root, capabilities_path or config_path("capabilities.toml"))
        runner.compute(compute_factory, fem_root)
        return runner.finish()
    except (Exception, KeyboardInterrupt) as error:
        record_stability_error(store, status, error, previous=previous)
        raise
