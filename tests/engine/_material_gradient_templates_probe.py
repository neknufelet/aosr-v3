"""換樣板（#561 庚三）雙精度子行程；只叫公開入口、寫帳本與原始數值，比對與判決在父行程。

兩組題：合成目標（每面一個加權二次式，再加一張「哪一面換成哪個目標」的固定偏移表，求值便宜，給每一個列界的接續、
五種停止原因、門檻邊界、求值次數與停止查詢次數用）與小房間替身（同 ``test_material_gradient`` 衍生的那一份、
六面 two_rigid 從預設樣板取規格起跑、``chain_objective`` 對每個配置重建）。合成目標包一層 ``jax.debug.callback``
數真的被執行了幾次（編譯後每叫一次跑一次，追蹤不算）；同一個配置的目標函式重用同一份（編譯一次），換配置才重建。
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np

from aosr.config.construction_presets import ConstructionPreset, load_construction_presets
from aosr.config.material_gradient import MaterialGradientSettings, load_material_gradient_settings
from aosr.config.paths import config_path
from aosr.config.physics_constants import load_physics_constants
from aosr.config.porous_model_ranges import load_porous_model_ranges
from aosr.config.quality_targets import load_quality_targets
from aosr.geometry.shoebox import Wall
from aosr.materials.construction_models import ConstantImpedance, SurfaceMaterial
from aosr.physics.fem_reduced_store import load_reduced
from aosr.scoring.receiver_set import ReceiverSet
from aosr.scoring.gradient_objective_jax import GradientObjectiveArrays, gradient_objective_constants, objective_reading
from aosr.scoring.gradient_objective_settings import gradient_objective_settings
from aosr.gradient import material_gradient as gradient_search
from aosr.gradient import material_gradient_ledger as ledger
from aosr.gradient import template_ledger as templates
from aosr.gradient import template_rounds as rounds
from aosr.gradient import template_segment as segment
from aosr.gradient.endpoint import material_digest
from aosr.gradient.run import template_header_for


Record = dict[str, object]


def _hex(values: object) -> list[str]:
    return [float(value).hex() for value in np.asarray(values).ravel()]


def settings_with(base: MaterialGradientSettings, override: object) -> MaterialGradientSettings:
    """``{"表名": {"鍵": 值}}`` 換掉那幾格、重新驗證（指紋跟著變）。"""
    document = base.model_dump()
    for section, changes in cast(dict[str, dict[str, object]], override).items():
        document[section] = {**document[section], **changes}
    return MaterialGradientSettings.model_validate(document)


def kept_walls(values: list[float]) -> dict[str, SurfaceMaterial]:
    """方案原本六面的牆：照 ``Wall.all()`` 的順序給固定阻抗（Pa·s/m）。"""
    return {wall.wall_name(): ConstantImpedance(impedance_pa_s_m=value) for wall, value in zip(Wall.all(), values,
                                                                                              strict=True)}


def synthetic_factory(settings: MaterialGradientSettings, offsets: dict[str, float],
                      counter: list[int]) -> rounds.ObjectiveFactory:
    """合成目標：Σ w·(θ − 目標)² ＋ 偏移。棉板目標 0.05（48K 起點與 24K、96K 的唯一合法厚度）、權重照面序 100、200…；
    空腔目標 0.03、權重 10；偏移是 ``offsets`` 裡「面->目標」每一個現在成立的加總。同一個配置回同一份函式。"""
    made: dict[rounds.Surfaces, gradient_search.ChainObjective] = {}

    def factory(designs: tuple[gradient_search.SurfaceDesign, ...]) -> gradient_search.ChainObjective:
        surfaces = rounds.surfaces_of(designs)
        if surfaces in made:
            return made[surfaces]
        weights, targets, offset = [], [], 0.0
        for index, design in enumerate(designs):
            offset += offsets.get(rounds.candidate_id(design.surface_id, rounds.target_of(design.preset_id, settings)),
                                  0.0)
            for name in design.variables:
                cavity = name == design.cavity_variable
                weights.append(10.0 if cavity else 100.0 * (index + 1))
                targets.append(0.03 if cavity else 0.05)
        weight, target = np.asarray(weights), np.asarray(targets)

        def tick(_: jax.Array) -> None:
            counter[0] += 1

        def objective(theta: jax.Array) -> tuple[jax.Array, GradientObjectiveArrays]:
            jax.debug.callback(tick, theta)
            shift = theta - jnp.asarray(target, dtype=theta.dtype)
            value = jnp.sum(jnp.asarray(weight, dtype=theta.dtype) * shift * shift) + offset
            return value, GradientObjectiveArrays(value[None], value[None], value, value * 0.5, value * 0.25,
                                          value * 0.125, value, value * 0.125)

        made[surfaces] = objective
        return objective

    return factory


# 合成目標的分數常數：0.125·v＋0.25·(v／2)＝v／4（舊合分）、v／4＋6·(v／8)＝v（新合分）。
SYNTHETIC_SCORES = ledger.ScoreConstants(component_weights=(0.125, 0.25), worse_references=(1.0, 1.0),
                                         proxy_weight=6.0, proxy_worse_reference=1.0, proxy_temperature_db=0.5)


def synthetic_header(settings: MaterialGradientSettings, surfaces: rounds.Surfaces,
                     kept: dict[str, SurfaceMaterial]) -> templates.TemplateHeader:
    """合成目標沒有替身、評分常數與庚二帳本：那幾格填固定字樣（摘要格填固定字樣的 SHA-256）；其他照正式算。
    分數常數配合成目標（分項 v、v／2，代理 v／8，舊合分 v／4，新合分 v）：見 :data:`SYNTHETIC_SCORES`。"""
    return templates.TemplateHeader(
        ledger_version=templates.TEMPLATE_LEDGER_VERSION, input_hash="synthetic", mesh_hash="synthetic",
        physics_identity="synthetic", settings_fingerprint=settings.full_fingerprint,
        objective_fingerprint=hashlib.sha256(b"synthetic").hexdigest(),
        base_ledger_sha256=hashlib.sha256(b"synthetic-base").hexdigest(), initial_surfaces=surfaces,
        kept_digests={name: material_digest(material) for name, material in kept.items()},
        score_constants=SYNTHETIC_SCORES)


def start_point(context: rounds.TemplateContext, surfaces: rounds.Surfaces) -> templates.TemplateStart:
    """起點：配置的樣板起點取規格那一點與那裡的值（庚二收尾那一列在考卷裡用這一點代替）。"""
    designs = rounds.designs_for_surfaces(surfaces, context)
    theta = gradient_search.snap_to_specs(gradient_search.baseline_theta(designs), designs).theta
    value, aux = context.objective_for(designs)(jnp.asarray(theta))
    reading = objective_reading(aux)
    return templates.TemplateStart(surfaces, tuple(_hex(theta)), float(value), reading.screening_total,
                                   reading.components)


def _cut_copy(source: Path, target: Path, rows: int) -> Path:
    """一口氣跑完的帳本留標頭與前 ``rows`` 列，另存一份。"""
    lines = source.read_bytes().splitlines(keepends=True)
    target.write_bytes(b"".join(lines[:rows + 1]))
    return target


def _result(result: segment.TemplateResult) -> Record:
    return {"surfaces": [list(item) for item in result.surfaces], "theta": _hex(result.theta),
            "value": result.value.hex(), "stop_reason": result.stop_reason, "rounds": result.rounds,
            "evaluations": result.evaluations, "seconds": result.seconds}


def _start_record(start: templates.TemplateStart) -> Record:
    return {"surfaces": [list(item) for item in start.surfaces], "theta_hex": list(start.theta_hex),
            "value": start.value.hex(), "screening_total": start.screening_total.hex(),
            "components": [value.hex() for value in start.components]}


def _counting_stop(checks: list[int], stop_at: int | None = None) -> Callable[[], bool]:
    def should_stop() -> bool:
        checks[0] += 1
        return stop_at is not None and checks[0] == stop_at
    return should_stop


class _Case:
    """一題合成：設定、計數、工廠、起點、標頭。"""

    def __init__(self, base: MaterialGradientSettings, presets: tuple[ConstructionPreset, ...], problem: Record,
                 override: object) -> None:
        self.settings = settings_with(base, override)
        self.counter = [0]
        kept = kept_walls(cast(list[float], problem["kept"]))
        offsets = cast(dict[str, float], problem["offsets"])
        self.context = rounds.TemplateContext(self.settings, presets, kept,
                                              synthetic_factory(self.settings, offsets, self.counter))
        self.surfaces = tuple((str(item[0]), item[1]) for item in cast(list[list[str | None]], problem["surfaces"]))
        self.start = start_point(self.context, self.surfaces)
        self.header = synthetic_header(self.settings, self.surfaces, kept)

    def run(self, path: Path, should_stop: Callable[[], bool] | None = None) -> segment.TemplateResult:
        return segment.run_template_rounds(self.context, path, self.header, self.start, should_stop=should_stop)

    def resume(self, path: Path, should_stop: Callable[[], bool] | None = None) -> segment.TemplateResult:
        return segment.resume_template_rounds(self.context, path, self.header, self.start, should_stop=should_stop)


def _resume_every(case: _Case, directory: Path, name: str, path: Path) -> Record:
    """在每一個列界（留前 0、1、…、全部列）各截一份接續。"""
    count = len(templates.read_template_ledger(path)[1])
    names, clock = [], time.perf_counter()
    for keep in range(count + 1):
        cut = _cut_copy(path, directory / f"{name}.cut{keep:04d}.jsonl", keep)
        case.resume(cut)
        names.append(cut.name)
    return {"resumed": names, "seconds": time.perf_counter() - clock}


def _stops(case: _Case, directory: Path, name: str, total_checks: int) -> list[Record]:
    """第 1 次、中間一次、最後一次查停止時叫停：要拋 SearchStopped、帳本另存一份，再接續到收尾。"""
    found = []
    for at in sorted({1, total_checks // 2, total_checks}):
        path = directory / f"{name}.stop{at:04d}.jsonl"
        stopped = False
        try:
            case.run(path, _counting_stop([0], at))
        except gradient_search.SearchStopped:
            stopped = True
        snapshot = directory / f"{path.stem}.stopped.jsonl"
        shutil.copyfile(path, snapshot)
        case.resume(path)
        found.append({"at": at, "stopped": stopped, "stopped_ledger": snapshot.name, "resumed": path.name})
    return found


def _with_seconds(path: Path, target: Path, keep: int, round_index: int, seconds: float) -> Path:
    """帳本留前 ``keep`` 列，第 ``round_index`` 輪決定列的秒數改成 ``seconds``（其他一字不動）。"""
    lines = path.read_bytes().splitlines(keepends=True)[:keep + 1]
    for index, line in enumerate(lines[1:], start=1):
        row = json.loads(line)
        if row["phase"] == "template_decision" and row["round"] == round_index:
            row["seconds"] = seconds
            lines[index] = (json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    target.write_bytes(b"".join(lines))
    return target


def _seconds_cases(case: _Case, directory: Path, name: str, path: Path) -> Record:
    """截在第 1 輪半途（第 0 輪決定列之後再留一列），第 0 輪決定列秒數改成恰等於上限：接續重做第 1 輪、記總時間到上限；
    改成比上限大：帳本與設定矛盾、接續拒收（帳本不動）。"""
    rows = templates.read_template_ledger(path)[1]
    decision = next(index for index, row in enumerate(rows) if row.decision is not None and row.round == 0)
    limit = float(case.settings.templates.max_seconds)
    equal = _with_seconds(path, directory / f"{name}.seconds-equal.jsonl", decision + 2, 0, limit)
    result = case.resume(equal)
    over = _with_seconds(path, directory / f"{name}.seconds-over.jsonl", decision + 2, 0, limit + 1.0)
    before = over.read_bytes()
    try:
        case.resume(over)
        error = None
    except ValueError as caught:
        error = str(caught)
    return {"equal": equal.name, "equal_result": _result(result), "over_error": error,
            "over_unchanged": over.read_bytes() == before, "limit": limit}


def _synthetic(directory: Path, name: str, spec: Record, base: MaterialGradientSettings,
               presets: tuple[ConstructionPreset, ...], problem: Record) -> Record:
    case = _Case(base, presets, problem, spec["override"])
    path = directory / f"{name}.jsonl"
    checks = [0]
    before, clock = case.counter[0], time.perf_counter()
    result = case.run(path, _counting_stop(checks))
    row: Record = {"ledger": path.name, "calls": case.counter[0] - before, "checks": checks[0],
                   "result": _result(result), "run_seconds": time.perf_counter() - clock,
                   "start": _start_record(case.start)}
    if spec.get("again"):
        again = directory / f"{name}.again.jsonl"
        case.run(again)
        row["again"] = again.name
    if spec.get("resume_every"):
        row["resume_every"] = _resume_every(case, directory, name, path)
    if spec.get("stops"):
        row["stops"] = _stops(case, directory, name, checks[0])
    if spec.get("seconds"):
        row["seconds"] = _seconds_cases(case, directory, name, path)
    if spec.get("single_round"):
        designs = rounds.designs_for_surfaces(case.surfaces, case.context)
        theta = np.asarray([float.fromhex(value) for value in case.start.theta_hex])
        point = rounds.TemplatePoint(designs, theta, case.start.value, case.start.screening_total, case.start.components)
        before = case.counter[0]
        single = rounds.template_round(case.context.objective_for(designs), point, case.context, budget_left=10**6)
        row["single_round"] = {"evaluations": single.evaluations, "calls": case.counter[0] - before,
                               "bound": single.bound, "quick": len(single.quick), "retuned": len(single.retuned)}
    return row


def _threshold_edge(base: MaterialGradientSettings, presets: tuple[ConstructionPreset, ...], problem: Record,
                    override: object) -> Record:
    """改善恰等於門檻不接受：同一輪先量最好的改善 X，門檻設 X 再跑一次、設比 X 小一個浮點數再跑一次。"""
    case = _Case(base, presets, problem, override)
    designs = rounds.designs_for_surfaces(case.surfaces, case.context)
    theta = np.asarray([float.fromhex(value) for value in case.start.theta_hex])
    point = rounds.TemplatePoint(designs, theta, case.start.value, case.start.screening_total, case.start.components)
    first = rounds.template_round(case.context.objective_for(designs), point, case.context, budget_left=10**6)
    best = max(item.improvement for item in first.retuned)
    found: Record = {"best": best.hex()}
    for label, threshold in (("equal", best), ("below", float(np.nextafter(best, -np.inf)))):
        settings = settings_with(case.settings, {"candidates": {"improvement_threshold": threshold}})
        context = case.context._replace(settings=settings)
        again = rounds.template_round(context.objective_for(designs), point, context, budget_left=10**6)
        found[label] = {"threshold": threshold.hex(), "best": max(item.improvement for item in again.retuned).hex(),
                        "accepted": again.accepted.quick.candidate_id if again.accepted is not None else None,
                        "stop_reason": again.stop_reason}
    return found


def _surrogate(directory: Path, problem: Record, base: MaterialGradientSettings,
               presets: tuple[ConstructionPreset, ...]) -> Record:
    spec = cast(Record, problem["surrogate"])
    settings = settings_with(base, spec["override"])
    model = load_reduced(directory / "model.npz")
    receivers = ReceiverSet.model_validate(problem["receivers"])
    score = gradient_objective_settings(load_quality_targets(config_path("quality_targets.toml")), str(problem["purpose"]))
    constants = gradient_objective_constants(np.asarray(model.frequencies_hz), score, receivers, len(model.source_names))
    air = load_physics_constants(config_path("physics_constants.toml"))
    ranges = load_porous_model_ranges(config_path("porous_model_ranges.toml"))
    builds: list[float] = []

    def factory(designs: tuple[gradient_search.SurfaceDesign, ...]) -> gradient_search.ChainObjective:
        clock = time.perf_counter()
        made = gradient_search.chain_objective(model, designs, constants, air=air, ranges=ranges,
                                               frequencies=model.frequencies_hz)
        builds.append(time.perf_counter() - clock)
        return made

    kept = kept_walls(cast(list[float], spec["kept"]))
    context = rounds.TemplateContext(settings, presets, kept, factory)
    surfaces = tuple(zip(gradient_search.wall_surface_ids(), cast(list[str | None], spec["presets"]), strict=True))
    clock = time.perf_counter()
    start = start_point(context, surfaces)
    start_seconds = time.perf_counter() - clock
    designs = rounds.designs_for_surfaces(surfaces, context)
    by_wall = {wall: kept[wall.wall_name()] for wall in Wall.all()}
    header = template_header_for(ledger.header_for(model, designs, settings, objective_constants=constants), settings,
                                 by_wall, base_ledger_sha256=hashlib.sha256(b"small-room").hexdigest())
    path = directory / "surrogate.jsonl"
    clock = time.perf_counter()
    result = segment.run_template_rounds(context, path, header, start)
    run_seconds = time.perf_counter() - clock
    rows = templates.read_template_ledger(path)[1]
    cut_keep = next(index for index, row in enumerate(rows) if row.phase == "baseline_retune") + 1
    cut = _cut_copy(path, directory / f"surrogate.cut{cut_keep:04d}.jsonl", cut_keep)
    clock = time.perf_counter()
    segment.resume_template_rounds(context, cut, header, start)
    return {"ledger": path.name, "resumed": cut.name, "result": _result(result),
            "start": _start_record(start),
            "x64": str(jnp.asarray(1.0).dtype),
            "seconds": {"start": start_seconds, "run": run_seconds, "resume": time.perf_counter() - clock,
                        "builds": builds}}


def main(directory: Path) -> None:
    problem = cast(Record, json.loads((directory / "problem.json").read_text(encoding="utf-8")))
    base = load_material_gradient_settings(config_path("material_gradient.toml"))
    presets = load_construction_presets(config_path("construction_presets.toml"))
    synthetic = cast(Record, problem["synthetic"])
    clock = time.perf_counter()
    output: Record = {"synthetic": {name: _synthetic(directory, name, cast(Record, spec), base, presets, synthetic)
                                    for name, spec in cast(dict[str, object], problem["cases"]).items()}}
    output["threshold_edge"] = _threshold_edge(base, presets, synthetic, problem["threshold_override"])
    output["synthetic_seconds"] = time.perf_counter() - clock
    if problem.get("surrogate"):
        output["surrogate"] = _surrogate(directory, problem, base, presets)
    (directory / "output.json").write_text(json.dumps(output), encoding="utf-8")
    shutil.rmtree(directory / "__pycache__", ignore_errors=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
