"""候選確認迴圈雙精度子行程；只叫公開入口、寫帳本與原始數值，比對與判決在父行程。

兩組題：小房間替身（同一份衍生替身、六面 two_rigid、從預設樣板取規格起跑、正式設定）與合成目標（加權二次式，
一面一個目標厚度與目標空腔，求值便宜，給每一個列界都接續一次與四種停止原因用）。合成目標包一層
``jax.debug.callback`` 數真的被執行了幾次（編譯後每叫一次跑一次，追蹤不算）。
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np

from aosr.config.construction_presets import load_construction_presets
from aosr.config.material_gradient import CandidateSettings, MaterialGradientSettings, load_material_gradient_settings
from aosr.config.paths import config_path
from aosr.config.physics_constants import load_physics_constants
from aosr.config.porous_model_ranges import load_porous_model_ranges
from aosr.config.quality_targets import load_quality_targets
from aosr.physics.fem_reduced_store import load_reduced
from aosr.scoring.receiver_set import ReceiverSet
from aosr.scoring.stage_one_objective_jax import StageOneObjectiveArrays, objective_constants
from aosr.scoring.stage_one_score import stage_one_score_settings
from aosr.gradient import material_gradient as gradient_search
from aosr.gradient import material_gradient_ledger as ledger
from aosr.gradient import material_gradient_loop as loop


Record = dict[str, object]


def _hex(values: object) -> list[str]:
    return [float(value).hex() for value in np.asarray(values).ravel()]


def _settings(base: MaterialGradientSettings, override: object) -> MaterialGradientSettings:
    changes = cast(dict[str, object], override)
    if not changes:
        return base
    candidates = CandidateSettings.model_validate({**base.candidates.model_dump(), **changes})
    return MaterialGradientSettings.model_validate({**base.model_dump(), "candidates": candidates.model_dump()})


def synthetic_header(designs: tuple[gradient_search.SurfaceDesign, ...],
                     settings: MaterialGradientSettings) -> ledger.MaterialGradientHeader:
    """合成目標沒有替身，標頭的替身三格填固定字樣；其他照正式算。"""
    return ledger.MaterialGradientHeader(
        ledger_version=ledger.LEDGER_VERSION, input_hash="synthetic", mesh_hash="synthetic",
        physics_identity="synthetic", surfaces=tuple((design.surface_id, design.preset_id) for design in designs),
        variable_names=gradient_search.design_names(designs), settings_fingerprint=settings.fingerprint)


def synthetic_objective(designs: tuple[gradient_search.SurfaceDesign, ...], counter: list[int],
                        ) -> gradient_search.ChainObjective:
    """Σ w·(θ − 目標)²：棉板目標 0.1、權重照面序 100、200、300…；空腔目標 0.03、權重 10。每執行一次計數加一。"""
    weights, targets = [], []
    for index, design in enumerate(designs):
        for name in design.variables:
            cavity = name == design.cavity_variable
            weights.append(10.0 if cavity else 100.0 * (index + 1))
            targets.append(0.03 if cavity else 0.1)
    weight, target = np.asarray(weights), np.asarray(targets)

    def tick(_: jax.Array) -> None:
        counter[0] += 1

    def objective(theta: jax.Array) -> tuple[jax.Array, StageOneObjectiveArrays]:
        jax.debug.callback(tick, theta)
        shift = theta - jnp.asarray(target, dtype=theta.dtype)
        value = jnp.sum(jnp.asarray(weight, dtype=theta.dtype) * shift * shift)
        return value, StageOneObjectiveArrays(value[None], value[None], value, value, value)

    return objective


def _start_ledger(path: Path, header: ledger.MaterialGradientHeader, designs: tuple[gradient_search.SurfaceDesign, ...],
                  objective: gradient_search.ChainObjective, theta: np.ndarray) -> np.ndarray:
    """前段只有一列：預設樣板取規格的可行點與那裡的值。"""
    snapped = gradient_search.snap_to_specs(theta, designs)
    value, aux = objective(jnp.asarray(snapped.theta))
    ledger.write_ledger(path, header, (ledger.snap_row(
        snapped, float(value), (float(aux.mean_own_line_ripple_rms_db), float(aux.mean_seat_robustness_rms_db)),
        0.0, 0),))
    return snapped.theta


def _cut_copy(source: Path, target: Path, rows: int) -> Path:
    """一口氣跑完的帳本留標頭與前 ``rows`` 列，另存一份給接續。"""
    lines = source.read_bytes().splitlines(keepends=True)
    target.write_bytes(b"".join(lines[:rows + 1]))
    return target


def _result(result: loop.LoopResult) -> Record:
    return {"theta": _hex(result.theta), "stop_reason": result.stop_reason, "evaluations": result.evaluations,
            "rounds": len(result.rounds)}


def _synthetic(directory: Path, name: str, case: Record, base: MaterialGradientSettings) -> Record:
    presets = load_construction_presets(config_path("construction_presets.toml"))
    surfaces = tuple((str(item[0]), item[1]) for item in cast(list[list[str | None]], case["surfaces"]))
    designs = gradient_search.designs_from_presets(surfaces, base, presets)
    settings = _settings(base, case["override"])
    header = synthetic_header(designs, settings)
    counter = [0]
    objective = synthetic_objective(designs, counter)
    path = directory / f"{name}.jsonl"
    start = _start_ledger(path, header, designs, objective, gradient_search.baseline_theta(designs))
    before, clock = counter[0], time.perf_counter()
    result = loop.run_rounds(objective, start, designs, settings, path, header)
    row: Record = {"ledger": path.name, "calls": counter[0] - before, "result": _result(result),
                   "seconds": {"run": time.perf_counter() - clock}}
    seconds = cast(dict[str, object], row["seconds"])
    if case.get("again"):
        again = directory / f"{name}.again.jsonl"
        _start_ledger(again, header, designs, objective, gradient_search.baseline_theta(designs))
        loop.run_rounds(objective, start, designs, settings, again, header)
        row["again"] = again.name
    if case.get("resume_every_row"):
        total = len(path.read_bytes().splitlines()) - 1
        resumed, each = [], []
        for keep in range(1, total + 1):
            cut = _cut_copy(path, directory / f"{name}.cut{keep:04d}.jsonl", keep)
            clock = time.perf_counter()
            loop.resume_rounds(objective, designs, settings, cut, header)
            each.append(time.perf_counter() - clock)
            resumed.append(cut.name)
        row["resumed"] = resumed
        seconds["resume_each"] = each
    if case.get("single_round"):
        before = counter[0]
        single = loop.confirmation_round(objective, start, designs, settings, budget_left=10**6)
        row["single_round"] = {"evaluations": single.evaluations, "calls": counter[0] - before,
                               "bound": single.bound, "candidates": len(single.candidates)}
    return row


def _surrogate(directory: Path, problem: Record, base: MaterialGradientSettings) -> Record:
    model = load_reduced(directory / "model.npz")
    presets = load_construction_presets(config_path("construction_presets.toml"))
    receivers = ReceiverSet.model_validate(problem["receivers"])
    score = stage_one_score_settings(load_quality_targets(config_path("quality_targets.toml")), str(problem["purpose"]))
    constants = objective_constants(np.asarray(model.frequencies_hz), score, receivers, len(model.source_names))
    designs = gradient_search.wall_designs_from_presets(cast(list[str | None], problem["presets"]), base, presets)
    objective = gradient_search.chain_objective(
        model, designs, constants, air=load_physics_constants(config_path("physics_constants.toml")),
        ranges=load_porous_model_ranges(config_path("porous_model_ranges.toml")), frequencies=model.frequencies_hz)
    header = ledger.header_for(model, designs, base)
    main, again = directory / "surrogate.jsonl", directory / "surrogate.again.jsonl"
    seconds: dict[str, object] = {}
    clock = time.perf_counter()
    start = _start_ledger(main, header, designs, objective, gradient_search.baseline_theta(designs))
    result = loop.run_rounds(objective, start, designs, base, main, header)
    seconds["run_first_with_compile"] = time.perf_counter() - clock
    clock = time.perf_counter()
    _start_ledger(again, header, designs, objective, gradient_search.baseline_theta(designs))
    loop.run_rounds(objective, start, designs, base, again, header)
    seconds["run_again"] = time.perf_counter() - clock
    _, rows = ledger.read_ledger(main)
    after_first = next(index for index, item in enumerate(rows) if item.phase == "round_decision") + 1
    mid_second = next(index for index, item in enumerate(rows) if item.round == 1 and item.phase == "candidate_jump") + 1
    resumed, each = [], []
    for keep in (after_first, mid_second):
        cut = _cut_copy(main, directory / f"surrogate.cut{keep:04d}.jsonl", keep)
        clock = time.perf_counter()
        loop.resume_rounds(objective, designs, base, cut, header)
        each.append(time.perf_counter() - clock)
        resumed.append(cut.name)
    seconds["resume_each"] = each
    clock = time.perf_counter()
    single = loop.confirmation_round(objective, start.astype(np.float32), designs, base, budget_left=10**6)
    seconds["float32_round_with_compile"] = time.perf_counter() - clock
    return {"ledger": main.name, "again": again.name, "resumed": resumed, "result": _result(result),
            "seconds": seconds,
            "single": {"dtype": str(single.theta.dtype), "theta": _hex(single.theta),
                       "baseline_dtype": str(single.baseline.theta.dtype) if single.baseline is not None else None,
                       "value": float(cast(float, single.value)).hex(), "evaluations": single.evaluations,
                       "candidate_values": [item.value.hex() for item in single.confirmed]}}


def _edge(problem: Record, base: MaterialGradientSettings) -> Record:
    """審查 B1 的重現參數：空腔切到下界那個角。投影後在那裡量投影梯度範數（下降 0 步那一列記的），
    再用線性目標 −2t−c 下降，看停在哪一個原因、走了幾步。"""
    case = cast(Record, problem["edge"])
    top = float.fromhex(cast(str, case["top"]))
    lower = np.asarray([float.fromhex(value) for value in cast(list[str], case["lower"])])
    upper = np.asarray([top, top])
    limits = (gradient_search.DepthLimit(0, 1, top),)
    theta = gradient_search.project_to_limits(np.asarray([1.0, -1.0]), lower, upper, limits)
    settings = base.descent.model_copy(update={"max_steps": 3, "gradient_norm_stop": 0.1})
    rows: Record = {"theta": _hex(theta)}
    for name, weights in (("flat", (1.0, 1.0)), ("linear", (2.0, 1.0))):
        def objective(values: jax.Array, weights: tuple[float, float] = weights) -> tuple[
                jax.Array, StageOneObjectiveArrays]:
            value = -weights[0] * values[0] - weights[1] * values[1]
            return value, StageOneObjectiveArrays(value[None], value[None], value, value, value)
        result = gradient_search.projected_descent(objective, theta, (lower, upper), settings, limits=limits)
        rows[name] = {"stop_reason": result.stop_reason, "steps": len(result.steps) - 1,
                      "norms": [step.gradient_norm.hex() for step in result.steps]}
    return rows


def _threshold_edge(problem: Record, base: MaterialGradientSettings) -> Record:
    """改善恰等於門檻不接受：同一輪先量最好的改善 X，門檻設 X 再跑一次、設比 X 小一個浮點數再跑一次。"""
    presets = load_construction_presets(config_path("construction_presets.toml"))
    surfaces = tuple((str(item[0]), item[1]) for item in cast(list[list[str | None]], problem["threshold_surfaces"]))
    designs = gradient_search.designs_from_presets(surfaces, base, presets)
    objective = synthetic_objective(designs, [0])
    start = gradient_search.snap_to_specs(gradient_search.baseline_theta(designs), designs).theta
    first = loop.confirmation_round(objective, start, designs, base, budget_left=10**6)
    best = max(item.improvement for item in first.confirmed)
    rows: Record = {"best": best.hex()}
    for name, threshold in (("equal", best), ("below", float(np.nextafter(best, -np.inf)))):
        settings = _settings(base, {"improvement_threshold": threshold})
        again = loop.confirmation_round(objective, start, designs, settings, budget_left=10**6)
        rows[name] = {"threshold": threshold.hex(), "best": max(item.improvement for item in again.confirmed).hex(),
                      "accepted": again.accepted.candidate_id if again.accepted is not None else None,
                      "stop_reason": again.stop_reason}
    return rows


def main(directory: Path) -> None:
    problem = cast(Record, json.loads((directory / "problem.json").read_text(encoding="utf-8")))
    base = load_material_gradient_settings(config_path("material_gradient.toml"))
    clock = time.perf_counter()
    output: Record = {"synthetic": {name: _synthetic(directory, name, cast(Record, case), base)
                                    for name, case in cast(dict[str, object], problem["synthetic"]).items()}}
    synthetic_seconds = time.perf_counter() - clock
    output["edge"] = _edge(problem, base)
    output["threshold_edge"] = _threshold_edge(problem, base)
    if problem.get("surrogate"):
        clock = time.perf_counter()
        output["surrogate"] = _surrogate(directory, problem, base)
        output["seconds"] = {"synthetic_all": synthetic_seconds, "surrogate_all": time.perf_counter() - clock}
    (directory / "output.json").write_text(json.dumps(output), encoding="utf-8")
    shutil.rmtree(directory / "__pycache__", ignore_errors=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
