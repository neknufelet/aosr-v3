"""T20 與殘響代價的自足題目、NumPy/Python 裁判及冷子行程夾具。"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from aosr.config.art_lane import ART_WLS_T20_LO_DB
from aosr.config.paths import config_path
from aosr.config.quality_targets import QualityPurpose, QualityTargets, SettingEntry, load_quality_targets
from aosr.geometry.shoebox import Room, Wall
from aosr.physics import late_decay, late_energy
from aosr.runtime import child_process_env, configure_jax
from aosr.scoring.reverberation_cost import cost_reverberation_evaluation
from aosr.scoring.contract import CategoryEvaluation
from tests.engine.test_late_decay_t20_contract import _independent_case
from tests.engine.test_reverberation_cost import _reverberation


CASES = ("flat", "varied", "lowabs", "patches", "tail")
COST_CASES = {
    "mixed": (0.18, 1.4, 0.42, 0.9, 0.2, 1.1, 0.35),
    "missing": (0.18, float("nan"), 0.42, 0.9, 0.2, 1.1, 0.35),
    "isolated": (0.18, float("nan"), float("nan"), 0.9, float("nan"), float("nan"), 0.35),
    "one": (0.18,) + (float("nan"),) * 6,
    "none": (float("nan"),) * 7,
    "inside": (0.45,) * 7,
}
COST_CASES.update({f"random_{i}": tuple(row) for i, row in enumerate(np.random.default_rng(562).uniform(0.05, 4.0, (8, 7)))})
ALPHA_FAILURES = {"alpha_nan": float("nan"), "alpha_inf": float("inf"), "alpha_negative": -0.1,
                  "alpha_above_one": 1.1, "alpha_one_patch": 1.0, "alpha_rounded_away": float(np.finfo(np.float64).eps / 4.0)}
FLAGS = ("invalid_absorption", "insufficient_decay_range", "insufficient_weight", "nonnegative_slope", "numerical_failure")
REGISTRY_CHANGES = ("scales", "weights", "jump_threshold", "lower_bounds", "upper_bounds", "logarithm_base")


def purpose() -> QualityPurpose:
    return load_quality_targets(config_path("quality_targets.toml")).purpose("dedicated_two_channel_listening_room")


def changed_purpose(name: str) -> QualityPurpose:
    document = load_quality_targets(config_path("quality_targets.toml")).model_dump(mode="json", by_alias=True)
    selected = document["purpose"][0]
    targets = {row["key"]: row for row in selected["target"]}
    settings = {row["key"]: row for row in selected["setting"]}
    if name == "scales":
        for key in ("reverberation.t20_interval_excess_s", "reverberation.adjacent_t20_log_ratio_excess"):
            targets[key]["worse_reference"] *= 2.0
    elif name == "weights":
        for table in selected["weight"]:
            if table["key"] == "reverberation.within_category_weights":
                table["item"][0]["value"] *= 3.0
    elif name == "jump_threshold":
        targets["reverberation.adjacent_t20_log_ratio_excess"]["value"] *= 2.0
    elif name in ("lower_bounds", "upper_bounds"):
        nominal = settings["reverberation.target_t20_nominal_s_by_band"]["value"]
        width = settings["reverberation.target_t20_tolerance_s_by_band"]["value"]
        for i, value in enumerate(width):
            nominal[i] += value / 2.0
            width[i] += (-1.0 if name == "lower_bounds" else 1.0) * value / 2.0
    elif name == "logarithm_base":
        settings["reverberation.adjacent_t20_logarithm_base"]["value"] *= 2.0
    else:
        raise ValueError(name)
    return QualityTargets.model_validate(document).purpose("dedicated_two_channel_listening_room")


def registry_values(name: str) -> NDArray[np.float64]:
    # 上下限各用每帶都在界外的題目，避免某一帶寫死卻因落在容許帶內而漏過。
    from aosr.scoring.reverberation_cost import target_intervals

    bounds = np.asarray(tuple(target_intervals(purpose()).values()))
    if name == "lower_bounds":
        return bounds[:, 0] / 2.0
    if name == "upper_bounds":
        return bounds[:, 1] * 2.0
    return np.asarray(COST_CASES["mixed"])


def inputs(name: str) -> tuple[late_energy.LateEnergyInputs, float, NDArray[np.float64]]:
    base, speed, _answer = _independent_case(name if name not in ("patches", "tail") else "varied")
    if name in ("patches", "tail"):
        base = replace(base, room=Room(5.1, 3.7, 2.6), n_per_wall=2, frequencies_hz=(125.0, 500.0, 2000.0),
                       absorption_by_wall={w: row[:3] for w, row in base.absorption_by_wall.items()})
        bounds = (0.025, 0.035) if name == "tail" else (0.04, 0.7)
        alpha = np.random.default_rng(562).uniform(*bounds, (24, 3))
    else:
        alpha = np.repeat(np.asarray([base.absorption_by_wall[w] for w in Wall.wall_names()]), base.n_per_wall**2, axis=0)
    return base, speed, alpha


def numpy_t20(name: str, alpha: NDArray[np.float64] | None = None) -> NDArray[np.float64]:
    base, speed, default = inputs(name)
    values = default if alpha is None else alpha
    problem = late_energy._reflection_problem(base)
    form = late_energy._form_factors(problem.patches)
    transfer = (1.0 - values)[:, None, :] * form[:, :, None]
    levels = late_decay._decay_level(late_decay._order_decay(transfer, problem.patches.areas), late_decay._exact_roots(transfer))
    late_decay._require_decay_reaches_lower_bound(levels, base.frequencies_hz, lower_db=ART_WLS_T20_LO_DB)
    return late_decay._fit_decay(levels, late_decay._collision_frequency(base, speed), lower_db=ART_WLS_T20_LO_DB).t60_s


def python_cost(values: NDArray[np.float64], unavailable: NDArray[np.bool_] | None = None,
                quality: QualityPurpose | None = None) -> tuple[float, dict[str, float]]:
    missing = ~np.isfinite(values) | (values <= 0.0)
    if unavailable is not None:
        missing |= unavailable
    rows = tuple(None if missing[i] else float(v) for i, v in enumerate(values))
    selected = purpose() if quality is None else quality
    evaluation = _reverberation(rows)
    # 舊夾具把底數與相鄰比寫死成 2；一起重建 payload（評估資料）後才餵正式 Python 裁判。
    document = evaluation.model_dump(mode="json")
    payload = document["payload"]
    entry = selected.entry("reverberation.adjacent_t20_logarithm_base")
    assert isinstance(entry, SettingEntry)
    base = entry.value
    assert isinstance(base, float)
    payload["logarithm_base"] = base
    for pair, lower, upper in zip(payload["adjacent_band_changes"], rows[:-1], rows[1:], strict=True):
        pair["signed_log_ratio"] = None if lower is None or upper is None else math.log(upper / lower, base)
    evaluation = CategoryEvaluation.model_validate(document)
    result = cost_reverberation_evaluation(evaluation, selected, "t20-jax-fixture")
    if result.category_cost is None:
        return float("nan"), {}
    return result.category_cost.value, dict(result.category_cost.components)


def decode(value: object) -> NDArray[np.float64]:
    return np.asarray(value, dtype=np.float64)


@pytest.fixture(scope="session")
def t20_runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    directory = tmp_path_factory.mktemp("t20-jax")
    cases = {}
    for name in CASES:
        base, speed, alpha = inputs(name)
        cases[name] = {"room": [base.room.Lx, base.room.Ly, base.room.Lz], "n": base.n_per_wall,
                       "speed": speed, "alpha": alpha.tolist()}
    problem = directory / "problem.json"
    output = directory / "output.json"
    problem.write_text(json.dumps({"decay": cases, "cost": COST_CASES}), encoding="utf-8")
    env = child_process_env(threads=1)
    env["PYTHONPATH"] = os.pathsep.join((env["PYTHONPATH"], str(Path(__file__).resolve().parents[2])))
    configure_jax(platforms="cpu", enable_x64=True, env=env)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["TMPDIR"] = str(directory)
    process = subprocess.run([sys.executable, str(Path(__file__).with_name("_t20_jax_probe.py")), str(problem), str(output)],
                             cwd=directory, env=env, capture_output=True, text=True, timeout=300)
    assert process.returncode == 0, process.stdout + process.stderr
    return cast(dict[str, object], json.loads(output.read_text(encoding="utf-8")))


def section(runs: dict[str, object], key: str, name: str) -> dict[str, object]:
    return cast(dict[str, dict[str, object]], runs[key])[name]


def mutant(runs: dict[str, object], name: str) -> object:
    return cast(dict[str, object], runs["mutants"])[name]
