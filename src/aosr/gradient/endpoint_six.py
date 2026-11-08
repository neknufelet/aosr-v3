"""六類終點的三組數；只消費既有聲壓，不求解完整模型。"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from aosr.config.capabilities import CapabilityTable
from aosr.config.construction_environment import construction_environment
from aosr.config.directivity_defaults import DirectivityDefaults
from aosr.config.paths import config_path
from aosr.config.precision_contracts import default_precision_contracts_path, load_precision_contracts
from aosr.config.quality_targets import load_quality_targets
from aosr.gradient.material_gradient import SurfaceDesign
from aosr.gradient.six_category_chain import _impedance_function, _materials
from aosr.physics.fem_reduced import ReducedModel, evaluate_reduced, system_coefficients
from aosr.reporting.pipeline import run_scheme
from aosr.reporting.calculation_fingerprint import calculation_fingerprint
from aosr.reporting.physics_identity import physics_identity
from aosr.reporting.scheme import Scheme
from aosr.reporting.validation import checked_inputs
from aosr.scoring import ranking
from aosr.scoring.gradient_objective_jax import gradient_objective_from_pressures
from aosr.scoring.six_category_jax import SixCategoryArrays, SixCategoryConstants, build_constants, constant_material_flags, six_category_costs

_FROZEN = ConfigDict(frozen=True, extra='forbid', allow_inf_nan=False)
type Categories = tuple[float | None, float | None, float | None, float | None, float | None, float | None]
CATEGORY_NAMES = ('timbre_balance', 'listening_area_stability', 'reflections_and_echo',
                  'reverberation', 'channel_matching', 'low_frequency_response')
_CONTRACTS = ('timbre_cost_jax_vs_python', 'listening_area_cost_jax_vs_python',
              'reflections_cost_jax_vs_python', 'reverberation_cost_jax_vs_python',
              'channel_matching_cost_jax_vs_python', 'stage_one_objective_jax_vs_python')


class SixScores(BaseModel):
    """產品六類、搜尋 J（含代理）、產品合分與低頻代理；不可估明記空值。"""
    model_config = _FROZEN
    categories: Categories
    objective_total: float | None
    product_total: float | None
    low_frequency_objective: float | None
    proxy_db: float | None


class CrossoverNotes(BaseModel):
    """房間的交接頻率、上限截斷旗標，以及殘響可估頻帶數。"""
    model_config = _FROZEN
    schroeder_frequency_hz: float | None
    capped_by_upper_limit: bool
    reverberation_band_count: int


class SixComparison(BaseModel):
    """一套材料的替身誤差、實作差及兩邊逐類值；正差為完整較高或 JAX 較高。"""
    model_config = _FROZEN
    reduced: SixScores
    full: SixScores
    surrogate_difference: float | None
    numpy_categories: Categories
    numpy_total: float | None
    numpy_unavailable_reason: str | None
    implementation_difference: float | None
    implementation_bound: float | None
    implementation_over_bound: bool | None
    reduced_crossover: CrossoverNotes
    full_crossover: CrossoverNotes


class SixImprovement(BaseModel):
    """同一定義的原牆減終點，替身、完整與 numpy 各自相減。"""
    model_config = _FROZEN
    reduced: SixScores
    full: SixScores
    numpy_categories: Categories
    numpy_total: float | None


class EndpointSix(BaseModel):
    """v5 必填的終點、原牆基準與改善三組數。"""
    model_config = _FROZEN
    endpoint: SixComparison
    baseline: SixComparison
    improvement: SixImprovement


class ScoringContext(NamedTuple):
    scheme: Scheme
    constants: SixCategoryConstants
    capabilities: CapabilityTable
    directivity: DirectivityDefaults
    program_fingerprint: str
    physics_identity: str
    engine_commit: str


def scoring_context(scheme: Scheme, model: ReducedModel, purpose: str, *, capabilities: CapabilityTable,
                    directivity: DirectivityDefaults, capabilities_path: Path, engine_commit: str | None) -> ScoringContext:
    """沿搜尋的聲源序、目的與逐對輸入建六類常數；全程保持雙精度。"""
    channels = {channel.speaker_id: channel for channel in scheme.channel_group.channels}
    group = scheme.channel_group.model_copy(update={'channels': tuple(channels[name] for name in model.source_names)})
    scored = scheme.model_copy(update={'purpose': purpose, 'channel_group': group})
    _, pairs = checked_inputs(scored, capabilities=capabilities, directivity=directivity)
    constants = build_constants(scored, pair_inputs={key: value[1] for key, value in pairs.items()},
                                quality_targets_path=config_path('quality_targets.toml'))
    return ScoringContext(scored, constants, capabilities, directivity,
        calculation_fingerprint(capabilities_path=capabilities_path),
        physics_identity(capabilities=capabilities, directivity=directivity), engine_commit or '未記錄')


def _finite(value: float) -> float | None:
    return value if np.isfinite(value) else None


def _categories(values: tuple[float | None, ...]) -> Categories:
    a, b, c, d, e, f = values
    return a, b, c, d, e, f


def _reading(arrays: SixCategoryArrays, pressure: NDArray[np.complex128], c: SixCategoryConstants) -> SixScores:
    low = gradient_objective_from_pressures(c.objective(), jnp.asarray(pressure, dtype=jnp.complex128))
    return SixScores(categories=_categories(tuple(_finite(float(v)) for v in arrays[:6])),
        objective_total=_finite(float(arrays.objective_total)), product_total=_finite(float(arrays.product_total)),
        low_frequency_objective=_finite(float(arrays.low_frequency_objective)), proxy_db=_finite(float(low.big_peak_proxy_db)))


def _notes(arrays: SixCategoryArrays) -> CrossoverNotes:
    return CrossoverNotes(schroeder_frequency_hz=_finite(float(arrays.schroeder_frequency_hz)),
        capped_by_upper_limit=bool(arrays.capped_by_upper_limit), reverberation_band_count=int(arrays.reverberation_band_count))


def _numpy_scores(context: ScoringContext, scheme: Scheme, pressure: NDArray[np.complex128]) -> tuple[Categories, float | None, str | None]:
    energy = np.abs(pressure) ** 2
    injected = {(channel.speaker_id, point.receiver_id): tuple(float(v) for v in energy[:, s, r])
        for s, channel in enumerate(context.scheme.channel_group.channels) for r, point in enumerate(scheme.receiver_set.points)}
    try:
        result = run_scheme(scheme, capabilities=context.capabilities, directivity=context.directivity,
            quality_targets_path=config_path('quality_targets.toml'), engine_commit=context.engine_commit,
            program_fingerprint=context.program_fingerprint, physics_identity=context.physics_identity,
            run_date=date.today(), fem_energies=injected)
    except ValueError as error:
        return (None, None, None, None, None, None), None, str(error)
    rules = ranking.read_ranking_rules(load_quality_targets(config_path('quality_targets.toml')), scheme.purpose)
    settled = tuple(ranking._settle(item, rules) for item in result.candidate.evaluations)
    values = {item.category.value: None if item.category_cost is None else item.category_cost.value for item in settled}
    categories = _categories(tuple(values.get(name) for name in CATEGORY_NAMES))
    unavailable = [name for name, value in zip(CATEGORY_NAMES, categories, strict=True) if value is None]
    if unavailable:
        evaluations = {item.category.value: item for item in settled}
        reasons = [f"{name}: " + (','.join(code.value for code in evaluations[name].reason_codes)
                   or evaluations[name].state.value if name in evaluations else '沒有評估列') for name in unavailable]
        return categories, None, '；'.join(reasons)
    lines, uncovered, missing = ranking._classify(settled, rules)
    assessment = ranking._Assessment(result.candidate, settled, tuple(lines), tuple(uncovered), tuple(missing),
                                     (), (), ranking.ExternalAcceptance.NOT_CHECKED)
    return categories, ranking._total_cost(assessment), None


def implementation_bound(c: SixCategoryConstants, categories: Categories) -> float | None:
    """照支 A 端到端考卷 Σ w·tol；低頻沿既有相對尺以 1 dB 作近零尺度。"""
    if categories[-1] is None:
        return None
    registry = load_precision_contracts(default_precision_contracts_path())
    tolerances = np.asarray([registry[name].value for name in _CONTRACTS])
    tolerances[-1] *= max(abs(categories[-1]), 1.)
    return float(np.dot(c.category_weights, tolerances))


def _difference(base: float | None, end: float | None) -> float | None:
    return None if base is None or end is None else base - end


def score_comparison(context: ScoringContext, model: ReducedModel, designs: tuple[SurfaceDesign, ...],
                     theta: NDArray[np.float64], full: NDArray[np.complex128],
                     reduced: NDArray[np.complex128] | None, material_scheme: Scheme) -> SixComparison:
    """四路阻抗同一 θ 只建一份供兩種聲壓共用；基準替身只解縮減矩陣，不增加完整解。"""
    c = context.constants
    air, ranges = construction_environment(density_kg_m3=model.density_kg_m3, sound_speed_m_s=model.sound_speed_m_s)
    impedances = _impedance_function(designs, c, air=air, ranges=ranges)(jnp.asarray(theta, dtype=jnp.float64), c)
    flags = jnp.asarray(constant_material_flags(_materials(designs)))
    if reduced is None:
        rows = np.asarray(impedances[0])
        reduced_rows = []
        for index, frequency in enumerate(model.frequencies_hz):
            k2, coefficients = system_coefficients(frequency, dict(zip(model.patch_keys, rows[:, index], strict=True)),
                density_kg_m3=model.density_kg_m3, sound_speed_m_s=model.sound_speed_m_s, patch_keys=model.patch_keys)
            reduced_rows.append(evaluate_reduced(model.matrices, k2, coefficients))
        reduced = np.asarray(reduced_rows)
    def evaluate(pressure: NDArray[np.complex128]) -> SixCategoryArrays:
        return six_category_costs(c, *impedances, jnp.asarray(pressure.transpose(1, 2, 0), dtype=jnp.complex128), flags)
    reduced_arrays, full_arrays = evaluate(reduced), evaluate(full)
    reduced_scores, full_scores = _reading(reduced_arrays, reduced, c), _reading(full_arrays, full, c)
    numpy_categories, numpy_total, unavailable = _numpy_scores(context, material_scheme, full)
    bound = implementation_bound(c, numpy_categories)
    difference = _difference(full_scores.product_total, numpy_total)
    return SixComparison(reduced=reduced_scores, full=full_scores,
        surrogate_difference=_difference(full_scores.objective_total, reduced_scores.objective_total),
        numpy_categories=numpy_categories, numpy_total=numpy_total, numpy_unavailable_reason=unavailable,
        implementation_difference=difference,
        implementation_bound=bound, implementation_over_bound=None if difference is None or bound is None else abs(difference) > bound,
        reduced_crossover=_notes(reduced_arrays), full_crossover=_notes(full_arrays))


def _score_improvement(base: SixScores, end: SixScores) -> SixScores:
    return SixScores(categories=_categories(tuple(_difference(b, e) for b, e in zip(base.categories, end.categories, strict=True))),
        objective_total=_difference(base.objective_total, end.objective_total), product_total=_difference(base.product_total, end.product_total),
        low_frequency_objective=_difference(base.low_frequency_objective, end.low_frequency_objective), proxy_db=_difference(base.proxy_db, end.proxy_db))


def six_improvement(base: SixComparison, end: SixComparison) -> SixImprovement:
    """逐類與兩種總分均為基準減終點，不把 J 與產品定義交叉比較。"""
    return SixImprovement(reduced=_score_improvement(base.reduced, end.reduced), full=_score_improvement(base.full, end.full),
        numpy_categories=_categories(tuple(_difference(b, e) for b, e in zip(base.numpy_categories, end.numpy_categories, strict=True))),
        numpy_total=_difference(base.numpy_total, end.numpy_total))


def score_endpoint(context: ScoringContext, model: ReducedModel, designs: tuple[SurfaceDesign, ...],
                   theta: NDArray[np.float64], full: NDArray[np.complex128], reduced: NDArray[np.complex128],
                   original_designs: tuple[SurfaceDesign, ...], baseline: NDArray[np.complex128], exported: Scheme) -> EndpointSix:
    """兩套既有完整聲壓各評一次三組數，原牆減終點另記；不求完整解。"""
    material_scheme = context.scheme.model_copy(update={'scene': exported.scene})
    end = score_comparison(context, model, designs, theta, full, reduced, material_scheme)
    base = score_comparison(context, model, original_designs, np.empty(0, dtype=np.float64), baseline, None, context.scheme)
    return EndpointSix(endpoint=end, baseline=base, improvement=six_improvement(base, end))
