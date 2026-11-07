"""冷子行程量三類；只回原始量測，契約與中央差分由父考卷判。"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from aosr.config.quality_targets import load_quality_targets
from aosr.scoring import timbre_cost_jax as costs
from aosr.scoring.channel_group import ChannelComparison, ChannelDefinition, ChannelGroup
from aosr.scoring.receiver_set import ReceiverRole, ReceiverSet
from aosr.scoring.channel_matching_settings import _load_settings
from aosr.scoring.timbre_jax import response_constants
from tests.engine._timbre_cost_jax_cases import Case, PURPOSE, curves, numpy_evaluations, numpy_metrics, small_room

_COMPILED = jax.jit(costs.cost_arrays)


def constants(case: Case) -> costs.CostConstants:
    return costs.cost_constants(case.frequencies, case.receivers, case.group, quality_targets_path=case.targets,
                                purpose=PURPOSE, sound_speed_m_s=case.sound_speed)


def _metric_array(result: costs.CostArrays) -> np.ndarray:
    response = result.responses
    return np.asarray(jnp.stack((response.tilt_db_per_octave, response.residual_rms_db,
                                 response.matching_rms_db, response.broadband_level_db), axis=-1))


def _record(case: Case) -> dict[str, object]:
    fixed = constants(case)
    energy, distance = jnp.asarray(case.energies), jnp.asarray(case.distances)
    eager = costs.cost_arrays(fixed, energy, distance)
    compiled = _COMPILED(fixed, energy, distance)
    single = _COMPILED(fixed, energy.astype(jnp.float32), distance.astype(jnp.float32))
    batch = jax.jit(jax.vmap(costs.cost_arrays, in_axes=(None, 0, None)))(fixed, jnp.stack((energy, energy * 1.7)), distance)
    reference = numpy_evaluations(case, case.energies)
    area, matching = reference[1].category_cost, reference[2].category_cost
    assert area is not None and matching is not None
    area_names = [f"{metric}_{suffix}" for suffix in ("weighted_mean_deviation", "worst_deviation")
                  for metric in ("tilt", "ripple_rms", "overall_level")]
    area_reference = [[area.components[f"{channel.role}.{name}"] for name in area_names] for channel in case.group.channels]
    matching_reference = [[matching.components.get(f"{metric}.{suffix}", 0.0)
                            for metric in ("tilt_difference", "ripple_rms_difference", "broadband_level_difference", "direct_time_difference")]
                           for suffix in ("weighted_mean", "worst")]
    if _load_settings(case.targets, PURPOSE).direct_time_cost_enabled:
        for row, suffix in enumerate(("weighted_mean", "worst")):
            assert matching.components[f"direct_time_difference.{suffix}"] > 0.0
            assert compiled.matching_components[row, -1] > 0.0
    return {"python": [item.category_cost.value for item in reference if item.category_cost is not None], "eager": np.asarray(eager.category_costs).tolist(),
            "jit": np.asarray(compiled.category_costs).tolist(), "single": np.asarray(single.category_costs).tolist(),
            "batch": np.asarray(batch.category_costs).tolist(), "metrics_python": numpy_metrics(case, case.energies).tolist(),
            "metrics64": _metric_array(compiled).tolist(), "metrics32": _metric_array(single).tolist(),
            "double_dtypes": [str(value.dtype) for value in compiled[:5]],
            "single_dtypes": [str(value.dtype) for value in single[:5]],
            "listening": np.asarray(compiled.listening_components).tolist(),
            "matching": np.asarray(compiled.matching_components).tolist(), "listening_python": area_reference,
            "matching_python": matching_reference, "peer_group_missing": bool(compiled.flags.peer_group_missing)}


def _cases(directory: Path) -> dict[str, Case]:
    base = curves()
    seat_order = [2, 0, 1]
    reordered = replace(base, receivers=ReceiverSet(points=tuple(base.receivers.points[i] for i in seat_order)),
                        energies=base.energies[::-1, seat_order], distances=base.distances[::-1, seat_order],
                        group=base.group.model_copy(update={"channels": base.group.channels[::-1]}))
    group = ChannelGroup(channels=(*base.group.channels, ChannelDefinition(role="center", speaker_id="center")),
                         comparisons=(*base.group.comparisons, ChannelComparison(left_role="center", right_role="left")),
                         feature_match_tolerance_hz=base.group.feature_match_tolerance_hz)
    multiple = replace(base, group=group, energies=np.concatenate((base.energies, (base.energies[:1] ** 1.4) * 1.7)),
                       distances=np.concatenate((base.distances, base.distances[:1] + 1.0)))
    text = base.targets.read_text().replace('key = "channel_matching.direct_time_cost_enabled"\nvalue = 0',
                                          'key = "channel_matching.direct_time_cost_enabled"\nvalue = 1')
    changed = directory / "time-enabled.toml"
    changed.write_text(text)
    assert _load_settings(changed, PURPOSE).direct_time_cost_enabled
    extra = base.receivers.points[-1].model_copy(update={"receiver_id": "side", "importance": 1.5,
                                                        "position_m": (1.0, 2.2, 1.2)})
    seats = list(base.receivers.points) + [extra]
    seats[2] = seats[2].model_copy(update={"importance": 0.0})
    zero = replace(base, receivers=ReceiverSet(points=tuple(seats)),
                   energies=np.concatenate((base.energies, base.energies[:, :1] * 0.7), axis=1),
                   distances=np.concatenate((base.distances, base.distances[:, :1]), axis=1))
    other = extra.model_copy(update={"role": ReceiverRole.OTHER_SEAT})
    other_energy = np.stack((base.energies[0, 0], base.energies[1, 2] ** 1.4 * 1.7))[:, None, :]
    return {"curves": base, "small_room": small_room(), "reordered": reordered, "multiple_pairs": multiple,
            "time_enabled": replace(base, targets=changed), "zero_weight_worst": zero,
            "other_seat": replace(zero, receivers=ReceiverSet(points=(*base.receivers.points, other)),
                                  energies=np.concatenate((base.energies, other_energy), axis=1))}


def _ignored_seat_records(case: Case) -> dict[str, dict[str, object]]:
    """只破壞 OTHER_SEAT；三類仍須對上 numpy，距離旗標不得誤報。"""
    fixed = constants(case)
    records = {}
    # numpy 的輸入模型先拒絕任何壞能量；裁判用未破壞、同樣忽略 OTHER_SEAT 的有效案例。
    reference = numpy_evaluations(case, case.energies)
    assert all(item.category_cost is not None for item in reference), reference

    def objective(values: jax.Array, distances: jax.Array) -> jax.Array:
        return jnp.sum(costs.cost_arrays(fixed, values, distances).category_costs)

    differentiate = jax.jit(jax.grad(objective))
    for name, bad in (("energy_zero", 0.0), ("energy_negative", -1.0), ("energy_nan", np.nan), ("energy_inf", np.inf),
                      ("distance_negative", -1.0), ("distance_nan", np.nan), ("distance_inf", np.inf)):
        energy, distances = case.energies.copy(), case.distances.copy()
        if name.startswith("energy"):
            energy[0, -1, 0] = bad
        else:
            distances[0, -1] = bad
        result = _COMPILED(fixed, jnp.asarray(energy), jnp.asarray(distances))

        gradient = differentiate(jnp.asarray(energy), jnp.asarray(distances))
        records[name] = {"jit": np.asarray(result.category_costs).tolist(),
                         "python": [item.category_cost.value for item in reference if item.category_cost is not None],
                         "distance_flag_clear": not bool(jnp.any(result.flags.invalid_direct_distance)),
                         "gradient_finite": bool(jnp.isfinite(gradient).all()),
                         "ignored_gradient_zero": bool(jnp.all(gradient[:, -1] == 0.0))}
    return records


def _invalid_category_records(case: Case) -> dict[str, dict[str, bool]]:
    """主位及單一周圍曲線無效；逐類查 NaN，另一聲道仍有有限非零梯度。"""
    fixed = constants(case)
    records = {}

    def valid_term(values: jax.Array, good_channel: jax.Array) -> jax.Array:
        good = costs.cost_arrays(fixed, values, jnp.asarray(case.distances))
        return jnp.sum(good.timbre_components[good_channel])

    differentiate = jax.jit(jax.grad(valid_term))
    for category, seat in (("timbre", 0), ("listening", 1)):
        for channel in range(len(case.group.channels)):
            energy = case.energies.copy()
            energy[channel, seat, 0] = np.nan
            result = _COMPILED(fixed, jnp.asarray(energy), jnp.asarray(case.distances))
            expected = np.zeros(energy.shape[:2], dtype=bool)
            expected[channel, seat] = True

            good_channel = jnp.asarray(1 - channel)
            gradient = differentiate(jnp.asarray(energy), good_channel)
            records[f"{category}_{channel}"] = {
                "nan": bool(jnp.isnan(result.category_costs[0 if category == "timbre" else 1])),
                "flag_exact": bool(np.array_equal(result.responses.flags.non_positive_energy, expected)),
                "valid_curves_exact": bool(np.array_equal(result.responses.valid, ~expected)),
                "other_finite": bool(jnp.isfinite(valid_term(jnp.asarray(energy), good_channel))),
                "gradient_finite": bool(jnp.isfinite(gradient).all()),
                "gradient_nonzero": bool(jnp.any(gradient != 0.0)),
            }
    return records


def _gradients(case: Case) -> dict[str, object]:
    fixed = constants(case)
    distance = jnp.asarray(case.distances)

    def objective(energy: jax.Array) -> jax.Array:
        return costs.cost_arrays(fixed, energy, distance).category_costs

    jacobian = jax.jit(jax.jacrev(objective))(jnp.asarray(case.energies))
    single = jax.jit(jax.jacrev(objective))(jnp.asarray(case.energies, dtype=jnp.float32))
    return {"double": np.asarray(jacobian).tolist(), "single_finite": bool(jnp.isfinite(single).all())}


def _flag_record(fixed: costs.CostConstants, energy: np.ndarray, distances: np.ndarray, name: str) -> dict[str, bool]:
    result = _COMPILED(fixed, jnp.asarray(energy), jnp.asarray(distances))
    response = result.responses
    chosen = {
        "energy_zero": response.flags.non_positive_energy,
        "energy_negative": response.flags.non_positive_energy,
        "energy_nan": response.flags.non_positive_energy,
        "energy_inf": response.flags.non_positive_energy,
        "coverage": response.flags.insufficient_coverage,
        "gap": response.flags.scoring_range_gap,
        "matching_coverage": response.flags.matching_coverage,
        "numerical_failure": response.flags.numerical_failure,
        "zero_primary_importance": result.flags.zero_pair_importance,
        "zero_peer_importance": result.flags.zero_pair_importance,
        "invalid_distance": result.flags.invalid_direct_distance,
        "no_comparisons": result.flags.matching_unavailable,
        "invalid_sound_speed": result.flags.invalid_sound_speed,
        "zero_matching_weight": result.flags.zero_matching_weight,
        "zero_matching_importance": result.flags.zero_matching_importance,
        "no_broadband_support": response.flags.broadband_coverage,
    }[name]
    nan_value = result.category_costs[1 if name.startswith("zero_p") else 2]
    if name in {"coverage", "gap", "numerical_failure"}:
        nan_value = response.residual_rms_db[0, 1]
    if name == "no_broadband_support":
        nan_value = response.broadband_level_db[0, 1]

    def valid_term(values: jax.Array) -> jax.Array:
        result = costs.cost_arrays(fixed, values, jnp.asarray(distances))
        return result.responses.broadband_level_db[1, 0] if name in {"coverage", "gap"} else result.category_costs[0]

    gradient = jax.jit(jax.grad(valid_term))(jnp.asarray(energy))
    record = {"flag": bool(jnp.any(chosen)), "nan": bool(jnp.isnan(nan_value)),
              "other_finite": bool(jnp.isfinite(valid_term(jnp.asarray(energy)))),
              "gradient_finite": bool(jnp.isfinite(gradient).all()), "gradient_nonzero": bool(jnp.any(gradient != 0.0))}
    if name.startswith("energy_") or name in {"numerical_failure", "invalid_distance"}:
        expected = np.zeros(energy.shape[:2], dtype=bool)
        expected[0, 1] = True
        record["flag_exact"] = bool(np.array_equal(chosen, expected))
    if name == "no_broadband_support":
        record.update(listening_flag=bool(jnp.all(result.flags.listening_unavailable)),
                      listening_nan=bool(jnp.isnan(result.category_costs[1])),
                      matching_nan=bool(jnp.isnan(result.category_costs[2])))
    return record


def _flags(case: Case) -> dict[str, dict[str, bool]]:
    fixed = constants(case)
    records = {}
    for name, bad in (("energy_zero", 0.0), ("energy_negative", -1.0), ("energy_nan", np.nan), ("energy_inf", np.inf)):
        energy = case.energies.copy()
        energy[0, 1, 0] = bad
        records[name] = _flag_record(fixed, energy, case.distances, name)
    for name, mask in (("coverage", np.arange(len(case.frequencies)) < 6),
                       ("gap", ~((case.frequencies > 500.0) & (case.frequencies < 1000.0)))):
        changed_case = replace(case, frequencies=case.frequencies[mask], energies=case.energies[..., mask])
        records[name] = _flag_record(constants(changed_case), changed_case.energies, case.distances, name)
    settings = _load_settings(case.targets, PURPOSE)
    missing_low = (settings.ripple_range_hz[0] / 2.0, settings.ripple_range_hz[1])
    response = response_constants(case.frequencies, settings.timbre, matching_range_hz=missing_low,
                                  broadband_range_hz=settings.broadband_range_hz)
    records["matching_coverage"] = _flag_record(fixed._replace(responses=response), case.energies, case.distances, "matching_coverage")
    response = response_constants(case.frequencies, settings.timbre, matching_range_hz=settings.ripple_range_hz,
                                  broadband_range_hz=(case.frequencies[-1] * 2.0, case.frequencies[-1] * 4.0))
    records["no_broadband_support"] = _flag_record(fixed._replace(responses=response), case.energies, case.distances, "no_broadband_support")
    tiny = case.energies.copy()
    tiny[0, 1] = np.nextafter(0.0, 1.0)
    tiny[0, 1, 0] = np.finfo(np.float64).max
    records["numerical_failure"] = _flag_record(fixed, tiny, case.distances, "numerical_failure")
    for name, importance in (("zero_primary_importance", (1.0, 0.0, 0.0)), ("zero_peer_importance", (1.0, 2.0, 0.0)),
                             ("zero_matching_importance", (0.0, 0.0, 0.0))):
        points = tuple(point.model_copy(update={"importance": weight})
                       for point, weight in zip(case.receivers.points, importance, strict=True))
        changed = constants(replace(case, receivers=ReceiverSet(points=points)))
        records[name] = _flag_record(changed, case.energies, case.distances, name)
    distances = case.distances.copy()
    distances[0, 1] = np.nan
    records["invalid_distance"] = _flag_record(fixed, case.energies, distances, "invalid_distance")
    for name, changed in (("no_comparisons", fixed._replace(channel_pairs=np.empty((0, 2), dtype=np.int64))),
                          ("invalid_sound_speed", fixed._replace(sound_speed_m_s=np.asarray(0.0))),
                          ("zero_matching_weight", fixed._replace(matching_weights=np.zeros_like(fixed.matching_weights)))):
        records[name] = _flag_record(changed, case.energies, case.distances, name)
    return records


def _bands(case: Case) -> dict[str, object]:
    entries = tuple(entry for entry in load_quality_targets(case.targets).purpose(PURPOSE).target
                    if entry.cost_shape == "in_range_best" and entry.key.startswith(("timbre_balance.", "listening_area_stability.", "channel_matching.")))
    fixed = costs.shape_constants(entries)
    center = jnp.asarray(fixed.center)
    tolerance = jnp.asarray(fixed.tolerance)
    inside = center + tolerance * 0.5
    boundary = center + tolerance
    outside = center + tolerance + jnp.asarray(fixed.worse_reference)

    def objective(values: jax.Array) -> jax.Array:
        return jnp.sum(costs.shape_cost(fixed, values))

    gradient = jax.jit(jax.grad(objective))
    return {"inside_cost": np.asarray(costs.shape_cost(fixed, inside)).tolist(),
            "inside_gradient": np.asarray(gradient(inside)).tolist(), "boundary_gradient": np.asarray(gradient(boundary)).tolist(),
            "outside_gradient": np.asarray(gradient(outside)).tolist(), "outside_values": np.asarray(outside).tolist(),
            "keys": [entry.key for entry in entries]}


def _shape_records(case: Case) -> dict[str, dict[str, bool]]:
    fixed = constants(case)
    energy, distance = jnp.asarray(case.energies), jnp.asarray(case.distances)
    records = {}
    for name in ("channel", "seat", "frequency", "distance"):
        def inputs(values: jax.Array) -> tuple[jax.Array, jax.Array]:
            if name == "channel":
                return values[:1], distance[:1]
            if name == "seat":
                return values[:, :1], distance[:, :1]
            if name == "frequency":
                return values[..., :-1], distance
            return values, distance[:1]

        bad_energy, bad_distance = inputs(energy)
        result = _COMPILED(fixed, bad_energy, bad_distance)

        def objective(values: jax.Array) -> jax.Array:
            invalid = costs.cost_arrays(fixed, *inputs(values)).category_costs[0]
            good = costs.cost_arrays(fixed, values, distance).category_costs[0]
            return good + jnp.where(jnp.isfinite(invalid), invalid, 0.0)

        gradient = jax.jit(jax.grad(objective))(energy)
        records[name] = {"flag": bool(result.flags.invalid_input_shape),
                         "nan": bool(jnp.isnan(result.category_costs).all()),
                         "response_nan": bool(jnp.isnan(_metric_array(result)).all()),
                         "gradient_finite": bool(jnp.isfinite(gradient).all()),
                         "gradient_nonzero": bool(jnp.any(gradient != 0.0))}
    return records


def _mutants(case: Case) -> dict[str, list[float]]:
    fixed = constants(case)
    responses = fixed.responses
    mutants = {
        "smoothing": fixed._replace(responses=responses._replace(tilt_smoothing=np.eye(len(case.frequencies)))),
        "fit_weights": fixed._replace(responses=responses._replace(fit_weights=(responses.fit_weights > 0.0).astype(float))),
        "ripple_range": fixed._replace(responses=responses._replace(matching_weights=responses.ripple_weights)),
        "broadband": fixed._replace(responses=responses._replace(broadband_weights=np.ones_like(responses.broadband_weights))),
        "category_weights": fixed._replace(timbre_weights=fixed.timbre_weights * 2.0, listening_weights=fixed.listening_weights * 2.0),
        "matching_scale": fixed._replace(matching_shapes=fixed.matching_shapes._replace(worse_reference=fixed.matching_shapes.worse_reference * 2.0)),
    }
    return {name: np.asarray(_COMPILED(changed, jnp.asarray(case.energies), jnp.asarray(case.distances)).category_costs).tolist()
            for name, changed in mutants.items()}


def main() -> None:
    directory = Path(sys.argv[1]).parent
    cases = _cases(directory)
    data = {"x64": bool(getattr(jax.config, "x64_enabled")), "cases": {name: _record(case) for name, case in cases.items()},
            "gradient": _gradients(cases["curves"]), "flags": _flags(cases["curves"]),
            "ignored_seat": _ignored_seat_records(cases["other_seat"]),
            "invalid_categories": _invalid_category_records(cases["curves"]),
            "bands": _bands(cases["curves"]), "mutants": _mutants(cases["small_room"]),
            "curves_mutants": _mutants(cases["curves"]), "shapes": _shape_records(cases["curves"])}
    Path(sys.argv[1]).write_text(json.dumps(data))


if __name__ == "__main__":
    main()
