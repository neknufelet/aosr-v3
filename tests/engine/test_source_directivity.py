"""#505 聲源指向性第一刀：只考模型與逐路徑倍率，不驗求解接線。"""

import inspect
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from pydantic import ValidationError
from scipy.special import j1

from aosr.config.directivity_defaults import DirectivityDefaults, load_directivity_defaults
from aosr.config.frequency_axis import (
    GEOMETRIC_BAND_FREQUENCIES_HZ, GEOMETRIC_LANE_FREQUENCIES_HZ,
    VERIFICATION_REPORT_FREQUENCIES_HZ,
)
from aosr.config.speaker_directivity import DIRECTIVITY_REAR_GAIN_MIN, SPEAKER_PRESETS
from aosr.geometry.shoebox import Point, Room, WALL_SEQUENCE_ORDER
from aosr.physics.room_paths import _one_path, image_source_paths
from aosr.physics.source_directivity import (
    SourceModel, TwoParameterValues, apply_pressure_factor, departure_direction,
    one_minus_cos, speaker_axis, two_parameter_power_ratio, two_parameter_pressure_factor,
    unit_vector, v2_compat_power_ratio, v2_compat_pressure_factor,
)
from tests.engine._precision_contracts import contract_value


_DEFAULTS = Path(__file__).resolve().parents[2] / "src/aosr/config/data/directivity_defaults.toml"
_PARAMS = load_directivity_defaults(_DEFAULTS)
_TOL = contract_value("source_directivity_power_ratio_consistency")


def _fixed(beta: float, floor_db: float) -> TwoParameterValues:
    return TwoParameterValues(beta, floor_db, _PARAMS.allowed_range)


def _manual_values(frequency: float) -> tuple[float, float]:
    curve = _PARAMS.two_parameter
    beta = curve.beta_limit / (1.0 + (curve.beta_corner_hz / frequency) ** curve.beta_exponent)
    floor_db = curve.power_floor_limit_db / (
        1.0 + (curve.power_floor_corner_hz / frequency) ** curve.power_floor_exponent
    )
    return beta, 10.0 ** (floor_db / 10.0)


def _quadrature(axis: tuple[float, ...], params: DirectivityDefaults | TwoParameterValues) -> np.ndarray:
    nodes, weights = leggauss(64)
    result = np.zeros(len(axis))
    for node, weight in zip(nodes, weights, strict=True):
        factor = two_parameter_pressure_factor(1.0 - float(node), axis, params)
        result += float(weight) * factor**2 / 2.0
    return result


def test_default_on_axis_is_bitwise_one() -> None:
    """正前方整條頻率軸逐位為 1。"""
    for axis in (GEOMETRIC_LANE_FREQUENCIES_HZ, GEOMETRIC_BAND_FREQUENCIES_HZ,
                 VERIFICATION_REPORT_FREQUENCIES_HZ):
        values = two_parameter_pressure_factor(0.0, axis, _PARAMS)
        assert np.array_equal(values, np.ones_like(values))
    for beta in (_PARAMS.allowed_range.beta_min, _PARAMS.allowed_range.beta_max):
        for floor_db in (_PARAMS.allowed_range.power_floor_min_db,
                         _PARAMS.allowed_range.power_floor_max_db):
            values = two_parameter_pressure_factor(0.0, GEOMETRIC_LANE_FREQUENCIES_HZ,
                                                   _fixed(beta, floor_db))
            assert np.array_equal(values, np.ones_like(values))


def test_angle_monotonic_and_low_frequency_limit() -> None:
    axis = GEOMETRIC_LANE_FREQUENCIES_HZ
    rear = two_parameter_pressure_factor(2.0, axis, _PARAMS)
    side = two_parameter_pressure_factor(1.0, axis, _PARAMS)
    front30 = two_parameter_pressure_factor(1.0 - math.cos(math.pi / 6), axis, _PARAMS)
    assert np.all(rear <= side) and np.all(side <= front30) and np.all(front30 <= 1.0)
    very_low = two_parameter_pressure_factor(2.0, (1e-8,), _PARAMS)[0]
    assert abs(very_low - 1.0) < 1e-5


def test_300_hz_hand_calculated_points() -> None:
    beta, floor = _manual_values(300.0)
    for x in (1.0, 2.0):
        expected = math.sqrt((1.0 - floor) * math.exp(-2.0 * beta * x) + floor)
        observed = two_parameter_pressure_factor(x, (300.0,), _PARAMS)[0]
        assert math.isclose(observed, expected, rel_tol=1e-14)
    power = (1.0 - floor) * (-math.expm1(-4.0 * beta)) / (4.0 * beta) + floor
    assert math.isclose(two_parameter_power_ratio((300.0,), _PARAMS)[0], power, rel_tol=1e-14)
    assert math.isclose(20.0 * math.log10(math.sqrt((1-floor)*math.exp(-4*beta)+floor)), -2.956, abs_tol=0.01)
    assert math.isclose(20.0 * math.log10(math.sqrt((1-floor)*math.exp(-2*beta)+floor)), -1.917, abs_tol=0.01)
    assert math.isclose(10.0 * math.log10(power), -1.683, abs_tol=0.01)


def test_power_formula_matches_independent_quadrature() -> None:
    axis = GEOMETRIC_LANE_FREQUENCIES_HZ
    expected = _quadrature(axis, _PARAMS)
    observed = two_parameter_power_ratio(axis, _PARAMS)
    assert np.all(np.abs(observed - expected) / expected <= _TOL)
    for beta in (0.0, math.ulp(0.0), 1e-320, 1e-300, 1e-6, _PARAMS.allowed_range.beta_max):
        for floor_db in (_PARAMS.allowed_range.power_floor_max_db, -45.0,
                         _PARAMS.allowed_range.power_floor_min_db):
            params = _fixed(beta, floor_db)
            expected = _quadrature((1000.0,), params)
            observed = two_parameter_power_ratio((1000.0,), params)
            assert np.all(np.abs(observed - expected) / expected <= _TOL)
            assert np.all(observed <= 1.0)


def test_mutant_power_ratio_beyond_tolerance_is_red() -> None:
    """列出的幾種典型錯法會紅；不宣稱能抓到所有可能的錯誤。"""
    beta, floor = _manual_values(3000.0)
    truth = _quadrature((3000.0,), _PARAMS)[0]
    base = -math.expm1(-4.0 * beta) / (4.0 * beta)
    mutants = (
        (1.0 - floor) * base,
        (1.0 - floor) * (-math.expm1(-2.0 * beta)) / (2.0 * beta) + floor,
        (1.0 - math.sqrt(floor)) * base + math.sqrt(floor),
        base + floor,
    )
    assert all(abs(mutant - truth) / truth > _TOL for mutant in mutants)


def test_range_boundaries_and_rejections() -> None:
    for beta in (_PARAMS.allowed_range.beta_min, _PARAMS.allowed_range.beta_max):
        for floor in (_PARAMS.allowed_range.power_floor_min_db,
                      _PARAMS.allowed_range.power_floor_max_db):
            assert np.all(np.isfinite(two_parameter_pressure_factor(1.0, (1000.0,), _fixed(beta, floor))))
    for beta, floor, name in ((-1.0, -45.0, "beta"), (11.0, -45.0, "beta"),
                              (1.0, -81.0, "power_floor_db"), (1.0, 1.0, "power_floor_db")):
        with pytest.raises(ValueError, match=name + ".*1000.0 Hz.*值"):
            two_parameter_pressure_factor(1.0, (1000.0,), _fixed(beta, floor))


def test_loader_rejects_missing_extra_and_missing_path(tmp_path: Path) -> None:
    import tomllib
    source = _DEFAULTS.read_text(encoding="utf-8")
    kept = [line for line in source.splitlines(keepends=True) if not line.startswith("beta_limit = ")]
    assert "".join(kept) != source
    missing = tmp_path / "missing.toml"
    missing.write_text("".join(kept), encoding="utf-8")
    extra = tmp_path / "extra.toml"
    extra.write_text(source + "\n[unlisted]\nextra = 1\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_directivity_defaults(missing)
    with pytest.raises(ValidationError):
        load_directivity_defaults(extra)
    assert inspect.signature(load_directivity_defaults).parameters["path"].default is inspect.Parameter.empty
    assert tomllib.loads(source)["two_parameter"]["beta_limit"] == _PARAMS.two_parameter.beta_limit


def test_curve_limits_and_positive_shape_parameters_are_validated() -> None:
    curve = _PARAMS.two_parameter
    for changes in (
        {"beta_limit": _PARAMS.allowed_range.beta_max + 1.0},
        {"power_floor_limit_db": _PARAMS.allowed_range.power_floor_min_db - 1.0},
        {"beta_corner_hz": 0.0},
        {"power_floor_exponent": float("inf")},
    ):
        with pytest.raises(ValidationError):
            DirectivityDefaults.model_validate({
                "two_parameter": {**curve.model_dump(), **changes},
                "allowed_range": _PARAMS.allowed_range.model_dump(),
                "provenance": _PARAMS.provenance.model_dump(),
            })
    with pytest.raises(ValidationError):
        DirectivityDefaults.model_validate({**_PARAMS.model_dump(), "unlisted": True})


def test_unit_vector_and_axis_reject_zero() -> None:
    with pytest.raises(ValueError, match="零向量"):
        unit_vector((0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="零向量"):
        speaker_axis(Point(1.0, 1.0, 1.0), Point(1.0, 1.0, 1.0))
    direction = speaker_axis(Point(0.5, 1.0, 1.0), Point(1.5, 2.0, 2.0))
    assert one_minus_cos(direction, direction) == 0.0


def test_departure_matches_first_geometric_bounce_including_edges() -> None:
    assert "最後一面" in WALL_SEQUENCE_ORDER
    cases = (
        (Room(2.0, 2.0, 2.0), Point(0.5, 0.5, 1.0), Point(1.5, 1.5, 1.0), (0,-1,0,-1,0,1)),
        (Room(2.0, 2.0, 2.0), Point(0.5, 0.5, 0.5), Point(1.5, 1.5, 1.5), (0,-1,0,-1,0,-1)),
    )
    for room, source, receiver, identity in cases:
        path = _one_path(room, source, receiver, 343.0, identity, None)
        expected_point = path.bounces[-1].point
        expected = unit_vector(tuple(b-a for a,b in zip(source.as_tuple(), expected_point, strict=True)))
        np.testing.assert_allclose(departure_direction(path, receiver), expected, rtol=0, atol=1e-15)
    # #305 的對稱三階房，加一間三軸不對稱、聲源與接收點三個座標都不同的一般房（上下分量不為 0）。
    for room, source, receiver in ((Room(6.0,4.0,3.0), Point(1.0,2.0,1.5), Point(4.0,2.0,1.5)),
                                   (Room(5.3,3.7,2.9), Point(0.9,1.3,1.1), Point(3.8,2.6,1.25))):
        for path in image_source_paths(room, source, receiver, 343.0, max_order=3):
            target = receiver.as_tuple() if path.order == 0 else path.bounces[-1].point
            expected = unit_vector(tuple(b-a for a,b in zip(source.as_tuple(), target, strict=True)))
            np.testing.assert_allclose(departure_direction(path, receiver), expected, rtol=0, atol=1e-14)


def test_departure_keeps_elevation_on_paths_without_floor_or_ceiling() -> None:
    """聲源與接收點不同高時，沒碰地板天花板的路徑（直達、只碰側牆）出發方向的上下分量照幾何算。"""
    room, source, receiver = Room(5.3, 3.7, 2.9), Point(0.9, 1.3, 1.1), Point(3.8, 2.6, 1.25)
    side_only = [path for path in image_source_paths(room, source, receiver, 343.0, max_order=3)
                 if path.identity[4:] == (0, 1)]
    assert side_only
    for path in side_only:
        target = receiver.as_tuple() if path.order == 0 else path.bounces[-1].point
        rise = target[2] - source.z
        assert rise > 0.0
        direction = departure_direction(path, receiver)
        length = math.dist(source.as_tuple(), target)
        assert math.isclose(direction[2], rise / length, rel_tol=1e-12)


def test_three_dimensional_rotational_symmetry() -> None:
    axis = unit_vector((1.0, 0.0, 0.0))
    cosine = math.cos(math.pi / 3)
    sine = math.sin(math.pi / 3)
    directions = ((cosine, sine, 0.0), (cosine, 0.0, sine),
                  (cosine, sine / math.sqrt(2), sine / math.sqrt(2)))
    values = [two_parameter_pressure_factor(one_minus_cos(unit_vector(d), axis),
                                            (1000.0,), _PARAMS)[0] for d in directions]
    np.testing.assert_allclose(values, values[0], rtol=0, atol=1e-15)


def test_mirrored_room_path_factors_are_bitwise_equal() -> None:
    room, receiver, aim = Room(6.0,4.0,3.0), Point(4.0,2.0,1.5), Point(5.0,2.0,1.5)
    left, right = Point(1.0,1.0,1.5), Point(1.0,3.0,1.5)
    left_paths = image_source_paths(room, left, receiver, 343.0, max_order=3)
    right_paths = {path.identity: path for path in image_source_paths(room, right, receiver, 343.0, max_order=3)}
    for path in left_paths:
        nx,sx,ny,sy,nz,sz = path.identity
        mirror_id = (nx,sx, -ny if sy == 1 else 1-ny, sy, nz,sz)
        other = right_paths[mirror_id]
        x_left = one_minus_cos(departure_direction(path, receiver), speaker_axis(left, aim))
        x_right = one_minus_cos(departure_direction(other, receiver), speaker_axis(right, aim))
        assert np.array_equal(two_parameter_pressure_factor(x_left,(1000.0,),_PARAMS),
                              two_parameter_pressure_factor(x_right,(1000.0,),_PARAMS))


def test_apply_pressure_factor_direct_omni_and_reflection() -> None:
    room, source, receiver = Room(4.0,4.0,3.0), Point(1.0,1.0,1.0), Point(3.0,2.0,1.0)
    paths = [replace(path, path_pressure=(1.0+2.0j,)) for path in
             image_source_paths(room, source, receiver, 343.0, max_order=1)]
    same = apply_pressure_factor(paths, receiver, (1000.0,), SourceModel.OMNIDIRECTIONAL)
    assert all(a is b for a,b in zip(paths,same,strict=True))
    axis = speaker_axis(source, receiver)
    adjusted = apply_pressure_factor(paths, receiver, (1000.0,), SourceModel.TWO_PARAMETER,
                                     params=_PARAMS, source=source, aim=receiver)
    assert adjusted[0].path_pressure == paths[0].path_pressure
    for before, after in zip(paths[1:], adjusted[1:], strict=True):
        x = one_minus_cos(departure_direction(before, receiver), axis)
        beta, floor = _manual_values(1000.0)
        expected = math.sqrt((1-floor)*math.exp(-2*beta*x)+floor)
        assert abs(after.path_pressure[0] - before.path_pressure[0] * expected) < 1e-15


def test_v2_compat_known_side_null_is_a_comparison_only() -> None:
    """上一代側面零點附近小於正後方，是對照性質，不是新模型要求。"""
    preset = SPEAKER_PRESETS["bookshelf"]
    for frequency in (1000.0, 3000.0):
        for x in (1.0, 2.0):
            arg = 2*math.pi*frequency*preset.piston_radius_m/343.0*math.sqrt(x*(2-x))
            piston = 1.0 if arg == 0.0 else 2*j1(arg)/arg
            corner = 343.0/(math.pi*preset.width_m)
            rear = DIRECTIVITY_REAR_GAIN_MIN+(1-DIRECTIVITY_REAR_GAIN_MIN)/(1+(frequency/corner)**2)
            expected = piston*(1-(1-rear)*x/2)
            got = v2_compat_pressure_factor(x,(frequency,),preset.width_m,preset.piston_radius_m,343.0)[0]
            assert math.isclose(got,expected,rel_tol=1e-14)
    assert v2_compat_pressure_factor(0.0,(1000.0,),preset.width_m,preset.piston_radius_m,343.0)[0] == 1.0
    side = v2_compat_pressure_factor(1.0,(3200.0,),preset.width_m,preset.piston_radius_m,343.0)[0]
    rear = v2_compat_pressure_factor(2.0,(3200.0,),preset.width_m,preset.piston_radius_m,343.0)[0]
    assert abs(side) < abs(rear)
    assert v2_compat_power_ratio((1000.0,),preset.width_m,preset.piston_radius_m,343.0)[0] > 0.0
