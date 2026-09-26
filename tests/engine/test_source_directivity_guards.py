"""#505 聲源指向性第一刀的守門與獨立對照：1−cosθ 對獨立算法、拒收、相容對照的積分、登記簿驗證。

這一支補的是自動突變抓到的缺口：主考卷的期望值有幾處是呼叫被測函式自己算的
（``one_minus_cos``），拒收與登記簿驗證也沒有考卷。這裡的期望值一律用 ``math``、
``numpy`` 內積或 ``scipy`` 的自適應積分另算，不呼叫被測函式。
"""

import math
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError
from scipy.integrate import quad
from scipy.special import j1

from aosr.config.directivity_defaults import AllowedRange, DirectivityDefaults, load_directivity_defaults
from aosr.config.speaker_directivity import DIRECTIVITY_REAR_GAIN_MIN, SPEAKER_PRESETS
from aosr.geometry.shoebox import Point, Room
from aosr.physics.room_paths import RoomPath, image_source_paths
from aosr.physics.source_directivity import (
    SourceModel, apply_pressure_factor, one_minus_cos, speaker_axis, two_parameter_pressure_factor,
    unit_vector, v2_compat_power_ratio, v2_compat_pressure_factor,
)


_DEFAULTS = Path(__file__).resolve().parents[2] / "src/aosr/config/data/directivity_defaults.toml"
_PARAMS = load_directivity_defaults(_DEFAULTS)
_SPEED = 343.0


def _independent_x(source: Point, target: tuple[float, float, float], axis: tuple[float, float, float]) -> float:
    """幾何直接算的出發方向（聲源指向目標點）跟軸線的 1−cosθ，用內積，只當對照。"""
    delta = np.subtract(target, source.as_tuple())
    return 1.0 - float(np.dot(delta / np.linalg.norm(delta), axis))


def _curve_at(frequency: float) -> tuple[float, float]:
    """登記簿的六個標量照決策紙的曲線另算 β 與功率下限（線性），不呼叫被測函式。"""
    curve = _PARAMS.two_parameter
    beta = curve.beta_limit / (1 + (curve.beta_corner_hz / frequency) ** curve.beta_exponent)
    floor_db = curve.power_floor_limit_db / (1 + (curve.power_floor_corner_hz / frequency) ** curve.power_floor_exponent)
    return beta, 10.0 ** (floor_db / 10.0)


def _v2_reference(x: float, frequency: float, width: float, radius: float) -> float:
    """上一代公式照原始碼另寫一次：活塞項乘前後比，正前方＝1。"""
    argument = 2 * math.pi * frequency * radius / _SPEED * math.sqrt(x * (2 - x))
    piston = 1.0 if argument == 0.0 else 2 * float(j1(argument)) / argument
    rear = DIRECTIVITY_REAR_GAIN_MIN + (1 - DIRECTIVITY_REAR_GAIN_MIN) / (1 + (frequency * math.pi * width / _SPEED) ** 2)
    return piston * (1 - (1 - rear) * x / 2)


def _paths() -> tuple[Point, Point, list[RoomPath]]:
    source, receiver = Point(1.0, 1.0, 1.0), Point(3.0, 2.0, 1.0)
    paths = [replace(path, path_pressure=(1.0 + 2.0j,)) for path in
             image_source_paths(Room(4.0, 4.0, 3.0), source, receiver, _SPEED, max_order=1)]
    return source, receiver, paths


def test_one_minus_cos_matches_independent_cosine() -> None:
    """已知夾角：差平方法對 math 的 1−cos；斜向軸線對 numpy 內積。"""
    for degrees in (15.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0):
        theta = math.radians(degrees)
        observed = one_minus_cos((math.cos(theta), math.sin(theta), 0.0), (1.0, 0.0, 0.0))
        assert math.isclose(observed, 1.0 - math.cos(theta), rel_tol=1e-12)
    axis = unit_vector((1.0, 2.0, 3.0))
    for raw in ((3.0, -1.0, 0.5), (-2.0, 0.25, 4.0), (0.0, -1.0, -1.0)):
        direction = unit_vector(raw)
        assert math.isclose(one_minus_cos(direction, axis), 1.0 - float(np.dot(direction, axis)),
                            rel_tol=1e-12, abs_tol=1e-15)


def test_rear_direction_is_clamped_and_accepted() -> None:
    """反向的兩個單位向量，差平方除 2 會捨入到 2 多一格；要夾回 2，正後方才不會被拒收。"""
    raw = (1.0, 1.0, 1.0)
    forward, backward = unit_vector(raw), unit_vector(tuple(-v for v in raw))
    unclamped = sum((a - b) ** 2 for a, b in zip(forward, backward, strict=True)) / 2.0
    assert unclamped > 2.0
    x = one_minus_cos(forward, backward)
    assert x == 2.0
    beta, floor = _curve_at(1000.0)
    expected = math.sqrt((1 - floor) * math.exp(-4 * beta) + floor)
    assert math.isclose(two_parameter_pressure_factor(x, (1000.0,), _PARAMS)[0], expected, rel_tol=1e-12)


def test_direction_and_axis_inputs_are_rejected() -> None:
    for bad in ((1.0, 0.0), (1.0, 0.0, 0.0, 0.0), (math.nan, 0.0, 1.0), (math.inf, 0.0, 1.0)):
        with pytest.raises(ValueError, match="三個有限數"):
            unit_vector(bad)
    with pytest.raises(ValueError, match="三維"):
        one_minus_cos((1.0, 0.0), (1.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="三維"):
        one_minus_cos((1.0, 0.0, 0.0), (1.0, 0.0))
    for axis in ((0.0,), (-1.0,), (math.nan,), (math.inf,), ((1000.0,),)):
        with pytest.raises(ValueError, match="頻率軸"):
            two_parameter_pressure_factor(1.0, axis, _PARAMS)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="頻率軸"):
            v2_compat_pressure_factor(1.0, axis, 0.2, 0.065, _SPEED)  # type: ignore[arg-type]


def test_x_outside_zero_to_two_is_rejected() -> None:
    """夾角量超出 [0, 2]（含只超一格與 NaN）兩個模型都拒收；邊界本身收下。"""
    just_above = math.nextafter(2.0, 3.0)
    just_below = math.nextafter(0.0, -1.0)
    for x in (just_above, just_below, math.nan, math.inf):
        with pytest.raises(ValueError, match="x 必須在"):
            two_parameter_pressure_factor(x, (1000.0,), _PARAMS)
        with pytest.raises(ValueError, match="x 必須在"):
            v2_compat_pressure_factor(x, (1000.0,), 0.2, 0.065, _SPEED)
    for x in (0.0, 2.0):
        assert np.all(np.isfinite(two_parameter_pressure_factor(x, (1000.0,), _PARAMS)))
        assert np.all(np.isfinite(v2_compat_pressure_factor(x, (1000.0,), 0.2, 0.065, _SPEED)))


def test_v2_compat_dimensions_must_be_positive_and_finite() -> None:
    for width, radius, speed in ((0.0, 0.065, _SPEED), (0.2, -0.065, _SPEED), (0.2, 0.065, math.nan),
                                 (0.2, 0.065, 0.0), (math.inf, 0.065, _SPEED)):
        with pytest.raises(ValueError, match="有限正數"):
            v2_compat_pressure_factor(1.0, (1000.0,), width, radius, speed)


def test_v2_compat_power_ratio_matches_adaptive_integration() -> None:
    """上一代相容對照的功率倍率（64 點高斯–勒讓德）對 scipy 自適應積分。

    容差 1e-11 只檢查「這一版的積分點數在細軸上限內夠不夠」，不是精度契約；已量 64 點最大相對差
    約 1.4e-13，換成 16 點在 8 kHz 就差 2e-7 以上。極低頻時兩項都趨近 1，功率倍率也是 1。
    """
    for preset in SPEAKER_PRESETS.values():
        for frequency in (100.0, 1000.0, 3000.0, 8000.0, 11166.8):
            def integrand(mu: float) -> float:
                return _v2_reference(1.0 - mu, frequency, preset.width_m, preset.piston_radius_m) ** 2
            truth = quad(integrand, -1.0, 1.0, epsabs=0.0, epsrel=1e-13, limit=500)[0] / 2.0
            observed = v2_compat_power_ratio((frequency,), preset.width_m, preset.piston_radius_m, _SPEED)[0]
            assert math.isclose(observed, truth, rel_tol=1e-11)
        low = v2_compat_power_ratio((1e-6,), preset.width_m, preset.piston_radius_m, _SPEED)[0]
        assert math.isclose(low, 1.0, rel_tol=1e-12)


def test_apply_two_parameter_uses_geometric_departure_angle() -> None:
    """逐路徑的倍率對「幾何直接算的出發方向」與手寫公式，不經 one_minus_cos。"""
    source, receiver, paths = _paths()
    axis = speaker_axis(source, Point(3.0, 3.0, 2.0))
    adjusted = apply_pressure_factor(paths, receiver, (1000.0,), SourceModel.TWO_PARAMETER,
                                     params=_PARAMS, speaker_direction=axis)
    beta, floor = _curve_at(1000.0)
    assert any(path.order == 0 for path in paths) and any(path.order > 0 for path in paths)
    for before, after in zip(paths, adjusted, strict=True):
        target = receiver.as_tuple() if before.order == 0 else before.bounces[-1].point
        x = _independent_x(source, target, axis)
        expected = math.sqrt((1 - floor) * math.exp(-2 * beta * x) + floor)
        assert x > 1e-3
        assert after.path_pressure[0] / before.path_pressure[0] == pytest.approx(expected, rel=1e-12)


def test_apply_v2_compat_uses_given_dimensions() -> None:
    """相容對照走 apply 那一支：面板寬、活塞半徑、聲速照呼叫端給的算，對手寫上一代公式。"""
    source, receiver, paths = _paths()
    axis = speaker_axis(source, receiver)
    preset = SPEAKER_PRESETS["floorstanding"]
    adjusted = apply_pressure_factor(paths, receiver, (3000.0,), SourceModel.V2_COMPAT, speaker_direction=axis,
                                     baffle_width_m=preset.width_m, piston_radius_m=preset.piston_radius_m,
                                     sound_speed_m_s=_SPEED)
    assert adjusted[0].path_pressure == paths[0].path_pressure
    for before, after in zip(paths[1:], adjusted[1:], strict=True):
        x = _independent_x(source, before.bounces[-1].point, axis)
        expected = _v2_reference(x, 3000.0, preset.width_m, preset.piston_radius_m)
        assert after.path_pressure[0] / before.path_pressure[0] == pytest.approx(expected, rel=1e-10)


def test_apply_rejects_missing_inputs_before_touching_paths() -> None:
    """非全向缺軸向、兩參數缺 params、相容對照缺任一尺寸：空路徑也拒收；頻率軸與聲壓長度不符拒收。"""
    source, receiver, paths = _paths()
    axis = speaker_axis(source, receiver)
    with pytest.raises(ValueError, match="軸向"):
        apply_pressure_factor([], receiver, (1000.0,), SourceModel.TWO_PARAMETER, params=_PARAMS)
    with pytest.raises(ValueError, match="params"):
        apply_pressure_factor([], receiver, (1000.0,), SourceModel.TWO_PARAMETER, speaker_direction=axis)
    for width, radius, speed in ((None, 0.065, _SPEED), (0.2, None, _SPEED), (0.2, 0.065, None)):
        with pytest.raises(ValueError, match="明給"):
            apply_pressure_factor([], receiver, (1000.0,), SourceModel.V2_COMPAT, speaker_direction=axis,
                                  baffle_width_m=width, piston_radius_m=radius, sound_speed_m_s=speed)
    with pytest.raises(ValueError, match="長度不符"):
        apply_pressure_factor(paths, receiver, (1000.0, 2000.0), SourceModel.TWO_PARAMETER,
                              params=_PARAMS, speaker_direction=axis)


def _document(**changes: Mapping[str, object]) -> dict[str, object]:
    base = _PARAMS.model_dump()
    return {section: {**values, **changes.get(section, {})} for section, values in base.items()}


def test_registry_rejects_unphysical_or_inverted_allowed_range() -> None:
    for section in ({"beta_min": -0.1}, {"power_floor_max_db": 0.5},
                    {"beta_min": 4.0, "beta_max": 3.9}, {"power_floor_min_db": -10.0, "power_floor_max_db": -20.0}):
        with pytest.raises(ValidationError):
            DirectivityDefaults.model_validate(_document(allowed_range=section))
    equal = AllowedRange(beta_min=3.2, beta_max=3.2, power_floor_min_db=-45.0, power_floor_max_db=-45.0)
    assert equal.beta_min == equal.beta_max and equal.power_floor_min_db == equal.power_floor_max_db


def test_registry_curve_limits_on_the_bounds_are_accepted() -> None:
    bounds = _PARAMS.allowed_range
    for curve in ({"beta_limit": bounds.beta_max}, {"beta_limit": bounds.beta_min},
                  {"power_floor_limit_db": bounds.power_floor_min_db},
                  {"power_floor_limit_db": bounds.power_floor_max_db}):
        loaded = DirectivityDefaults.model_validate(_document(two_parameter=curve))
        assert loaded.two_parameter.model_dump() == {**_PARAMS.two_parameter.model_dump(), **curve}


def test_registry_shape_parameters_must_be_strictly_positive() -> None:
    for field in ("beta_corner_hz", "beta_exponent", "power_floor_corner_hz", "power_floor_exponent"):
        with pytest.raises(ValidationError):
            DirectivityDefaults.model_validate(_document(two_parameter={field: 0.0}))
        small = DirectivityDefaults.model_validate(_document(two_parameter={field: 1e-3}))
        assert getattr(small.two_parameter, field) == 1e-3


def test_registry_rejects_booleans_and_uncontrolled_provenance() -> None:
    """TOML 的 true 不准被讀成 1；出處狀態與來源種類只收受控字。"""
    for section, change in (("two_parameter", {"beta_limit": True}), ("allowed_range", {"beta_max": True}),
                            ("provenance", {"status": "calibrated"}),
                            ("provenance", {"source_kind": "product_choice"}),
                            ("provenance", {"source": ""})):
        with pytest.raises(ValidationError):
            DirectivityDefaults.model_validate(_document(**{section: change}))
