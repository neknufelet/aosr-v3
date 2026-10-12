"""鏡像法逐路徑構造阻抗：純量逐位契約與四個讀者接線。"""
from __future__ import annotations

from aosr.physics import three_lane_report_geometry as report_geometry

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pytest

import aosr
from aosr.geometry.shoebox import Point, Room, Wall, wall_count_signature
from aosr.materials.construction_models import (
    AirLayer, Assembly, BoardLayer, ConstantImpedance, SurfaceMaterial)
from aosr.physics.amplitude import reflection_coefficient
from aosr.runtime import child_process_env, configure_jax


def test_vector_coefficient_matches_python_complex_bits() -> None:
    """抓 numpy 直接除法、Smith 分支、零號或乘法重排造成最後一位不同。"""
    assert importlib.util.find_spec("aosr.physics.wall_reflections") is not None
    from aosr.physics.wall_reflections import reflection_coefficient_row
    rng = np.random.default_rng(309)
    z = rng.uniform(0., 30000., 800) + 1j * rng.uniform(-1e6, 1e6, 800)
    z = np.concatenate((z, np.array([complex(0., -0.), complex(411.6, 0.), complex(50., -0.)])))
    for cos in (0., .0001, .25, .8, 1.):
        actual = reflection_coefficient_row(z, cos, 411.6)
        expected = np.array([reflection_coefficient(complex(value), cos, 411.6) for value in z])
        assert actual.tobytes() == expected.tobytes()
    # 實部加 ρc 恰等於虛部絕對值時兩個分支都走得通；CPython 走「實部大於等於」那支，純虛的那格零號會不同。
    ties = np.array([complex(0., -400.), complex(0., 400.), complex(100., 500.), complex(100., -500.)])
    expected = np.array([reflection_coefficient(complex(value), 1., 400.) for value in ties])
    assert reflection_coefficient_row(ties, 1., 400.).tobytes() == expected.tobytes()


_CPU_PIN_VARS = ("NPY_DISABLE_CPU_FEATURES", "OPENBLAS_CORETYPE", "XLA_FLAGS")


def _child(directory: Path, mode: str) -> dict[str, str]:
    env = child_process_env(threads=1).copy()
    env.update({name: os.environ[name] for name in _CPU_PIN_VARS if name in os.environ})
    configure_jax(platforms="cpu", enable_x64=True, env=env)
    env["TMPDIR"], env["PYTHONDONTWRITEBYTECODE"] = str(directory), "1"
    # 保留 runtime 給的來源路徑，只追加考卷根，不蓋掉被測程式來源。
    root = Path(__file__).resolve().parents[2]
    env["PYTHONPATH"] = os.pathsep.join((env["PYTHONPATH"], str(root)))
    probe = Path(__file__).with_name("_wall_reflections_probe.py")
    output = directory / "result.json"
    result = subprocess.run([sys.executable, str(probe), mode, str(aosr.__file__), str(output)], cwd=directory,
        env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    return cast(dict[str, str], json.loads(output.read_text()))


# 子行程每一段檢查各抓什麼；名字要跟探針的 modes 一致。
_SUITE = {
    "direct": "每路每面角度或 Z 錯誤、批次求值或反射次方漂移（含兩面構造在同一軸）",
    "readers": "細密軸、路徑表、時間窗的反射來源分岔",
    "both-lanes": "兩軸總入口把覆寫只交給細軸或只交給密軸",
    "override-inputs": "讀者入口用 ρc÷c 反推密度、或把覆寫裡的固定牆換回入口的固定阻抗",
    "normal": "散射與篩檢的垂直入射列誤用路徑角、篩檢把逐頻值排成常數、垂直入射列不等於核心直接算的",
    "outputs": "構造接線未改變輸出、忽略空腔參數",
    "reuse": "混合房只打固定牆時失去捷徑或核心按每路重算",
    "constant": "固定牆的細密軸、路徑表、時間窗、篩檢觸碰 JAX",
    "batch": "內部構造覆寫沒有送到逐對鏡像法、污染有限元素或晚期家族的材料",
    "zero-angle": "植錯：全部路徑用 0° 要被抓到",
    "wrong-axis": "植錯：入射角取錯軸要被抓到",
    "scattering-angle": "植錯：房間散射用路徑角要被抓到",
    "screen-angle": "植錯：篩檢用路徑角要被抓到",
}


def test_construction_reflection_suite(tmp_path: Path) -> None:
    """子行程一次跑完全部檢查。

    分成很多題共用模組夾具時，平行工人每個都會各跑一次子行程（實測 11 份），所以併成一題。
    """
    results = _child(tmp_path, "suite")
    assert set(results) == set(_SUITE)
    failures = {name: message for name, message in results.items() if message}
    assert not failures, "\n\n".join(f"[{name}] 抓的是：{_SUITE[name]}\n{message}" for name, message in failures.items())


def test_legacy_rho_c_is_not_rounded_through_density() -> None:
    """抓舊呼叫只給ρc時，先除聲速再乘回來損失一位而改掉固定牆結果。"""
    from aosr.physics import geometric_lane, three_lane_report
    from aosr.physics.report_source import SourceModelKind, SourceModelSpec
    rho_c, c = float.fromhex("0x1.f5a1b50ebc70fp+8"), float.fromhex("0x1.595fdff9071a1p+8")
    walls = {w: 500. + i * 173. for i, w in enumerate(Wall.all())}
    model, room = SourceModelSpec(kind=SourceModelKind.OMNIDIRECTIONAL), Room(6., 4., 3.)
    source, receiver, axis = Point(1.1, .9, 1.2), Point(3.7, 2.1, 1.45), (125., 503., 2017.)
    actual = report_geometry._solve_geometric_report_lane(source_model=model, room=room, source=source,
        receiver=receiver, sound_speed_m_s=c, rho_c_pa_s_per_m=rho_c, frequencies_hz=axis,
        reflection_order_k=3, wall_impedances=walls, scattering_by_wall=None)
    expected = geometric_lane.solve_geometric_lane(source_model=model, room=room, source=source,
        receiver=receiver, sound_speed_m_s=c, rho_c_pa_s_per_m=rho_c, frequencies_hz=axis,
        reflection_order_k=3, impedance_by_wall={w.wall_name(): complex(z) for w, z in walls.items()})
    assert actual == expected


def test_construction_override_requires_scene_density() -> None:
    """抓構造覆寫沒帶場景密度時悄悄從 ρc 反推（構造的 Z 會因此差最後一位）。"""
    from aosr.physics.wall_reflections import lane_reflection_sources
    walls = {wall: 900. + 173. * i for i, wall in enumerate(Wall.all())}
    board = Assembly(layers=(BoardLayer(surface_mass_kg_m2=.15, resistance_pa_s_m=80.), AirLayer(thickness_m=.43)),
                     backing="rigid")
    materials: dict[Wall, SurfaceMaterial] = {
        **{wall: ConstantImpedance(impedance_pa_s_m=z) for wall, z in walls.items()}, Wall.X0: board}
    with pytest.raises(ValueError, match="場景密度"):
        lane_reflection_sources(walls, materials, (125., 503.), rho_c_pa_s_per_m=1.19 * 347.,
                                sound_speed_m_s=347., density_kg_m3=None)


def test_bounce_power_matches_python_zero_signs() -> None:
    """抓同面反彈次方改回 np.power：二次方不乘 1+0j，負實數 R 的虛部零號會跟純量不同。"""
    from aosr.physics.wall_reflections import _power
    values = [complex(a, b) for a in (.5, -.5, -0., 5e-324, -5e-324) for b in (0., -0., .25, -.25, 5e-324)]
    for count in (2, 3, 4, 5, 6):
        expected = np.array([value ** count for value in values])
        assert _power(np.array(values), count).tobytes() == expected.tobytes()


def _board_room() -> tuple[Room, Point, Point, dict[Wall, SurfaceMaterial]]:
    board = Assembly(layers=(BoardLayer(surface_mass_kg_m2=.15, resistance_pa_s_m=80.), AirLayer(thickness_m=.43)),
                     backing="rigid")
    materials: dict[Wall, SurfaceMaterial] = {
        **{wall: ConstantImpedance(impedance_pa_s_m=900. + 173. * i) for i, wall in enumerate(Wall.all())},
        Wall.X0: board}
    return Room(6., 4., 3.), Point(1.1, .9, 1.2), Point(3.7, 2.1, 1.45), materials


def test_path_bounce_powers_go_through_python_power(monkeypatch: pytest.MonkeyPatch) -> None:
    """抓整條路徑的同面次方改回 np.power：乘積對得上純量，而且每個兩次以上的反彈都走 Python 次方那支。

    正負零的差別會在連乘時被吃掉，光比乘積看不出來，所以另外記下次方那支有沒有被叫到。
    """
    from aosr.physics import amplitude, wall_reflections
    from aosr.physics.room_paths import image_source_paths
    room, source, receiver, materials = _board_room()
    axis = (125., 503.)
    sources = wall_reflections.wall_reflection_sources(materials, axis, density_kg_m3=1.19, sound_speed_m_s=347.)
    monkeypatch.setattr(wall_reflections.WallReflectionSources, "_evaluate_wall",
        lambda self, wall, material, cosines: {cos: np.full(len(axis), complex(300., 0.)) for cos in cosines})
    seen: list[int] = []
    original = wall_reflections._power

    def recording(r: np.ndarray, count: int) -> np.ndarray:
        seen.append(count)
        return original(r, count)

    monkeypatch.setattr(wall_reflections, "_power", recording)
    paths = sources.apply(image_source_paths(room, source, receiver, 347., max_order=3), receiver.as_tuple(), 347.)
    scalar = amplitude.Materials(sources.fixed_materials.rho_c, axis, {wall.wall_name(): (
        complex(300., 0.) if wall is Wall.X0 else complex(cast(ConstantImpedance, m).impedance_pa_s_m),) * len(axis)
        for wall, m in materials.items()})
    expected_powers = []
    for path in paths:
        counts = wall_count_signature(path.identity)
        expected = amplitude.reflection_product(scalar, path.dist_m, receiver.as_tuple(), path.image, counts)
        assert np.array(path.reflection_product).tobytes() == np.array(expected).tobytes(), path.identity
        if counts["x0"]:
            expected_powers += [count for wall in amplitude.CANONICAL_WALLS if (count := counts[wall]) > 1]
    assert seen == expected_powers and seen


def test_overflowing_construction_reflection_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """抓阻抗大到溢位時反射係數變成非有限數，卻照樣流進路徑壓力與能量結果。"""
    from aosr.physics import wall_reflections
    from aosr.physics.room_paths import image_source_paths
    room, source, receiver, materials = _board_room()
    axis = (125., 503.)
    sources = wall_reflections.wall_reflection_sources(materials, axis, density_kg_m3=1.19, sound_speed_m_s=347.)
    monkeypatch.setattr(wall_reflections.WallReflectionSources, "_evaluate_wall",
        lambda self, wall, material, cosines: {cos: np.full(len(axis), complex(1e308, 1e308)) for cos in cosines})
    paths = image_source_paths(room, source, receiver, 347., max_order=1)
    with np.errstate(over="ignore", invalid="ignore"), pytest.raises(ValueError, match="x0 在 125.0 Hz.*不是有限數"):
        sources.apply(paths, receiver.as_tuple(), 347.)


def test_reflection_sources_must_share_the_lane_air() -> None:
    """抓反射來源的聲速、ρc 或頻率軸跟幾何路不同時照算（構造 Z、相位、房間散射各用一組空氣或兩條軸）。"""
    from aosr.physics import geometric_lane, wall_reflections
    from aosr.physics.report_source import SourceModelKind, SourceModelSpec
    from aosr.physics.room_paths import image_source_paths
    room, source, receiver, _materials = _board_room()
    axis, fixed = (125., 503.), {wall: ConstantImpedance(impedance_pa_s_m=900. + 173. * i)
                                 for i, wall in enumerate(Wall.all())}
    sources = wall_reflections.wall_reflection_sources(fixed, axis, density_kg_m3=1.19, sound_speed_m_s=347.)
    with pytest.raises(ValueError, match="聲速"):
        image_source_paths(room, source, receiver, 343., max_order=1, reflection_sources=sources)
    impedance = {w.wall_name(): complex(m.impedance_pa_s_m) for w, m in fixed.items()}
    for rho_c, lane_axis, message in ((1.2 * 347., axis, "rho_c"), (1.19 * 347., (125., 251.), "頻率軸")):
        with pytest.raises(ValueError, match=message):
            geometric_lane.solve_geometric_early_lane(source_model=SourceModelSpec(kind=SourceModelKind.OMNIDIRECTIONAL),
                room=room, source=source, receiver=receiver, sound_speed_m_s=347., rho_c_pa_s_per_m=rho_c,
                frequencies_hz=lane_axis, impedance_by_wall=impedance, reflection_sources=sources)
