"""鏡像法逐路徑構造阻抗：純量逐位契約與四個讀者接線。"""
from __future__ import annotations

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


@pytest.fixture(scope="module")
def construction_checks(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    return _child(tmp_path_factory.mktemp("wall-reflections"), "suite")


def test_construction_impedance_and_product_match_direct_bits(construction_checks: dict[str, str]) -> None:
    """抓每路每面角度或Z錯誤、批次求值或反射次方漂移。"""
    assert construction_checks["direct"] == "", construction_checks["direct"]


def test_four_readers_share_path_reflections(construction_checks: dict[str, str]) -> None:
    """抓細密軸、路徑表、時間窗的反射來源分岔。"""
    assert construction_checks["readers"] == "", construction_checks["readers"]


def test_scattering_and_screen_use_normal_impedance(construction_checks: dict[str, str]) -> None:
    """抓兩處垂直入射誤用路徑角、篩檢把逐頻值排成常數。"""
    assert construction_checks["normal"] == "", construction_checks["normal"]


def test_construction_and_depth_change_both_axes_table_screen(construction_checks: dict[str, str]) -> None:
    """抓構造接線未改變輸出、忽略空腔參數。"""
    assert construction_checks["outputs"] == "", construction_checks["outputs"]


def test_mixed_room_keeps_scalar_reuse_and_batches_by_wall(construction_checks: dict[str, str]) -> None:
    """抓混合房只打固定牆時失去捷徑或核心按每路重算。"""
    assert construction_checks["reuse"] == "", construction_checks["reuse"]


def test_constant_readers_run_without_jax_calls(construction_checks: dict[str, str]) -> None:
    """抓固定牆的細密軸、路徑表、時間窗、篩檢觸碰JAX。"""
    assert construction_checks["constant"] == "", construction_checks["constant"]


def test_internal_batch_hands_override_to_both_geometric_lanes(construction_checks: dict[str, str]) -> None:
    """抓內部構造覆寫沒有從房間準備送到逐對鏡像法、污染有限元素材料。"""
    assert construction_checks["batch"] == "", construction_checks["batch"]


@pytest.mark.parametrize("mutant", ("zero-angle", "wrong-axis", "scattering-angle", "screen-angle"))
def test_angle_mutants_are_rejected(construction_checks: dict[str, str], mutant: str) -> None:
    """抓全部路徑0°、錯軸，以及散射／篩檢用路徑角四種植錯真的會紅。"""
    assert construction_checks[mutant] == "", construction_checks[mutant]


def test_legacy_rho_c_is_not_rounded_through_density() -> None:
    """抓舊呼叫只給ρc時，先除聲速再乘回來損失一位而改掉固定牆結果。"""
    from aosr.geometry.shoebox import Point, Room, Wall
    from aosr.physics import geometric_lane, three_lane_report
    from aosr.physics.report_source import SourceModelKind, SourceModelSpec
    rho_c, c = float.fromhex("0x1.f5a1b50ebc70fp+8"), float.fromhex("0x1.595fdff9071a1p+8")
    walls = {w: 500. + i * 173. for i, w in enumerate(Wall.all())}
    model, room = SourceModelSpec(kind=SourceModelKind.OMNIDIRECTIONAL), Room(6., 4., 3.)
    source, receiver, axis = Point(1.1, .9, 1.2), Point(3.7, 2.1, 1.45), (125., 503., 2017.)
    actual = three_lane_report._solve_geometric_report_lane(source_model=model, room=room, source=source,
        receiver=receiver, sound_speed_m_s=c, rho_c_pa_s_per_m=rho_c, frequencies_hz=axis,
        reflection_order_k=3, wall_impedances=walls, scattering_by_wall=None)
    expected = geometric_lane.solve_geometric_lane(source_model=model, room=room, source=source,
        receiver=receiver, sound_speed_m_s=c, rho_c_pa_s_per_m=rho_c, frequencies_hz=axis,
        reflection_order_k=3, impedance_by_wall={w.wall_name(): complex(z) for w, z in walls.items()})
    assert actual == expected


def test_construction_override_requires_scene_density() -> None:
    """抓構造覆寫沒帶場景密度時悄悄從 ρc 反推（構造的 Z 會因此差最後一位）。"""
    from aosr.geometry.shoebox import Wall
    from aosr.materials.construction_models import AirLayer, Assembly, BoardLayer, ConstantImpedance
    from aosr.physics.wall_reflections import lane_reflection_sources
    walls = {wall: 900. + 173. * i for i, wall in enumerate(Wall.all())}
    board = Assembly(layers=(BoardLayer(surface_mass_kg_m2=.15, resistance_pa_s_m=80.), AirLayer(thickness_m=.43)),
                     backing="rigid")
    materials = {**{wall: ConstantImpedance(impedance_pa_s_m=z) for wall, z in walls.items()}, Wall.X0: board}
    with pytest.raises(ValueError, match="場景密度"):
        lane_reflection_sources(walls, materials, (125., 503.), rho_c_pa_s_per_m=1.19 * 347.,
                                sound_speed_m_s=347., density_kg_m3=None)
