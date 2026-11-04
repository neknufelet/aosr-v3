"""#561：逐反彈查矩形、分界歸屬、舊答案契約與未支援下游。"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from aosr.geometry.shoebox import Point, Room, Wall
from aosr.materials.construction_models import (
    AirLayer, Assembly, BoardLayer, ConstantImpedance, MeasuredImpedance, MeasuredImpedancePoint, SurfaceMaterial,
)
from aosr.physics import geometric_lane, report_path_table, wall_reflections
from aosr.physics.amplitude import reflection_coefficient
from aosr.physics.report_surfaces import RectangleZone, RectangleZonesUnsupported, WholeWallZone, ZonesByWall, rectangle_zone_at
from aosr.physics.room_paths import RoomPath, image_source_paths, load_room_input
from aosr.runtime import child_process_env, configure_jax
from tests.engine._precision_contracts import contract_value
from tests.engine._zone_reflections_probe import AXIS, MODEL, RECEIVER, ROOM, SOURCE, materials
from tests.engine.test_patch_materials import _patch_input, _write_input

# 改產品前在 c6f5138a286de8a1c73cebd21a536de18759af53 實量；見 .evidence/baseline-snapshots.json。
_BASELINES = {
    "fixed": ["7248c6d32b530121014dbb42cf3ef052f2ebcc13492ea3345de20e93dd75e831",
              "440880f5a23e0ae05b1400f846681db7713f2cdbeccfd74571627c9786265d49"],
    "construction": ["ebf86dfba39ff6e7495a02f1282bd02460f5e597e75820ffc138721783278c18",
                     "b93b2565d4e6810d09ddf2d9d666de37c2810150d90a4976f810a3040bf1926a"],
}


def _whole() -> ZonesByWall:
    return {w.wall_name(): (WholeWallZone(area="whole_wall", material=m),) for w, m in materials(False).items()}


def _grid(width: float, height: float, split_u: float, split_v: float) -> tuple[RectangleZone, ...]:
    return tuple(RectangleZone(area="rectangle", id=f"r{row}c{col}", u_m=(u0, u1), v_m=(v0, v1),
        material=ConstantImpedance(impedance_pa_s_m=800. + 600. * (2 * row + col)))
        for row, (v0, v1) in enumerate(((0., split_v), (split_v, height)))
        for col, (u0, u1) in enumerate(((0., split_u), (split_u, width))))


def _sources(zones: ZonesByWall, room: Room = ROOM) -> wall_reflections.WallReflectionSources:
    return wall_reflections.zoned_reflection_sources(room, zones, AXIS, density_kg_m3=1.19, sound_speed_m_s=347.)


def _paths(zones: ZonesByWall, source: Point = SOURCE, receiver: Point = RECEIVER,
           room: Room = ROOM, order: int = 3) -> list[RoomPath]:
    return image_source_paths(room, source, receiver, 347., order, reflection_sources=_sources(zones, room))


def _bits(values: tuple[complex, ...]) -> bytes:
    return np.asarray(values, dtype=np.complex128).tobytes()


def _probe(tmp_path: Path, mode: str) -> subprocess.CompletedProcess[str]:
    env = child_process_env(threads=1).copy()
    env.update({name: os.environ[name] for name in ("NPY_DISABLE_CPU_FEATURES", "OPENBLAS_CORETYPE", "XLA_FLAGS")
                if name in os.environ})
    configure_jax(platforms="cpu", enable_x64=True, env=env)
    env["PYTHONPATH"] = os.pathsep.join((env["PYTHONPATH"], str(Path(__file__).resolve().parents[2])))
    return subprocess.run([sys.executable, "-m", "tests.engine._zone_reflections_probe", mode, str(tmp_path)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)


def test_whole_wall_fixed_and_construction_match_starting_commit(tmp_path: Path) -> None:
    """抓整面一區誤走分區乘法，含固定純量捷徑及構造運算順序。"""
    run = _probe(tmp_path, "checks")
    assert run.returncode == 0, run.stdout + run.stderr
    assert json.loads((tmp_path / "zone-result.json").read_text()) == _BASELINES


def test_internal_boundary_selects_only_upper_right_zone() -> None:
    """點在 u、v 兩條內界上，必須只用右上區一次；兩區相乘會紅。"""
    zones = _whole()
    zones["floor"] = _grid(6., 4., 3., 2.)
    path = next(p for p in _paths(zones, Point(2., 1., 1.), Point(4., 3., 1.))
                if p.identity == (0, 1, 0, 1, 0, -1))
    assert path.bounces[0].point == (3., 2., 0.)
    expected = reflection_coefficient(complex(2600.), 2. / math.sqrt(12.), 1.19 * 347.)
    assert path.reflection_product == (expected, expected)


def test_two_partitioned_walls_at_edge_each_reflect_once() -> None:
    """交線上的兩面各查自己分區；外緣歸最後區，不能只乘第一面。"""
    zones = _whole()
    zones["x0"] = _grid(4., 3., 2., 1.5)
    y_grid = _grid(6., 3., 2., 2.5)
    zones["y0"] = (y_grid[0].model_copy(update={"material": ConstantImpedance(impedance_pa_s_m=1100.)}), *y_grid[1:])
    path = next(p for p in _paths(zones, Point(.5, .5, 2.), Point(1.5, 1.5, 2.))
                if p.identity == (0, -1, 0, -1, 0, 1))
    assert path.bounces[0].walls == ("x0", "y0")
    assert path.bounces[0].point == (0., 0., 2.)
    x = reflection_coefficient(complex(2000.), 2. / math.sqrt(8.), 1.19 * 347.)
    y = reflection_coefficient(complex(1100.), 2. / math.sqrt(8.), 1.19 * 347.)
    assert path.reflection_product == (x * y, x * y)


def test_asymmetric_partition_preserves_u_v_axes() -> None:
    """u 切三分之一、v 切一半，(u,v)=(3,1) 只能打右下；交換軸就紅。"""
    zones = _whole()
    zones["floor"] = _grid(6., 4., 2., 2.)
    path = next(p for p in _paths(zones, Point(2., .5, 1.), Point(4., 1.5, 1.), order=1)
                if p.identity == (0, 1, 0, 1, 0, -1))
    assert path.bounces[0].point == (3., 1., 0.)
    expected = reflection_coefficient(complex(1400.), 2. / 3., 1.19 * 347.)
    assert path.reflection_product == (expected, expected)


@pytest.mark.parametrize("wall", Wall.all())
def test_zone_lookup_uses_wall_axes_and_closes_only_outer_edges(wall: Wall) -> None:
    """六面含側牆的 u/v 軸、交點與外角都要正確；倒排輸入不能改歸屬。"""
    zones = tuple(reversed(_grid(6., 4., 2., 2.)))
    axes = [a for a in range(3) if a != wall.axis()]
    for u, v, expected in ((2., 2., "r1c1"), (6., 4., "r1c1"), (6., 0., "r0c1"),
                            (0., 4., "r1c0"), (0., 0., "r0c0"), (3., 1., "r0c1")):
        point = [0., 0., 0.]
        point[axes[0]], point[axes[1]] = u, v
        assert rectangle_zone_at(wall.wall_name(), (point[0], point[1], point[2]), zones).id == expected


@pytest.mark.parametrize("reader", ["jax-table", "jax-sources", "json", "human", "per-wall-rows"])
def test_whole_wall_readers_refuse_partitioned_sources(reader: str) -> None:
    """舊逐牆陣列與格號輸出不能把矩形當第一格；均須拋專用拒收型別。"""
    from aosr.physics import amplitude_jax, room_path_output
    zones = _whole()
    zones["floor"] = _grid(6., 4., 2., 2.)
    sources = _sources(zones)
    paths = _paths(zones)
    with pytest.raises(RectangleZonesUnsupported):
        if reader == "jax-table":
            amplitude_jax.build_path_table(paths, RECEIVER.as_tuple(), AXIS)
        elif reader == "jax-sources":
            geometry = image_source_paths(ROOM, SOURCE, RECEIVER, 347., 3)
            table = amplitude_jax.build_path_table(geometry, RECEIVER.as_tuple(), AXIS)
            amplitude_jax.bounce_impedances_from_sources(table, geometry, RECEIVER.as_tuple(), sources)
        elif reader == "json":
            room_path_output.paths_to_payload(paths)
        elif reader == "human":
            room_path_output._human_table(paths, AXIS)
        else:
            sources.impedances_for_paths(paths, RECEIVER.as_tuple())


def test_change_one_zone_only_changes_paths_hitting_it() -> None:
    """只改一區：打到的要變，沒打到的連壓力位元都不准變。"""
    zones = _whole()
    grid = _grid(6., 4., 2., 2.)
    zones["floor"] = grid
    before = _paths(zones)
    zones["floor"] = (grid[0].model_copy(update={"material": ConstantImpedance(impedance_pa_s_m=9000.)}), *grid[1:])
    after = _paths(zones)
    hit = miss = 0
    for old, new in zip(before, after, strict=True):
        touches = any("floor" in b.walls and b.point[0] < 2. and b.point[1] < 2. for b in old.bounces)
        if touches:
            hit += 1
            assert _bits(old.reflection_product) != _bits(new.reflection_product)
        else:
            miss += 1
            assert _bits(old.reflection_product) == _bits(new.reflection_product)
            assert _bits(old.path_pressure) == _bits(new.path_pressure)
    assert hit and miss


def _legacy_comparison(tmp_path: Path) -> dict[str, float]:
    """原檔三區是逐頻複數阻抗；逐點實測表精確保留它們，沒有拿首頻代替。"""
    inputs = load_room_input(_write_input(tmp_path, _patch_input()))
    assert inputs.materials is not None
    legacy = image_source_paths(inputs.room, inputs.source, inputs.receiver, inputs.sound_speed,
                                inputs.max_order, inputs.materials)
    zones: ZonesByWall = {}
    for wall in Wall.all():
        axes = [a for a in range(3) if a != wall.axis()]
        width, height = (inputs.room.length(a) for a in axes)
        rows, cols, cells = inputs.materials.grid(wall.wall_name())
        zone_materials: list[SurfaceMaterial] = []
        for cell in cells:
            if all(z == cell[0] and z.imag == 0 for z in cell):
                zone_materials.append(ConstantImpedance(impedance_pa_s_m=cell[0].real))
            else:
                zone_materials.append(MeasuredImpedance(points=tuple(MeasuredImpedancePoint(
                    frequency_hz=f, real_pa_s_m=z.real, imag_pa_s_m=z.imag)
                    for f, z in zip(inputs.materials.frequencies_hz, cell, strict=True)),
                    base_assembly=Assembly(layers=(BoardLayer(surface_mass_kg_m2=.15, resistance_pa_s_m=80.),
                        AirLayer(thickness_m=.43)), backing="rigid"),
                    source_and_conditions="reference_amplitude_patch_varied.json 既定逐頻固定阻抗；只在原頻點求值"))
        zones[wall.wall_name()] = tuple(RectangleZone(area="rectangle", id=f"r{r}c{c}",
            u_m=(width*c/cols, width*(c+1)/cols), v_m=(height*r/rows, height*(r+1)/rows),
            material=zone_materials[r*cols+c])
            for r in range(rows) for c in range(cols))
    sources = wall_reflections.zoned_reflection_sources(inputs.room, zones, inputs.materials.frequencies_hz,
        density_kg_m3=inputs.materials.rho_c / inputs.sound_speed, sound_speed_m_s=inputs.sound_speed,
        rho_c_pa_s_per_m=inputs.materials.rho_c)
    actual = image_source_paths(inputs.room, inputs.source, inputs.receiver, inputs.sound_speed,
                                inputs.max_order, reflection_sources=sources)
    largest = 0.
    largest_ulps = 0.
    for old, new in zip(legacy, actual, strict=True):
        assert old.identity == new.identity
        for a, b in zip(old.reflection_product, new.reflection_product, strict=True):
            for x, y in ((a.real, b.real), (a.imag, b.imag)):
                largest = max(largest, abs(x-y))
                largest_ulps = max(largest_ulps, abs(x-y) / math.ulp(x))
    assert largest <= contract_value("reflection_product_ulp"), (largest, largest_ulps)
    return {"max_absolute_difference": largest, "max_component_ulps": largest_ulps}


def test_uniform_rectangles_match_legacy_bounce_lookup(tmp_path: Path) -> None:
    """同凍結格網與固定阻抗，逐分量套既有 reflection_product_ulp 絕對契約。"""
    run = _probe(tmp_path, "legacy")
    assert run.returncode == 0, run.stdout + run.stderr
    stats = json.loads((tmp_path / "zone-result.json").read_text())
    assert stats["max_absolute_difference"] <= contract_value("reflection_product_ulp")


def test_path_table_uses_partitioned_reflection_products() -> None:
    """路徑表沿分區聲壓算相對能量，不拿第一區冒充整牆。"""
    zones = _whole()
    zones["floor"] = _grid(6., 4., 2., 2.)
    paths = _paths(zones)
    direct = next(p for p in paths if p.order == 0)
    table = report_path_table.build_path_table(room=ROOM, source=SOURCE, receiver=RECEIVER,
        sound_speed_m_s=347., rho_c_pa_s_per_m=1.19*347., density_kg_m3=1.19, frequencies_hz=AXIS,
        scattering_coefficient=(.21, .21), reflection_order_k=3, source_model=MODEL, furniture=None,
        materials_by_wall={}, surface_zones_by_wall=zones)
    for path, row in zip(paths, table.rows, strict=True):
        assert row.relative_direct_energy == tuple((1.-.21)**path.order * abs(p)**2 / abs(d)**2
            for p, d in zip(path.path_pressure, direct.path_pressure, strict=True))


@pytest.mark.parametrize("reader", ["_solve_geometric_report_lane", "_solve_dense_geometric_report_lane",
                                    "_solve_both_geometric_report_lanes"])
def test_report_geometry_readers_reject_rectangle_zones(reader: str) -> None:
    """報表幾何讀者尚會合成整房散射；明給分區必須是專用拒收，不能靜默忽略。"""
    from aosr.physics import three_lane_report_geometry
    zones = _whole()
    zones["floor"] = _grid(6., 4., 2., 2.)
    solve = getattr(three_lane_report_geometry, reader)
    with pytest.raises(RectangleZonesUnsupported):
        solve(source_model=MODEL, room=ROOM, source=SOURCE, receiver=RECEIVER, scattering_by_wall=None,
            rho_c_pa_s_per_m=1.19*347., sound_speed_m_s=347., density_kg_m3=1.19, reflection_order_k=3,
            materials_by_wall={}, surface_zones_by_wall=zones)


def test_partition_sources_validate_room_and_reject_ambiguous_materials() -> None:
    """低層來源不能繞過房間驗證，或同面既給整牆又給分區。"""
    rectangles = {Wall.FLOOR: _grid(6., 4., 2., 2.)}
    with pytest.raises(ValueError, match="房間"):
        wall_reflections.wall_reflection_sources({w: m for w, m in materials(False).items() if w is not Wall.FLOOR},
            AXIS, density_kg_m3=1.19, sound_speed_m_s=347., rectangle_zones_by_wall=rectangles)
    with pytest.raises(ValueError, match="同時"):
        wall_reflections.wall_reflection_sources(materials(False), AXIS, density_kg_m3=1.19, sound_speed_m_s=347.,
            rectangle_zones_by_wall=rectangles, room=ROOM)
    zones = _whole()
    zones["floor"] = rectangles[Wall.FLOOR]
    with pytest.raises(ValueError, match="房間"):
        image_source_paths(Room(7., 4., 3.), SOURCE, RECEIVER, 347., 3, reflection_sources=_sources(zones))


@pytest.mark.parametrize("lane", [geometric_lane.solve_geometric_early_lane, geometric_lane.solve_geometric_lane])
def test_scattering_lanes_reject_rectangle_zones(lane: object) -> None:
    """合成房間散射尚無分區定義，細軸與密軸的物理入口均須明確拒收。"""
    zones = _whole()
    zones["floor"] = _grid(6., 4., 2., 2.)
    assert callable(lane)
    with pytest.raises(RectangleZonesUnsupported):
        lane(room=ROOM, source=SOURCE, receiver=RECEIVER, source_model=MODEL, sound_speed_m_s=347.,
            rho_c_pa_s_per_m=1.19*347., frequencies_hz=AXIS,
            impedance_by_wall={w.wall_name(): 900. for w in Wall.all()}, reflection_sources=_sources(zones))


@pytest.mark.parametrize("downstream", ["screen", "window", "solver"])
def test_report_downstreams_reject_rectangle_zones(downstream: str) -> None:
    """牆對留存／顫動回聲、補算窗與三路總入口即使傳整牆覆寫也不得忽略分區。"""
    from aosr.physics.reflection_screen import build_reflection_screen
    from aosr.physics.reflection_window import build_reflection_window
    from aosr.physics.report_io import ReportInput, solver_inputs
    zones = _whole()
    zones["floor"] = _grid(6., 4., 2., 2.)
    inputs = ReportInput.model_validate({"room_m": ROOM, "source_m": SOURCE, "receiver_m": RECEIVER,
        "source_model": {"kind": "omnidirectional"}, "sound_speed_m_s": 347., "density_kg_m3": 1.19,
        "surface_zones_by_wall": zones, "reflection_order_k": 3})
    with pytest.raises(RectangleZonesUnsupported):
        if downstream == "screen":
            build_reflection_screen(inputs, AXIS, materials_by_wall=materials(False))
        elif downstream == "window":
            build_reflection_window(inputs, frequencies_hz=AXIS, scattering_coefficient=(.21, .21),
                window_s=.025, materials_by_wall=materials(False))
        else:
            solver_inputs(inputs)
