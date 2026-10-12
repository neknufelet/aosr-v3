"""牆面單一吸音率來源與晚期家族接線考卷。"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from aosr.geometry.shoebox import Wall
from aosr.materials.catalog_absorption import complex_random_incidence_absorption
from aosr.materials.construction_models import ConstantImpedance
from aosr.runtime import child_process_env, configure_jax

# 控制數字來自主線 2144db851a11a536f44d11ad8bf137a2eb1b3c0a，
# 由外部施工資料 309-materials 的 control 目錄內 pin_late_family.py 實跑，
# 以 float.hex() 記錄；此來源程式不在 repo 內。
# 選兩場景、六個含頭尾的分散能量頻點，以及每個報表八度帶一個衰減頻點。
CONTROL: dict[str, dict[str, object]] = {'unequal_hard': {'walls': {'floor': 8200.0,
                            'ceiling': 1646.4,
                            'x0': 3100.0,
                            'xL': 950.0,
                            'y0': 12500.0,
                            'yL': 2300.0},
                  't60': {'500.0': '0x1.ef44c8d6cb749p-4', '1000.0': '0x1.ef44c8d6cb749p-4'},
                  'fs': '0x1.6b2d640317784p+6',
                  'late': [('0x1.4000000000000p+4',
                            '0x1.131c27775a95fp-2',
                            '0x1.45d832f00ff40p-6',
                            ['0x1.441c47844fd24p-3', '0x1.0432d770c0527p-4', '0x1.ba3caaa0198f5p-6']),
                           ('0x1.b7aa486590c52p+5',
                            '0x1.131c27775a95fp-2',
                            '0x1.45d832f00ff40p-6',
                            ['0x1.441c47844fd24p-3', '0x1.0432d770c0527p-4', '0x1.ba3caaa0198f5p-6']),
                           ('0x1.2e0a2f06a0c0ap+7',
                            '0x1.131c27775a95fp-2',
                            '0x1.45d832f00ff40p-6',
                            ['0x1.441c47844fd24p-3', '0x1.0432d770c0527p-4', '0x1.ba3caaa0198f5p-6']),
                           ('0x1.9efd18a83b4b1p+8',
                            '0x1.131c27775a95fp-2',
                            '0x1.45d832f00ff40p-6',
                            ['0x1.441c47844fd24p-3', '0x1.0432d770c0527p-4', '0x1.ba3caaa0198f5p-6']),
                           ('0x1.1d166c49138b9p+10',
                            '0x1.131c27775a95fp-2',
                            '0x1.45d832f00ff40p-6',
                            ['0x1.441c47844fd24p-3', '0x1.0432d770c0527p-4', '0x1.ba3caaa0198f5p-6']),
                           ('0x1.5cf664b9ca5dap+13',
                            '0x1.131c27775a95fp-2',
                            '0x1.45d832f00ff40p-6',
                            ['0x1.441c47844fd24p-3', '0x1.0432d770c0527p-4', '0x1.ba3caaa0198f5p-6'])],
                  'decay': [('0x1.ed81e92b54912p+6', '0x1.e3381d66358ddp-4', '0x1.e4b0433cc87f7p-4'),
                            ('0x1.ed81e92b54912p+7', '0x1.e3381d66358ddp-4', '0x1.e4b0433cc87f7p-4'),
                            ('0x1.ed81e92b54912p+8', '0x1.e3381d66358ddp-4', '0x1.e4b0433cc87f7p-4'),
                            ('0x1.ed81e92b54912p+9', '0x1.e3381d66358ddp-4', '0x1.e4b0433cc87f7p-4'),
                            ('0x1.ed81e92b54912p+10', '0x1.e3381d66358ddp-4', '0x1.e4b0433cc87f7p-4'),
                            ('0x1.ed81e92b54912p+11', '0x1.e3381d66358ddp-4', '0x1.e4b0433cc87f7p-4'),
                            ('0x1.ed81e92b54912p+12', '0x1.e3381d66358ddp-4', '0x1.e4b0433cc87f7p-4')]},
 'unequal_soft': {'walls': {'floor': 520.0,
                            'ceiling': 430.0,
                            'x0': 900.0,
                            'xL': 610.0,
                            'y0': 2400.0,
                            'yL': 700.0},
                  't60': {'500.0': '0x1.6c0725057876fp-5', '1000.0': '0x1.6c0725057876fp-5'},
                  'fs': '0x1.b85515bed2f37p+5',
                  'late': [('0x1.4000000000000p+4',
                            '0x1.8f5f576c38970p-6',
                            '0x1.67d2d013df400p-16',
                            ['0x1.6b55c9fcb386ap-6', '0x1.02eccae488099p-9', '0x1.a8ffaf778bb00p-13']),
                           ('0x1.b7aa486590c52p+5',
                            '0x1.8f5f576c38970p-6',
                            '0x1.67d2d013df400p-16',
                            ['0x1.6b55c9fcb386ap-6', '0x1.02eccae488099p-9', '0x1.a8ffaf778bb00p-13']),
                           ('0x1.2e0a2f06a0c0ap+7',
                            '0x1.8f5f576c38970p-6',
                            '0x1.67d2d013df400p-16',
                            ['0x1.6b55c9fcb386ap-6', '0x1.02eccae488099p-9', '0x1.a8ffaf778bb00p-13']),
                           ('0x1.9efd18a83b4b1p+8',
                            '0x1.8f5f576c38970p-6',
                            '0x1.67d2d013df400p-16',
                            ['0x1.6b55c9fcb386ap-6', '0x1.02eccae488099p-9', '0x1.a8ffaf778bb00p-13']),
                           ('0x1.1d166c49138b9p+10',
                            '0x1.8f5f576c38970p-6',
                            '0x1.67d2d013df400p-16',
                            ['0x1.6b55c9fcb386ap-6', '0x1.02eccae488099p-9', '0x1.a8ffaf778bb00p-13']),
                           ('0x1.5cf664b9ca5dap+13',
                            '0x1.8f5f576c38970p-6',
                            '0x1.67d2d013df400p-16',
                            ['0x1.6b55c9fcb386ap-6', '0x1.02eccae488099p-9', '0x1.a8ffaf778bb00p-13'])],
                  'decay': [('0x1.ed81e92b54912p+6', '0x1.67603cc8cd9aap-5', '0x1.62eace906f261p-5'),
                            ('0x1.ed81e92b54912p+7', '0x1.67603cc8cd9aap-5', '0x1.62eace906f261p-5'),
                            ('0x1.ed81e92b54912p+8', '0x1.67603cc8cd9aap-5', '0x1.62eace906f261p-5'),
                            ('0x1.ed81e92b54912p+9', '0x1.67603cc8cd9aap-5', '0x1.62eace906f261p-5'),
                            ('0x1.ed81e92b54912p+10', '0x1.67603cc8cd9aap-5', '0x1.62eace906f261p-5'),
                            ('0x1.ed81e92b54912p+11', '0x1.67603cc8cd9aap-5', '0x1.62eace906f261p-5'),
                            ('0x1.ed81e92b54912p+12', '0x1.67603cc8cd9aap-5', '0x1.62eace906f261p-5')]}}


def _child(directory: Path, mode: str, *, x64: bool) -> dict[str, object]:
    env = child_process_env(threads=1).copy()
    configure_jax(platforms="cpu", enable_x64=x64, env=env)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["TMPDIR"] = str(directory)
    output = directory / "result.json"
    probe = Path(__file__).with_name("_wall_surfaces_probe.py")
    process = subprocess.run([sys.executable, str(probe), mode, str(output)], cwd=directory,
                             env=env, input=json.dumps(CONTROL), capture_output=True, text=True, timeout=180)
    assert process.returncode == 0, process.stdout + process.stderr
    return cast(dict[str, object], json.loads(output.read_text()))


def test_constant_absorption_bitwise() -> None:
    """抓固定阻抗除以場景ρc的算式或取樣廣播改掉最後一位。"""
    from aosr.physics.wall_surfaces import wall_surface_response
    rng = np.random.default_rng(309)
    for z, rho, c in zip(rng.uniform(100, 30000, 40), rng.uniform(.8, 1.6, 40),
                         rng.uniform(290, 380, 40), strict=True):
        materials = {wall: ConstantImpedance(impedance_pa_s_m=float(z)) for wall in Wall.all()}
        response = wall_surface_response(materials, (20., 500., 1000., 20000.),
                                         density_kg_m3=float(rho), sound_speed_m_s=float(c))
        expected = complex_random_incidence_absorption(complex(z) / (rho * c))
        assert response.absorption_by_wall == {wall: (expected,) * 4 for wall in Wall.all()}
        assert response.notes_by_wall == {wall: () for wall in Wall.all()}


@pytest.mark.parametrize("x64", (False, True))
def test_main_control_bitwise_without_jax_calls(tmp_path: Path, x64: bool) -> None:
    """抓正式固定阻抗準備碰JAX、兩帶抄錯頻率、晚期能量或T20/T30逐位漂移。"""
    actual = _child(tmp_path, "constant", x64=x64)
    expected = {name: {key: value for key, value in scene.items() if key != "walls"}
                for name, scene in CONTROL.items()}
    assert actual == json.loads(json.dumps(expected))


def test_construction_family_matches_direct(tmp_path: Path) -> None:
    """抓構造沒接進三種晚期消費者、兩帶抄同一α、空氣或逐面註記遺失。"""
    result = _child(tmp_path, "construction", x64=True)
    assert result["verified"] is True


def test_construction_requires_process_precision(tmp_path: Path) -> None:
    """抓未開雙精度的構造靜默降精度，或拒收訊息沒指向行程設定。"""
    result = _child(tmp_path, "precision", x64=False)
    assert "JAX" in str(result["error"])
    assert "精度" in str(result["error"])
    assert "執行環境" in str(result["error"])


@pytest.mark.parametrize("invalid", ((.2,), (float("nan"), .2), (-.1, .2), (1.1, .2)))
def test_late_input_rejects_invalid_alpha(invalid: tuple[float, ...]) -> None:
    """抓晚期輸入未驗頻點數、有限性或吸音率範圍。"""
    from aosr.geometry.shoebox import Room
    from aosr.physics.late_energy import LateEnergyInputs, _wall_absorption
    inputs = LateEnergyInputs(Room(5., 4., 3.), 411.6, (500., 1000.),
                              {wall: invalid for wall in Wall.wall_names()}, 1, 1.)
    with pytest.raises(ValueError):
        _wall_absorption(inputs)


def test_geometric_lane_consumes_alpha_without_cached_late() -> None:
    """抓單獨幾何求解忽略α、退回阻抗重算晚期，或改動鏡像法三欄。"""
    from aosr.geometry.shoebox import Point, Room
    from aosr.physics.geometric_lane import GeometricLaneResult, solve_geometric_lane, solve_geometric_late_energy
    from aosr.physics.report_source import SourceModelKind, SourceModelSpec
    room, axis = Room(5.3, 4.1, 2.7), (125., 500., 1000.)
    alpha = {wall: (.2, .35, .6) for wall in Wall.wall_names()}
    late = solve_geometric_late_energy(room=room, frequencies_hz=axis, rho_c_pa_s_per_m=411.6,
                                      absorption_by_wall=alpha, reflection_order_k=3)

    def lane(*, cached: bool, injected: bool) -> GeometricLaneResult:
        return solve_geometric_lane(source_model=SourceModelSpec(kind=SourceModelKind.OMNIDIRECTIONAL),
            room=room, source=Point(1., 1., 1.), receiver=Point(3., 2., 1.5),
            sound_speed_m_s=343., rho_c_pa_s_per_m=411.6, frequencies_hz=axis,
            impedance_by_wall={wall: 8200.+0j for wall in Wall.wall_names()},
            absorption_by_wall=alpha if injected else None, late_result=late if cached else None,
            scattering_by_wall={wall: 1. for wall in Wall.wall_names()}, reflection_order_k=3)

    actual, cached, baseline = lane(cached=False, injected=True), lane(cached=True, injected=False), lane(cached=False, injected=False)
    assert actual == cached
    assert actual.late_energy != baseline.late_energy
    assert (actual.direct_energy, actual.reflected_energy, actual.interference_energy) == (
        baseline.direct_energy, baseline.reflected_energy, baseline.interference_energy)
