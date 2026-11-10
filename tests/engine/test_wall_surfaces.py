"""牆面單一吸音率來源與晚期家族接線考卷。"""
from __future__ import annotations

import json
import os
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
from tests.engine._cross_cpu_tolerance import GEOMETRIC_REL, LINEAR_SOLVE_REL, Cell, assert_within, cell

# 控制數字來自主線 2144db851a11a536f44d11ad8bf137a2eb1b3c0a，
# 由外部施工資料 309-materials 的 control 目錄內 pin_late_family.py 實跑，
# 以 float.hex() 記錄；此來源程式不在 repo 內。
# 選兩場景、六個含頭尾的分散能量頻點，以及每個報表八度帶一個衰減頻點。
# #1057 物理變更：晚期一族家具後加空氣；零空氣逐位重播（x64 關／開）通過後，由同一個子行程重錄
# （工作區 1057-air-absorption 底下 record 目錄的 record_wall_control.py）。
CONTROL: dict[str, dict[str, object]] = {'unequal_hard': {'walls': {'floor': 8200.0,
                            'ceiling': 1646.4,
                            'x0': 3100.0,
                            'xL': 950.0,
                            'y0': 12500.0,
                            'yL': 2300.0},
                  't60': {'500.0': '0x1.ee68ea66b73a7p-4', '1000.0': '0x1.ed8f57a5c7954p-4'},
                  'fs': '0x1.6ab4cfdf687a6p+6',
                  'late': [('0x1.4000000000000p+4',
                            '0x1.131a590dea807p-2',
                            '0x1.45d30955bc958p-6',
                            ['0x1.441adc29acccep-3', '0x1.0430e7fdb5573p-4', '0x1.ba380644afadbp-6']),
                           ('0x1.b7aa486590c52p+5',
                            '0x1.130f475d5a0ebp-2',
                            '0x1.45b36763f1848p-6',
                            ['0x1.4412296247702p-3', '0x1.04250bce31647p-4', '0x1.ba1b9426ae53dp-6']),
                           ('0x1.2e0a2f06a0c0ap+7',
                            '0x1.12d73334b7056p-2',
                            '0x1.45133d9fb4110p-6',
                            ['0x1.43e614c2ed323p-3', '0x1.03e8f4b37d1ffp-4', '0x1.b98b7cc65e33bp-6']),
                           ('0x1.9efd18a83b4b1p+8',
                            '0x1.12606cd112f1dp-2',
                            '0x1.43c08a3f3b220p-6',
                            ['0x1.4388a61d5fa12p-3', '0x1.0369a972047dfp-4', '0x1.b85a6c1ee4fa8p-6']),
                           ('0x1.1d166c49138b9p+10',
                            '0x1.111bdef504ed8p-2',
                            '0x1.4026ad42e9c40p-6',
                            ['0x1.4288dc9b7a8d9p-3', '0x1.020dac69b1635p-4', '0x1.b519ab8acb1a0p-6']),
                           ('0x1.5cf664b9ca5dap+13',
                            '0x1.8913933a66c82p-3',
                            '0x1.1b209ac415480p-7',
                            ['0x1.0080b044e5052p-3', '0x1.5d7e83fc3c4a7p-5', '0x1.f81384a315bdcp-7'])],
                  'decay': [('0x1.ed81e92b54912p+6', '0x1.e3059b0b268d5p-4', '0x1.e47dafa25e9dbp-4'),
                            ('0x1.ed81e92b54912p+7', '0x1.e2c18d4b30e59p-4', '0x1.e4398a91b9e64p-4'),
                            ('0x1.ed81e92b54912p+8', '0x1.e265ebf30b28dp-4', '0x1.e3ddc9a7f0db4p-4'),
                            ('0x1.ed81e92b54912p+9', '0x1.e1984cffd0756p-4', '0x1.e30fe2dac676ep-4'),
                            ('0x1.ed81e92b54912p+10', '0x1.deb15d1ee226cp-4', '0x1.e027daa975f15p-4'),
                            ('0x1.ed81e92b54912p+11', '0x1.d3ee72f6f7504p-4', '0x1.d55e342739048p-4'),
                            ('0x1.ed81e92b54912p+12', '0x1.b250069990febp-4', '0x1.b39803d7a0963p-4')]},
 'unequal_soft': {'walls': {'floor': 520.0,
                            'ceiling': 430.0,
                            'x0': 900.0,
                            'xL': 610.0,
                            'y0': 2400.0,
                            'yL': 700.0},
                  't60': {'500.0': '0x1.6bcbb034105e4p-5', '1000.0': '0x1.6b90b95d7a116p-5'},
                  'fs': '0x1.b81f4818d8405p+5',
                  'late': [('0x1.4000000000000p+4',
                            '0x1.8f5d99463a8c8p-6',
                            '0x1.67cdedc328000p-16',
                            ['0x1.6b5450a9bbaedp-6', '0x1.02eaf34002d7cp-9', '0x1.a8fb5c86dc5e1p-13']),
                           ('0x1.b7aa486590c52p+5',
                            '0x1.8f52eb1b1fb1fp-6',
                            '0x1.67b0004e81c00p-16',
                            ['0x1.6b4b48425260bp-6', '0x1.02dfa8f123052p-9', '0x1.a8e0dd4aa817fp-13']),
                           ('0x1.2e0a2f06a0c0ap+7',
                            '0x1.8f1ccb6354837p-6',
                            '0x1.6718703d1bc00p-16',
                            ['0x1.6b1d816c65dedp-6', '0x1.02a674947bc57p-9', '0x1.a85aa427f2838p-13']),
                           ('0x1.9efd18a83b4b1p+8',
                            '0x1.8eaa142fc76a3p-6',
                            '0x1.65d7beda78c00p-16',
                            ['0x1.6abc78d067fbdp-6', '0x1.022d44475bc3bp-9', '0x1.a73e735eabdd7p-13']),
                           ('0x1.1d166c49138b9p+10',
                            '0x1.8d700dd4a4825p-6',
                            '0x1.626da7401d400p-16',
                            ['0x1.69b2c808bb80dp-6', '0x1.00e1ea2bf4257p-9', '0x1.a4368e4d3abd5p-13']),
                           ('0x1.5cf664b9ca5dap+13',
                            '0x1.3cb6ab477c7bep-6',
                            '0x1.5c2dbda55d800p-17',
                            ['0x1.2481ce04fd91ep-6',
                             '0x1.61c8e27cdf2b5p-10',
                             '0x1.ecc962fc4b942p-14'])],
                  'decay': [('0x1.ed81e92b54912p+6', '0x1.6752ace1e8f1ap-5', '0x1.62dd66a610cd8p-5'),
                            ('0x1.ed81e92b54912p+7', '0x1.674063df8d955p-5', '0x1.62cb53c9fa89bp-5'),
                            ('0x1.ed81e92b54912p+8', '0x1.6727bf7bffc72p-5', '0x1.62b2f8cdf8023p-5'),
                            ('0x1.ed81e92b54912p+9', '0x1.66f05b725f482p-5', '0x1.627c3b8d48c8bp-5'),
                            ('0x1.ed81e92b54912p+10', '0x1.6627241c8ee2cp-5', '0x1.61b5766be43cfp-5'),
                            ('0x1.ed81e92b54912p+11', '0x1.632ddbbea692ep-5', '0x1.5ec679a3bb527p-5'),
                            ('0x1.ed81e92b54912p+12', '0x1.5944587a448a7p-5', '0x1.5508c2f84182dp-5')]}}


# 雲端 verify 在 job 層把 numpy／OpenBLAS／XLA 釘在 AVX2（見 tests/conftest.py 的機器那一行）；
# child_process_env 不繼承它們，照樣帶下去，子行程才跟父行程走同一級指令（錄答案那題已改比容差）。
_CPU_PIN_VARS = ("NPY_DISABLE_CPU_FEATURES", "OPENBLAS_CORETYPE", "XLA_FLAGS")


def _child(directory: Path, mode: str, *, x64: bool) -> dict[str, object]:
    env = child_process_env(threads=1).copy()
    env.update({name: os.environ[name] for name in _CPU_PIN_VARS if name in os.environ})
    configure_jax(platforms="cpu", enable_x64=x64, env=env)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    root = Path(__file__).resolve().parents[2]
    env["PYTHONPATH"] = os.pathsep.join((env["PYTHONPATH"], str(root)))
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
def test_main_control_within_tolerance_without_jax_calls(tmp_path: Path, x64: bool) -> None:
    """抓正式固定阻抗準備碰JAX，或晚期能量、逐階、尾段、T60、f_s、T20/T30 超出跨處理器容差的漂移。

    頻率是鍵，照舊逐位（tests/engine/_cross_cpu_tolerance.py）。T60、f_s 是艾林閉式公式、不解方程組，走①級；
    晚期能量、逐階、尾段、T20/T30 走②級（含解線性方程組的晚期家族）。尾段是總量減逐階會相消，絕對尺取同列
    晚期總量答案 × ②級。控制場景是固定阻抗，500 與 1000 Hz 的 T60 只差空氣那一份（#1057），牆面吸音率兩帶
    相同，所以「兩帶抄錯牆面頻率」這題抓不到，那一類錯由 test_construction_family_matches_direct 守。
    """
    actual = cast(dict[str, dict[str, object]], _child(tmp_path, "constant", x64=x64))
    assert {name: set(scene) for name, scene in actual.items()} == \
        {name: set(scene) - {"walls"} for name, scene in CONTROL.items()}
    cells: list[Cell] = []
    for name, scene in CONTROL.items():
        got = actual[name]
        t60 = cast(dict[str, str], scene["t60"])
        assert cast(dict[str, str], got["t60"]).keys() == t60.keys()
        cells += [cell(f"{name}.t60[{band}]", cast(dict[str, str], got["t60"])[band], value,
                       rel=GEOMETRIC_REL, abs_=0.0) for band, value in t60.items()]
        cells.append(cell(f"{name}.fs", cast(str, got["fs"]), cast(str, scene["fs"]), rel=GEOMETRIC_REL, abs_=0.0))
        late = cast(list[tuple[str, str, str, list[str]]], scene["late"])
        for row, (have, want) in enumerate(zip(cast(list[list[object]], got["late"]), late, strict=True)):
            assert have[0] == want[0]
            total = float.fromhex(want[1])
            cells.append(cell(f"{name}.late[{row}].total", cast(str, have[1]), want[1], rel=LINEAR_SOLVE_REL, abs_=0.0))
            cells.append(cell(f"{name}.late[{row}].tail", cast(str, have[2]), want[2],
                              rel=LINEAR_SOLVE_REL, abs_=LINEAR_SOLVE_REL * total))
            cells += [cell(f"{name}.late[{row}].order[{k}]", h, w, rel=LINEAR_SOLVE_REL, abs_=0.0)
                      for k, (h, w) in enumerate(zip(cast(list[str], have[3]), want[3], strict=True))]
        decay = cast(list[tuple[str, str, str]], scene["decay"])
        for row, (seen, wanted) in enumerate(zip(cast(list[list[str]], got["decay"]), decay, strict=True)):
            assert seen[0] == wanted[0]
            cells += [cell(f"{name}.decay[{row}].{label}", seen[k], wanted[k], rel=LINEAR_SOLVE_REL, abs_=0.0)
                      for k, label in ((1, "t20"), (2, "t30"))]
    assert_within(cells)


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
