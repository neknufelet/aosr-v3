"""逐頻複數有限元素：替身解法器下的主線逐位控制（只驗組裝與接線）、真解法器的空氣層對照（時間慣例與誤差上限）、構造接線與鑰匙。"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

import aosr
from aosr.geometry.shoebox import Room, Wall
from aosr.geometry.shoebox_mesh import ShoeboxMesh
from aosr.physics.fem_helmholtz import FrequencyWallImpedances, WallImpedances
from aosr.runtime import child_process_env, configure_jax
from tests.engine._precision_contracts import MUTANT_MARGIN, contract_value

# 輸入取施工資料 main-fem-mkl-compatible.json（工作區 309-materials 的 control 目錄，pin_fem.py）；
# stand_in_energy 是主線 c2b7c4825916019f3930b6d8afa0c7fe0d2bdc25 在替身解法器
# （tests/engine/_solver_stand_in.py）下五條路（批次、正式逐對、正式批次、單對、搜尋分片）實跑、
# 彼此逐位相同的能量（施工資料 main-digest-pins.json，工作區 309-fem-complex，pin_standin.py）；
# MAIN_KEY_JSON 取 main-femkey.json。原數字逐字保留。
CONTROL = {'density_kg_m3': 1.2,
 'stand_in_energy': {'s1|r1': ['0x1.5deec82d3c7fap+3',
                               '0x1.3a5009b5bc0b3p+3',
                               '0x1.2674acb960156p+3',
                               '0x1.745093f77ebb0p+3',
                               '0x1.78d7770d48a43p+3',
                               '0x1.519a2d4097ed4p+3'],
                     's1|r2': ['0x1.f89ceed12116fp+2',
                               '0x1.c53fa72ebc586p+2',
                               '0x1.a8a6438038edbp+2',
                               '0x1.0c8f3f54119dap+3',
                               '0x1.102c9e3c04c33p+3',
                               '0x1.ea17b28f0c159p+2'],
                     's2|r1': ['0x1.4ce3239dfbcb5p+3',
                               '0x1.6d87db27a3778p+3',
                               '0x1.ec4cf89f158ffp+3',
                               '0x1.82bf6bfcba51cp+3',
                               '0x1.b05be5fff20f6p+3',
                               '0x1.53c830eb5c804p+3'],
                     's2|r2': ['0x1.e0086f680dd79p+2',
                               '0x1.078da6e5e5abcp+3',
                               '0x1.62fc5c33a80b9p+3',
                               '0x1.16f86d431aa4ep+3',
                               '0x1.3845902eedb62p+3',
                               '0x1.ed41c25e33b60p+2']},
 'frequencies_hz': ['0x1.4000000000000p+4',
                    '0x1.496070711ab45p+4',
                    '0x1.36e3e588d7b90p+5',
                    '0x1.36e3e588d7b90p+6',
                    '0x1.2e0a2f06a0c0ap+7',
                    '0x1.2570f8a1d46d4p+8'],
 'impedance_pa_s_per_m_by_wall': {'ceiling': 1646.4,
                                  'floor': 8200.0,
                                  'x0': 3100.0,
                                  'xL': 950.0,
                                  'y0': 12500.0,
                                  'yL': 2300.0},
 'receivers': {'r1': [1.5, 0.9, 0.7], 'r2': [0.7, 1.4, 1.1]},
 'room_m': [2.3, 1.7, 1.4],
 'sound_speed_m_s': 343.0,
 'sources': {'s1': [0.4, 0.5, 0.6], 's2': [1.9, 1.2, 0.9]}}
MAIN_KEY_JSON = (
    '{"room_hex":["0x1.2666666666666p+1","0x1.b333333333333p+0","0x1.6666666666666p+0"],"impedance_he'
    'x":["0x1.0040000000000p+13","0x1.9b9999999999ap+10","0x1.8380000000000p+11","0x1.db00000000000p+'
    '9","0x1.86a0000000000p+13","0x1.1f80000000000p+11"],"density_hex":"0x1.3333333333333p+0","speed_'
    'hex":"0x1.5700000000000p+8","frequency_hex":["0x1.4000000000000p+4","0x1.496070711ab45p+4","0x1.'
    '53073778fd7bcp+4","0x1.5cf664b9ca5dap+4","0x1.6730174b9cc70p+4","0x1.71b67e3083c7cp+4","0x1.7c8b'
    'd8cbe24dap+4","0x1.87b2775d4eda4p+4","0x1.932cbb7f0cf2ep+4","0x1.9efd18a83b4b1p+4","0x1.ab2614b2'
    'd2776p+4","0x1.b7aa486590c52p+4","0x1.c48c6001f0ac0p+4","0x1.d1cf1bd648293p+4","0x1.df7550d43039'
    'cp+4","0x1.ed81e92b54912p+4","0x1.fbf7e4e8cc9cbp+4","0x1.056d2d4d8fed4p+5","0x1.0d163bfd8c44cp+5'
    '","0x1.14f8c14cc461cp+5","0x1.1d166c49138bap+5","0x1.2570f8a1d46d4p+5","0x1.2e0a2f06a0c0ap+5","0'
    'x1.36e3e588d7b90p+5","0x1.4000000000000p+5","0x1.496070711ab45p+5","0x1.53073778fd7bcp+5","0x1.5'
    'cf664b9ca5dap+5","0x1.6730174b9cc70p+5","0x1.71b67e3083c7cp+5","0x1.7c8bd8cbe24dap+5","0x1.87b27'
    '75d4eda4p+5","0x1.932cbb7f0cf2ep+5","0x1.9efd18a83b4b1p+5","0x1.ab2614b2d2776p+5","0x1.b7aa48659'
    '0c52p+5","0x1.c48c6001f0ac0p+5","0x1.d1cf1bd648294p+5","0x1.df7550d43039cp+5","0x1.ed81e92b54912'
    'p+5","0x1.fbf7e4e8cc9ccp+5","0x1.056d2d4d8fed3p+6","0x1.0d163bfd8c44cp+6","0x1.14f8c14cc461cp+6"'
    ',"0x1.1d166c49138bap+6","0x1.2570f8a1d46d4p+6","0x1.2e0a2f06a0c0ap+6","0x1.36e3e588d7b90p+6","0x'
    '1.4000000000000p+6","0x1.496070711ab44p+6","0x1.53073778fd7bcp+6","0x1.5cf664b9ca5dap+6","0x1.67'
    '30174b9cc6fp+6","0x1.71b67e3083c7ep+6","0x1.7c8bd8cbe24dap+6","0x1.87b2775d4eda3p+6","0x1.932cbb'
    '7f0cf2ep+6","0x1.9efd18a83b4b1p+6","0x1.ab2614b2d2774p+6","0x1.b7aa486590c53p+6","0x1.c48c6001f0'
    'ac0p+6","0x1.d1cf1bd648293p+6","0x1.df7550d43039dp+6","0x1.ed81e92b54912p+6","0x1.fbf7e4e8cc9cbp'
    '+6","0x1.056d2d4d8fed4p+7","0x1.0d163bfd8c44cp+7","0x1.14f8c14cc461cp+7","0x1.1d166c49138bap+7",'
    '"0x1.2570f8a1d46d4p+7","0x1.2e0a2f06a0c0ap+7","0x1.36e3e588d7b91p+7","0x1.4000000000000p+7","0x1'
    '.496070711ab44p+7","0x1.53073778fd7bcp+7","0x1.5cf664b9ca5dap+7","0x1.6730174b9cc6fp+7","0x1.71b'
    '67e3083c7ep+7","0x1.7c8bd8cbe24dap+7","0x1.87b2775d4eda3p+7","0x1.932cbb7f0cf2ep+7","0x1.9efd18a'
    '83b4b1p+7","0x1.ab2614b2d2774p+7","0x1.b7aa486590c53p+7","0x1.c48c6001f0ac0p+7","0x1.d1cf1bd6482'
    '93p+7","0x1.df7550d43039dp+7","0x1.ed81e92b54912p+7","0x1.fbf7e4e8cc9cbp+7","0x1.056d2d4d8fed4p+'
    '8","0x1.0d163bfd8c44cp+8","0x1.14f8c14cc461cp+8","0x1.1d166c49138bap+8","0x1.2570f8a1d46d4p+8"]}'
)

_CPU_PIN_VARS = ("NPY_DISABLE_CPU_FEATURES", "OPENBLAS_CORETYPE", "XLA_FLAGS")


def _child(directory: Path, mode: str, *, x64: bool = True) -> dict[str, object]:
    env = child_process_env(threads=1).copy()
    env.update({name: os.environ[name] for name in _CPU_PIN_VARS if name in os.environ})
    configure_jax(platforms="cpu", enable_x64=x64, env=env)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["TMPDIR"] = str(directory)
    # 保留 child_process_env 指給的被測套件來源，再加 repo 根（探針要 import tests.engine）；
    # 整格蓋掉會讓子行程改從環境登記的安裝路徑載入，考的就不一定是這一份程式。
    env["PYTHONPATH"] = os.pathsep.join((env["PYTHONPATH"], str(Path(__file__).resolve().parents[2])))
    output = directory / "result.json"
    process = subprocess.run([sys.executable, str(Path(__file__).with_name("_fem_complex_probe.py")),
                              mode, str(output)], cwd=directory, env=env,
                             input=json.dumps(CONTROL), capture_output=True, text=True, timeout=180)
    assert process.returncode == 0, process.stdout + process.stderr
    result = cast(dict[str, object], json.loads(output.read_text()))
    assert result.pop("aosr_file") == str(Path(aosr.__file__).resolve()), "子行程載入的不是被測的那一份程式"
    return result


@pytest.mark.parametrize("x64", (False, True))
def test_stand_in_control_matches_main_without_jax_calls(tmp_path: Path, x64: bool) -> None:
    """抓正式單份、批次、搜尋分片固定阻抗碰 JAX，或替身解法器下六個控制頻點的能量跟主線（同一份替身錄的）不同。

    這是組裝與接線的回歸，不驗解法器、不驗聲場。

    真的 PARDISO 跨 CPU 差末幾位（雲端 9V74 量到），所以子行程換成替身解法器再釘主線：
    矩陣或右側變了一位，替身的解就整個換掉；比的是平方後的能量，取點後聲壓的末位差可能被捨入吃掉。
    """
    result = _child(tmp_path, "constant", x64=x64)
    assert result["energy"] == CONTROL["stand_in_energy"]
    assert result["key"] == MAIN_KEY_JSON


def test_construction_matches_direct_and_keys(tmp_path: Path) -> None:
    """抓構造未接到兩條路、用錯空氣或頻率軸、鑰匙漏掉實虛部（替身解法器：只驗接線與鑰匙，不驗聲場）。"""
    assert _child(tmp_path, "construction")["verified"] is True


def test_construction_reaches_single_pair_path(tmp_path: Path) -> None:
    """抓逐對求解那條正式路（非批次）還拿舊的六個實數、沒吃到構造（植錯曾存活；替身解法器：只驗接線與鑰匙，不驗聲場）。"""
    assert _child(tmp_path, "construction_single")["verified"] is True


def test_construction_uses_scene_air_not_config(tmp_path: Path) -> None:
    """抓構造在有限元素路誤用設定檔的密度或聲速（控制場景剛好等於設定檔，前一題分不出來；不解方程，只比阻抗列）。"""
    assert _child(tmp_path, "scene_air")["verified"] is True


AIR_GAP_TOLERANCE = contract_value("fem_air_gap_vs_extended_room")


def _air_gap_holds(correct: list[float], conjugated: list[float], tolerance: float) -> bool:
    """逐頻點：正確慣例對延長房的相對差在界線內，而且共軛那一版的差大於它十倍。"""
    return all(right <= tolerance and wrong > 10 * right
               for right, wrong in zip(correct, conjugated, strict=True))


def test_air_gap_equivalent_to_extended_room(tmp_path: Path) -> None:
    """真解法器。抓兩件事：時間慣例接反（共軛那一版的差要大於正確那一版十倍以上），以及有限元素吃逐頻純電抗
    阻抗時的量值錯（正確那一版四個頻點逐點不超過登記簿 ``fem_air_gap_vs_extended_room``）。

    證明的是有限元素對有限元素：兩間房在橫向截止以下物理上等價，差的只有兩張網格的離散誤差；不證明聲壓對
    真值的精度、不證明多孔層或板、不證明真實房間裡的延伸反應。
    """
    result = _child(tmp_path, "convention")
    correct = cast(list[float], result["correct"])
    conjugated = cast(list[float], result["conjugated"])
    assert _air_gap_holds(correct, conjugated, AIR_GAP_TOLERANCE)


def test_air_gap_mutant_beyond_tolerance_is_red() -> None:
    """正確慣例的差在界線內推 δ 判綠、有一點推到界線外 δ 判紅；有一點的共軛差不到十倍也判紅。"""
    inside = [(1.0 - MUTANT_MARGIN) * AIR_GAP_TOLERANCE] * 4
    conjugated = [20 * AIR_GAP_TOLERANCE] * 4
    assert _air_gap_holds(inside, conjugated, AIR_GAP_TOLERANCE)
    for index in range(4):
        outside = [*inside[:index], (1.0 + MUTANT_MARGIN) * AIR_GAP_TOLERANCE, *inside[index + 1:]]
        assert not _air_gap_holds(outside, conjugated, AIR_GAP_TOLERANCE)
        near = [*conjugated[:index], 9 * inside[index], *conjugated[index + 1:]]
        assert not _air_gap_holds(inside, near, AIR_GAP_TOLERANCE)


@pytest.mark.parametrize("invalid", (-1+2j, complex(float("inf"), 1), complex(1, float("nan")), 0j, None))
def test_gate_rejects_invalid_frequency_cell(invalid: complex | None) -> None:
    """抓負實部、非有限、零與同面混剛性；錯誤須指出牆與壞頻點。"""
    from aosr.physics.fem_helmholtz import checked_impedance_rows
    walls: dict[Wall, tuple[complex | None, ...]] = {wall: (1000+0j, 1000+0j) for wall in Wall.all()}
    walls[Wall.XL] = (1000+0j, invalid)
    with pytest.raises(ValueError, match=r"xL.*95"):
        checked_impedance_rows(walls, (45., 95.))


def test_gate_accepts_pure_reactance_and_rigid_rows() -> None:
    """抓把被動零實部誤拒收，或剛性整列誤當混合。"""
    from aosr.physics.fem_helmholtz import checked_impedance_rows
    walls: FrequencyWallImpedances = {**{wall: None for wall in Wall.all()}, Wall.XL: (-100j, -200j)}
    assert checked_impedance_rows(walls, (45., 95.))[Wall.XL] == (-100j, -200j)


@pytest.mark.parametrize("value", (("-0x1.0000000000000p+0", "0x1.0000000000000p+0"),
                                   ("0x0.0p+0", "0x0.0p+0"), ("inf", "0x1.0000000000000p+0")))
def test_key_rejects_invalid_complex_cell(value: tuple[str, str]) -> None:
    """抓快取讀回驗證未守有限、被動與非零；實虛部不套正數規則。"""
    from aosr.reporting.fem_slices import FemKey
    data = json.loads(MAIN_KEY_JSON)
    data["impedance_hex"][3] = [value] * len(data["frequency_hex"])
    with pytest.raises(ValueError):
        FemKey.model_validate(data)


@pytest.fixture(scope="module")
def tiny_mesh() -> ShoeboxMesh:
    from aosr.config.fem_lane import FEM_ELEMENTS_PER_WAVELENGTH, FEM_MESH_RANDOM_SEED
    from aosr.geometry.shoebox_mesh import generate_shoebox_mesh
    return generate_shoebox_mesh(Room(1.0, .8, .6), max_frequency_hz=80., sound_speed_m_s=343.,
        elements_per_wavelength=FEM_ELEMENTS_PER_WAVELENGTH, random_seed=FEM_MESH_RANDOM_SEED)


@pytest.mark.parametrize("bad", (-100j, 1000 + 5j, 0.0, -1.0, float("nan"), True))
def test_modal_entries_still_refuse_non_real_impedance(tiny_mesh: ShoeboxMesh, bad: object) -> None:
    """抓共用閘門放寬後模態悄悄收複數或純電抗：模態特徵束只取係數虛部，純電抗那面牆會被當成剛性（審查確證）。"""
    from aosr.physics.fem_helmholtz import assemble_p2_operators, checked_real_impedances
    from aosr.physics.fem_modal import FemModalSpectrum, solve_fem_modes
    from aosr.physics.fem_modal_participation import prepare_modal_expansion
    walls = cast(WallImpedances, {**{wall: 1000. for wall in Wall.all()}, Wall.XL: bad})
    with pytest.raises(ValueError, match="xL"):
        checked_real_impedances(walls)
    with pytest.raises(ValueError, match="xL"):
        solve_fem_modes(tiny_mesh, wall_impedances=walls, density_kg_m3=1.2, sound_speed_m_s=343.,
                        frequency_max_hz=50.)
    with pytest.raises(ValueError, match="xL"):
        prepare_modal_expansion(cast(FemModalSpectrum, None), assemble_p2_operators(tiny_mesh),
                                wall_impedances=walls, density_kg_m3=1.2, sound_speed_m_s=343.)


def test_single_frequency_accepts_passive_complex() -> None:
    """抓組裝閘門仍只接受正實數；純電抗不是主動材料。"""
    from aosr.physics.fem_helmholtz import _checked_impedances
    walls: dict[Wall, float | complex | None] = {wall: 1000. for wall in Wall.all()}
    walls[Wall.XL] = -100j
    assert _checked_impedances(walls)[Wall.XL] == -100j
