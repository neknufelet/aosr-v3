"""逐頻複數有限元素：主線逐位控制、時間慣例、構造接線與鑰匙。"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from aosr.geometry.shoebox import Wall
from aosr.physics.fem_helmholtz import FrequencyWallImpedances
from aosr.runtime import child_process_env, configure_jax

# 主線 eae1de332796a6582899fc6923d95e33883d1e2c 實跑；施工資料 control 目錄
# main-fem-mkl-compatible.json（pin_fem.py）與 main-femkey.json，原數字逐字保留。
CONTROL = {'density_kg_m3': 1.2,
 'fem_energy': {'s1|r1': ['0x1.8672aacf81600p+6',
                          '0x1.6de39ccd7392dp+6',
                          '0x1.5cc37c978f86ap+4',
                          '0x1.21c1c95d33534p+2',
                          '0x1.5ff6e5843d271p-1',
                          '0x1.e1038b0d9fa92p-1'],
                's1|r2': ['0x1.7292f067fb789p+6',
                          '0x1.5a45960afce6bp+6',
                          '0x1.302250bd86f28p+4',
                          '0x1.e5f7f480120fbp+3',
                          '0x1.448db0a0ba3d9p+0',
                          '0x1.81274fa790b1cp-5'],
                's2|r1': ['0x1.31115e6c94fe1p+6',
                          '0x1.198b0e4f3cef5p+6',
                          '0x1.f8f1f01d15a2dp+2',
                          '0x1.fb1674b682d54p+1',
                          '0x1.590c948d2f34fp-4',
                          '0x1.4c4e04b9f2e13p+1'],
                's2|r2': ['0x1.7747b613ce67fp+6',
                          '0x1.5ed1dd565458fp+6',
                          '0x1.27e5b13dbb6f5p+4',
                          '0x1.46b29e1c8b472p+0',
                          '0x1.d9bdb5ba1ce76p-1',
                          '0x1.0688939407768p+1']},
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

# MKL 預設依 CPU 選程式路徑，跨 AVX-512 機器可能差最後一位；COMPATIBLE
# 是 MKL 文件的跨廠牌 CPU 重現模式。本機已量，其他 CPU 要等雲端 verify。
# 改動在 MKL 上游；矩陣逐位相同，在任何固定模式下解都相同，故此比對有效。
MKL_REPRODUCIBILITY_MODE = "COMPATIBLE"
_CPU_PIN_VARS = ("NPY_DISABLE_CPU_FEATURES", "OPENBLAS_CORETYPE", "XLA_FLAGS")


def _child(directory: Path, mode: str, *, x64: bool = True) -> dict[str, object]:
    env = child_process_env(threads=1).copy()
    env.update({name: os.environ[name] for name in _CPU_PIN_VARS if name in os.environ})
    env["MKL_CBWR"] = MKL_REPRODUCIBILITY_MODE
    configure_jax(platforms="cpu", enable_x64=x64, env=env)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["TMPDIR"] = str(directory)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    output = directory / "result.json"
    process = subprocess.run([sys.executable, str(Path(__file__).with_name("_fem_complex_probe.py")),
                              mode, str(output)], cwd=directory, env=env,
                             input=json.dumps(CONTROL), capture_output=True, text=True, timeout=180)
    assert process.returncode == 0, process.stdout + process.stderr
    return cast(dict[str, object], json.loads(output.read_text()))


@pytest.mark.parametrize("x64", (False, True))
def test_main_control_bitwise_without_jax_calls(tmp_path: Path, x64: bool) -> None:
    """抓正式單份、批次、搜尋分片固定阻抗碰 JAX 或能量末位漂移。"""
    result = _child(tmp_path, "constant", x64=x64)
    assert result["energy"] == CONTROL["fem_energy"]
    assert result["key"] == MAIN_KEY_JSON


def test_construction_matches_direct_and_keys(tmp_path: Path) -> None:
    """抓構造未接到兩條路、用錯空氣或頻率軸、鑰匙漏掉實虛部。"""
    assert _child(tmp_path, "construction")["verified"] is True


def test_air_gap_equivalent_to_extended_room(tmp_path: Path) -> None:
    """抓正時間指數用成共軛：剛背空氣層必須等價於延長房間。"""
    result = _child(tmp_path, "convention")
    correct = cast(list[float], result["correct"])
    conjugated = cast(list[float], result["conjugated"])
    assert all(wrong > 10 * right for right, wrong in zip(correct, conjugated, strict=True))


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


def test_single_frequency_accepts_passive_complex() -> None:
    """抓組裝閘門仍只接受正實數；純電抗不是主動材料。"""
    from aosr.physics.fem_helmholtz import _checked_impedances
    walls: dict[Wall, float | complex | None] = {wall: 1000. for wall in Wall.all()}
    walls[Wall.XL] = -100j
    assert _checked_impedances(walls)[Wall.XL] == -100j
