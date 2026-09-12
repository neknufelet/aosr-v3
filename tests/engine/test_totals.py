"""票 #206 第 2a 段的引擎考卷：v3 把 63 條路徑加總，跟兩組答案檔 ``totals`` 依契約比。

**這一支是引擎考卷**（住 ``tests/engine/``，規矩卡 ``green-must-be-real-green`` 的引擎籃）。
它 import 新家的 ``aosr``：``aosr.physics.totals`` 是總量的家，``aosr.physics.amplitude`` 是
每條路徑界線的家，``aosr.physics.compare`` 是比對程式搬家後的家。兩組振幅答案
（flat／varied）的 ``totals`` 從唯讀的 ``blueprint/reference_amplitude_{flat,varied}.json`` 讀。

**契約（決策紙 docs/decisions/precision-contract-totals-root-sum-square.md，選項 1）。**
總壓力絕對差界線 ``sqrt(Σ_k (tol_k·|p_k|)²)``、反射能量相對界線
``2·sqrt(Σ_{k>0}(tol_k·|p_k|)²)/|Σ_{k>0} p_k| + 2^-23``、直達能量相對界線固定 ``2^-23``。
界線函式從 ``aosr.physics.totals`` 這一個地方來（``total_pressure_tolerance`` 等），判決函式
呼叫時走模組名，所以 ``monkeypatch.setattr(totals, "total_pressure_tolerance", …)`` 要真的生效。

**定義照上一代（決策紙 stage-five-ism-totals-and-energy）。** 總壓力是 63 條路徑壓力的複數
相加；直達能量是直達那一條壓力的模平方；反射能量是 order>0 的 62 條**先複數相加再取模平方**
（交叉項刻意不算）。借 ``test_amplitude`` 的 ``_input_case``／``_write_input``／``_v3_paths``
從答案檔參數組輸入、跑 v3，不複製。

**控制組（證明裁判咬得住）。** ①把三支界線函式各換 0 要對應的那一格全體超界；②一條 order 3
路徑壓力換共軛 → 總壓力超界；③答案檔壓力某格 hex 改到界線外 → 超界；④答案檔直達能量某格
乘 (1+2^-20) → 直達超界；⑤答案檔 totals 少一格（刪 ism_rev_E）→ ValueError 指名那一欄。

**不寫死條數、不碰真環境。** 頻帶數用 ``len(frequencies)``、63 條用 ``len(paths)``、反射那
62 條用 ``order > 0`` 過濾；只寫 ``tmp_path``（規矩卡 tests-isolated-from-real-env）。
"""
from __future__ import annotations

import copy
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from aosr.physics import amplitude as amp
from aosr.physics import totals
from aosr.physics.room_paths import RoomPath
from tests.engine.test_amplitude import (
    _amplitude_path,
    _answer_params,
    _as_float_list,
    _v3_paths,
)

_AMPLITUDE_CASES: tuple[str, ...] = ("flat", "varied")


def _answer_totals(case: str) -> dict[str, object]:
    """兩組振幅答案檔的 ``totals``（唯讀）。"""
    with _amplitude_path(case).open(encoding="utf-8") as handle:
        data = json.load(handle)
    totals_node = data["totals"]
    assert isinstance(totals_node, dict)
    return totals_node


def _frequencies(case: str) -> tuple[float, ...]:
    """答案檔參數的頻率清單（六個頻帶）。"""
    return tuple(_as_float_list(_answer_params(case)["frequencies_hz"], "frequencies_hz"))


def _direct_path(paths: list[RoomPath]) -> RoomPath:
    """找到 order 0 的直達路徑（「恰好一條」由 totals_from_paths 的 ValueError 守）。"""
    direct = [p for p in paths if p.order == 0]
    assert direct, "找不到 order 0 的直達路徑"
    return direct[0]


# ── 兩組答案檔：每格在契約內，三個 max_frac 都 > 0 且 ≤ 1 ─────────────────────────


@pytest.mark.parametrize("case", _AMPLITUDE_CASES)
def test_compare_totals_within_contract(tmp_path: Path, case: str) -> None:
    """兩組答案檔的 ``compare_totals`` 差異清單為空；三個 max_frac 印進失敗訊息、且 (0,1]。"""
    paths = _v3_paths(tmp_path, case)
    freqs = _frequencies(case)
    ans = _answer_totals(case)

    result = totals.compare_totals(paths, ans, freqs)
    assert result.diffs == [], (
        f"{case} 超出契約的格有 {len(result.diffs)} 個；總壓力用到界線 "
        f"{result.max_pressure_frac:.6f}、直達能量 {result.max_direct_frac:.6f}、"
        f"反射能量 {result.max_reflected_frac:.6f}。前幾筆：{result.diffs[:5]!r}"
    )
    # 決策紙 precision-contract-totals-root-sum-square 實測 16.5%／10.1%／5.8%，
    # 界線被放鬆十倍這裡就紅。
    assert result.max_pressure_frac >= 0.10
    assert result.max_reflected_frac >= 0.05
    assert result.max_direct_frac >= 0.03
    assert result.max_pressure_frac <= 1.0
    assert result.max_direct_frac <= 1.0
    assert result.max_reflected_frac <= 1.0


# ── 定義題（照階段五的定義）────────────────────────────────────────────────────


@pytest.mark.parametrize("case", _AMPLITUDE_CASES)
def test_totals_definition_matches_stage_five(tmp_path: Path, case: str) -> None:
    """直達與反射能量守定義；pressure 的順序相加誤差不超過標準相加上界。"""
    paths = _v3_paths(tmp_path, case)
    freqs = _frequencies(case)
    t = totals.totals_from_paths(paths)
    direct = _direct_path(paths)

    for i in range(len(freqs)):
        # |exp(-iωτ)|² 在浮點下差最多幾個 ulp。
        expected_direct_energy = (1.0 / direct.dist_m) ** 2
        assert math.isclose(
            t.direct_energy[i],
            expected_direct_energy,
            rel_tol=4 * sys.float_info.epsilon,
        ), (
            f"{case} 直達能量[{i}] 跟 1／距離平方對不上"
        )

        # 反射能量不准把直達也算進去。
        all_sum = sum(p.path_pressure[i] for p in paths)
        assert t.reflected_energy[i] != abs(all_sum) ** 2, (
            f"{case} 反射能量[{i}] 竟等於把直達也加進去的數（定義被突變）"
        )

        fsum = complex(
            math.fsum(p.path_pressure[i].real for p in paths),
            math.fsum(p.path_pressure[i].imag for p in paths),
        )
        # n 次相加，每次一個機器 epsilon：n·epsilon·Σ|p_k|。
        addition_bound = (
            len(paths)
            * sys.float_info.epsilon
            * sum(abs(p.path_pressure[i]) for p in paths)
        )
        assert abs(t.pressure[i] - fsum) <= addition_bound, (
            f"{case} 總壓力[{i}] 跟 math.fsum 分實虛的差超出相加誤差上界"
        )


def test_totals_from_paths_rejects_no_paths() -> None:
    """刪掉空清單守門就會讓這題失敗。"""
    with pytest.raises(ValueError, match="沒有路徑"):
        totals.totals_from_paths([])


def test_totals_from_paths_rejects_path_without_pressure(tmp_path: Path) -> None:
    """刪掉 path_pressure 空值守門就會讓這題失敗。"""
    paths = _v3_paths(tmp_path, "flat")
    empty = replace(paths[0], path_pressure=())

    with pytest.raises(ValueError, match="沒有振幅"):
        totals.totals_from_paths([empty, *paths[1:]])


def test_totals_from_paths_rejects_two_direct_paths(tmp_path: Path) -> None:
    """放行兩條 order 0 會讓這題失敗。"""
    paths = _v3_paths(tmp_path, "flat")
    second_direct = replace(paths[1], order=0)

    with pytest.raises(ValueError, match="恰好一條"):
        totals.totals_from_paths([paths[0], second_direct, *paths[2:]])


def test_totals_from_paths_rejects_mismatched_reflection_product(tmp_path: Path) -> None:
    """reflection_product 跟 path_pressure 不同長就要指名該路徑。"""
    paths = _v3_paths(tmp_path, "flat")
    victim = paths[1]
    tampered = replace(victim, reflection_product=())

    with pytest.raises(ValueError, match=str(victim.identity)):
        totals.totals_from_paths([paths[0], tampered, *paths[2:]])


def test_compare_totals_rejects_our_frequency_count(tmp_path: Path) -> None:
    """frequencies 少一格時先抓我方 Totals 的頻帶數。"""
    paths = _v3_paths(tmp_path, "flat")
    freqs = _frequencies("flat")

    with pytest.raises(
        ValueError,
        match=rf"我方頻帶數 {len(freqs)} 跟 frequencies 的 {len(freqs[:-1])} 對不上",
    ):
        totals.compare_totals(paths, _answer_totals("flat"), freqs[:-1])


# ── 控制組（每一條都要真的讓判決函式吃到壞東西）──────────────────────────────────


@pytest.mark.parametrize(
    "bound_name,worst_kind",
    [
        ("total_pressure_tolerance", "pressure"),
        ("reflected_energy_tolerance", "reflected_energy"),
        ("direct_energy_tolerance", "direct_energy"),
    ],
)
def test_control_group_zero_bound_goes_red(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bound_name: str, worst_kind: str
) -> None:
    """把三支界線函式各換 0，對應的那一格全體超界（界線真的被吃進去）。"""
    paths = _v3_paths(tmp_path, "flat")
    freqs = _frequencies("flat")
    ans = _answer_totals("flat")

    monkeypatch.setattr(totals, bound_name, lambda paths, i, freqs: 0.0)
    result = totals.compare_totals(paths, ans, freqs)
    assert len(result.diffs) == len(freqs), (
        f"{bound_name} 換 0 之後應該每個頻帶都紅，得到 {len(result.diffs)} 格"
    )
    assert result.worst is not None
    assert result.worst[0] == worst_kind, (
        f"{bound_name} 換 0 之後 worst 的量名應是 {worst_kind!r}，得到 {result.worst[0]!r}"
    )


def test_control_group_conjugated_order3_path_over_budget(tmp_path: Path) -> None:
    """把一條 order 3 路徑的壓力換共軛 → 總壓力那格超界。"""
    paths = _v3_paths(tmp_path, "flat")
    freqs = _frequencies("flat")
    ans = _answer_totals("flat")

    victims = [p for p in paths if p.order == 3]
    assert victims, "找不到 order 3 路徑"
    victim = victims[0]
    tampered = replace(
        victim,
        path_pressure=tuple(complex(z.real, -z.imag) for z in victim.path_pressure),
    )
    tampered_list = [tampered if p is victim else p for p in paths]

    result = totals.compare_totals(tampered_list, ans, freqs)
    assert any("pressure[" in d for d in result.diffs), (
        f"共軛一條 order 3 路徑之後總壓力沒超界：{result.diffs[:5]!r}"
    )


def test_control_group_pressure_hex_out_of_budget(tmp_path: Path) -> None:
    """答案檔 totals.pressure 某頻帶 real.hex 改到界線外 → 超界。"""
    paths = _v3_paths(tmp_path, "flat")
    freqs = _frequencies("flat")
    ans = copy.deepcopy(_answer_totals("flat"))

    pressure_cells = ans["pressure"]
    assert isinstance(pressure_cells, list)
    cell = pressure_cells[0]
    assert isinstance(cell, dict)
    real = cell["real"]
    assert isinstance(real, dict)
    orig = float.fromhex(real["hex"])
    real["hex"] = (orig * (1.0 + 0.01)).hex()

    result = totals.compare_totals(paths, ans, freqs)
    assert any("pressure[" in d for d in result.diffs), (
        f"壓力 real.hex 改到界線外之後沒超界：{result.diffs[:5]!r}"
    )


def test_control_group_direct_energy_scaled_over_budget(tmp_path: Path) -> None:
    """答案檔 ism_direct_E 某格乘 (1+2^-20) → 直達那格超界。"""
    paths = _v3_paths(tmp_path, "flat")
    freqs = _frequencies("flat")
    ans = copy.deepcopy(_answer_totals("flat"))

    direct_cells = ans["ism_direct_E"]
    assert isinstance(direct_cells, list)
    cell = direct_cells[0]
    assert isinstance(cell, dict)
    orig = float.fromhex(cell["hex"])
    cell["hex"] = (orig * (1.0 + 2.0 ** -20)).hex()

    result = totals.compare_totals(paths, ans, freqs)
    assert any("ism_direct_E[" in d for d in result.diffs), (
        f"直達能量乘 (1+2^-20) 之後沒超界：{result.diffs[:5]!r}"
    )


def test_control_group_reflected_energy_scaled_over_budget(tmp_path: Path) -> None:
    """答案檔 ism_rev_E 某格乘 (1+2^-10) → 反射能量超界且是 worst。"""
    paths = _v3_paths(tmp_path, "flat")
    freqs = _frequencies("flat")
    ans = copy.deepcopy(_answer_totals("flat"))

    reflected_cells = ans["ism_rev_E"]
    assert isinstance(reflected_cells, list)
    cell = reflected_cells[0]
    assert isinstance(cell, dict)
    orig = float.fromhex(cell["hex"])
    cell["hex"] = (orig * (1.0 + 2.0 ** -10)).hex()

    result = totals.compare_totals(paths, ans, freqs)
    assert result.diffs, "反射能量乘 (1+2^-10) 之後沒有超界"
    assert result.worst is not None
    assert result.worst[0] == "reflected_energy"


def test_control_group_missing_ism_rev_E(tmp_path: Path) -> None:
    """答案檔 totals 少一格（刪 ism_rev_E）→ ValueError 指名那一欄。"""
    paths = _v3_paths(tmp_path, "flat")
    freqs = _frequencies("flat")
    ans = copy.deepcopy(_answer_totals("flat"))
    del ans["ism_rev_E"]

    with pytest.raises(ValueError, match="ism_rev_E"):
        totals.compare_totals(paths, ans, freqs)


# ── 界線函式 ──────────────────────────────────────────────────────────────────


def _tol_abs_p(p: RoomPath, freqs: tuple[float, ...], i: int) -> float:
    """一條路徑一個頻帶的 ``tol_k·|p_k|``。"""
    tol = amp.pressure_tolerance(freqs[i], p.delay_s, abs(p.reflection_product[i]))
    return tol * abs(p.path_pressure[i])


def test_total_pressure_tolerance_single_path_equals_tol_abs_p(tmp_path: Path) -> None:
    """一條路徑的清單，總壓力界線等於 tol·|p|（跟每條路徑的界線接得上）。"""
    paths = _v3_paths(tmp_path, "flat")
    freqs = _frequencies("flat")
    p = paths[0]
    i = 0
    expected = _tol_abs_p(p, freqs, i)
    assert totals.total_pressure_tolerance([p], i, freqs) == expected


def test_total_pressure_tolerance_rss_less_than_linear(tmp_path: Path) -> None:
    """63 條的平方相加再開根小於線性和 Σ tol_k·|p_k|（統計尺不是嚴格上界）。"""
    paths = _v3_paths(tmp_path, "flat")
    freqs = _frequencies("flat")
    i = 0
    linear = sum(_tol_abs_p(p, freqs, i) for p in paths)
    rss = totals.total_pressure_tolerance(paths, i, freqs)
    assert rss < linear, f"平方相加再開根 {rss!r} 不小於線性和 {linear!r}"


def test_reflected_energy_tolerance_rejects_zero_pressure_sum(tmp_path: Path) -> None:
    """只有直達路徑時，反射能量界線應回有意義的 ValueError。"""
    paths = _v3_paths(tmp_path, "flat")

    with pytest.raises(ValueError, match="反射路徑壓力和為 0，反射能量界線沒定義"):
        totals.reflected_energy_tolerance([_direct_path(paths)], 0, _frequencies("flat"))


def test_totals_to_payload_round_trips_every_cell(tmp_path: Path) -> None:
    """三塊每格的 dec 與 hex 都逐位還原 Totals，鍵名則由答案檔推得。"""
    computed = totals.totals_from_paths(_v3_paths(tmp_path, "flat"))
    payload = totals.totals_to_payload(computed)
    expected_keys = set(_answer_totals("flat")) - {"reproduced_pressure_sum"}
    assert set(payload) == expected_keys

    pressure_cells = payload["pressure"]
    assert isinstance(pressure_cells, list)
    for cell, value in zip(pressure_cells, computed.pressure, strict=True):
        assert isinstance(cell, dict)
        expected_parts = {"real": value.real, "imag": value.imag, "abs": abs(value)}
        for part, expected in expected_parts.items():
            encoded = cell[part]
            assert isinstance(encoded, dict)
            assert float.fromhex(encoded["hex"]) == expected
            assert float(encoded["dec"]) == expected

    energy_blocks = {
        "ism_direct_E": computed.direct_energy,
        "ism_rev_E": computed.reflected_energy,
    }
    for key, expected_values in energy_blocks.items():
        cells = payload[key]
        assert isinstance(cells, list)
        for cell, expected in zip(cells, expected_values, strict=True):
            assert isinstance(cell, dict)
            assert float.fromhex(cell["hex"]) == expected
            assert float(cell["dec"]) == expected


# ── 搬家題：比對程式搬到 compare，room_paths 不再定義它 ───────────────────────────


def test_compare_moved_out_of_room_paths() -> None:
    """compare 搬家後的公開相容名字都由 room_paths 原物 re-export。"""
    from aosr.physics import compare
    from aosr.physics import room_paths as rp

    assert compare.compare_paths is rp.compare_paths
    assert compare._load_answer is rp._load_answer
    assert compare.PathComparison is rp.PathComparison
    assert compare.wall_name_seq_from_identity is rp.wall_name_seq_from_identity
