"""鏡像法的總壓力、直達能量、反射能量，以及總量的精度契約界線與判決。

**定義照上一代（決策紙 ``docs/decisions/engine-stage-order-three-to-ten.md``）。** 把 63 條
路徑加起來（含相位）得每個頻帶的總壓力，直達能量是直達那一條（order 0，恰好一條）壓力的
模平方，反射能量是 order 大於 0 的 62 條**先複數相加再取模平方**（交叉項刻意不算——這是
上一代的定義，v3 照比）。路徑順序照 ``image_source_paths`` 給的順序相加。

**界線照決策紙 ``precision-contract-totals-root-sum-square.md`` 與
``precision-contracts-geometry-energy-amplitude-fem.md``。**
每條路徑的契約界線 ``tol_k = amplitude.pressure_tolerance(f, τ_k, |refl_k|)``（``τ_k``＝該路徑
到達時間 ``delay_s``、``|refl_k|``＝該頻帶反射乘積的大小）；總壓力的容許差是
``sqrt(Σ_k (tol_k·|p_k|)²)``（63 條全算、含直達），反射能量的相對界線是
``2·sqrt(Σ_{k>0}(tol_k·|p_k|)²)/|Σ_{k>0} p_k| + floor``，直達能量用固定相對界線。
界線函式**從這一個地方來**：判決函式（:func:`compare_totals`）呼叫界線時走模組全域名
（``total_pressure_tolerance(...)``），讓 ``monkeypatch.setattr(totals, "total_pressure_tolerance", …)``
真的打到判決程式去查的那個名字。

**純函式、不 print。** 只 import 標準庫與同層／下層（``amplitude``、``compare`` 都是
``physics`` 同層，往下是 ``geometry``；規矩卡 ``layers-import-downward-only``）。命令列
輸出全住在 ``room_paths.py``，這一支回值、不解讀、不對人說話。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from aosr.physics.amplitude import pressure_tolerance
from aosr.physics.compare import _complex_hex_dec, _dec_hex, _mapping

if TYPE_CHECKING:
    from aosr.physics.room_paths import RoomPath

@dataclass(frozen=True)
class Totals:
    """一個接收點的總量：每頻帶的總壓力、直達能量、反射能量。

    ``pressure`` 每個頻帶一個複數（63 條路徑壓力的複數和）；``direct_energy``、
    ``reflected_energy`` 每個頻帶一個非負 float（模平方）。
    """

    pressure: tuple[complex, ...]
    direct_energy: tuple[float, ...]
    reflected_energy: tuple[float, ...]


@dataclass(frozen=True)
class PathPressureSums:
    """同一次 totals 計算裡已用過的直達壓力、反射同調和，以及逐階的反射同調和。

    ``reflected_pressure_by_order`` 第 ``k-1`` 格是第 ``k`` 階（``k`` 從 1 起算到路徑
    集合裡最高的那一階）全部反射路徑壓力的複數和，同階內的干涉本來就在裡面；每一格
    每個頻帶一個複數。逐階和相加等於 ``reflected_pressure``——只差浮點相加的結合律
    （``reflected_pressure`` 照路徑順序一條一條加，逐階那一路先分組再相加），所以
    考卷比的是「在捨入量級之內相等」，不是逐位相等。
    """

    direct_pressure: tuple[complex, ...]
    reflected_pressure: tuple[complex, ...]
    reflected_pressure_by_order: tuple[tuple[complex, ...], ...]


@dataclass(frozen=True)
class TotalsComparison:
    """``compare_totals`` 的判決：差異清單（空＝在契約內）與三個「用到界線幾成」的最大值。

    ``max_pressure_frac``／``max_direct_frac``／``max_reflected_frac`` 是差／界線比值的最大值
    （≤1 在契約內，>1 就超界）。``worst`` 記「超界最多」的那一格
    ``(量名, 頻帶, 差, 界線, 倍數)``，只在有超界時非空。
    """

    diffs: list[str]
    max_pressure_frac: float
    max_direct_frac: float
    max_reflected_frac: float
    worst: tuple[str, int, float, float, float] | None


def _direct_path(paths: list[RoomPath]) -> RoomPath:
    """恰好一條 order 0 的直達路徑；不是一條就 ValueError。"""
    direct = [p for p in paths if p.order == 0]
    if len(direct) != 1:
        raise ValueError(f"order 0 的直達路徑不是恰好一條，是 {len(direct)} 條")
    return direct[0]


def _checked_paths(paths: list[RoomPath]) -> None:
    """路徑集合算得出總量嗎：不是空的、每條都有振幅、反射乘積格數對得上。"""
    if not paths:
        raise ValueError("沒有路徑，算不出總量")
    for p in paths:
        if not p.path_pressure:
            raise ValueError(f"路徑 {p.identity!r} 沒有振幅（path_pressure 空），算不出總量")
        if len(p.reflection_product) != len(p.path_pressure):
            raise ValueError(
                f"路徑 {p.identity!r} 的 reflection_product 有 {len(p.reflection_product)} 格，"
                f"跟 path_pressure 的 {len(p.path_pressure)} 格對不上"
            )


def _paths_by_order(paths: list[RoomPath]) -> tuple[tuple[RoomPath, ...], ...]:
    """第 1 階到最高階各一組路徑（照原本的路徑順序），沒有路徑的那一階是空組。"""
    highest = max(p.order for p in paths)
    return tuple(
        tuple(p for p in paths if p.order == order)
        for order in range(1, highest + 1)
    )


def totals_and_pressure_sums_from_paths(
    paths: list[RoomPath],
) -> tuple[Totals, PathPressureSums]:
    """同一次加總回 :class:`Totals` 與其中已算過的直達／反射／逐階複數壓力。

    路徑沒有振幅（``path_pressure`` 空）→ ValueError 說沒有振幅；order 0 的直達不是恰好一條
    → ValueError。壓力相加**照路徑順序**（直達排最前，其餘按 ``(order, identity)``）；逐階和
    另外分組相加，決策紙 ``stage-nine-reflection-order-is-a-setting.md`` 第 5 條的 ``p_k``
    就是這一格。
    """
    _checked_paths(paths)
    direct = _direct_path(paths)
    reflections = tuple(p for p in paths if p.order > 0)
    order_groups = _paths_by_order(paths)
    n_freq = len(paths[0].path_pressure)

    pressure: list[complex] = []
    direct_pressure: list[complex] = []
    reflected_pressure: list[complex] = []
    by_order: tuple[list[complex], ...] = tuple([] for _group in order_groups)
    direct_energy: list[float] = []
    reflected_energy: list[float] = []
    for f_idx in range(n_freq):
        pressure.append(sum(p.path_pressure[f_idx] for p in paths))
        zd = direct.path_pressure[f_idx]
        direct_pressure.append(zd)
        direct_energy.append(abs(zd) ** 2)
        zr = sum(p.path_pressure[f_idx] for p in reflections)
        reflected_pressure.append(zr)
        reflected_energy.append(abs(zr) ** 2)
        for slot, group in enumerate(order_groups):
            by_order[slot].append(sum(p.path_pressure[f_idx] for p in group))
    return (
        Totals(tuple(pressure), tuple(direct_energy), tuple(reflected_energy)),
        PathPressureSums(
            tuple(direct_pressure),
            tuple(reflected_pressure),
            tuple(tuple(column) for column in by_order),
        ),
    )


def totals_from_paths(paths: list[RoomPath]) -> Totals:
    """把路徑加總成既有答案契約的 :class:`Totals`，不增加契約欄位。"""
    result, _pressure_sums = totals_and_pressure_sums_from_paths(paths)
    return result


def total_pressure_tolerance(
    paths: list[RoomPath],
    f_index: int,
    frequencies: tuple[float, ...],
    reflection_tolerance_ulp: float,
) -> float:
    """總壓力的絕對差界線：``sqrt(Σ_k (tol_k·|p_k|)²)``（``tol_k=pressure_tolerance``）。

    ``tol_k = amplitude.pressure_tolerance(f, τ_k, |refl_k|)``，``f=frequencies[f_index]``、
    ``τ_k=delay_s``、``|refl_k|``＝該頻帶反射乘積大小。63 條全算、含直達；這是決策紙選項 1
    的統計尺（平方相加再開根），不是線性疊加。
    """
    f = frequencies[f_index]
    acc = 0.0
    for p in paths:
        tol = pressure_tolerance(
            f,
            p.delay_s,
            abs(p.reflection_product[f_index]),
            reflection_tolerance_ulp,
        )
        acc += (tol * abs(p.path_pressure[f_index])) ** 2
    return math.sqrt(acc)


def reflected_energy_tolerance(
    paths: list[RoomPath],
    f_index: int,
    frequencies: tuple[float, ...],
    reflection_tolerance_ulp: float,
    tolerance_rel_floor: float,
) -> float:
    """反射能量的相對差界線：相位尺度的統計項加上呼叫端給定的相對地板。

    分子只算 order>0 的 62 條；分母是反射路徑壓力的複數和取模（先相加再取模平方的那個和）。
    地板與振幅基底都由呼叫端從精度契約登記簿取得。
    """
    f = frequencies[f_index]
    acc = 0.0
    rev: list[complex] = []
    for p in paths:
        if p.order <= 0:
            continue
        rev.append(p.path_pressure[f_index])
        tol = pressure_tolerance(
            f,
            p.delay_s,
            abs(p.reflection_product[f_index]),
            reflection_tolerance_ulp,
        )
        acc += (tol * abs(p.path_pressure[f_index])) ** 2
    denom = abs(sum(rev))
    if denom == 0.0:
        raise ValueError("反射路徑壓力和為 0，反射能量界線沒定義")
    return 2.0 * math.sqrt(acc) / denom + tolerance_rel_floor


def direct_energy_tolerance(
    paths: list[RoomPath],
    f_index: int,
    frequencies: tuple[float, ...],
    tolerance_rel: float,
) -> float:
    """直達能量的固定相對差界線，由呼叫端傳入。

    決策紙 ``precision-contracts-geometry-energy-amplitude-fem.md``；三個參數保留是為了跟另外兩支
    同一張簽名、讓「界線不隨路徑／頻率變」在呼叫點看得見。
    """
    del paths, f_index, frequencies
    return tolerance_rel


def _hex_float(node: object, where: str) -> float:
    """答案檔一個 ``{dec, hex}`` 能量格的 float，用 ``hex`` 讀（任務要求不用 dec）。"""
    cell = _mapping(node, where)
    hexed = cell.get("hex")
    if not isinstance(hexed, str):
        raise ValueError(f"{where} 沒有 hex 那一格")
    return float.fromhex(hexed)


def _hex_complex(node: object, where: str) -> complex:
    """答案檔一個複數格 ``{real:{hex,…}, imag:{hex,…}, …}``，用 hex 讀成複數。"""
    cell = _mapping(node, where)
    real = _hex_float(cell.get("real"), f"{where}.real")
    imag = _hex_float(cell.get("imag"), f"{where}.imag")
    return complex(real, imag)


def _cell_list(node: object, where: str, n_freq: int) -> list[object]:
    """答案檔 totals 的一欄（``pressure``／``ism_direct_E``／``ism_rev_E``）清單，頻帶數
    對不上就 ValueError 指名是哪一欄。"""
    if not isinstance(node, list):
        raise ValueError(f"{where} 不是一串東西：{node!r}")
    if len(node) != n_freq:
        raise ValueError(f"{where} 有 {len(node)} 格，不是 {n_freq} 格（頻帶數對不上）")
    return node


def _frac(diff: float, bound: float) -> float:
    """差／界線，界線為 0 且差非 0 回 inf。"""
    if bound == 0.0:
        return float("inf") if diff != 0.0 else 0.0
    return diff / bound


def _total_band_differences(
    paths: list[RoomPath],
    computed: Totals,
    expected: tuple[complex, float, float],
    index: int,
    frequencies: tuple[float, ...],
    reflection_tolerance_ulp: float,
    direct_tolerance_rel: float,
    reflected_tolerance_rel_floor: float,
) -> tuple[tuple[str, str, float, float], ...]:
    """回一個頻帶三個量的（欄名、worst 名、差、界線）。"""
    p2, d2, r2 = expected
    p_bound = total_pressure_tolerance(
        paths, index, frequencies, reflection_tolerance_ulp
    )
    d_bound = direct_energy_tolerance(
        paths, index, frequencies, direct_tolerance_rel
    ) * d2
    r_bound = reflected_energy_tolerance(
        paths,
        index,
        frequencies,
        reflection_tolerance_ulp,
        reflected_tolerance_rel_floor,
    ) * r2
    return (
        ("pressure", "pressure", abs(computed.pressure[index] - p2), p_bound),
        ("ism_direct_E", "direct_energy", abs(computed.direct_energy[index] - d2), d_bound),
        ("ism_rev_E", "reflected_energy", abs(computed.reflected_energy[index] - r2), r_bound),
    )


def compare_totals(
    paths: list[RoomPath],
    answer_totals: dict[str, object],
    frequencies: tuple[float, ...],
    reflection_tolerance_ulp: float,
    direct_tolerance_rel: float,
    reflected_tolerance_rel_floor: float,
) -> TotalsComparison:
    """比 v3 的總量跟答案檔 ``totals`` 依契約，照頻帶逐格判。

    讀答案檔 ``totals`` 的 ``pressure``（複數 real/imag 用 hex）、``ism_direct_E``、
    ``ism_rev_E``（energy 用 hex），每頻帶判三條：``|P3−P2| ≤ 界線``、
    ``|D3−D2| ≤ 界線·D2``、``|R3−R2| ≤ 界線·R2``。界線走模組全域名的界線函式（
    monkeypatch 打得到）。缺欄或頻帶數對不上 → ValueError 指名哪一欄。
    """
    totals = totals_from_paths(paths)
    n_freq = len(frequencies)
    if len(totals.pressure) != n_freq:
        raise ValueError(
            f"我方頻帶數 {len(totals.pressure)} 跟 frequencies 的 {n_freq} 對不上"
        )
    table = _mapping(answer_totals, "totals")
    for key in ("pressure", "ism_direct_E", "ism_rev_E"):
        if key not in table:
            raise ValueError(f"totals 缺欄位：{key}")
    p2_cells = _cell_list(table["pressure"], "totals.pressure", n_freq)
    d2_cells = _cell_list(table["ism_direct_E"], "totals.ism_direct_E", n_freq)
    r2_cells = _cell_list(table["ism_rev_E"], "totals.ism_rev_E", n_freq)

    diffs: list[str] = []
    max_fractions = {"pressure": 0.0, "direct_energy": 0.0, "reflected_energy": 0.0}
    worst: tuple[str, int, float, float, float] | None = None

    def _note_worst(kind: str, i: int, diff: float, bound: float) -> None:
        nonlocal worst
        frac = _frac(diff, bound)
        if worst is None or frac > worst[4]:
            worst = (kind, i, diff, bound, frac)

    for i in range(n_freq):
        p2 = _hex_complex(p2_cells[i], f"totals.pressure[{i}]")
        d2 = _hex_float(d2_cells[i], f"totals.ism_direct_E[{i}]")
        r2 = _hex_float(r2_cells[i], f"totals.ism_rev_E[{i}]")

        measurements = _total_band_differences(
            paths,
            totals,
            (p2, d2, r2),
            i,
            frequencies,
            reflection_tolerance_ulp,
            direct_tolerance_rel,
            reflected_tolerance_rel_floor,
        )
        for field, kind, difference, bound in measurements:
            if difference > bound:
                diffs.append(f"{field}[{i}] 超界：差 {difference!r} > 界線 {bound!r}")
                _note_worst(kind, i, difference, bound)
            max_fractions[kind] = max(max_fractions[kind], _frac(difference, bound))

    return TotalsComparison(
        diffs=diffs,
        max_pressure_frac=max_fractions["pressure"],
        max_direct_frac=max_fractions["direct_energy"],
        max_reflected_frac=max_fractions["reflected_energy"],
        worst=worst,
    )


def totals_to_payload(t: Totals) -> dict[str, object]:
    """把 :class:`Totals` 攤成答案檔 ``totals`` 的同形（壓力每格 real/imag/abs 各 dec+hex，
    能量 dec+hex）——命令列 ``--json`` 下一段要用。"""
    return {
        "pressure": [_complex_hex_dec(z) for z in t.pressure],
        "ism_direct_E": [_dec_hex(e) for e in t.direct_energy],
        "ism_rev_E": [_dec_hex(e) for e in t.reflected_energy],
    }
