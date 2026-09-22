"""晚期混響能量的人看命令列輸出層。"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path

from aosr.config.capabilities import (
    capability_for,
    load_capabilities,
    status_for,
)
from aosr.config.precision_contracts import load_precision_contracts
from aosr.geometry.shoebox import Wall
from aosr.physics import capability_report, late_energy
from aosr.physics.late_energy import (
    LateEnergyBandJudgment,
    LateEnergyContractReport,
    LateEnergyInputs,
    LateEnergyResult,
    judge_late_energy,
    load_late_energy_inputs,
    load_legacy_late_energy_bands,
    solve_late_energy,
)


# 這一節在能力表上的名字。材料形式**不寫死**：從讀進來的輸入判斷——六面各一個與頻率無關
# 的實數阻抗是 ``real_frequency_independent_impedance``，阻抗帶虛部或逐面逐頻不同就是別的形式。
_ENTRY = "late_energy"
_ROOM = "shoebox"
_REAL_MATERIALS = "real_frequency_independent_impedance"


def materials_for(inputs: LateEnergyInputs) -> str:
    """從輸入判斷材料形式。

    阻抗帶虛部是 ``complex_impedance_by_wall``；同一面在不同頻率給不同值才是逐頻阻抗
    （``frequency_dependent_impedance``）。**各面數值不同仍是實數阻抗**——表上那一條寫的
    是「六面各一個與頻率無關的實數阻抗」，各面可以各自一個值；六面不同不是逐頻。
    """
    for wall in inputs.impedance_by_wall:
        row = inputs.impedance_by_wall[wall]
        if any(value.imag != 0.0 for value in row):
            return "complex_impedance_by_wall"
        if len(set(row)) > 1:
            return "frequency_dependent_impedance"
    return _REAL_MATERIALS


def capability_line(path: Path, materials: str) -> str:
    """查這條組合本人，回傳一行給人看的 capability 節。

    印出來的那一行帶著表上那一條宣告的頻率範圍與輸出欄，不是只有狀態與收據：
    晚期能量的表印的欄位比驗過的兩欄多，只印 ``status=validated`` 會讓沒驗過的
    欄位看起來也驗過了。
    """
    table = load_capabilities(path)
    record = capability_for(table, _ENTRY, room=_ROOM, materials=materials)
    return capability_report.capability_line(_ENTRY, _ROOM, materials, record)


def _contract_cells(
    point: late_energy.LateEnergyBandJudgment,
    tolerance_rel: float,
) -> tuple[str, ...]:
    return (
        f"{point.expected_energy:.17g}",
        f"{point.absolute_difference:.17g}",
        f"{point.allowed_difference:.17g}",
        f"{point.relative_difference:.6e}",
        f"{tolerance_rel:.6e}",
        f"{point.contract_fraction * 100.0:.6f}%",
        "舊界內" if point.within_contract else "舊界外",
    )


def late_energy_table(
    result: LateEnergyResult,
    report: LateEnergyContractReport | None = None,
    *,
    missing_legacy_frequencies_hz: tuple[float, ...] = (),
) -> str:
    """純函式回傳逐頻吸收率、能量，以及可選的上一代相容紀錄。"""
    wall_names = Wall.wall_names()
    headings = [
        "frequency_hz",
        *(f"alpha_{wall}" for wall in wall_names),
        "alpha_bar",
        "legacy_domain",
        "raw_energy",
        "eyring_ratio",
        "late_energy",
    ]
    points = report.points if report is not None else ()
    points_by_frequency = {point.frequency_hz: point for point in points}
    if report is not None:
        _check_legacy_band_alignment(result, points_by_frequency, missing_legacy_frequencies_hz)
        headings.extend(
            (
                "legacy_energy",
                "absolute_difference",
                "allowed_difference",
                "relative_difference",
                "relative_limit",
                "contract_used_pct",
                "comparison",
            )
        )
    lines = []
    if report is not None:
        lines.append(
            "上一代答案是第二類相容紀錄；差距照量、照留，不作通過判決"
        )
    lines.append(" ".join(headings))
    for band in result.bands:
        cells = [
            f"{band.frequency_hz:g}",
            *(f"{band.alpha_by_wall[wall]:.12g}" for wall in wall_names),
            f"{band.alpha_bar:.12g}",
            "是" if band.in_domain else "否",
            f"{band.raw_reverberant_energy:.17g}",
            f"{band.eyring_ratio:.17g}",
            f"{band.late_reverberant_energy:.17g}",
        ]
        if report is not None:
            point = points_by_frequency.get(band.frequency_hz)
            if point is None:
                cells.extend(("-", "-", "-", "-", "-", "-", "這一帶沒有上一代答案"))
            else:
                cells.extend(_contract_cells(point, report.tolerance_rel))
        lines.append(" ".join(cells))
    if report is not None:
        lines.append(_contract_summary_line(report))
    return "\n".join(lines) + "\n"


def _check_legacy_band_alignment(
    result: LateEnergyResult,
    points_by_frequency: Mapping[float, LateEnergyBandJudgment],
    missing_legacy_frequencies_hz: tuple[float, ...],
) -> None:
    """契約報告的帶要都在能量結果裡；結果裡沒答案的帶要等於宣告的「沒有上一代答案」清單。"""
    result_frequencies = {band.frequency_hz for band in result.bands}
    if not set(points_by_frequency) <= result_frequencies:
        raise ValueError("契約報告含有能量結果不存在的頻帶")
    if set(missing_legacy_frequencies_hz) != result_frequencies - set(points_by_frequency):
        raise ValueError("沒有上一代答案的頻帶清單與能量結果不一致")


def _contract_summary_line(report: LateEnergyContractReport) -> str:
    """相容紀錄的收尾一行：全在舊界內或幾格舊界外，加最壞那一帶。"""
    points = report.points
    worst = max(points, key=lambda point: point.contract_fraction)
    if report.within_contract:
        summary = "相容紀錄：全部在舊界內（不擋）"
    else:
        failed = sum(not point.within_contract for point in points)
        summary = f"相容紀錄：{failed} 格舊界外（不擋）"
    return (
        f"{summary}；最壞={worst.frequency_hz:g} Hz；"
        f"相對差={worst.relative_difference:.6e}；"
        f"用掉={worst.contract_fraction * 100.0:.6f}%"
    )


def _compare_with_legacy(
    result: LateEnergyResult, answer_path: Path, tolerance_rel: float
) -> tuple[LateEnergyContractReport, tuple[float, ...]]:
    """只比上一代答案有的那幾帶；報表多出來的帶列成「沒有上一代答案」，答案多出來的帶是錯。"""
    legacy_by_frequency = dict(load_legacy_late_energy_bands(answer_path))
    result_frequencies = {band.frequency_hz for band in result.bands}
    legacy_only = set(legacy_by_frequency) - result_frequencies
    if legacy_only:
        frequencies = ", ".join(f"{frequency:g} Hz" for frequency in sorted(legacy_only))
        raise ValueError(f"上一代答案含有 v3 結果沒有的頻帶：{frequencies}")
    matched = LateEnergyResult(
        bands=tuple(band for band in result.bands if band.frequency_hz in legacy_by_frequency)
    )
    expected = tuple(legacy_by_frequency[band.frequency_hz] for band in matched.bands)
    missing = tuple(
        band.frequency_hz for band in result.bands if band.frequency_hz not in legacy_by_frequency
    )
    return judge_late_energy(matched, expected, tolerance_rel), missing


def main(argv: list[str]) -> int:
    """印逐頻答案；第二類差距不擋而回 0，讀檔或求解失敗回 2。"""
    parser = argparse.ArgumentParser(description="晚期混響能量逐頻核對表")
    parser.add_argument("input", type=Path, nargs="?", help="答案 JSON 或只含 parameters 的 JSON")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="用同一輸入檔的 bands[].late_rev_E 比對上一代答案",
    )
    parser.add_argument("--contracts", type=Path, help="精度契約 TOML 登記簿")
    parser.add_argument(
        "--capabilities",
        type=Path,
        required=True,
        help="能力與驗證範圍表 TOML；必給，物理層不設隱含預設",
    )
    args = parser.parse_args(argv)
    try:
        if args.input is None:
            raise ValueError("要給答案 JSON 或只含 parameters 的 JSON")
        inputs = load_late_energy_inputs(args.input)
        materials = materials_for(inputs)
        table = load_capabilities(args.capabilities)
        try:
            status = status_for(table, _ENTRY, room=_ROOM, materials=materials)
        except KeyError as exc:
            raise ValueError(
                f"{_ENTRY} × {_ROOM} × {materials}：能力表沒有這個組合，"
                "這一版不支援這種材料形式"
            ) from exc
        if status != "validated":
            raise ValueError(
                f"{_ENTRY} × {_ROOM} × {materials}：能力表標 {status}，"
                "這一版不支援這種材料形式"
            )
        line = capability_line(args.capabilities, materials)
        result = solve_late_energy(inputs)
        report = None
        missing_legacy_frequencies_hz: tuple[float, ...] = ()
        if args.compare:
            if args.contracts is None:
                raise ValueError("--compare 模式必須給 --contracts")
            tolerance_rel = load_precision_contracts(args.contracts)[
                "late_energy_vs_legacy"
            ].value
            report, missing_legacy_frequencies_hz = _compare_with_legacy(
                result, args.input, tolerance_rel
            )
        print(line)
        print(
            late_energy_table(
                result,
                report,
                missing_legacy_frequencies_hz=missing_legacy_frequencies_hz,
            ),
            end="",
        )
        return 0
    except Exception as exc:
        print(f"晚期混響能量算不出來：{exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
