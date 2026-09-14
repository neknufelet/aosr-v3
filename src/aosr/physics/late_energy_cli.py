"""晚期混響能量的人看命令列輸出層。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aosr.physics import late_energy
from aosr.geometry.shoebox import Wall
from aosr.physics.late_energy import (
    LateEnergyContractReport,
    LateEnergyResult,
    judge_late_energy,
    load_late_energy_inputs,
    load_legacy_late_energies,
    solve_late_energy,
)


def _contract_cells(point: late_energy.LateEnergyBandJudgment) -> tuple[str, ...]:
    return (
        f"{point.expected_energy:.17g}",
        f"{point.absolute_difference:.17g}",
        f"{point.allowed_difference:.17g}",
        f"{point.relative_difference:.6e}",
        f"{late_energy.LATE_ENERGY_CONTRACT_REL:.6e}",
        f"{point.contract_fraction * 100.0:.6f}%",
        "過" if point.within_contract else "不過",
    )


def late_energy_table(
    result: LateEnergyResult,
    report: LateEnergyContractReport | None = None,
) -> str:
    """純函式回傳逐頻吸收率、能量，以及可選的上一代契約比對表。"""
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
    if report is not None:
        if len(points) != len(result.bands):
            raise ValueError("能量結果與契約報告的頻帶數不同")
        headings.extend(
            (
                "legacy_energy",
                "absolute_difference",
                "allowed_difference",
                "relative_difference",
                "relative_limit",
                "contract_used_pct",
                "verdict",
            )
        )
    lines = []
    if report is not None:
        lines.append("差與容許差是能量的絕對值；相對差與相對界線是比例")
    lines.append(" ".join(headings))
    for index, band in enumerate(result.bands):
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
            point = points[index]
            if point.frequency_hz != band.frequency_hz:
                raise ValueError("能量結果與契約報告的頻帶沒有對齊")
            cells.extend(_contract_cells(point))
        lines.append(" ".join(cells))
    if report is not None:
        worst = max(points, key=lambda point: point.contract_fraction)
        if report.within_contract:
            summary = "判決：全部過"
        else:
            failed = sum(not point.within_contract for point in points)
            summary = f"判決：{failed} 格超界"
        lines.append(
            f"{summary}；最壞={worst.frequency_hz:g} Hz；"
            f"相對差={worst.relative_difference:.6e}；"
            f"用掉={worst.contract_fraction * 100.0:.6f}%"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    """印逐頻答案；全過回 0、超界回 1、讀檔或求解失敗回 2。"""
    parser = argparse.ArgumentParser(description="晚期混響能量逐頻核對表")
    parser.add_argument("input", type=Path, nargs="?", help="答案 JSON 或只含 parameters 的 JSON")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="用同一輸入檔的 bands[].late_rev_E 比對上一代答案",
    )
    args = parser.parse_args(argv)
    try:
        if args.input is None:
            raise ValueError("要給答案 JSON 或只含 parameters 的 JSON")
        result = solve_late_energy(load_late_energy_inputs(args.input))
        report = None
        if args.compare:
            expected = load_legacy_late_energies(
                args.input,
                frequencies_hz=tuple(band.frequency_hz for band in result.bands),
            )
            report = judge_late_energy(result, expected)
        print(late_energy_table(result, report), end="")
        return 0 if report is None or report.within_contract else 1
    except Exception as exc:
        print(f"晚期混響能量算不出來：{exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
