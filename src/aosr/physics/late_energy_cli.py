"""晚期混響能量的人看命令列輸出層。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aosr.config.capabilities import (
    evidence_for,
    load_capabilities,
    status_for,
)
from aosr.config.paths import config_path
from aosr.config.precision_contracts import load_precision_contracts
from aosr.geometry.shoebox import Wall
from aosr.physics import late_energy
from aosr.physics.late_energy import (
    LateEnergyContractReport,
    LateEnergyResult,
    judge_late_energy,
    load_late_energy_inputs,
    load_legacy_late_energies,
    solve_late_energy,
)


# 這一節在能力表上的名字，以及這次輸入的材料形式：六面各一個與頻率無關的實數阻抗。
_ENTRY = "late_energy"
_ROOM = "shoebox"
_MATERIALS = "real_frequency_independent_impedance"


def capability_line(path: Path) -> str:
    """查這條組合的狀態與收據，回傳一行給人看的 capability 節。"""
    table = load_capabilities(path)
    status = status_for(table, _ENTRY, room=_ROOM, materials=_MATERIALS)
    evidence = evidence_for(table, _ENTRY, room=_ROOM, materials=_MATERIALS)
    joined = ",".join(evidence) if evidence else "none"
    return (
        f"capability entry={_ENTRY} room={_ROOM} materials={_MATERIALS} "
        f"status={status} evidence={joined}"
    )


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
                "comparison",
            )
        )
    lines = []
    if report is not None:
        lines.append(
            "上一代答案是第二類相容紀錄；差距照量、照留，不作通過判決"
        )
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
            cells.extend(_contract_cells(point, report.tolerance_rel))
        lines.append(" ".join(cells))
    if report is not None:
        worst = max(points, key=lambda point: point.contract_fraction)
        if report.within_contract:
            summary = "相容紀錄：全部在舊界內（不擋）"
        else:
            failed = sum(not point.within_contract for point in points)
            summary = f"相容紀錄：{failed} 格舊界外（不擋）"
        lines.append(
            f"{summary}；最壞={worst.frequency_hz:g} Hz；"
            f"相對差={worst.relative_difference:.6e}；"
            f"用掉={worst.contract_fraction * 100.0:.6f}%"
        )
    return "\n".join(lines) + "\n"


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
        help="能力與驗證範圍表 TOML；預設 src/aosr/config/data/capabilities.toml",
    )
    args = parser.parse_args(argv)
    try:
        if args.input is None:
            raise ValueError("要給答案 JSON 或只含 parameters 的 JSON")
        line = capability_line(
            args.capabilities
            if args.capabilities is not None
            else config_path("capabilities.toml")
        )
        result = solve_late_energy(load_late_energy_inputs(args.input))
        report = None
        if args.compare:
            if args.contracts is None:
                raise ValueError("--compare 模式必須給 --contracts")
            tolerance_rel = load_precision_contracts(args.contracts)[
                "late_energy_vs_legacy"
            ].value
            expected = load_legacy_late_energies(
                args.input,
                frequencies_hz=tuple(band.frequency_hz for band in result.bands),
            )
            report = judge_late_energy(result, expected, tolerance_rel)
        print(line)
        print(late_energy_table(result, report), end="")
        return 0
    except Exception as exc:
        print(f"晚期混響能量算不出來：{exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
