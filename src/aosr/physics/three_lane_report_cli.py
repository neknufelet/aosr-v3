"""三路接合物理量報表的人看命令列輸出層。

幾何能量含干涉項（票 #302），不是上一代定義。

輸入 JSON 格式如下；六個牆名固定是 ``floor``、``ceiling``、``x0``、``xL``、
``y0``、``yL``。``scattering_by_wall`` 整格可省略，省略時由幾何路套既有預設值。
這份輸入由 :mod:`aosr.physics.report_io` 的輸入模型驗（票 #316）；欄位形狀與
每一欄的物理量／單位／參考基準／有效狀態匯出在 ``blueprint/schemas/``。

.. code-block:: json

   {
     "room_m": {"Lx": 6.0, "Ly": 4.0, "Lz": 3.0},
     "source_m": {"x": 1.2, "y": 1.3, "z": 1.1},
     "receiver_m": {"x": 4.7, "y": 2.8, "z": 1.4},
     "sound_speed_m_s": 343.0,
     "density_kg_m3": 1.2,
     "impedance_pa_s_per_m_by_wall": {
       "floor": 1646.4, "ceiling": 1646.4,
       "x0": 1646.4, "xL": 1646.4, "y0": 1646.4, "yL": 1646.4
     },
     "scattering_by_wall": {
       "floor": 0.1, "ceiling": 0.1,
       "x0": 0.1, "xL": 0.1, "y0": 0.1, "yL": 0.1
     }
   }

執行 ``uv run python -m aosr.physics.three_lane_report_cli input.json``；加
``--points`` 會在頂層與六頻帶表後再印完整細軸逐點表。加 ``--format json``
改印輸出契約的 JSON（印之前用模型自己反解一次，證明它真的合那份契約）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from aosr.config.capabilities import (
    CapabilityTable,
    capability_for,
    load_capabilities,
)
from aosr.geometry.shoebox import Wall
from aosr.physics import capability_report, report_io
from aosr.physics.report_io import ReportOutput
from aosr.physics.three_lane_report import (
    ReportCapability,
    ThreeLaneReport,
    solve_three_lane_report,
)


# 這一節在能力表上的名字，以及這次輸入的材料形式：六面各一個與頻率無關的實數阻抗。
_MATERIALS = "real_frequency_independent_impedance"
_ROOM = "shoebox"
_ENTRY = "three_lane_report"

# 收到複數或逐頻阻抗時，拒收訊息要提的兩條組合。名單、訊息與「從哪一張表讀」全部住在
# report_io：命令列只把 --capabilities 那一張表傳進去，不在這裡拼第二份。


def _capability_section(capability: ReportCapability) -> str:
    """把這份報表落在哪一條能力組合、什麼狀態、憑什麼、範圍與欄位，印成一行人話。"""
    return capability_report.capability_line(
        capability.entry,
        capability.room,
        capability.materials,
        capability.record,
    )


def _capability_for(table: CapabilityTable) -> ReportCapability:
    """從能力表查這條組合本人；查不到或標 unsupported 就報錯。

    拿的是整條組合（狀態、收據、頻率範圍、輸出欄），不是只有狀態字串——
    印出來的那一行要能讓人看出 validated 蓋到哪裡為止。
    """
    record = capability_for(table, _ENTRY, room=_ROOM, materials=_MATERIALS)
    if record.status == "unsupported":
        hint = report_io.unsupported_materials_hint(table)
        raise ValueError(f"{_ENTRY} × {_ROOM} × {_MATERIALS}：{hint}")
    return ReportCapability(
        entry=_ENTRY,
        room=_ROOM,
        materials=_MATERIALS,
        record=record,
    )


def _value(value: float | None) -> str:
    return "none" if value is None else f"{value:.17g}"


def _top_table(report: ThreeLaneReport) -> str:
    rows = (
        ("f_s_hz", _value(report.f_s_hz)),
        ("crossover_lower_hz", _value(report.crossover_lower_hz)),
        ("crossover_upper_hz", _value(report.crossover_upper_hz)),
        ("capped_by_upper_limit", str(report.capped_by_upper_limit).lower()),
        ("eyring_t60_500_hz_s", _value(report.eyring_t60_by_band_s[500.0])),
        ("eyring_t60_1000_hz_s", _value(report.eyring_t60_by_band_s[1000.0])),
    )
    lines = [f"{name} {value}" for name, value in rows]
    if report.capped_by_upper_limit:
        lines.append("被上限硬切（f_s ≥ 300 Hz，在 300 Hz 硬切換）")
    return "\n".join(lines)


def _decay_value(value: float | None, reason: str | None) -> str:
    if reason is not None:
        return f"算不出：{reason}"
    return _value(value)


def _band_table(report: ThreeLaneReport) -> str:
    sampling_note = (
        "頻帶取樣：直達／反射／干涉／s 欄為 0.5 Hz 密頻率點平均；"
        "晚期／T20／T30／權重欄為 1/24 八度細軸點平均，權重只供閱讀；"
        "請用 fem_contribution 與 geometric_contribution 驗算 total_energy。"
    )
    headings = (
        "center_frequency_hz fem_energy_fem_points_only fem_point_count "
        "direct_energy_all_points reflected_energy_all_points "
        "interference_energy_all_points "
        "late_energy_all_points geometric_energy_all_points "
        "fem_contribution_all_points geometric_contribution_all_points "
        "total_energy w_fem w_geo t20_s t30_s"
    )
    rows = []
    for band in report.bands:
        values = (
            band.center_frequency_hz,
            band.fem_energy,
            band.fem_point_count,
            band.direct_energy,
            band.reflected_energy,
            band.interference_energy,
            band.late_energy,
            band.geometric_energy,
            band.fem_contribution,
            band.geometric_contribution,
            band.total_energy,
            band.w_fem,
            band.w_geo,
        )
        cells = [_value(value) for value in values]
        cells.append(_decay_value(band.t20_s, band.t20_unavailable_reason))
        cells.append(_decay_value(band.t30_s, band.t30_unavailable_reason))
        rows.append(" ".join(cells))
    return "\n".join((sampling_note, headings, *rows))


def _point_table(report: ThreeLaneReport) -> str:
    headings = (
        "frequency_hz fem_energy direct_energy reflected_energy "
        "interference_energy late_energy scattering geometric_energy "
        "w_fem w_geo total_energy"
    )
    rows = []
    for point in report.points:
        values = (
            point.frequency_hz,
            point.fem_energy,
            point.direct_energy,
            point.reflected_energy,
            point.interference_energy,
            point.late_energy,
            point.scattering,
            point.geometric_energy,
            point.w_fem,
            point.w_geo,
            point.total_energy,
        )
        rows.append(" ".join(_value(value) for value in values))
    return "\n".join((headings, *rows))


def _text_sections(report: ThreeLaneReport, *, with_points: bool) -> list[str]:
    sections = [
        _capability_section(report.capability),
        _top_table(report),
        _band_table(report),
    ]
    if with_points:
        sections.append(_point_table(report))
    return sections


def main(argv: list[str]) -> int:
    """印報表；成功回 0，讀檔、輸入或求解失敗回 2。"""
    parser = argparse.ArgumentParser(description="三路接合物理量報表")
    parser.add_argument("input", type=Path, help="輸入 JSON")
    parser.add_argument("--points", action="store_true", help="另印完整細軸逐點表")
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="text 是原本的人看表格；json 印輸出契約的 JSON",
    )
    parser.add_argument(
        "--capabilities",
        type=Path,
        required=True,
        help="能力與驗證範圍表 TOML；必給，物理層不設隱含預設",
    )
    args = parser.parse_args(argv)
    try:
        table = load_capabilities(args.capabilities)
        capability = _capability_for(table)
        inputs = report_io.load_input(args.input, table)
        solved = report_io.solver_inputs(inputs)
        report = solve_three_lane_report(
            room=solved.room,
            source=solved.source,
            receiver=solved.receiver,
            sound_speed_m_s=solved.sound_speed_m_s,
            density_kg_m3=solved.density_kg_m3,
            impedance_by_wall=solved.impedance_by_wall,
            scattering_by_wall=solved.scattering_by_wall,
            capability=capability,
        )
        if args.format == "json":
            parsed = report_io.output_from_report(
                report, room=inputs.room_m, with_points=args.points
            )
            payload = parsed.model_dump_json()
            if ReportOutput.model_validate_json(payload) != parsed:
                raise ValueError("輸出契約反解回來的結果跟收成的結果不同")
            print(json.dumps(json.loads(payload), ensure_ascii=False, sort_keys=True))
            return 0
        print("\n".join(_text_sections(report, with_points=args.points)))
        return 0
    except Exception as exc:
        print(f"三路接合報表算不出來：{exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

