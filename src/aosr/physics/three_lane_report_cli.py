"""三路接合物理量報表的人看命令列輸出層。

幾何能量含干涉項（票 #302），不是上一代定義。

輸入 JSON 格式如下；六個牆名固定是 ``floor``、``ceiling``、``x0``、``xL``、
``y0``、``yL``。``scattering_by_wall`` 整格可省略，省略時由幾何路套既有預設值。

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
``--points`` 會在頂層與六頻帶表後再印完整細軸逐點表。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

from aosr.geometry.shoebox import Point, Room, Wall
from aosr.physics.three_lane_report import ThreeLaneReport, solve_three_lane_report


@dataclass(frozen=True)
class _CliInput:
    room: Room
    source: Point
    receiver: Point
    sound_speed_m_s: float
    density_kg_m3: float
    impedance_by_wall: dict[Wall, float]
    scattering_by_wall: dict[Wall, float] | None


def _mapping(value: object, where: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{where} 必須是 JSON 物件")
    return {str(key): item for key, item in value.items()}


def _number(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{where} 必須是有限數字")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{where} 必須是有限數字")
    return result


def _point(value: object, where: str) -> Point:
    fields = _mapping(value, where)
    return Point(
        _number(fields.get("x"), f"{where}.x"),
        _number(fields.get("y"), f"{where}.y"),
        _number(fields.get("z"), f"{where}.z"),
    )


def _wall_values(value: object, where: str) -> dict[Wall, float]:
    fields = _mapping(value, where)
    return {
        wall: _number(fields.get(wall.wall_name()), f"{where}.{wall.wall_name()}")
        for wall in Wall.all()
    }


def _load_input(path: Path) -> _CliInput:
    with path.open(encoding="utf-8") as handle:
        document: object = json.load(handle)
    fields = _mapping(document, "輸入")
    room_fields = _mapping(fields.get("room_m"), "room_m")
    scattering = fields.get("scattering_by_wall")
    return _CliInput(
        room=Room(
            _number(room_fields.get("Lx"), "room_m.Lx"),
            _number(room_fields.get("Ly"), "room_m.Ly"),
            _number(room_fields.get("Lz"), "room_m.Lz"),
        ),
        source=_point(fields.get("source_m"), "source_m"),
        receiver=_point(fields.get("receiver_m"), "receiver_m"),
        sound_speed_m_s=_number(fields.get("sound_speed_m_s"), "sound_speed_m_s"),
        density_kg_m3=_number(fields.get("density_kg_m3"), "density_kg_m3"),
        impedance_by_wall=_wall_values(
            fields.get("impedance_pa_s_per_m_by_wall"),
            "impedance_pa_s_per_m_by_wall",
        ),
        scattering_by_wall=(
            None
            if scattering is None
            else _wall_values(scattering, "scattering_by_wall")
        ),
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


def main(argv: list[str]) -> int:
    """印報表；成功回 0，讀檔、輸入或求解失敗回 2。"""
    parser = argparse.ArgumentParser(description="三路接合物理量報表")
    parser.add_argument("input", type=Path, help="輸入 JSON")
    parser.add_argument("--points", action="store_true", help="另印完整細軸逐點表")
    args = parser.parse_args(argv)
    try:
        inputs = _load_input(args.input)
        report = solve_three_lane_report(
            room=inputs.room,
            source=inputs.source,
            receiver=inputs.receiver,
            sound_speed_m_s=inputs.sound_speed_m_s,
            density_kg_m3=inputs.density_kg_m3,
            impedance_by_wall=inputs.impedance_by_wall,
            scattering_by_wall=inputs.scattering_by_wall,
        )
        sections = [_top_table(report), _band_table(report)]
        if args.points:
            sections.append(_point_table(report))
        print("\n".join(sections))
        return 0
    except Exception as exc:
        print(f"三路接合報表算不出來：{exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
