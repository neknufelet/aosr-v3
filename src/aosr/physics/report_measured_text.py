"""命令列的實測牆表頭、接縫小表與補值註記；差值只在顯示時計算。"""
from __future__ import annotations

from typing import TYPE_CHECKING

from aosr.materials.construction_models import ConstructionNote, MeasuredImpedance
from aosr.physics.report_measured import measured_seams
from aosr.physics.wall_surfaces import extrapolated_frequency_text

if TYPE_CHECKING:
    from aosr.physics.report_io import ReportInput


def measured_text(inputs: ReportInput) -> list[str]:
    sections = []
    for wall, zones in (inputs.surface_zones_by_wall or {}).items():
        material = zones[0].material
        if not isinstance(material, MeasuredImpedance):
            continue
        lower, upper = material.points[0].frequency_hz, material.points[-1].frequency_hz
        sections.append(f"{wall}：實測表涵蓋 {lower:g}–{upper:g} Hz；{lower:g} Hz 以下與 {upper:g} Hz 以上是模型補值（底構造）；"
                        "實測段假設與角度無關；實測表只代表樣品在管子裡量到的結果，不代表現場安裝")
    seams = measured_seams(inputs.surface_zones_by_wall, density_kg_m3=inputs.density_kg_m3,
                           sound_speed_m_s=inputs.sound_speed_m_s)
    for wall, rows in (seams or {}).items():
        lines = [f"{wall} 接縫報告（差＝底構造減實測）", "頻率 Hz｜項目｜實測｜底構造｜差"]
        for row in rows:
            for field, label in (("real_pa_s_m", "阻抗實部 Pa·s/m"), ("imag_pa_s_m", "阻抗虛部 Pa·s/m"),
                                 ("normal_absorption", "垂直入射吸音率"), ("random_absorption", "無規入射吸音率")):
                measured, base = getattr(row.measured, field), getattr(row.base_assembly, field)
                lines.append(f"{row.frequency_hz:g}｜{label}｜{measured:.12g}｜{base:.12g}｜{base - measured:.12g}")
        sections.append("\n".join(lines))
    return sections


def construction_note_lines(inputs: ReportInput, notes_by_wall: dict[str, tuple[ConstructionNote, ...]]) -> list[str]:
    lines = ["構造註記"]
    for wall, notes in notes_by_wall.items():
        zones = inputs.surface_zones_by_wall
        material = zones[wall][0].material if zones is not None else None
        for note in notes:
            if note.code == "measured_table_model_fill" and isinstance(material, MeasuredImpedance):
                bounds = material.points[0].frequency_hz, material.points[-1].frequency_hz
                frequency = extrapolated_frequency_text(note.frequencies_hz, table_bounds_hz=bounds)
                lines.append(f"{wall}：{note.message}；模型補值頻點 {frequency}")
            else:
                frequency = f"；外推頻點 {extrapolated_frequency_text(note.frequencies_hz)}" if note.frequencies_hz else ""
                prefix = "底構造" if isinstance(material, MeasuredImpedance) else ""
                lines.append(f"{wall} {prefix}第 {note.layer_index} 層：{note.message}{frequency}")
    return lines
