"""命令列的實測牆表頭、接縫小表與補值註記；差值只在顯示時計算。"""
from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from aosr.materials.construction_models import ConstructionNote, MeasuredImpedance, MountedOnWall
from aosr.physics.report_measured import measured_seams
from aosr.physics.wall_surfaces import extrapolated_frequency_text

if TYPE_CHECKING:
    from aosr.physics.report_io import ReportInput


def _base_note_text(notes: tuple[ConstructionNote, ...]) -> str:
    """表頭照實寫底構造自己的 Komatsu 註記：表外頻點同時是模型補值與材料模型外推時兩個都寫。"""
    parts = []
    for note in notes:
        if note.code == "komatsu_frequency_extrapolation":
            parts.append(f"其中 {extrapolated_frequency_text(note.frequencies_hz)} 底構造第 {note.layer_index} 層另有材料模型外推")
        elif note.code == "komatsu_material_class_outside_fit":
            parts.append(f"底構造第 {note.layer_index} 層的材料類別不在 Komatsu（小松）纖維模型擬合範圍")
        elif note.code == "komatsu_flow_below_samples":
            parts.append(f"底構造第 {note.layer_index} 層流阻低於原研究樣品範圍，照算但不宣稱準確性")
    return "".join(f"；{part}" for part in parts)


def measured_text(inputs: ReportInput, notes_by_wall: Mapping[str, tuple[ConstructionNote, ...]] | None = None) -> list[str]:
    sections = []
    for wall, zones in (inputs.surface_zones_by_wall or {}).items():
        material = zones[0].material
        if not isinstance(material, MeasuredImpedance):
            continue
        lower, upper = material.points[0].frequency_hz, material.points[-1].frequency_hz
        sections.append(f"{wall}：實測表涵蓋 {lower:g}–{upper:g} Hz；{lower:g} Hz 以下與 {upper:g} Hz 以上是模型補值（底構造）"
                        f"{_base_note_text((notes_by_wall or {}).get(wall, ()))}；"
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
    """貼在原牆上的構造：補值註記同實測牆（界線取原牆的表），其他註記的訊息自帶「第 k 層／原牆第 k 層」。"""
    lines = ["構造註記"]
    for wall, notes in notes_by_wall.items():
        zones = inputs.surface_zones_by_wall
        material = zones[wall][0].material if zones is not None else None
        table = material.wall if isinstance(material, MountedOnWall) else material
        for note in notes:
            if note.code == "measured_table_model_fill" and isinstance(table, MeasuredImpedance):
                bounds = table.points[0].frequency_hz, table.points[-1].frequency_hz
                frequency = extrapolated_frequency_text(note.frequencies_hz, table_bounds_hz=bounds)
                lines.append(f"{wall}：{note.message}；模型補值頻點 {frequency}")
            else:
                frequency = f"；外推頻點 {extrapolated_frequency_text(note.frequencies_hz)}" if note.frequencies_hz else ""
                if isinstance(material, MeasuredImpedance | MountedOnWall):
                    lines.append(f"{wall} {note.message}{frequency}")
                else:  # 既有構造牆照舊（文字輸出逐位不變）
                    lines.append(f"{wall} 第 {note.layer_index} 層：{note.message}{frequency}")
    return lines
