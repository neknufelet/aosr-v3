"""實測表的純顯示：只讀當次保存的接縫與註記，不重新求材料響應。"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from aosr.gui.labels import WALLS
from aosr.geometry.shoebox import Wall
from aosr.materials.construction_models import ConstructionNote, MeasuredImpedance, SurfaceMaterial
from aosr.physics.report_measured import MeasuredSeam
from aosr.physics.wall_surfaces import extrapolated_frequency_text, merge_construction_notes
from aosr.reporting.result import SchemeResult
from aosr.reporting.scheme import Scheme


class MeasuredWallView(BaseModel):
    """每面牆一份表頭與端點表；所有數字都已格式化。"""
    model_config = ConfigDict(frozen=True, extra='forbid')
    wall_name: str
    header_lines: tuple[str, ...]
    seam_heading: str = '接縫報告（差＝底構造減實測）'
    seam_note: str = ('即使表完全正確，接縫處的無規吸音率本來就會跳一下'
                      '（表內不分角度、底構造分角度）。')
    rows: tuple[tuple[str, ...], ...]


def measured_materials(scheme: Scheme) -> dict[str, MeasuredImpedance]:
    """沿用完整方案驗證過的模型，不另建材料物件。"""
    return {wall: entries[0].material for wall, entries in (scheme.scene.surface_zones_by_wall or {}).items()
            if isinstance(entries[0].material, MeasuredImpedance)}


def _saved_seams(result: SchemeResult, materials: dict[str, MeasuredImpedance]) -> dict[str, tuple[MeasuredSeam, ...]]:
    reference: dict[str, tuple[MeasuredSeam, ...]] | None = None
    for pair in result.pairs:
        saved = pair.report.measured_seams_by_wall
        if saved is None or saved.keys() != materials.keys():
            raise ValueError('保存的接縫牆面與方案的實測牆不一致，無法顯示接縫')
        for wall, material in materials.items():
            if tuple(s.frequency_hz for s in saved[wall]) != (material.points[0].frequency_hz, material.points[-1].frequency_hz):
                raise ValueError(f'{WALLS[wall]}保存的接縫端點與實測表不一致')
        if reference is not None and saved != reference:
            raise ValueError('各位置對保存的接縫不一致，無法顯示接縫')
        reference = saved
    if reference is None:
        raise ValueError('結果沒有保存任何位置對的接縫')
    return reference


def _header(result: SchemeResult, wall: str, material: MeasuredImpedance) -> tuple[str, ...]:
    low, high = material.points[0].frequency_hz, material.points[-1].frequency_hz
    extra = []
    merged: dict[Wall, tuple[ConstructionNote, ...]] = {}
    key = Wall.from_name(wall)
    for pair in result.pairs:
        merge_construction_notes(merged, key, (pair.report.construction_notes_by_wall or {}).get(wall, ()))
    for note in merged.get(key, ()):
        if note.code == 'komatsu_frequency_extrapolation':
            extra.append(f'底構造第 {note.layer_index} 層另有材料模型外推（小松纖維模型）：'
                         f'{extrapolated_frequency_text(note.frequencies_hz)}')
        elif note.code == 'komatsu_material_class_outside_fit':
            extra.append(f'底構造第 {note.layer_index} 層的材料類別不在小松纖維模型擬合範圍')
        elif note.code == 'komatsu_flow_below_samples':
            extra.append(f'底構造第 {note.layer_index} 層流阻低於原研究樣品範圍，照算但不宣稱準確性')
    return (f'{WALLS[wall]}：實測表涵蓋 {low:g}–{high:g} Hz；{low:g} Hz 以下與 {high:g} Hz 以上是模型補值（底構造）',
            '實測段假設與角度無關。', '實測表只代表樣品在管子裡量到的結果，不代表現場安裝（邊緣、龍骨、板會彎）。',
            *dict.fromkeys(extra))


def measured_note_text(text: str) -> str:
    """僅翻實測牆保存註記的模型名；保留原有構造牆文字與當次註記內容。"""
    return text.replace('Komatsu（小松）纖維模型', '小松纖維模型')


def _seam_rows(seams: tuple[MeasuredSeam, ...]) -> tuple[tuple[str, ...], ...]:
    fields = (('real_pa_s_m', '阻抗實部（Pa·s/m）'), ('imag_pa_s_m', '阻抗虛部（Pa·s/m）'),
              ('normal_absorption', '垂直入射吸音率'), ('random_absorption', '無規入射吸音率'))
    return tuple((f'{seam.frequency_hz:g} Hz', label, f'{measured:.6g}', f'{base:.6g}', f'{base - measured:.6g}')
                 for seam in seams for field, label in fields
                 for measured, base in [(getattr(seam.measured, field), getattr(seam.base_assembly, field))])


def build_measured_view(result: SchemeResult) -> tuple[MeasuredWallView, ...] | None:
    """先核全部位置對相等，才顯示結果檔保存的數字；無實測牆時整塊省略。"""
    materials = measured_materials(result.scheme)
    if not materials:
        return None
    seams = _saved_seams(result, materials)
    return tuple(MeasuredWallView(wall_name=WALLS[wall], header_lines=_header(result, wall, material),
                                 rows=_seam_rows(seams[wall])) for wall, material in materials.items())


def _point_text(material: MeasuredImpedance) -> dict[float, str]:
    return {p.frequency_hz: f'阻抗實部 {p.real_pa_s_m:g}、阻抗虛部 {p.imag_pa_s_m:g} Pa·s/m' for p in material.points}


def measured_differences(a: SurfaceMaterial | None, b: SurfaceMaterial | None) -> tuple[tuple[str, str, str], ...]:
    """用頻率作鍵，新增或刪除一列不牽動後續列；細項留在原本那面牆的列內。"""
    if not isinstance(a, MeasuredImpedance) or not isinstance(b, MeasuredImpedance):
        return ()
    left, right = _point_text(a), _point_text(b)
    raw_left = {point.frequency_hz: point for point in a.points}
    raw_right = {point.frequency_hz: point for point in b.points}
    rows = []
    for frequency in sorted(raw_left.keys() | raw_right.keys()):
        if raw_left.get(frequency) == raw_right.get(frequency):
            continue
        before, after = left.get(frequency, '未設定'), right.get(frequency, '未設定')
        if before == after:
            after += '（顯示位數以下有差異）'
        rows.append((f'{frequency:g} Hz', before, after))
    if a.source_and_conditions != b.source_and_conditions:
        rows.append(('量測出處與條件', a.source_and_conditions, b.source_and_conditions))
    if a.base_assembly != b.base_assembly:
        from aosr.gui.problem_text import material_summary
        rows.append(('同一面牆的底構造', material_summary(a.base_assembly), material_summary(b.base_assembly)))
    if a.fill_rule != b.fill_rule:
        rows.append(('補值規則版本', '原補值規則', '補值規則已變更'))
    return tuple(rows)


def wall_difference_details(a: Scheme, b: Scheme, path: str) -> tuple[tuple[str, str, str], ...]:
    prefix = 'scene.surface_zones_by_wall.'
    if not path.startswith(prefix):
        return ()
    wall = path.removeprefix(prefix)
    left, right = measured_materials(a), measured_materials(b)
    return measured_differences(left.get(wall), right.get(wall))
