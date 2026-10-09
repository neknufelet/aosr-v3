"""家具材質與報告說明只讀當次存檔路徑表，不查現行材質登記簿。"""
from aosr.reporting.display import (
    FURNITURE_ESTIMATE_NOTE, FURNITURE_MATERIALS, FURNITURE_MATERIAL_NOTES, FURNITURE_WOOD_CLOUD_NOTE,
    FURNITURE_MODEL_TITLE, FURNITURE_REASON, FURNITURE_REFLECTION_NOTE, FURNITURE_TRANSMISSION_NOTE,
    FURNITURE_REVERBERATION_REPORT_NOTE, FURNITURE_FINITE_SIZE_NOTE, FURNITURE_DIRECTIVITY_NOTE,
    furniture_name, furniture_unknown_text,
)
from aosr.reporting.result import SchemeResult


def furniture_rows(result: SchemeResult) -> tuple[tuple[str, str, str], ...]:
    """取主位的存檔路徑表材質；未知頻帶不可改查現在的登記簿。"""
    primary = result.scheme.receiver_set.primary.receiver_id
    table = next((pair.report.path_table for pair in result.pairs
                  if pair.receiver_id == primary and pair.report.path_table is not None), None)
    if not result.scheme.furniture or table is None:
        return ()
    items = {item.furniture_id: item for item in result.scheme.furniture}
    return tuple((furniture_name(items[row.furniture_id].kind, row.furniture_id),
                  _material_text(items[row.furniture_id].kind, row.material),
                  furniture_unknown_text(row.unknown_bands_hz)) for row in table.furniture_materials or ())


def _material_text(kind: str, material: str) -> str:
    text = f"{FURNITURE_MATERIALS[material]}；{FURNITURE_ESTIMATE_NOTE}"
    if material in FURNITURE_MATERIAL_NOTES:
        text += f"；{FURNITURE_MATERIAL_NOTES[material]}"
    if kind == "ceiling_cloud" and material == "wood":
        text += f"；{FURNITURE_WOOD_CLOUD_NOTE}"
    return text


def furniture_material_lines(result: SchemeResult) -> tuple[str, ...]:
    return tuple("；".join(value for value in row if value) for row in furniture_rows(result))


def furniture_report_notes(result: SchemeResult | None) -> tuple[str, ...]:
    """缺第一名結果時保留模型範圍，不以專案或現行登記簿猜材質。"""
    return (FURNITURE_MODEL_TITLE, FURNITURE_REASON, FURNITURE_REFLECTION_NOTE,
            *(furniture_material_lines(result) if result is not None else ()),
            FURNITURE_FINITE_SIZE_NOTE, FURNITURE_DIRECTIVITY_NOTE,
            FURNITURE_TRANSMISSION_NOTE, FURNITURE_REVERBERATION_REPORT_NOTE)
