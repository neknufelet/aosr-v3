"""家具顯示只讀當次存下的表頭，不重算材質或修改評分資料。"""
from aosr.reporting.display import (
    FURNITURE_ESTIMATE_NOTE, FURNITURE_FACES, FURNITURE_MATERIALS, FURNITURE_WOOD_CLOUD_NOTE,
    furniture_name, furniture_unknown_text,
)
from aosr.reporting.result import PairResult, SchemeResult
from aosr.reporting.scheme import Scheme
from aosr.scoring.reflections_contract import ReflectionPath, ReflectionSource


def furniture_surface(path: ReflectionPath, pair: PairResult, scheme: Scheme) -> str:
    """source_index 是原始 table.rows（含直達）的索引；補算列不查這張表。"""
    table = pair.report.path_table
    if path.source is not ReflectionSource.PATH_TABLE or table is None:
        return ""
    row = table.rows[path.source_index]
    if row.furniture_id is None or row.furniture_face is None:
        return ""
    item = next(item for item in scheme.furniture or () if item.furniture_id == row.furniture_id)
    return f"{furniture_name(item.kind, row.furniture_id)}{FURNITURE_FACES[row.furniture_face]}"


def has_furniture_paths(pair: PairResult) -> bool:
    table = pair.report.path_table
    return table is not None and any(row.furniture_id is not None for row in table.rows)


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
    if kind == "ceiling_cloud" and material == "wood":
        text += f"；{FURNITURE_WOOD_CLOUD_NOTE}"
    return text
