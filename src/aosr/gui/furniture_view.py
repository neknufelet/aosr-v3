"""家具顯示只讀當次存下的表頭，不重算材質或修改評分資料。"""
from aosr.reporting.display import (
    FURNITURE_FACES, furniture_name,
)
from aosr.reporting.result import PairResult
from aosr.reporting.furniture_display import furniture_rows as furniture_rows
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
