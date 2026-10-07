"""家具近似的唯一評分來源：存得下、重評時仍在的路徑表表頭。"""
from __future__ import annotations

from typing import Final

from aosr.physics.report_io import PathTableSection
from aosr.scoring.contract_base import Flag


_MODEL_FLAGS: Final[dict[str | None, tuple[Flag, ...]]] = {
    None: (),
    "single_bounce_finite_size_v1": (Flag.FURNITURE_MODEL_APPROXIMATE,),
}


def furniture_model_state(table: PathTableSection | None) -> tuple[str | None, tuple[Flag, ...]]:
    """回表頭模型與共用旗標；表外模型直接拒收，沒有路徑表視為沒有家具。"""
    model = table.furniture_model if table is not None else None
    try:
        return model, _MODEL_FLAGS[model]
    except KeyError as exc:
        raise ValueError(f"未知家具模型：{model}") from exc
