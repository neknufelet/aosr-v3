"""分區方案的逐區接縫表：矩形區用實測阻抗表時，逐區報表格兩端的實測與底構造值。

數字逐區借整面那一份（:func:`aosr.physics.report_measured.measured_seams`）求值，兩份同一套算法；只報告、不訂警告門檻。
整面一區的牆照舊進報表的 ``measured_seams_by_wall``，這裡只收有矩形的牆。
獨立成一支、不改 ``report_measured`` 與 ``report_io``：那兩支在有限元素分片身分（fem-v1）的閉包裡，
改了低頻分片就得重解（#1066）。貼在實測原牆上的構造區跟整面一樣不報接縫。
"""
from __future__ import annotations

from typing import Annotated

from pydantic import AfterValidator

from aosr.geometry.shoebox import Wall
from aosr.materials.construction_models import MeasuredImpedance
from aosr.physics.report_measured import MeasuredSeam, measured_seams
from aosr.physics.report_surfaces import RectangleZone, WholeWallZone, ZonesByWall

SeamPair = tuple[MeasuredSeam, MeasuredSeam]


def _canonical(table: dict[str, dict[str, SeamPair]]) -> dict[str, dict[str, SeamPair]]:
    """讀回存檔也守同一個形狀：六面牆名、每面至少一區、區代號非空、兩端由低到高。"""
    if not table or not set(table) <= set(Wall.wall_names()):
        raise ValueError("逐區接縫表必須用六面牆名，且不可為空物件")
    for wall, zones in table.items():
        if not zones or not all(zones):
            raise ValueError(f"{wall} 的逐區接縫表不可為空，區代號也不可為空字串")
        for zone_id, (low, high) in zones.items():
            if low.frequency_hz >= high.frequency_hz:
                raise ValueError(f"{wall} 分區 {zone_id} 的接縫須依低、高端點排列")
    return table


ZoneSeamTable = Annotated[dict[str, dict[str, SeamPair]], AfterValidator(_canonical)]


def zone_measured_seams(zones: ZonesByWall | None, *, density_kg_m3: float,
                        sound_speed_m_s: float) -> dict[str, dict[str, SeamPair]] | None:
    """矩形區逐區取表端點；沒有任何矩形實測區就回 None（輸出省略這一塊）。"""
    result: dict[str, dict[str, SeamPair]] = {}
    for wall, cells in (zones or {}).items():
        for zone in cells:
            if not isinstance(zone, RectangleZone) or not isinstance(zone.material, MeasuredImpedance):
                continue
            try:
                seams = measured_seams({wall: (WholeWallZone(area="whole_wall", material=zone.material),)},
                                       density_kg_m3=density_kg_m3, sound_speed_m_s=sound_speed_m_s)
            except ValueError as exc:
                raise ValueError(f"分區 {zone.id}：{exc}") from exc
            assert seams is not None
            low, high = seams[wall]
            result.setdefault(wall, {})[zone.id] = (low, high)
    return result or None
