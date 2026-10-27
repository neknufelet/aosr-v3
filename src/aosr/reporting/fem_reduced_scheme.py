"""從一份方案交出替身方程式要的完整模型算子（#561）。

方案先過 ``checked_inputs``，房間與空氣取驗過的第一對輸入（跟 :mod:`aosr.reporting.fem_slices` 同一個取法），
喇叭取方案的兩支、座位取座位組的每一點；物理身分照現行入口算好一起交出去，存替身時記進 manifest。
網格上限預設是正式的有限元素交界頻率。
"""
from __future__ import annotations

from aosr.config.capabilities import CapabilityTable
from aosr.config.directivity_defaults import DirectivityDefaults
from aosr.config.frequency_axis import FEM_GEOMETRIC_CROSSOVER_CAP_HZ
from aosr.geometry.shoebox import Point
from aosr.physics import report_io
from aosr.physics.fem_reduced import FullOperators, export_room_operators
from aosr.reporting.physics_identity import physics_identity
from aosr.reporting.scheme import Scheme
from aosr.reporting.validation import checked_inputs


def export_operators(
    scheme: Scheme, *, capabilities: CapabilityTable, directivity: DirectivityDefaults,
    max_frequency_hz: float = FEM_GEOMETRIC_CROSSOVER_CAP_HZ,
) -> FullOperators:
    """K、M、六面 B、每支喇叭的點源右側、每個座位的取值列、網格雜湊與物理身分。"""
    checked, documents = checked_inputs(scheme, capabilities=capabilities, directivity=directivity)
    solved = report_io.solver_inputs(next(iter(documents.values()))[1])
    return export_room_operators(
        solved.room, dict(checked.speakers),
        {point.receiver_id: Point(*point.position_m) for point in checked.receiver_set.points},
        density_kg_m3=solved.density_kg_m3, sound_speed_m_s=solved.sound_speed_m_s,
        physics_identity=physics_identity(capabilities=capabilities, directivity=directivity),
        max_frequency_hz=max_frequency_hz)
