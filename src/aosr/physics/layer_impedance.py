"""自訂房間（#1327）每個圖層的垂直入射阻抗：跟六面牆 ``wall_normal_impedances`` 同一套算法，只算一面。

另開一個模組、不改六面牆那一支：六面牆在物理身分（結果快取的鑰匙）的程式範圍裡，動了它，
算過的結果全部要重算；這一支只有自訂房間的背景計算用得到。
"""
from __future__ import annotations

import numpy as np

from aosr.config.construction_environment import construction_environment
from aosr.materials.construction_models import ConstantImpedance, ConstructionNote, SurfaceMaterial


def normal_impedance(
    material: SurfaceMaterial | None, frequencies_hz: tuple[float, ...], *, density_kg_m3: float, sound_speed_m_s: float,
) -> tuple[float | None | tuple[complex, ...], tuple[ConstructionNote, ...] | None]:
    """一面的垂直入射 Z(f,0°)：固定值保留原 float，None 表整軸剛性；構造另回物理的提醒（固定值與剛性回 None）。

    構造直接在此軸雙精度求值、不內插；密度與聲速來自呼叫端，其餘空氣常數來自產品設定。
    """
    if material is None:
        return None, None
    if isinstance(material, ConstantImpedance):
        return material.impedance_pa_s_m, None
    import jax.numpy as jnp
    from aosr.materials.construction import surface_impedance

    air, ranges = construction_environment(density_kg_m3=density_kg_m3, sound_speed_m_s=sound_speed_m_s)
    values, notes = surface_impedance(material, jnp.asarray(np.asarray(frequencies_hz, dtype=np.float64)),
                                      air=air, ranges=ranges)
    # 先整排轉成 numpy 再交出 Python 複數：位元相同，免得一格一格向 JAX 取值。
    return tuple(np.asarray(values, dtype=np.complex128).tolist()), notes
