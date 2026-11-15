"""#1171 黏住鋼板的背襯（複合板共振器 VPR）考卷共用的原始欄位字典；不 import 被測程式。

數字出處：
- 背襯 10 cm 美耐皿的流阻 8,644 Pa·s/m²：Sgard、Atalla、Robin 2024 Frontiers in Acoustics 2:1414356 表 1。
- 骨架楊氏模數 0.020／0.05125／0.0825 MPa：用 1 mm 與 2.5 mm 鋼板加 10 cm 美耐皿的有效吸音率讀圖值（Kristl 2023 轉載
  Brandstätt 等的量測，二手）反推，兩片板峰位都落在實測範圍內的 E 交集兩端與中間（#1171 工單的證據檔）。
- 骨架損耗 0.08：Bonfiglio、Pompoli、Shravage 2008 量美耐皿 0.07–0.09 的中間。
- 鋼板面密度＝厚度 × 7,850 kg/m³（鋼密度常用值）；板阻力 50 Pa·s/m 只是讓阻力欄能被微擾，不是量測值。
"""
from __future__ import annotations

E_LOW, E_MID, E_HIGH = 20_000.0, 51_250.0, 82_500.0
LOSS = 0.08
STEEL_1MM: dict[str, object] = {"kind": "board", "surface_mass_kg_m2": 7.85, "resistance_pa_s_m": 0.0}
STEEL_2_5MM: dict[str, object] = {"kind": "board", "surface_mass_kg_m2": 19.625, "resistance_pa_s_m": 50.0}
MELAMINE_100: dict[str, object] = {"kind": "porous", "thickness_m": 0.10, "flow_resistivity_pa_s_m2": 8644.0,
                                   "model": "komatsu", "material_class": "foam"}
MELAMINE_BONDED: dict[str, object] = {**MELAMINE_100, "frame_young_modulus_pa": E_MID, "frame_loss_factor": LOSS}
# 老闆 11-15 問的前面加板：長縫板（16 mm、縫 3 mm、縫距 32 mm）＋5 cm 美耐皿擺在鋼板前面；尺寸是常見值的例子。
SLOTTED_FRONT: dict[str, object] = {"kind": "slotted", "thickness_m": 0.016, "slot_width_m": 0.003, "slot_spacing_m": 0.032}
MELAMINE_50: dict[str, object] = {**MELAMINE_100, "thickness_m": 0.05}
BONDED_VPR: dict[str, object] = {"kind": "assembly", "backing": "rigid", "layers": [STEEL_2_5MM, MELAMINE_BONDED]}
