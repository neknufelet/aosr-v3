"""合成材料庫 12 筆（#1183，給 #1176「每一筆材料能用在哪幾面」的考卷用）：**全部是編的**，不對應任何真實房間、真實材料庫或量測。
代號、顯示名、每一層的厚度與參數、實測表每一列都是自己編的；只照「分組規則要什麼形狀」造（規則在
``aosr.gui.material_surfaces`` 模組說明，前面的先吃）：每一筆打哪一條、還符合哪幾條但被前面先吃掉，寫在那一筆上面的註解。
``demo109`` 前綴、``ceiling``／``floor`` 這類字樣是規則要看的，照留；名字長的那幾筆（VPR 三筆）字數跟寬窄字數刻意做得一樣長，
寬度考題才有東西撐。順序有意義（考卷取第 3 筆當門）。材料要過 ``LibraryEntry`` 驗證，數字一律寫成小數（嚴格浮點不收整數）。"""
from __future__ import annotations

# 第 2 筆的合成實測表（頻率、實部、虛部；Pa·s/m），用一條自訂的平滑曲線編出來再四捨五入，不是量的。
_PET35_TABLE: tuple[tuple[float, float, float], ...] = (
    (105.0, 3146.3, -6912.0), (133.0, 2611.5, -5375.6), (171.0, 2158.1, -4116.8), (207.0, 1877.8, -3362.5),
    (259.0, 1606.6, -2653.5), (317.0, 1406.1, -2144.7), (389.0, 1238.0, -1730.2), (461.0, 1120.9, -1448.8),
    (557.0, 1010.5, -1190.1), (683.0, 911.3, -964.1), (811.0, 841.1, -808.4), (997.0, 770.4, -655.5),
    (1203.0, 716.9, -542.9), (1461.0, 670.5, -447.9), (1777.0, 631.5, -370.3), (2131.0, 601.2, -311.4),
    (2593.0, 573.7, -259.4), (3109.0, 552.3, -220.2),
)
# 第 6 筆的合成表列值（頻率、實部；Pa·s/m）：整張沒有虛部，兩頭持平。
_GLASS_TABLE: tuple[tuple[float, float], ...] = (
    (21.0, 8400.5), (71.0, 8400.5), (141.0, 13950.5), (283.0, 44800.5), (567.0, 70150.5), (1131.0, 98700.5),
    (2263.0, 141500.5), (4527.0, 141500.5), (11001.0, 141500.5),
)
_SOURCE = "合成：考卷自己編的數字，不對應任何真實材料或量測"
_WOOL30 = {"kind": "porous", "thickness_m": 0.03, "flow_resistivity_pa_s_m2": 14800.0, "model": "komatsu", "material_class": "fibrous"}
_FIBRE_CEMENT = {"kind": "board", "surface_mass_kg_m2": 11.2, "resistance_pa_s_m": 64.0}
_GLASS_12MM = {"kind": "board", "surface_mass_kg_m2": 30.0, "resistance_pa_s_m": 0.0}


def _fixed(key: str, label: str, provenance: str, material: dict[str, object]) -> dict[str, object]:
    return {"id": key, "label": label, "provenance": provenance, "source": _SOURCE, "links": [], "notes": [], "kind": "fixed",
            "material": material}


SAMPLE_USER_ENTRIES: list[dict[str, object]] = [
    # 1. 可最佳化、代號 demo109 開頭 → 第 1 條（不然是第 6 條牆面）；全庫唯一一筆可最佳化。
    {"id": "demo109_rockwool60k_capped", "label": "109示範-60K岩棉板（試算上限）", "provenance": "literature", "source": _SOURCE,
     "links": [], "notes": [], "kind": "optimizable",
     "porous": {"kind": "porous", "thickness_m": 0.075, "flow_resistivity_pa_s_m2": 23500.0, "flow_resistivity_source": "catalog",
                "model": "komatsu", "material_class": "fibrous"},
     "porous_thickness": {"adjustable": False, "range_m": [0.075, 0.075], "choices_m": None, "initial_m": 0.075},
     "cavity": {"adjustable": True, "range_m": [0.0, 0.12], "choices_m": None, "initial_m": 0.035},
     "max_total_depth_m": 0.18, "suggested_total_depth_m": 0.11, "preset_members": [], "template_preset": None},
    # 2. 固定實測表（每列都有虛部）、代號 demo109 開頭 → 第 1 條（也符合第 2 條，被第 1 條先吃）。
    _fixed("demo109_measured_pet35_synthetic_b7", "109示範-實測聚酯棉 35mm（自編 synthetic B-7）", "measured", {
        "kind": "measured_impedance",
        "points": [{"frequency_hz": f, "real_pa_s_m": re, "imag_pa_s_m": im} for f, re, im in _PET35_TABLE],
        "base_assembly": {"kind": "assembly", "layers": [
            {"kind": "porous", "thickness_m": 0.035, "flow_resistivity_pa_s_m2": 31700.0, "model": "wilson_2015",
             "material_class": "fibrous", "porosity": 0.94, "tortuosity": 1.35}], "backing": "rigid"},
        "source_and_conditions": "合成表：一條自訂曲線編出來的，沒有量測條件", "fill_rule": "log_linear_table_base_assembly_outside_v1"}),
    # 3. 多層構造、背後 air → 第 3 條（門）。兩片板夾空腔，層跟別筆都不同。
    _fixed("hall_door_hollow_core", "木門 中空 18 kg/m²（自編）", "model", {"kind": "assembly", "layers": [
        {"kind": "board", "surface_mass_kg_m2": 9.0, "resistance_pa_s_m": 0.0}, {"kind": "air", "thickness_m": 0.038},
        {"kind": "board", "surface_mass_kg_m2": 9.0, "resistance_pa_s_m": 0.0}], "backing": "air"}),
    # 4. 固定阻抗、代號含 floor → 第 2 條（也符合第 5 條，被第 2 條先吃）。值不准跟 109 號房原牆的 1646.4 相同。
    _fixed("joist_floor_plank_fixed", "木地板 架高格柵（自編，Z=47k）", "literature",
           {"kind": "constant_impedance", "impedance_pa_s_m": 47250.0}),
    # 5. 多層構造、背後 air → 第 3 條（窗）。
    _fixed("side_window_glass_12mm_single", "玻璃窗 12 mm 單層（合成版；厚度是自填值，不是實物）", "model",
           {"kind": "assembly", "layers": [_GLASS_12MM], "backing": "air"}),
    # 6. 實測表（虛部整張沒有），底構造背後 air → 第 2 條；第 3 條看的是材料本身，實測表不是多層構造，輪不到。
    _fixed("side_window_glass_table_upper", "玻璃窗 表列上限（自編大片玻璃值，非實測）", "literature", {
        "kind": "measured_impedance",
        "points": [{"frequency_hz": f, "real_pa_s_m": re, "imag_pa_s_m": None} for f, re in _GLASS_TABLE],
        "base_assembly": {"kind": "assembly", "layers": [_GLASS_12MM], "backing": "air"},
        "source_and_conditions": "合成表：照形狀編的表列值，沒有量測條件", "fill_rule": "log_linear_table_base_assembly_outside_v1"}),
    # 7～9. 剛背多層構造，代號含 ceiling、顯示名「天花」開頭 → 第 4 條（兩個條件都中；不然是第 6 條牆面）。
    _fixed("ceiling_gypsum9mm_gap300", "天花 石膏板 9 mm 單層噴白漆＋30 cm 空腔（剛背）", "model", {"kind": "assembly", "layers": [
        {"kind": "board", "surface_mass_kg_m2": 7.4, "resistance_pa_s_m": 52.5}, {"kind": "air", "thickness_m": 0.3}],
        "backing": "rigid"}),
    # 第 8 筆：#1175 那題拿它當天花原牆，構造在內建＋合成裡唯一。
    _fixed("ceiling_fibrecement8mm_wool30_on_board_gap180", "天花 纖維水泥板 8 mm＋岩棉毯 30 mm 平鋪在板背＋18 cm 空腔（剛背）", "model",
           {"kind": "assembly", "layers": [_FIBRE_CEMENT, _WOOL30, {"kind": "air", "thickness_m": 0.18}], "backing": "rigid"}),
    # 第 9 筆：跟第 8 筆同三層、順序不同（認名字比的是有順序的層）。
    _fixed("ceiling_fibrecement8mm_gap180_wool30_on_slab", "天花 纖維水泥板 8 mm＋18 cm 空腔＋岩棉毯 30 mm 貼在上方樓板（對照組）", "model",
           {"kind": "assembly", "layers": [_FIBRE_CEMENT, {"kind": "air", "thickness_m": 0.18}, _WOOL30], "backing": "rigid"}),
    # 10～12. 剛背多層構造、代號與顯示名都沒有規則字樣 → 第 6 條（牆面四面與天花）。VPR：骨架兩格只在最後一層、前一層是板。
    _fixed("vpr_alu2mm_foam85_sample", "VPR 合成：鋁板 2 mm 黏貼彈性泡棉 85 mm（自編的尺寸，總厚約 8.7 cm）", "model", {
        "kind": "assembly", "layers": [
            {"kind": "board", "surface_mass_kg_m2": 5.4, "resistance_pa_s_m": 0.0},
            {"kind": "porous", "thickness_m": 0.085, "flow_resistivity_pa_s_m2": 12300.0, "model": "komatsu", "material_class": "foam",
             "frame_compression_modulus_pa": 38500.0, "frame_loss_factor": 0.12}], "backing": "rigid"}),
    _fixed("pet60_vpr_steel1p2mm_foam90_sample", "VPR＋前面聚酯棉 60 mm 合成版（1.2 mm 鋼板、背後泡棉 90 mm，總厚約 15 cm）", "model", {
        "kind": "assembly", "layers": [
            {"kind": "porous", "thickness_m": 0.06, "flow_resistivity_pa_s_m2": 15700.0, "model": "komatsu", "material_class": "fibrous"},
            {"kind": "board", "surface_mass_kg_m2": 9.42, "resistance_pa_s_m": 0.0},
            {"kind": "porous", "thickness_m": 0.09, "flow_resistivity_pa_s_m2": 11800.0, "model": "komatsu", "material_class": "foam",
             "frame_compression_modulus_pa": 44500.0, "frame_loss_factor": 0.06}], "backing": "rigid"}),
    # 第 12 筆：有板（只給面密度、沒有厚度）；顯示名的總厚把 2.25 mm 鋼板算進去（約 14 cm），小字只加有厚度的三層。
    _fixed("slotted18_pet45_vpr_steel2p25mm_foam70_sample",
           "VPR＋長縫板合成（長縫板 18 mm、聚酯棉 45 mm、鋼板 2.25 mm、泡棉墊 70 mm，總厚約 14 cm）", "model", {
        "kind": "assembly", "layers": [
            {"kind": "slotted", "thickness_m": 0.018, "slot_width_m": 0.004, "slot_spacing_m": 0.036},
            {"kind": "porous", "thickness_m": 0.045, "flow_resistivity_pa_s_m2": 15700.0, "model": "komatsu", "material_class": "fibrous"},
            {"kind": "board", "surface_mass_kg_m2": 17.66, "resistance_pa_s_m": 0.0},
            {"kind": "porous", "thickness_m": 0.07, "flow_resistivity_pa_s_m2": 11800.0, "model": "komatsu", "material_class": "foam",
             "frame_compression_modulus_pa": 29800.0, "frame_loss_factor": 0.09}], "backing": "rigid"}),
]
