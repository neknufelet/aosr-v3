"""八組材料的手寫答案；按施工二 decisions.md 第一節與 presets-spec.md 第二節逐欄核對。"""
NEW_ROWS: dict[str, tuple[str, list[dict[str, object]]]] = {
    "fabric_pet_a_gap100": ("PET（聚酯）織布 A＋100 mm 空腔", [
        {"kind": "fabric", "surface_mass_kg_m2": .065, "flow_resistance_pa_s_m": 196.},
        {"kind": "air", "thickness_m": .100}]),
    "fabric_pet_b_gap100": ("PET（聚酯）織布 B＋100 mm 空腔", [
        {"kind": "fabric", "surface_mass_kg_m2": .120, "flow_resistance_pa_s_m": 462.},
        {"kind": "air", "thickness_m": .100}]),
    "perforated_gypsum_8_18r_backed_gw30_ceiling": ("穿孔石膏板 8/18R 背面貼布＋30 mm 玻璃棉＋吊頂空腔", [
        {"kind": "perforated", "thickness_m": .0125, "hole_diameter_m": .008,
         "hole_spacing_m": .018, "hole_pattern": "square", "backing_fabric_surface_mass_kg_m2": .063,
         "backing_fabric_flow_resistance_pa_s_m": 236.},
        {"kind": "porous", "thickness_m": .030, "model": "komatsu", "material_class": "fibrous",
         "flow_resistivity_pa_s_m2": 6900., "flow_resistivity_source": "density_table"},
        {"kind": "air", "thickness_m": .170}]),
    "perforated_0p8mm_d10_gw24k_50mm": ("0.8 mm 圓孔面板（孔 10 mm）＋24K 玻璃棉 50 mm 貼牆", [
        {"kind": "perforated", "thickness_m": .0008, "hole_diameter_m": .010, "open_area": .31},
        {"kind": "porous", "thickness_m": .050, "model": "komatsu", "material_class": "fibrous",
         "flow_resistivity_pa_s_m2": 6900., "flow_resistivity_source": "density_table"}]),
    "mpp_t1mm_d0p5_gap50": ("微孔板 1 mm 厚、孔 0.5 mm＋50 mm 空腔", [
        {"kind": "micro_perforated", "thickness_m": .001, "hole_diameter_m": .0005,
         "open_area": .0077, "surface_mass_kg_m2": 1.13},
        {"kind": "air", "thickness_m": .050}]),
    "mpp_t0p5mm_d0p5_gap50": ("微孔板 0.5 mm 厚、孔 0.5 mm＋50 mm 空腔", [
        {"kind": "micro_perforated", "thickness_m": .0005, "hole_diameter_m": .0005, "open_area": .00785},
        {"kind": "air", "thickness_m": .050}]),
    "slotted_gypsum_b6_backed_gap65": ("開縫石膏板 Slotline B6 背面貼布＋65 mm 空腔", [
        {"kind": "slotted", "thickness_m": .0125, "slot_width_m": .005, "open_area": .157,
         "backing_fabric_surface_mass_kg_m2": .063, "backing_fabric_flow_resistance_pa_s_m": 236.},
        {"kind": "air", "thickness_m": .065}]),
    "slotted_wood_typ9_absorber40_gap98": ("開縫木板 Typ 9＋40 mm 吸音材＋98 mm 空腔", [
        {"kind": "slotted", "thickness_m": .031, "slot_width_m": .0055, "open_area": .133},
        {"kind": "porous", "thickness_m": .040, "model": "komatsu", "material_class": "fibrous",
         "flow_resistivity_pa_s_m2": 19100.},
        {"kind": "air", "thickness_m": .098}]),
}
