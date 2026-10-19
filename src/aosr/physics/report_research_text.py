"""材料研究範圍的報告文字；只讀方案與登記簿，位於物理計算身分閉包外。"""
from __future__ import annotations

from aosr.config.paths import config_path
from aosr.config.porous_model_ranges import load_porous_model_ranges
from aosr.materials.construction_models import Assembly, MeasuredImpedance, PorousLayer
from aosr.physics.report_surfaces import ZonesByWall


def komatsu_research_notice(zones_by_wall: ZonesByWall | None, *, web: bool = False) -> str | None:
    """任一牆含小松層就提醒；實測表看同一面牆的底構造，不依賴已存外推註記。"""
    for zones in (zones_by_wall or {}).values():
        for zone in zones:
            material = zone.material
            if isinstance(material, MeasuredImpedance):
                material = material.base_assembly
            if isinstance(material, Assembly) and any(isinstance(layer, PorousLayer) and layer.model == "komatsu"
                                                     for layer in material.layers):
                lower = load_porous_model_ranges(config_path("porous_model_ranges.toml")).komatsu.measured_frequency_min_hz
                name = "小松纖維模型" if web else "Komatsu（小松）纖維模型"
                return f"{name}：原論文只量到 {lower:g} Hz 以上"
    return None
