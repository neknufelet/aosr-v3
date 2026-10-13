"""表單構造數字與範圍提示；不解析材料模型，正式驗證只走完整方案。"""
from __future__ import annotations

from typing import cast

from aosr.config.paths import config_path
from aosr.config.porous_model_ranges import load_porous_model_ranges
from aosr.gui.problem_text import MATERIAL_MODELS
from aosr.physics.report_surfaces import ZONE_WALL_FIELD
from aosr.reporting.validation import SchemeProblem


def wall_materials(document: dict[str, object]) -> dict[str, dict[str, object]]:
    """已檢查容器的表單資料，只讀原始值；舊六數字仍由原表單處理。"""
    scene = document.get("scene")
    if not isinstance(scene, dict):
        return {}
    zones = scene.get(ZONE_WALL_FIELD)
    if not isinstance(zones, dict):
        return {}
    return {name: cast(dict[str, object], entries[0]["material"])
            for name, entries in zones.items() if isinstance(entries, list) and entries
            and isinstance(entries[0], dict) and isinstance(entries[0].get("material"), dict)}


def _layers(material: dict[str, object]) -> list[object]:
    layers = material.get("layers")
    return layers if isinstance(layers, list) else []


def material_input_problems(document: object, problems: tuple[SchemeProblem, ...]) -> tuple[SchemeProblem, ...]:
    """補上核心同一登記簿的流阻拒收；格式錯先由 validate_scheme（方案驗證）定案。"""
    located = tuple(problem for problem in problems if problem.path.startswith(f"scene.{ZONE_WALL_FIELD}"))
    if located or not isinstance(document, dict):
        return located
    ranges = load_porous_model_ranges(config_path("porous_model_ranges.toml")).komatsu
    low, high = ranges.sample_flow_resistivity_min_pa_s_m2, ranges.sample_flow_resistivity_max_pa_s_m2
    issues = []
    for wall, material in wall_materials(document).items():
        for index, layer in enumerate(_layers(material)):
            if not isinstance(layer, dict) or layer.get("kind") != "porous":
                continue
            model = layer.get("model", MATERIAL_MODELS["porous"].model_fields["model"].default)
            sigma = layer.get("flow_resistivity_pa_s_m2")
            if model == "komatsu" and isinstance(sigma, int | float) and not low <= sigma <= high:
                issues.append(SchemeProblem(f"scene.{ZONE_WALL_FIELD}.{wall}.0.material.assembly.layers.{index}.porous.flow_resistivity_pa_s_m2",
                    f"小松纖維模型的流阻可用範圍為 {low:g}–{high:g} Pa·s/m²；這是原研究樣品範圍，請改流阻或選合適模型"))
    return tuple(issues)


def material_field_values(document: dict[str, object]) -> dict[str, object]:
    """混合格式裡的固定面與逐層數字沿用原值保存，欄位由模型反查。"""
    values = {}
    for wall, material in wall_materials(document).items():
        model = MATERIAL_MODELS.get(str(material.get("kind")))
        if model is None:
            continue
        if material.get("kind") == "constant_impedance":
            name = next(name for name in model.model_fields if name != "kind")
            values[f"wall-{wall}"] = material.get(name)
        for index, layer in enumerate(_layers(material)):
            if not isinstance(layer, dict):
                continue
            for name, value in layer.items():
                if isinstance(value, int | float) or value is None:
                    values[f"layer-{wall}-{index}-{name}"] = value
    return values
