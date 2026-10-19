"""表單構造數字與範圍提示；不解析材料模型，正式驗證只走完整方案。"""
from __future__ import annotations

from typing import cast

from aosr.config.capabilities import CapabilityTable
from aosr.config.directivity_defaults import DirectivityDefaults
from aosr.config.paths import config_path
from aosr.config.porous_model_ranges import load_porous_model_ranges
from aosr.gui.problem_text import MATERIAL_MODELS
from aosr.physics.report_surfaces import ZONE_WALL_FIELD
from aosr.reporting.validation import SchemeProblem, validate_scheme
from aosr.materials.construction_models import layer_open_area


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
    if material.get("kind") == "measured_impedance":
        base = material.get("base_assembly")
        material = base if isinstance(base, dict) else {}
    layers = material.get("layers")
    return layers if isinstance(layers, list) else []


def material_input_problems(document: object, problems: tuple[SchemeProblem, ...]) -> tuple[SchemeProblem, ...]:
    """只回正式驗證之外補上的流阻拒收（核心同一登記簿）；正式驗證已點名牆面時容器形狀未定，不補。"""
    if any(problem.path.startswith(f"scene.{ZONE_WALL_FIELD}") for problem in problems) or not isinstance(document, dict):
        return ()
    ranges = load_porous_model_ranges(config_path("porous_model_ranges.toml")).komatsu
    low, high = ranges.manual_flow_resistivity_min_pa_s_m2, ranges.sample_flow_resistivity_max_pa_s_m2
    issues = []
    for wall, material in wall_materials(document).items():
        for index, layer in enumerate(_layers(material)):
            if not isinstance(layer, dict) or layer.get("kind") != "porous":
                continue
            model = layer.get("model", MATERIAL_MODELS["porous"].model_fields["model"].default)
            sigma = layer.get("flow_resistivity_pa_s_m2")
            if model == "komatsu" and isinstance(sigma, int | float) and not low <= sigma <= high:
                branch = "measured_impedance.base_assembly" if material.get("kind") == "measured_impedance" else "assembly"
                issues.append(SchemeProblem(f"scene.{ZONE_WALL_FIELD}.{wall}.0.material.{branch}.layers.{index}.porous.flow_resistivity_pa_s_m2",
                    f"小松纖維模型的流阻可用範圍為 {low:g}–{high:g} Pa·s/m²；手動下限是工程界線，上限沿用原研究樣品值，請改流阻或選合適模型"))
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
        points = material.get("points")
        if isinstance(points, list):
            for index, point in enumerate(points):
                if isinstance(point, dict):
                    values.update({f"point-{wall}-{index}-{name}": value for name, value in point.items()
                                   if isinstance(value, int | float) or value is None})
        for index, layer in enumerate(_layers(material)):
            if not isinstance(layer, dict):
                continue
            for name, value in layer.items():
                if isinstance(value, int | float) or value is None:
                    values[f"layer-{wall}-{index}-{name}"] = value
    return values


def web_scheme_problems(document: object, *, capabilities: CapabilityTable,
                        directivity: DirectivityDefaults, check_material_range: bool = True) -> tuple[SchemeProblem, ...]:
    """網頁驗證唯一入口：移除模型上選填的 null；必填與未知鍵原樣交正式驗證。

    只處理網頁傳來的層，包含實測底構造；不改方案檔格式的拒收契約。
    """
    if isinstance(document, dict):
        for material in wall_materials(document).values():
            for layer in _layers(material):
                if not isinstance(layer, dict):
                    continue
                model = MATERIAL_MODELS.get(str(layer.get("kind")))
                if model is None:
                    continue
                for name, field in model.model_fields.items():
                    if name in layer and layer[name] is None and not field.is_required():
                        layer.pop(name)
    problems = validate_scheme(document, capabilities=capabilities, directivity=directivity)
    return problems + material_input_problems(document, problems) if check_material_range else problems


def material_open_area_texts(document: dict[str, object]) -> dict[str, str]:
    """只在幾何填法驗證完成後呼叫唯一換算；直接給開孔率的層不回傳提示。"""
    texts = {}
    for wall, material in wall_materials(document).items():
        for index, layer in enumerate(_layers(material)):
            if not isinstance(layer, dict) or layer.get("open_area") is not None:
                continue
            kind = layer.get("kind")
            if kind in ("micro_perforated", "perforated"):
                area = layer_open_area(cast(float, layer["hole_diameter_m"]), cast(float, layer["hole_spacing_m"]),
                    str(layer["hole_pattern"]), cast(float | None, layer.get("hole_spacing_2_m")))
            elif kind == "slotted":
                area = layer_open_area(cast(float, layer["slot_width_m"]), cast(float, layer["slot_spacing_m"]), "slotted")
            else:
                continue
            texts[f"open-area-{wall}-{index}"] = f"算出的開孔率約 {area:.2%}"
    return texts
