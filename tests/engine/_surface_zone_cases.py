"""整面分區的輸入案例；固定值來自參考方案，不呼叫材料求值核心。"""
from copy import deepcopy
from typing import cast

from aosr.reporting.scheme import Scheme
from tests.engine._furniture_cases import reference_document

OLD = "impedance_pa_s_per_m_by_wall"
NEW = "surface_zones_by_wall"
ASSEMBLY: dict[str, object] = {"kind": "assembly", "layers": [
    {"kind": "board", "surface_mass_kg_m2": 0.15, "resistance_pa_s_m": 80.0},
    {"kind": "air", "thickness_m": 0.43}], "backing": "rigid"}


def zone_scene(*, mixed: bool = False) -> dict[str, object]:
    scene = cast(dict[str, object], deepcopy(reference_document()["scene"]))
    walls = cast(dict[str, float], scene.pop(OLD))
    zones = {name: [{"area": "whole_wall", "material": {
        "kind": "constant_impedance", "impedance_pa_s_m": value}}] for name, value in walls.items()}
    if mixed:
        zones["floor"] = [{"area": "whole_wall", "material": deepcopy(ASSEMBLY)}]
    scene[NEW] = zones
    return scene


def zone_document(*, mixed: bool = False) -> dict[str, object]:
    return reference_document() | {"scene": zone_scene(mixed=mixed)}


def mixed_scheme() -> Scheme:
    return Scheme.model_validate(zone_document(mixed=True))


def invalid_scene(case: str) -> tuple[dict[str, object], str, str]:
    scene = zone_scene(mixed=True)
    zones = cast(dict[str, list[dict[str, object]]], scene[NEW])
    path = NEW
    if case == "both":
        scene[OLD] = cast(dict[str, object], reference_document()["scene"])[OLD]
        hint = "恰好"
    elif case == "neither":
        scene.pop(NEW)
        hint = "恰好"
    elif case == "missing":
        zones.pop("ceiling")
        path += ".ceiling"
        hint = "缺少"
    elif case == "extra":
        zones["ceilling"] = deepcopy(zones["ceiling"])
        path += ".ceilling"
        hint = "不認識"
    elif case == "two":
        zones["floor"].append(deepcopy(zones["floor"][0]))
        path += ".floor"
        hint = "這一版每面只收整面一區；矩形分區是 #309 第二步"
    elif case == "area":
        zones["floor"][0]["area"] = "rectangle"
        path += ".floor.0.area"
        hint = "這一版每面只收整面一區；矩形分區是 #309 第二步"
    elif case == "measured":
        zones["floor"][0]["material"] = {"kind": "measured_impedance", "arbitrary": True}
        path += ".floor.0.material"
        hint = "實測阻抗表這一版還不收"
    else:
        zones["floor"][0]["material"] = {"kind": "assembly", "layers": [
            {"kind": "air", "thickness_m": 0.0}], "backing": "rigid"}
        path += ".floor.0.material.assembly.layers.0.air.thickness_m"
        hint = "要大於 0"
    return scene, path, hint
