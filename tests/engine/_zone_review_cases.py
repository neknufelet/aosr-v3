"""第一批審查的獨立壞資料與三入口操作；所有落盤只用考卷暫存目錄。"""
from copy import deepcopy
from pathlib import Path
from typing import cast

from aosr.physics import report_io
from aosr.reporting.scheme import Scene, Scheme
from aosr.reporting.validation import validated_scheme
from tests.engine._directivity import DIRECTIVITY
from tests.engine._furniture_cases import CAPABILITIES, pair, reference_document
from tests.engine._surface_zone_cases import NEW, OLD, invalid_scene, zone_scene
from tests.engine.test_scheme_furniture import _store

ORIGINAL_CASES = ("both", "neither", "null-old", "null-new", "missing", "extra", "two", "area", "measured", "layer")
MATERIAL_CASES: dict[str, tuple[str, object, str]] = {
    "missing-area": ("area", None, "必填"),
    "null-thickness": ("thickness_m", None, "必填"),
    "number": ("thickness_m", "厚", "數字"),
    "finite": ("thickness_m", float("inf"), "有限"),
    "empty-layers": ("layers", [], "至少"),
    "missing-material-kind": ("kind", None, "種類"),
    "unknown-material-kind": ("kind", "unknown", "種類"),
    "missing-layer-kind": ("kind", None, "種類"),
    "unknown-layer-kind": ("kind", "unknown", "種類"),
    "backing": ("backing", "wall", "只能"),
    "model": ("model", "unknown", "只能"),
    "material-class": ("material_class", "wood", "只能"),
    "komatsu-porosity": ("porosity", .98, "不讀"),
    "miki-tortuosity": ("tortuosity", 1.1, "不讀"),
    "wilson-missing": ("porosity", None, "必須"),
    "mechel-missing": ("porosity", None, "必須"),
    "mechel-large": ("porosity", 1.2, "孔隙率"),
}


def porous(model: str = "komatsu") -> dict[str, object]:
    return {"kind": "porous", "thickness_m": .1, "flow_resistivity_pa_s_m2": 10000.,
            "model": model, "material_class": "fibrous"}


def review_scene(case: str) -> tuple[dict[str, object], str, str]:
    if case in ORIGINAL_CASES:
        return invalid_scene(case)
    scene = zone_scene(mixed=True)
    zones = cast(dict[str, list[dict[str, object]]], scene[NEW])
    zone = zones["floor"][0]
    material = cast(dict[str, object], zone["material"])
    layers = cast(list[dict[str, object]], material["layers"])
    field, value, hint = MATERIAL_CASES[case]
    target = layers[1]
    path = f"{NEW}.floor.0.material.assembly.layers.1.air"
    if case == "missing-area":
        target, path = zone, f"{NEW}.floor.0"
    elif case in ("empty-layers", "missing-material-kind", "unknown-material-kind", "backing"):
        target, path = material, f"{NEW}.floor.0.material.assembly"
    elif case in ("model", "material-class") or case.startswith(("komatsu", "miki", "wilson", "mechel")):
        model = {"miki": "miki", "wilson": "wilson_2015", "mechel": "mechel_1976"}.get(case.split("-")[0], "komatsu")
        target = porous(model)
        layers[1] = target
        path = f"{NEW}.floor.0.material.assembly.layers.1.porous"
    if value is None and case != "null-thickness":
        target.pop(field, None)
    else:
        target[field] = value
    if "kind" in case or case.startswith(("komatsu", "miki", "wilson", "mechel")):
        path = path.removesuffix(".assembly").removesuffix(".air").removesuffix(".porous")
    else:
        path += f".{field}"
    return scene, path, hint


def report_document(scene: dict[str, object]) -> dict[str, object]:
    document = pair(Scheme.model_validate(reference_document()))
    for field in Scene.model_fields:
        document.pop(field, None)
    return document | scene


def reject_at(entry: str, scene: dict[str, object], tmp_path: Path) -> None:
    if entry == "scheme":
        validated_scheme(reference_document() | {"scene": scene})
    elif entry == "report":
        report_io.load_input_document(report_document(scene), CAPABILITIES, DIRECTIVITY)
    else:
        project = Scheme.model_validate(reference_document()).model_copy(update={
            "scene": Scene.model_construct(_fields_set=set(scene), **deepcopy(scene))})
        _store(tmp_path / "search", project)


def reverse_walls(scene: dict[str, object], field: str) -> dict[str, object]:
    walls = cast(dict[str, object], scene[field])
    return scene | {field: dict(reversed(tuple(walls.items())))}
