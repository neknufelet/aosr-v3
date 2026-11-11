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


def rectangle_scene(*, only_wall: str | None = None) -> dict[str, object]:
    """6×4×3 公尺房間：五面各 2×2，地板整面；只指定一面時其餘也整面。"""
    scene = zone_scene()
    zones = cast(dict[str, list[dict[str, object]]], scene[NEW])
    sizes = {"ceiling": (6.0, 4.0), "x0": (4.0, 3.0), "xL": (4.0, 3.0),
             "y0": (6.0, 3.0), "yL": (6.0, 3.0)}
    for name, (u, v) in sizes.items():
        if only_wall is not None and name != only_wall:
            continue
        material = deepcopy(zones[name][0]["material"])
        zones[name] = [{"area": "rectangle", "id": f"r{row}_c{col}",
                        "u_m": [col * u / 2, (col + 1) * u / 2],
                        "v_m": [row * v / 2, (row + 1) * v / 2], "material": deepcopy(material)}
                       for row in range(2) for col in range(2)]
    return scene


def rectangle_document() -> dict[str, object]:
    return reference_document() | {"scene": rectangle_scene()}


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
    elif case == "null-old":
        scene.pop(NEW)
        scene[OLD] = None
        hint = "恰好"
    elif case == "null-new":
        scene[NEW] = None
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
        hint = "每面只收整面一區，或一組蓋滿整面、互不重疊的矩形分區"
    elif case == "area":
        zones["floor"][0]["area"] = "rectangle"
        path += ".floor.0.id"
        hint = "必填"
    elif case == "measured":
        zones["floor"][0]["material"] = {"kind": "measured_impedance", "arbitrary": True}
        path += ".floor.0.material.measured_impedance.points"
        hint = "必填"
    else:
        zones["floor"][0]["material"] = {"kind": "assembly", "layers": [
            {"kind": "air", "thickness_m": 0.0}], "backing": "rigid"}
        path += ".floor.0.material.assembly.layers.0.air.thickness_m"
        hint = "要大於 0"
    return scene, path, hint


# 109 號房的長寬高（#562 示範方案的 room_m 原值）：牆寬不在 5 公分格上，畫框貼牆邊的浮點問題要在這種尺寸上考。
ROOM_109: dict[str, float] = {"Lx": 6.98792656080558, "Ly": 4.41988145469311, "Lz": 3.1554034223396052}


def room109_document(scheme_id: str = "room109") -> dict[str, object]:
    """參考方案換成 109 號房的長寬高，六面整面一區；喇叭與座位都在房內（參考房比 109 小）。"""
    document = zone_document() | {"scheme_id": scheme_id}
    cast(dict[str, object], document["scene"])["room_m"] = dict(ROOM_109)
    return document


def room109_quarters(document: dict[str, object], wall: str, impedances: tuple[float, float, float, float]) -> None:
    """把一面牆切成 109 示範用的 2×2（代號 u0v0…，順序同 #562 方案），四區各給一個固定阻抗。"""
    sizes = {"x0": ("Ly", "Lz"), "xL": ("Ly", "Lz"), "y0": ("Lx", "Lz"), "yL": ("Lx", "Lz"),
             "floor": ("Lx", "Ly"), "ceiling": ("Lx", "Ly")}
    room = cast(dict[str, float], cast(dict[str, object], document["scene"])["room_m"])
    width, height = ROOM_109[sizes[wall][0]], ROOM_109[sizes[wall][1]]
    assert room == ROOM_109
    zones = cast(dict[str, list[dict[str, object]]], cast(dict[str, object], document["scene"])[NEW])
    zones[wall] = [{"area": "rectangle", "id": f"u{col}v{row}",
                    "u_m": [col * width / 2, width if col else width / 2],
                    "v_m": [row * height / 2, height if row else height / 2],
                    "material": {"kind": "constant_impedance", "impedance_pa_s_m": impedances[2 * col + row]}}
                   for col in range(2) for row in range(2)]
