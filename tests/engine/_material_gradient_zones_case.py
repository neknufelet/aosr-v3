"""小房間分區搜尋案例；網格與分區沿用 _zone_rom_case，材料各區不同。"""
from __future__ import annotations

from aosr.reporting.scheme import Scheme
from tests.engine._zone_rom_case import PARTITIONS, ROOM
from tests.engine.test_material_gradient import _receivers
from tests.engine.test_scheme_model import _document


def scheme() -> Scheme:
    document = _document()
    scene = document["scene"]
    assert isinstance(scene, dict)
    scene = {key: value for key, value in scene.items() if key != "impedance_pa_s_per_m_by_wall"}
    scene["room_m"] = {"Lx": ROOM.Lx, "Ly": ROOM.Ly, "Lz": ROOM.Lz}
    scene["low_frequency_axis"] = "search_octave_24"
    zones: dict[str, list[dict[str, object]]] = {name: [{"area": "whole_wall", "material": {"kind": "constant_impedance", "impedance_pa_s_m": 2100.0}}]
             for name in ("floor", "ceiling", "x0", "xL", "y0", "yL")}
    zones["x0"] = [{"area": "rectangle", "id": name, "u_m": list(u), "v_m": list(v),
                     "material": {"kind": "constant_impedance", "impedance_pa_s_m": 3200.0 + 1100.0 * index}}
                    for index, (name, u, v) in enumerate(next(iter(PARTITIONS.values())))]
    scene["surface_zones_by_wall"] = zones
    receivers = _receivers().model_dump(mode="json")
    for index, point in enumerate(receivers["points"]):
        point["position_m"] = [1.0 + (index % 3) * 0.1, 0.6 + (index // 3) * 0.1, 0.5]
    return Scheme.model_validate({**document, "scene": scene,
        "speakers": {"left": {"x": 0.3, "y": 0.3, "z": 0.4}, "right": {"x": 0.3, "y": 0.9, "z": 0.4}}, "receiver_set": receivers})
