"""家具考卷的手定擺法；答案由各題明寫，不從換算器取。"""
import json
from pathlib import Path

from aosr.config.capabilities import load_capabilities
from aosr.config.paths import config_path
from aosr.geometry.shoebox import Point
from aosr.physics import report_io
from aosr.reporting.scheme import Scheme, pair_input_document
from tests.engine._directivity import DIRECTIVITY


REFERENCE = Path(__file__).resolve().parents[2] / "blueprint" / "scheme_reference_room.json"
CAPABILITIES = load_capabilities(config_path("capabilities.toml"))


def relative_item(**changes: object) -> dict[str, object]:
    return {"furniture_id": "seat", "kind": "sofa", "material": "fabric",
            "width_m": 1.0, "depth_m": 0.5, "height_m": 0.6,
            "placement": {"forward_m": 1.0, "left_m": 0.5, "bottom_height_m": 0.0, "yaw_deg": 0},
            **changes}


def cloud_item(**changes: object) -> dict[str, object]:
    return {"furniture_id": "cloud", "kind": "ceiling_cloud", "material": "wood",
            "width_m": 1.0, "depth_m": 0.5, "height_m": 0.1,
            "placement": {"bottom_center_m": [2.0, 2.0, 2.5], "yaw_deg": 90}, **changes}


def reference_document() -> dict[str, object]:
    document: dict[str, object] = json.loads(REFERENCE.read_text(encoding="utf-8"))
    return document


def document(*items: dict[str, object], facing: str = "north",
             primary: tuple[float, float, float] = (3.0, 3.0, 1.2)) -> dict[str, object]:
    content = reference_document()
    scene = content["scene"]
    assert isinstance(scene, dict)
    content["scene"] = scene | {"room_m": {"Lx": 8.0, "Ly": 8.0, "Lz": 4.0}}
    speakers = {
        "north": ((2.0, 5.0, 1.2), (4.0, 5.0, 1.2)),
        "west": ((1.0, 2.0, 1.2), (1.0, 4.0, 1.2)),
        "south": ((4.0, 1.0, 1.2), (2.0, 1.0, 1.2)),
        "east": ((5.0, 4.0, 1.2), (5.0, 2.0, 1.2)),
    }[facing]
    content["speakers"] = {name: dict(zip(("x", "y", "z"), point, strict=True))
                           for name, point in zip(("left", "right"), speakers, strict=True)}
    content["receiver_set"] = {"points": [
        {"receiver_id": "main", "role": "primary", "position_m": primary, "importance": 1.0},
        {"receiver_id": "side", "role": "surrounding", "position_m": (3.1, 3.0, 1.2),
         "importance": 1.0, "direction_relative_to_primary": "left"},
    ]}
    content["furniture"] = list(items)
    return content


def pair(scheme: Scheme) -> dict[str, object]:
    return pair_input_document(scheme, scheme.speakers["left"],
                               Point(*scheme.receiver_set.primary.position_m), {"kind": "omnidirectional"})


def fingerprint(scheme: Scheme) -> str:
    return report_io.scene_fingerprint(report_io.load_input_document(pair(scheme), CAPABILITIES, DIRECTIVITY))
