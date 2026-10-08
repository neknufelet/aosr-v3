"""#559 第七支第四步手寫輸入；尺寸與高度答案照施工單，不讀代表資料。"""
from typing import cast

from tests.engine import _furniture_cases as furniture


def setup(kind: str = "bookshelf", mount: str = "stand", *, representative: bool = True) -> dict[str, object]:
    return {"kind": kind, "mount": mount, "representative": representative,
            "cabinet": {"width_m": 0.21, "depth_m": 0.28, "height_m": 1.105 if kind == "floorstanding" else 0.355,
                        "acoustic_center_behind_front_m": 0.0,
                        "acoustic_center_above_bottom_m": 0.8 if kind == "floorstanding" else 0.205}}


def table(kind: str = "desk", identifier: str = "table") -> dict[str, object]:
    return furniture.relative_item(furniture_id=identifier, kind=kind, material="wood", height_m=0.03,
        placement={"forward_m": 1.0, "left_m": 0.0, "bottom_height_m": 0.70, "yaw_deg": 0})


def document(mount: str = "stand", *, left_z: float | None = None,
             right_z: float | None = None) -> dict[str, object]:
    items = (table(),) if mount == "desk" else ()
    result = furniture.document(*items)
    result["speaker_setup"] = setup("floorstanding" if mount == "floor" else "bookshelf", mount)
    height = {"stand": 1.2, "desk": 0.935, "floor": 0.8}[mount]
    speakers = cast(dict[str, dict[str, object]], result["speakers"])
    speakers["left"]["z"] = height if left_z is None else left_z
    speakers["right"]["z"] = height if right_z is None else right_z
    return result
