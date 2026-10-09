"""圖面考卷的寫入樣本；幾何答案另由各題手算。"""
from tests.engine._furniture_cases import reference_document

CABINET = {"width_m": 0.21, "depth_m": 0.28, "height_m": 0.355,
           "acoustic_center_behind_front_m": 0.0, "acoustic_center_above_bottom_m": 0.205}


def listening_document() -> dict[str, object]:
    document = reference_document()
    document["scheme_id"] = "listen"
    document["furniture"] = [
        {"furniture_id": "sofa", "kind": "sofa", "material": "fabric", "width_m": 1.8,
         "depth_m": 0.75, "height_m": 0.65,
         "placement": {"forward_m": -0.2, "left_m": 0.0, "bottom_height_m": 0.0, "yaw_deg": 0}},
        {"furniture_id": "table", "kind": "coffee_table", "material": "wood", "width_m": 1.0,
         "depth_m": 0.5, "height_m": 0.04,
         "placement": {"forward_m": 0.9, "left_m": 0.0, "bottom_height_m": 0.4, "yaw_deg": 0}},
        {"furniture_id": "cloud", "kind": "ceiling_cloud", "material": "absorptive_cloud",
         "width_m": 1.2, "depth_m": 1.2, "height_m": 0.1,
         "placement": {"bottom_center_m": [2.1, 1.9, 2.55], "yaw_deg": 0}},
    ]
    document["speaker_setup"] = {"kind": "bookshelf", "mount": "stand", "cabinet": CABINET,
                                 "representative": True}
    return document


def working_document() -> dict[str, object]:
    document = listening_document()
    document["scheme_id"] = "work"
    document["speakers"] = {"left": {"x": 1.7, "y": 1.5, "z": 0.955},
                             "right": {"x": 1.7, "y": 2.3, "z": 0.955}}
    document["receiver_set"] = {"points": [
        {"receiver_id": "main", "role": "primary", "position_m": [2.4, 1.9, 1.2], "importance": 1.0},
        {"receiver_id": "side", "role": "surrounding", "position_m": [2.4, 2.0, 1.2],
         "importance": 1.0, "direction_relative_to_primary": "left"},
    ]}
    document["furniture"] = [
        {"furniture_id": "desk", "kind": "desk", "material": "wood", "width_m": 1.6,
         "depth_m": 0.9, "height_m": 0.03,
         "placement": {"forward_m": 0.6, "left_m": 0.0, "bottom_height_m": 0.72, "yaw_deg": 0}},
        {"furniture_id": "chair", "kind": "chair", "material": "fabric", "width_m": 0.55,
         "depth_m": 0.55, "height_m": 0.48,
         "placement": {"forward_m": -0.1, "left_m": 0.0, "bottom_height_m": 0.0, "yaw_deg": 0}},
    ]
    document["speaker_setup"] = {"kind": "bookshelf", "mount": "desk", "cabinet": CABINET,
                                 "representative": True}
    return document
