"""第八支第三步之二：照老闆施工單手寫的答案，不讀產品預設。"""

FURNITURE: dict[str, dict[str, object]] = {
    "sofa": {"furniture_id": "sofa", "kind": "sofa", "material": "fabric",
             "width_m": 1.8, "depth_m": 0.75, "height_m": 0.65,
             "placement": {"forward_m": 0.075, "left_m": 0.0, "bottom_height_m": 0.0, "yaw_deg": 0}},
    "coffee_table": {"furniture_id": "table", "kind": "coffee_table", "material": "wood",
                     "width_m": 1.2, "depth_m": 0.6, "height_m": 0.03,
                     "placement": {"forward_m": 1.15, "left_m": 0.0, "bottom_height_m": 0.47, "yaw_deg": 0}},
    "desk": {"furniture_id": "table", "kind": "desk", "material": "wood",
             "width_m": 1.4, "depth_m": 0.75, "height_m": 0.03,
             "placement": {"forward_m": 0.825, "left_m": 0.0, "bottom_height_m": 0.72, "yaw_deg": 0}},
    "ceiling_cloud": {"furniture_id": "cloud", "kind": "ceiling_cloud", "material": "absorptive_cloud",
                      "width_m": 1.35, "depth_m": 1.8, "height_m": 0.05,
                      "placement": {"bottom_center_m": [3.0, 4.0, 3.85], "yaw_deg": 0}},
}
CABINETS: dict[str, dict[str, float]] = {
    "bookshelf": {"width_m": 0.21, "depth_m": 0.28, "height_m": 0.355,
                  "acoustic_center_behind_front_m": 0.0, "acoustic_center_above_bottom_m": 0.205},
    "floorstanding": {"width_m": 0.21, "depth_m": 0.38, "height_m": 1.105,
                      "acoustic_center_behind_front_m": 0.0, "acoustic_center_above_bottom_m": 0.8},
}
