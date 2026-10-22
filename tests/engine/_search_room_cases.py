"""房間候選的合成輸入；不讀工作區、不啟動計算。"""
from aosr.reporting.scheme import Scheme
from aosr.search.layout_settings import Cabinet, LayoutSettings, Span


def project() -> Scheme:
    return Scheme.model_validate({
        "schema_version": "aosr.scheme.v1", "scheme_id": "synthetic", "purpose": "listening",
        "source_model": "product_default",
        "scene": {"room_m": {"Lx": 6.0, "Ly": 4.0, "Lz": 3.0}, "sound_speed_m_s": 343.0,
                  "density_kg_m3": 1.2, "impedance_pa_s_per_m_by_wall": {
                      wall: 1646.4 for wall in ("floor", "ceiling", "x0", "xL", "y0", "yL")}},
        "speakers": {"left": {"x": 1.0, "y": 1.5, "z": 1.25}, "right": {"x": 1.0, "y": 2.5, "z": 1.25}},
        "channel_group": {"channels": [{"role": "left", "speaker_id": "left"},
                                       {"role": "right", "speaker_id": "right"}],
                          "comparisons": [{"left_role": "left", "right_role": "right"}],
                          "feature_match_tolerance_hz": 10.0},
        "receiver_set": {"points": [{"receiver_id": "main", "position_m": [3.0, 2.0, 1.25],
                                     "role": "primary", "importance": 1.0},
                                    {"receiver_id": "near", "position_m": [3.125, 2.0, 1.25],
                                     "role": "surrounding", "importance": 1.0,
                                     "direction_relative_to_primary": "back"}]},
    })


def settings(**fields: object) -> LayoutSettings:
    original = LayoutSettings(
        front_wall="x0", speaker_height_m=1.25, ear_height_m=1.25,
        front_distance_m=Span(low=0.5, high=1.5), spacing_m=Span(low=0.5, high=2.5),
        listening_distance_m=Span(low=0.5, high=2.5),
        cabinet=Cabinet(width_m=0.25, depth_m=0.25, height_m=0.5),
    )
    return LayoutSettings.model_validate(original.model_dump() | fields)


def search_document(layout: LayoutSettings) -> dict[str, object]:
    return {"purpose": "listening", "layout": layout, "seed": 0, "n_startup_trials": 2,
            "batch_size": 3, "max_workers": 1, "budget": 11, "convergence_run": 5}


def header_document() -> dict[str, object]:
    return {"ledger_version": "aosr.search_ledger.v1", "search_id": "synthetic",
            "settings_fingerprint": "settings", "project_fingerprint": "project",
            "sampler": {"seed": 19, "n_startup_trials": 2, "constant_liar": True},
            "batch_size": 3, "search_space": {name: (0.0, 1.0) for name in
                ("front_distance", "spacing", "listening_distance")},
            "physics_identity": "physical", "program_fingerprint": "program", "purpose_fingerprint": "purpose"}
