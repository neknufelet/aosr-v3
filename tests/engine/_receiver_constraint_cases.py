"""座位硬限制的手寫題目；方案與座標不讀真實專案。"""
from aosr.reporting.scheme import Scheme
from aosr.search.layout_settings import Cabinet, LayoutSettings, Span


def project() -> Scheme:
    return Scheme.model_validate(dict(
        schema_version="aosr.scheme.v1", scheme_id="receiver-control", source_model="omnidirectional",
        purpose="dedicated_two_channel_listening_room",
        scene=dict(room_m=dict(Lx=8.0, Ly=8.0, Lz=4.0), sound_speed_m_s=343.0, density_kg_m3=1.2,
            impedance_pa_s_per_m_by_wall=dict.fromkeys(("x0", "xL", "y0", "yL", "floor", "ceiling"), 1646.4)),
        speakers=dict(left=dict(x=2.0, y=2.0, z=1.25), right=dict(x=2.0, y=4.0, z=1.25)),
        channel_group=dict(channels=[dict(role=key, speaker_id=key) for key in ("left", "right")],
            comparisons=[dict(left_role="left", right_role="right")], feature_match_tolerance_hz=10.0),
        receiver_set=dict(points=[
            dict(receiver_id="main", role="primary", position_m=(4.0, 2.0, 1.25), importance=1.0),
            dict(receiver_id="front", role="surrounding", position_m=(3.9, 2.0, 1.25), importance=1.0,
                 direction_relative_to_primary="front")])))


def settings(**changes: object) -> LayoutSettings:
    base = LayoutSettings(front_wall="x0", speaker_height_m=1.25, ear_height_m=1.25,
        front_distance_m=Span(low=0.5, high=3.0), spacing_m=Span(low=0.2, high=3.0),
        listening_distance_m=Span(low=0.01, high=3.0),
        cabinet=Cabinet(width_m=0.5, depth_m=0.5, height_m=0.5, acoustic_center_behind_front_m=0.125))
    return LayoutSettings.model_validate(base.model_dump() | changes)
