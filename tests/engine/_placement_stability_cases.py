"""純函式考卷的手寫小房間；不讀案例檔、不開搜尋資料夾。"""
from typing import Literal

from aosr.reporting.scheme import Scheme
from aosr.search.layout_settings import Cabinet, LayoutSettings, Span
from aosr.search.settings import SearchSettings


def scheme(facing: Literal["west", "north"] = "west", mount: str = "stand", *, furniture: bool = True) -> Scheme:
    speakers = ((1.0, 2.0), (1.0, 4.0)) if facing == "west" else ((2.0, 5.0), (4.0, 5.0))
    height = {"stand": 1.2, "desk": 0.935, "floor": 0.8}[mount]
    cabinet = dict(width_m=0.21, depth_m=0.28, height_m=1.105 if mount == "floor" else 0.355,
                   acoustic_center_behind_front_m=0.0, acoustic_center_above_bottom_m=0.8 if mount == "floor" else 0.205)
    items = [
        dict(furniture_id="table", kind="desk", material="wood", width_m=4.0, depth_m=3.0, height_m=0.03,
             placement=dict(forward_m=1.0, left_m=0.0, bottom_height_m=0.70, yaw_deg=0)),
        dict(furniture_id="cloud", kind="ceiling_cloud", material="wood", width_m=2.0, depth_m=2.0, height_m=0.1,
             placement=dict(bottom_center_m=(3.0, 3.0, 2.5), yaw_deg=0)),
    ] if furniture else None
    return Scheme.model_validate(dict(
        schema_version="aosr.scheme.v1", scheme_id=f"hand-{facing}-{mount}", purpose="dedicated_two_channel_listening_room",
        scene=dict(room_m=dict(Lx=8.0, Ly=8.0, Lz=4.0), sound_speed_m_s=343.0, density_kg_m3=1.2,
                   impedance_pa_s_per_m_by_wall=dict.fromkeys(("x0", "xL", "y0", "yL", "floor", "ceiling"), 1646.4),
                   low_frequency_axis="verification_linear_1hz"),
        source_model="omnidirectional", speakers={key: dict(x=x, y=y, z=height)
            for key, (x, y) in zip(("left", "right"), speakers, strict=True)},
        channel_group=dict(channels=[dict(role=key, speaker_id=key) for key in ("left", "right")],
                           comparisons=[dict(left_role="left", right_role="right")], feature_match_tolerance_hz=10.0),
        receiver_set=dict(points=[
            dict(receiver_id="main", role="primary", position_m=(3.0, 3.0, 1.2), importance=1.0),
            *[dict(receiver_id=key, role="surrounding", position_m=position, importance=1.0,
                   direction_relative_to_primary=key) for key, position in (
                       ("left", (3.1, 3.0, 1.2)), ("up", (3.0, 3.0, 1.3)), ("down", (3.0, 3.0, 1.1)))]]),
        furniture=items, speaker_setup=dict(kind="floorstanding" if mount == "floor" else "bookshelf",
                                            mount=mount, representative=True, cabinet=cabinet)))


def settings(project: Scheme, *, locked: bool = False) -> SearchSettings:
    setup = project.speaker_setup
    assert setup is not None
    return SearchSettings(purpose=project.purpose, seed=7, n_startup_trials=4, batch_size=4, max_workers=2,
        budget=12, convergence_run=4, layout=LayoutSettings(
            front_wall="x0" if "west" in project.scheme_id else "yL", speaker_height_m=project.speakers["left"].z,
            ear_height_m=1.2, front_distance_m=Span(low=0.2, high=6.0), spacing_m=Span(low=0.2, high=4.0),
            listening_distance_m=None if locked else Span(low=0.2, high=4.0), seat_locked=locked,
            cabinet=Cabinet.model_validate(setup.cabinet.model_dump())))
