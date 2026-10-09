"""家具反射集合與桌緣距離的手算題；牆面列變化不得冒充模型事件。"""
import pytest

from aosr.config.paths import config_path
from aosr.config.quality_targets import load_quality_targets
from aosr.physics.report_path_output import PathRow, PathTableSection
from aosr.reporting.scheme import Scheme
from aosr.scoring.direction_zones import zone_limits
from aosr.search.placement_stability_events import baseline_boundary_distances, furniture_events
from tests.engine._placement_stability_cases import scheme
from tests.engine._precision_contracts import contract_value


def path(identifier: str | None, face: str = "top", *, elevation: float = -25.0,
         point: tuple[float, float, float] = (0.75, 1.4, 0.73)) -> PathRow:
    return PathRow.model_validate(dict(order=1, wall_sequence=("furniture",) if identifier else ("x0",),
        delay_s=0.01, distance_m=3.43, direction_vector=(-1.0, 0.0, -1.0),
        direction_angles=dict(azimuth_deg=0.0, elevation_deg=elevation), departure_off_axis_deg=None,
        relative_direct_energy=(0.1,), **(dict(furniture_id=identifier, furniture_face=face,
                                            reflection_point_m=point) if identifier else {})))


def document(base: Scheme, pairs: dict[tuple[str, str], tuple[PathRow, ...]]) -> dict[str, object]:
    result = []
    for (speaker, receiver), rows in pairs.items():
        ids = sorted({r.furniture_id for r in rows if r.furniture_id is not None})
        metadata = dict(furniture_ids=ids, furniture_model="single_bounce_finite_size_v1", blocked_wall_paths=(),
                        furniture_materials=[dict(furniture_id=i, material="wood", unknown_bands_hz=()) for i in ids]) if ids else {}
        table = PathTableSection.model_validate(dict(reflection_order_k=1, frequencies_hz=(100.0,),
            scattering_coefficient=(0.0,), source_model_kind="omnidirectional", rows=rows, **metadata))
        result.append(dict(speaker_id=speaker, receiver_id=receiver, report=dict(path_table=table.model_dump())))
    return dict(scheme=base.model_dump(), pairs=result)


def test_furniture_appears_disappears_primary_marked_and_wall_changes_ignored() -> None:
    base = scheme()
    before = document(base, {("left", "main"): (path("table"), path(None)),
        ("right", "main"): (path("cloud", "bottom"),), ("left", "up"): (), ("right", "up"): (path(None),)})
    after = document(base, {("right", "up"): (), ("left", "up"): (path("table"),),
        ("right", "main"): (path("cloud", "bottom"),), ("left", "main"): (path("cloud", "bottom"),)})
    events = furniture_events(before, after)
    assert {(e.speaker_id, e.receiver_id, e.furniture_id, e.face, e.change, e.is_primary) for e in events} == {
        ("left", "main", "table", "top", "disappeared", True),
        ("left", "main", "cloud", "bottom", "appeared", True),
        ("left", "up", "table", "top", "appeared", False)}
    assert furniture_events(before, before) == ()
    walls_before = document(base, {("left", "main"): (path(None),)})
    walls_after = document(base, {("left", "main"): ()})
    assert furniture_events(walls_before, walls_after) == ()


def test_primary_identity_comes_from_scheme_and_face_changes_are_two_events() -> None:
    base = scheme()
    points = tuple(p.model_copy(update={"receiver_id": "focus"}) if p.role == "primary" else p for p in base.receiver_set.points)
    base = base.model_copy(update={"receiver_set": base.receiver_set.model_copy(update={"points": points})})
    before = document(base, {("left", "focus"): (path("table"),)})
    after = document(base, {("left", "focus"): (path("table", "-x", point=(0.5, 1.4, 0.71)),)})
    events = furniture_events(before, after)
    assert {(e.face, e.change, e.is_primary) for e in events} == {("top", "disappeared", True), ("-x", "appeared", True)}


def test_boundary_distance_to_rotated_table_edges_and_vertical_registry_limit() -> None:
    base = scheme()
    purpose = load_quality_targets(config_path("quality_targets.toml")).purpose(base.purpose)
    limit = zone_limits(purpose)[0].vertical_min_abs_elevation_deg
    result = document(base, {("left", "main"): (path("table", elevation=-limit + 2.0), path(None)),
        ("right", "main"): (path("table", "-x", elevation=limit + 3.0, point=(0.5, 1.4, 0.71)),),
        ("left", "up"): (path("table"),)})
    distances = baseline_boundary_distances(result, purpose=purpose, contact_rel=contract_value("furniture_geometry_contact"))
    values = {(d.speaker_id, d.face): d for d in distances}
    assert set(values) == {("left", "top"), ("right", "-x")}
    # 90° 桌子：x 在 0.5 到 3.5，y 在 1 到 5；頂面點 x=.75，最近邊距 .25。
    assert values["left", "top"].edge_distance_m == pytest.approx(0.25)
    assert values["left", "top"].vertical_boundary_distance_deg == 2.0
    # -x 側面看 y,z；板底 .70，點 .71，最近邊 .01。
    assert values["right", "-x"].edge_distance_m == pytest.approx(0.01)
    assert values["right", "-x"].vertical_boundary_distance_deg == 3.0
    # 改傳入登記簿的界線，排除產品偷偷寫死角度。
    changed = purpose.model_copy(update={"setting": tuple(
        entry.model_copy(update={"value": limit + 7.0}) if entry.key == "direction_zones.vertical_min_abs_elevation_deg"
        else entry for entry in purpose.setting)})
    other = baseline_boundary_distances(result, purpose=changed, contact_rel=contract_value("furniture_geometry_contact"))
    assert {(d.speaker_id, d.vertical_boundary_distance_deg) for d in other} == {("left", 9.0), ("right", 4.0)}


def test_missing_path_table_is_unavailable_not_an_empty_furniture_set() -> None:
    base = scheme()
    missing: dict[str, object] = dict(scheme=base.model_dump(), pairs=[dict(speaker_id="left", receiver_id="main", report={})])
    with pytest.raises(ValueError):
        furniture_events(missing, document(base, {("left", "main"): ()}))
