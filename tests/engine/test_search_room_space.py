"""房間候選與版本落帳；期待值由合成輸入手算。"""
from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError

from aosr.geometry.shoebox import Point, Room
from aosr.search.constraints import Reason, Violation, check
from aosr.search.labels import COUNT_REASONS
from aosr.search.layout import (
    LOCKED_UNIT_SPACE, UNIT_SPACE, LayoutParams, meter_quantities_for, params_from_unit, params_to_meters,
    place, space_for, standard_start, to_scheme, unit_from_params,
)
from aosr.search.layout_settings import LayoutSettings
from aosr.search.ledger import Ledger, LedgerHeader, LedgerHeaderV2, LedgerRow, LedgerRowV2, row_from_outcome
from aosr.search.sampler import Excluded, Illegal, Outcome, Proposal, RankingZone, Scored
from aosr.search.settings import SearchSettings
from aosr.search.store import candidate_name
from tests.engine._search_room_cases import header_document, project, search_document, settings
from tests.engine._search_store_cases import purpose_settings


PLACEMENT_KEYS = ("front_distance", "spacing", "listening_distance")
ROOM_KEYS = ("room_Lx", "room_Ly", "room_Lz")
ALL_METERS = (*PLACEMENT_KEYS, *ROOM_KEYS)
RANGES = {"Lx": {"low": 4.0, "high": 8.0}, "Ly": {"low": 2.0, "high": 6.0},
          "Lz": {"low": 2.0, "high": 4.0}}
UNIT = {"front_distance": 0.5, "spacing": 0.25, "listening_distance": 0.75}


@pytest.fixture(autouse=True)
def temporary_contact_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """產品讀取接觸界線時只讀本題的合成登記簿；不借工作區的精度檔。"""
    path = tmp_path / "contracts"
    path.write_text('[[contract]]\nname = "furniture_geometry_contact"\n'
                    'value = 9.094947017729282e-13\ndisplay = "2^-40"\nunit = "relative"\n'
                    'decision_paper = "synthetic"\ntruth = "synthetic"\nmutant_test = "synthetic"\n')
    monkeypatch.setattr("aosr.search.constraints.default_precision_contracts_path", lambda: path)
    monkeypatch.setattr("aosr.search.seat_lock.default_precision_contracts_path", lambda: path)


def test_room_settings_accept_declared_range() -> None:
    chosen = settings(room_range={"Lx": RANGES["Lx"]})
    assert chosen.model_dump()["room_range"] == {
        "Lx": RANGES["Lx"], "Ly": None, "Lz": None, "min_floor_area_m2": None}


@pytest.mark.parametrize("locked", [False, True])
def test_unset_room_is_bit_exact(locked: bool) -> None:
    from aosr.search.layout import placement_space_for

    chosen = settings(seat_locked=locked, listening_distance_m=None) if locked else settings()
    expected_space = LOCKED_UNIT_SPACE if locked else UNIT_SPACE
    assert space_for(chosen) is expected_space
    assert placement_space_for(chosen) is expected_space
    unit = {"front_distance": 0.5, "spacing": 0.25} if locked else UNIT
    expected = LayoutParams(1.0, 1.0, 2.0)
    assert params_from_unit(unit, chosen, project=project()) == expected
    assert dict(unit_from_params(expected, chosen)) == unit
    assert params_to_meters(expected) == {"front_distance": 1.0, "spacing": 1.0, "listening_distance": 2.0}
    assert meter_quantities_for(chosen) == PLACEMENT_KEYS
    placed = place(project(), chosen, expected)
    assert check(project(), chosen, placed) == ()
    outside = replace(placed, receivers=(("main", Point(7.0, 2.0, 1.25)),))
    assert check(project(), chosen, outside) == (Violation(Reason.SEAT_OUTSIDE_ROOM, 1.0),)


@pytest.mark.parametrize("edges,keys", [
    (("Lx",), (*PLACEMENT_KEYS, "room_Lx")),
    (("Ly",), (*PLACEMENT_KEYS, "room_Ly")),
    (("Lz",), (*PLACEMENT_KEYS, "room_Lz")),
    (("Lx", "Ly"), (*PLACEMENT_KEYS, "room_Lx", "room_Ly")),
    (("Lx", "Lz"), (*PLACEMENT_KEYS, "room_Lx", "room_Lz")),
    (("Ly", "Lz"), (*PLACEMENT_KEYS, "room_Ly", "room_Lz")),
    (("Lx", "Ly", "Lz"), ALL_METERS),
])
def test_room_space_only_samples_adjustable_edges(edges: tuple[str, ...], keys: tuple[str, ...]) -> None:
    from aosr.search.layout import ROOM_QUANTITIES, placement_space_for

    chosen = settings(room_range={edge: RANGES[edge] for edge in reversed(edges)})
    assert tuple(space_for(chosen)) == keys
    assert dict(space_for(chosen)) == dict.fromkeys(keys, (0.0, 1.0))
    assert placement_space_for(chosen) is UNIT_SPACE
    assert meter_quantities_for(chosen) == ALL_METERS
    assert ROOM_QUANTITIES == ROOM_KEYS


@pytest.mark.parametrize("u,edges", [(0.0, (4.0, 2.0, 2.0)), (1.0, (8.0, 6.0, 4.0)), (0.25, (5.0, 3.0, 2.5))])
def test_room_conversion_hand_calculated(u: float, edges: tuple[float, float, float]) -> None:
    from aosr.search.layout import room_from_unit

    chosen = settings(room_range=RANGES)
    unit = UNIT | dict.fromkeys(ROOM_KEYS, u)
    room = room_from_unit(unit, chosen, project())
    assert (room.Lx.hex(), room.Ly.hex(), room.Lz.hex()) == tuple(value.hex() for value in edges)
    params = params_from_unit(unit, chosen)
    assert params == LayoutParams(1.0, 1.0, 2.0)
    assert dict(unit_from_params(params, chosen, room=room)) == unit
    assert params_to_meters(params, room) == dict(zip(ALL_METERS, (1.0, 1.0, 2.0, *edges), strict=True))


def test_fixed_edges_copy_project_and_endpoint_is_clamped() -> None:
    from aosr.search.layout import room_from_unit

    # 這組外框在 u=1 真的會多一位；先守前提，才能考到夾回。
    assert 1.24 + (3.9 - 1.24) > 3.9
    chosen = settings(room_range={"Lx": {"low": 1.24, "high": 3.9}})
    room = room_from_unit(UNIT | {"room_Lx": 1.0}, chosen, project())
    assert (room.Lx.hex(), room.Ly.hex(), room.Lz.hex()) == (3.9.hex(), 4.0.hex(), 3.0.hex())


@pytest.mark.parametrize("value", [-0.01, 1.01, math.inf, math.nan])
def test_room_units_reject_invalid_values(value: float) -> None:
    from aosr.search.layout import room_from_unit

    chosen = settings(room_range={"Lx": RANGES["Lx"]})
    unit = UNIT | {"room_Lx": value}
    with pytest.raises(ValueError, match="room_Lx"):
        room_from_unit(unit, chosen, project())
    with pytest.raises(ValueError, match="room_Lx"):
        params_from_unit(unit, chosen)


@pytest.mark.parametrize("unit", [UNIT, UNIT | {"room_Lx": 0.5, "extra": 0.5},
                                 {"room_Lx": 0.5}, UNIT | {"room_Ly": 0.5}])
def test_conversion_requires_exact_keys(unit: dict[str, float]) -> None:
    from aosr.search.layout import room_from_unit

    chosen = settings(room_range={"Lx": RANGES["Lx"]})
    with pytest.raises(ValueError, match="exactly"):
        params_from_unit(unit, chosen)
    with pytest.raises(ValueError, match="exactly"):
        room_from_unit(unit, chosen, project())


@pytest.mark.parametrize("edge,value", [("Lx", 3.0), ("Lx", 9.0), ("Ly", 1.0), ("Ly", 7.0),
                                       ("Lz", 1.0), ("Lz", 5.0)])
def test_inverse_rejects_room_outside_ranges(edge: str, value: float) -> None:
    chosen = settings(room_range=RANGES)
    room = Room(**({"Lx": 6.0, "Ly": 4.0, "Lz": 3.0} | {edge: value}))
    with pytest.raises(ValueError, match=f"room_{edge}"):
        unit_from_params(LayoutParams(1.0, 1.0, 2.0), chosen, room=room)


def test_room_argument_is_required_only_with_room_range() -> None:
    from aosr.search.layout import room_from_unit

    params = LayoutParams(1.0, 1.0, 2.0)
    with pytest.raises(ValueError):
        unit_from_params(params, settings(room_range=RANGES))
    with pytest.raises(ValueError):
        unit_from_params(params, settings(), room=project().scene.room_m)
    with pytest.raises(ValueError):
        room_from_unit(UNIT, settings(), project())


def test_candidate_project_skips_validation_but_formal_scheme_checks_bounds() -> None:
    from aosr.search.layout import project_in

    original = project()
    before = original.model_dump_json()
    room = Room(Lx=2.0, Ly=4.0, Lz=3.0)
    candidate = project_in(original, room)
    assert candidate.scene.room_m == room
    assert candidate.receiver_set.primary.position_m == (3.0, 2.0, 1.25)
    assert original.model_dump_json() == before
    placed = place(candidate, settings(), LayoutParams(0.5, 1.0, 0.5))
    formal = to_scheme(candidate, placed, "candidate")
    assert formal.scene.room_m == room
    assert formal.receiver_set.primary.position_m == (1.0, 2.0, 1.25)
    with pytest.raises(ValidationError, match="喇叭"):
        to_scheme(candidate, replace(placed, left=Point(2.5, 1.5, 1.25)), "invalid")


@pytest.mark.parametrize("wall,left,right,primary", [
    ("xL", (7.0, 3.5, 1.25), (7.0, 2.5, 1.25), (5.0, 3.0, 1.25)),
    ("yL", (3.5, 5.0, 1.25), (4.5, 5.0, 1.25), (4.0, 3.0, 1.25)),
])
def test_far_front_wall_moves_with_room(wall: str, left: tuple[float, ...], right: tuple[float, ...],
                                      primary: tuple[float, ...]) -> None:
    from aosr.search.layout import project_in

    chosen = settings(front_wall=wall, room_range=RANGES)
    candidate = project_in(project(), Room(Lx=8.0, Ly=6.0, Lz=4.0))
    placed = place(candidate, chosen, LayoutParams(1.0, 1.0, 2.0))
    assert (placed.left.as_tuple(), placed.right.as_tuple(), placed.primary.as_tuple()) == (left, right, primary)


@pytest.mark.parametrize("minimum", [23.0, 24.0])
def test_floor_area_at_or_above_minimum_is_legal(minimum: float) -> None:
    chosen = settings(room_range=RANGES | {"min_floor_area_m2": minimum})
    assert check(project(), chosen, place(project(), chosen, LayoutParams(1.0, 1.0, 2.0))) == ()


def test_floor_area_equal_by_product_is_legal_even_if_division_overshoots() -> None:
    # 乘積剛好等於下限就合法；這組邊反除回去兩邊各多出最末一位，不准因此記成面積不足。
    from aosr.search.layout import project_in

    room = Room(Lx=6.975, Ly=3.665, Lz=3.0)
    minimum = room.Lx * room.Ly
    assert minimum / room.Ly - room.Lx > 0.0 and minimum / room.Lx - room.Ly > 0.0
    chosen = settings(room_range={"Lx": RANGES["Lx"], "Ly": RANGES["Ly"], "min_floor_area_m2": minimum})
    candidate = project_in(project(), room)
    assert check(candidate, chosen, place(candidate, chosen, LayoutParams(1.0, 1.0, 2.0))) == ()


@pytest.mark.parametrize("edges,amount", [(("Lx", "Ly"), 2.0), (("Lx",), 3.0),
                                          (("Ly",), 2.0), (("Lz",), 2.0)])
def test_floor_deficit_is_shortest_single_adjustable_edge_extension(edges: tuple[str, ...], amount: float) -> None:
    # 面積 24、最低 36：拉 Lx 要 36/4-6=3 m，拉 Ly 要 36/6-4=2 m。
    chosen = settings(room_range={edge: RANGES[edge] for edge in edges} | {"min_floor_area_m2": 36.0})
    violations = check(project(), chosen, place(project(), chosen, LayoutParams(1.0, 1.0, 2.0)))
    assert [(v.reason.value, v.amount_m.hex()) for v in violations] == [("floor_area_below_minimum", amount.hex())]
    assert COUNT_REASONS[violations[0].reason] == "地板面積不足"


@pytest.mark.parametrize("range_fields,minimum,message", [
    ({"Lz": RANGES["Lz"]}, 25.0, "地板兩邊都不可調"),
    ({"Lx": RANGES["Lx"], "Ly": RANGES["Ly"]}, 49.0, "最大地板面積"),
])
def test_precheck_rejects_impossible_floor(range_fields: dict[str, object], minimum: float, message: str) -> None:
    from aosr.search.seat_lock import check_room_range

    chosen = settings(room_range=range_fields | {"min_floor_area_m2": minimum})
    with pytest.raises(ValueError, match=message):
        check_room_range(project(), chosen)


@pytest.mark.parametrize("range_fields,minimum", [({"Lz": RANGES["Lz"]}, 24.0), (RANGES, 48.0),
    ({"Lx": {"low": 7.0, "high": 8.0}}, 32.0), ({"Lx": RANGES["Lx"]}, None)])
def test_precheck_accepts_possible_floor_even_when_project_is_outside(range_fields: dict[str, object],
                                                                    minimum: float | None) -> None:
    from aosr.search.seat_lock import check_room_range

    check_room_range(project(), settings(room_range=range_fields | {"min_floor_area_m2": minimum}))


@pytest.mark.parametrize("accepted", [False, True])
def test_precheck_listening_reach(accepted: bool) -> None:
    from aosr.search.seat_lock import check_room_range

    limits = {"low": 0.625, "high": 1.25} if accepted else {"low": 1.5, "high": 4.0}
    chosen = settings(room_range=RANGES, spacing_m={"low": 1.0, "high": 2.0},
                      listening_distance_m={"low": 0.375, "high": 0.75}, listening_range_m=limits)
    if accepted:
        check_room_range(project(), chosen)
    else:
        with pytest.raises(ValueError, match="型號適用聆聽距離"):
            check_room_range(project(), chosen)


def test_listening_precheck_uses_largest_room_contact_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.reporting.scheme import Scheme
    from aosr.search.layout import Placement
    from aosr.search.seat_lock import check_listening_reach, check_room_range

    chosen = settings(room_range={"Lx": {"low": 0.1189, "high": 63.9}},
        spacing_m={"low": 1.0, "high": 2.0}, listening_distance_m={"low": 0.375, "high": 0.75},
        listening_range_m={"low": 1.25 + 1e-11, "high": 4.0})
    # 最大房必須直接取 high；單位端點反算可能略低於 high。
    assert 0.1189 + (63.9 - 0.1189) < 63.9
    seen: list[Room] = []

    def observe(candidate: Scheme, layout_settings: LayoutSettings, params: LayoutParams) -> Placement:
        seen.append(candidate.scene.room_m)
        return place(candidate, layout_settings, params)

    monkeypatch.setattr("aosr.search.seat_lock.layout.place", observe)
    # 原房最長邊 6 的接觸帶小於缺口；最大房 63.9 的接觸帶大於缺口。
    check_listening_reach(project(), chosen)
    check_room_range(project(), chosen)
    assert seen and all(room == Room(Lx=63.9, Ly=4.0, Lz=3.0) for room in seen)


@pytest.mark.parametrize("fields", [
    {"seat_locked": True, "listening_distance_m": None},
    {"keep_out": [{axis: {"low": 0.5, "high": 1.0} for axis in ("x", "y", "z")}]},
    {"speaker_areas": [{axis: {"low": 0.5, "high": 1.0} for axis in ("x", "y", "z")}]},
])
def test_room_range_rejects_absolute_coordinate_constraints(fields: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="room_range"):
        settings(**(fields | {"room_range": RANGES}))


@pytest.mark.parametrize("room_range", [{}, {"min_floor_area_m2": 12.0},
    {"Lx": {"low": 0.0, "high": 1.0}}, {"Lz": {"low": -1.0, "high": 1.0}},
    {"Lx": RANGES["Lx"], "extra": True},
    *({"Lx": RANGES["Lx"], "min_floor_area_m2": value} for value in (0.0, -1.0, math.nan, math.inf)),
])
def test_room_range_rejects_invalid_envelopes(room_range: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        settings(room_range=room_range)


def test_room_range_is_frozen() -> None:
    from aosr.search.layout_settings import RoomRange

    declared = RoomRange.model_validate({"Lx": RANGES["Lx"]})
    with pytest.raises(ValidationError, match="frozen"):
        declared.min_floor_area_m2 = 30.0


@pytest.mark.parametrize("stage", [None, "stage_one_fem_only"])
def test_search_rejects_room_range_before_creating_folder(tmp_path: Path, stage: str | None) -> None:
    from aosr.search.store import SearchIdentity, SearchStore

    document = search_document(settings(room_range=RANGES)) | {"stage": stage}
    with pytest.raises(ValidationError, match="6b-2"):
        SearchSettings.model_validate(document)
    # 即使呼叫者繞過總表驗證，建檔入口仍須重驗，不能留下半接線資料夾。
    original = SearchSettings.model_validate(search_document(settings()))
    forged = original.model_copy(update={"layout": settings(room_range=RANGES), "stage": stage})
    root = tmp_path / "searches"
    with pytest.raises(ValidationError, match="6b-2"):
        SearchStore.create(root, project=project(), settings=forged,
            identity=SearchIdentity("physical", "program", purpose_settings("listening")),
            versions={"python": "test", "optuna": "test", "numpy": "test"})
    assert not root.exists()


def test_room_start_contains_project_room_and_unchanged_placement() -> None:
    from aosr.search.layout import room_start

    chosen = settings(room_range=RANGES | {"min_floor_area_m2": 24.0})
    start = room_start(project(), chosen)
    assert start is not None
    assert tuple(start) == ALL_METERS
    assert {name: start[name] for name in ROOM_KEYS} == dict.fromkeys(ROOM_KEYS, 0.5)
    # 原間距 1、60 度起點聆聽距離 sqrt(3)/2；以輸入手算單位座標。
    assert start["front_distance"].hex() == 0.5.hex()
    assert start["spacing"].hex() == 0.25.hex()
    assert start["listening_distance"].hex() == ((math.sqrt(3.0) / 2.0 - 0.5) / 2.0).hex()
    standard = standard_start(project(), chosen)
    assert standard is not None
    assert dict(unit_from_params(standard, chosen, room=project().scene.room_m)) == dict(start)


@pytest.mark.parametrize("fields", [
    {"room_range": {"Lx": {"low": 7.0, "high": 8.0}}},
    {"room_range": {"Ly": {"low": 1.0, "high": 3.0}}},
    {"room_range": {"Lz": {"low": 4.0, "high": 5.0}}},
    {"room_range": RANGES | {"min_floor_area_m2": 25.0}},
    {"room_range": RANGES, "front_distance_m": {"low": 1.1, "high": 1.5}},
    {"room_range": RANGES, "listening_distance_m": {"low": 0.1, "high": 0.2}},
])
def test_room_start_returns_none_when_project_cannot_start(fields: dict[str, object]) -> None:
    from aosr.search.layout import room_start

    assert room_start(project(), settings(**fields)) is None


@pytest.mark.parametrize("outcome", [Illegal("floor_area_below_minimum", 2.0),
                                     Scored(0.5), Excluded(RankingZone.ELIMINATED)])
def test_row_uses_v2_header_and_roundtrips(tmp_path: Path, outcome: Outcome) -> None:
    header = LedgerHeaderV2.model_validate(header_document() | {
        "ledger_version": "aosr.search_ledger.v2", "search_space": dict.fromkeys(ALL_METERS, (0.0, 1.0)),
        "meter_quantities": ALL_METERS})
    unit = UNIT | dict.fromkeys(ROOM_KEYS, 0.5)
    meters = dict(zip(ALL_METERS, (1.0, 1.0, 2.0, 6.0, 4.0, 3.0), strict=True))
    row = row_from_outcome(batch_index=0, proposal=Proposal(0, unit), params_m=meters, outcome=outcome,
                           seconds=0.25, result_file=None if isinstance(outcome, Illegal) else candidate_name(0), header=header)
    assert type(row) is LedgerRowV2
    assert row.unit_params_hex == {key: value.hex() for key, value in unit.items()}
    assert row.params_m == meters
    path = tmp_path / "ledger"
    Ledger.create(path, header).append(row)
    assert Ledger.read(path) == (header, (row,))


def test_runner_records_against_its_snapshot_header(tmp_path: Path) -> None:
    from datetime import date

    from aosr.config.quality_targets import QualityTargets
    from aosr.search.ledger import header_for
    from aosr.search.run import SearchStatus, _Runner
    from aosr.search.store import SearchIdentity, SearchStore, _IdentitySnapshot
    from aosr.search.timings import WallClock

    original = SearchSettings.model_validate(search_document(settings()))
    # 只建立記憶體快照供落帳測試；不走搜尋建立入口、不啟動尚未接好的房間執行迴圈。
    future = original.model_copy(update={"layout": settings(room_range=RANGES)})
    identity = SearchIdentity("physical", "program", purpose_settings("listening"))
    snapshot = _IdentitySnapshot.model_validate({"store_version": "aosr.search_store.v1", "search_id": "synthetic",
        "physics_identity": "physical", "program_fingerprint": "program",
        "versions": {"python": "test", "optuna": "test", "numpy": "test"}})
    store = SearchStore(tmp_path, project(), future, identity, snapshot)
    header = header_for(store)
    assert isinstance(header, LedgerHeaderV2)
    book = Ledger.create(store.ledger_path, header)
    registry = QualityTargets.model_validate({"schema_version": 1,
        "purpose": [{"name": "listening", "setting": [], "target": [], "weight": [], "qualification": []}]})
    runner = _Runner(store, lambda jobs, workers: iter(()), lambda: identity, registry,
                     date(2026, 10, 22), "program", book, SearchStatus(), WallClock())
    unit = UNIT | dict.fromkeys(ROOM_KEYS, 0.5)
    meters = dict(zip(ALL_METERS, (1.0, 1.0, 2.0, 6.0, 4.0, 3.0), strict=True))
    runner.record(0, Proposal(0, unit), meters, Illegal("wall_gap", 0.125), 0.25, None)
    read_header, rows = Ledger.read(store.ledger_path)
    assert read_header == header
    assert tuple(type(row) for row in rows) == (LedgerRowV2,)
    assert tuple(row.params_m for row in rows) == (meters,)


@pytest.mark.parametrize("explicit_header", [False, True])
def test_v1_row_bytes_stay_identical(explicit_header: bool) -> None:
    header = LedgerHeader.model_validate(header_document()) if explicit_header else None
    row = row_from_outcome(batch_index=0, proposal=Proposal(0, UNIT),
        params_m={"front_distance": 1.0, "spacing": 1.0, "listening_distance": 2.0},
        outcome=Illegal("wall_gap", 0.125), seconds=0.25, result_file=None, header=header)
    assert type(row) is LedgerRow
    expected = {"batch_index": 0, "trial_number": 0,
        "unit_params_hex": {"front_distance": "0x1.0000000000000p-1", "spacing": "0x1.0000000000000p-2",
                            "listening_distance": "0x1.8000000000000p-1"},
        "params_m": {"front_distance": 1.0, "spacing": 1.0, "listening_distance": 2.0},
        "outcome": "illegal", "score": None, "reason": "wall_gap", "violation_m": 0.125,
        "seconds": 0.25, "result_file": None}
    assert json.dumps(row.model_dump(mode="json"), ensure_ascii=False, allow_nan=False).encode() == json.dumps(expected).encode()
