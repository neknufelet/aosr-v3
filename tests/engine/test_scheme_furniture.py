"""#559 第四支：相對規格、手算換位、兩份既有指紋及輸入關。"""
from pathlib import Path

import pytest
from pydantic import ValidationError

from aosr.geometry.shoebox import Point
from aosr.reporting.scheme import Scheme
from aosr.reporting.validation import SchemeValidationError, checked_inputs
from aosr.search import ledger
from aosr.search.settings import SearchSettings
from aosr.search.store import SearchIdentity, SearchStore
from tests.engine._directivity import DIRECTIVITY
from tests.engine._furniture_cases import (
    CAPABILITIES, GATE_MESSAGE, cloud_item, document, fingerprint, pair, reference_document, relative_item,
)
from tests.engine._search_store_cases import purpose_settings, settings_document


def _store(tmp_path: Path, project: Scheme) -> SearchStore:
    identity = SearchIdentity("phys-test", "program-test", purpose_settings(project.purpose))
    return SearchStore.create(tmp_path, project=project,
                              settings=SearchSettings.model_validate(settings_document() | {"purpose": project.purpose}),
                              identity=identity, versions={"python": "test", "numpy": "test", "optuna": "test"})


def test_no_furniture_reference_fingerprints_are_unchanged(tmp_path: Path) -> None:
    from aosr.physics.report_io import scene_fingerprint

    scheme, pairs = checked_inputs(reference_document(), capabilities=CAPABILITIES, directivity=DIRECTIVITY)
    for pair_document, inputs in pairs.values():
        assert "furniture" not in pair_document
        assert scene_fingerprint(inputs) == "ef718eb5ecfa13d9d0c84320d18aeb99566db02021bdc16d7b7cbb2644b74de2"
    assert ledger.header_for(_store(tmp_path, scheme)).project_fingerprint == (
        "0cd0e215f3fa0fe6d0de2798098fbc57b283caa35e946c68160690406c2b4d1f")


@pytest.mark.parametrize("facing,center,yaw", [
    ("north", [2.5, 4.0, 0.0], 90.0), ("west", [2.0, 2.5, 0.0], 180.0),
    ("south", [3.5, 2.0, 0.0], 270.0), ("east", [4.0, 3.5, 0.0], 0.0),
])
def test_listener_coordinates_four_hand_calculated_facings(facing: str, center: list[float], yaw: float) -> None:
    item = relative_item(placement={"forward_m": 1.0, "left_m": 0.5, "bottom_height_m": 0.0, "yaw_deg": 90})
    furniture = pair(Scheme.model_validate(document(item, facing=facing)))["furniture"]
    assert furniture == [{"furniture_id": "seat", "kind": "sofa", "material": "fabric",
                          "width_m": 1.0, "depth_m": 0.5, "height_m": 0.6,
                          "bottom_center_m": center, "yaw_deg": yaw}]


def test_cloud_keeps_room_coordinates_even_when_listener_moves() -> None:
    for facing, primary in (("north", (3.0, 3.0, 1.2)), ("west", (4.0, 4.0, 1.6))):
        furniture = pair(Scheme.model_validate(document(cloud_item(), facing=facing, primary=primary)))["furniture"]
        assert isinstance(furniture, list)
        assert furniture[0]["bottom_center_m"] == [2.0, 2.0, 2.5]
        assert furniture[0]["yaw_deg"] == 90.0


def test_every_pair_uses_the_primary_furniture_scene() -> None:
    from aosr.reporting.scheme import pair_input_document

    scheme = Scheme.model_validate(document(relative_item()))
    for source in scheme.speakers.values():
        for receiver in scheme.receiver_set.points:
            pair_document = pair_input_document(scheme, source, Point(*receiver.position_m), {"kind": "omnidirectional"})
            assert pair_document["furniture"] == [{
                "furniture_id": "seat", "kind": "sofa", "material": "fabric",
                "width_m": 1.0, "depth_m": 0.5, "height_m": 0.6,
                "bottom_center_m": [2.5, 4.0, 0.0], "yaw_deg": 0.0}]


def test_relative_furniture_moves_and_turns_with_primary() -> None:
    item = relative_item()
    shifted = pair(Scheme.model_validate(document(item, primary=(4.0, 3.0, 1.8))))["furniture"]
    turned = pair(Scheme.model_validate(document(item, facing="west")))["furniture"]
    assert isinstance(shifted, list) and isinstance(turned, list)
    assert shifted[0]["bottom_center_m"] == [3.5, 4.0, 0.0]
    assert shifted[0]["yaw_deg"] == 0.0
    assert turned[0]["bottom_center_m"] == [2.0, 2.5, 0.0]
    assert turned[0]["yaw_deg"] == 90.0


def test_furniture_is_sorted_and_declaration_order_does_not_change_scene() -> None:
    first = Scheme.model_validate(document(relative_item(furniture_id="z-seat"), cloud_item()))
    second = Scheme.model_validate(document(cloud_item(), relative_item(furniture_id="z-seat")))
    furniture = pair(first)["furniture"]
    assert isinstance(furniture, list)
    assert [item["furniture_id"] for item in furniture] == ["cloud", "z-seat"]
    assert fingerprint(first) == fingerprint(second)


def test_empty_and_omitted_furniture_are_the_same_scene_and_project(tmp_path: Path) -> None:
    omitted = Scheme.model_validate(reference_document())
    empty = Scheme.model_validate(reference_document() | {"furniture": []})
    assert empty.furniture is None
    assert "furniture" not in pair(empty)
    assert fingerprint(empty) == fingerprint(omitted)
    assert ledger.header_for(_store(tmp_path / "empty", empty)).project_fingerprint == (
        ledger.header_for(_store(tmp_path / "omitted", omitted)).project_fingerprint)


@pytest.mark.parametrize("change,reason", [
    ({"furniture_id": "   "}, "furniture_id.*不可.*空白"),
    ({"kind": "unknown"}, "kind"), ({"material": "unknown"}, "五種材質"),
    ({"material": "glass"}, "材質.*不適用.*sofa"),
    ({"width_m": 0.0}, "有限正數"), ({"depth_m": -1.0}, "有限正數"),
    ({"height_m": float("nan")}, "有限正數"), ({"width_m": float("inf")}, "有限正數"),
    ({"width_m": True}, "有限正數"), ({"width_m": "1"}, "有限正數"),
    ({"width_m": 10**1000}, "有限正數"),
    ({"placement": {"bottom_center_m": [2, 2, 0], "yaw_deg": 0}}, "跟著主位"),
    ({"placement": None}, "placement"), ({"tilt_deg": 0}, "Extra inputs"),
])
def test_invalid_furniture_spec_reports_its_reason(change: dict[str, object], reason: str) -> None:
    with pytest.raises(ValidationError, match=reason):
        Scheme.model_validate(document(relative_item(**change)))


@pytest.mark.parametrize("change,reason", [
    ({"yaw_deg": 45}, "0／90／180／270"), ({"yaw_deg": True}, "0／90／180／270"),
    ({"yaw_deg": "90"}, "0／90／180／270"), ({"yaw_deg": float("inf")}, "0／90／180／270"),
    ({"forward_m": float("inf")}, "有限數字"), ({"left_m": float("nan")}, "有限數字"),
    ({"bottom_height_m": float("inf")}, "有限數字"),
    ({"bottom_center_m": [2, 2, 0]}, "Extra inputs"), ({"tilt_deg": 0}, "Extra inputs"),
])
def test_invalid_relative_placement_reports_its_reason(change: dict[str, object], reason: str) -> None:
    placement = {"forward_m": 0, "left_m": 0, "bottom_height_m": 0, "yaw_deg": 0} | change
    with pytest.raises(ValidationError, match=reason):
        Scheme.model_validate(document(relative_item(placement=placement)))


def test_cloud_requires_room_placement_and_finite_coordinates() -> None:
    with pytest.raises(ValidationError, match="天雲.*房間座標"):
        Scheme.model_validate(document(cloud_item(placement=relative_item()["placement"])))
    with pytest.raises(ValidationError, match="有限數字"):
        Scheme.model_validate(document(cloud_item(placement={"bottom_center_m": [2, 2, float("inf")], "yaw_deg": 0})))


@pytest.mark.parametrize("kind,materials", [
    ("sofa", {"fabric", "leather"}), ("chair", {"fabric", "leather"}),
    ("coffee_table", {"wood", "glass"}), ("desk", {"wood", "glass"}),
    ("ceiling_cloud", {"wood", "absorptive_cloud"}),
])
def test_material_choices_for_every_furniture_kind(kind: str, materials: set[str]) -> None:
    for material in ("fabric", "leather", "wood", "glass", "absorptive_cloud"):
        item = cloud_item(kind=kind, material=material) if kind == "ceiling_cloud" else relative_item(kind=kind, material=material)
        if material in materials:
            Scheme.model_validate(document(item))
        else:
            with pytest.raises(ValidationError, match=f"材質.*不適用 {kind}"):
                Scheme.model_validate(document(item))


def test_duplicate_furniture_id_is_rejected() -> None:
    with pytest.raises(ValidationError, match="furniture_id.*重複.*seat"):
        Scheme.model_validate(document(relative_item(), relative_item()))


def test_furniture_spec_and_placement_are_frozen() -> None:
    item = Scheme.model_validate(document(relative_item())).furniture
    assert item is not None
    with pytest.raises(ValidationError, match="frozen"):
        item[0].width_m = 2.0
    with pytest.raises(ValidationError, match="frozen"):
        item[0].placement.yaw_deg = 90.0


def test_checked_inputs_rejects_furniture_before_building_pairs(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden_pair(*args: object, **kwargs: object) -> dict[str, object]:
        raise AssertionError("輸入關應在組報表文件之前")

    monkeypatch.setattr("aosr.reporting.validation.pair_input_document", forbidden_pair)
    with pytest.raises(SchemeValidationError, match=GATE_MESSAGE) as caught:
        checked_inputs(document(relative_item()), capabilities=CAPABILITIES, directivity=DIRECTIVITY)
    assert [(problem.path, problem.message) for problem in caught.value.problems] == [("furniture", GATE_MESSAGE)]


def test_solver_inputs_refuse_furniture_so_no_physics_entry_ignores_it() -> None:
    from aosr.physics import report_io

    with_furniture = report_io.load_input_document(
        pair(Scheme.model_validate(document(relative_item()))), CAPABILITIES, DIRECTIVITY)
    with pytest.raises(ValueError, match=GATE_MESSAGE):
        report_io.solver_inputs(with_furniture)
    plain = report_io.load_input_document(pair(Scheme.model_validate(reference_document())), CAPABILITIES, DIRECTIVITY)
    assert report_io.solver_inputs(plain).room == plain.room_m


def test_search_store_rejects_furniture_before_making_any_directory(tmp_path: Path) -> None:
    root = tmp_path / "new-search"
    with pytest.raises(ValueError, match=GATE_MESSAGE):
        _store(root, Scheme.model_validate(document(relative_item())))
    assert not root.exists()


def test_search_project_fingerprint_includes_furniture(tmp_path: Path) -> None:
    baseline = _store(tmp_path, Scheme.model_validate(reference_document()))
    # 搜尋入口這支刻意拒收家具；唯讀的表頭仍要能核家具身分，以供後續解除輸入關。
    snapshot = baseline.path / "project.json"
    changed = Scheme.model_validate(reference_document() | {"furniture": [relative_item()]})
    snapshot.write_text(changed.model_dump_json(), encoding="utf-8")
    assert ledger.header_for(SearchStore.open(baseline.path)).project_fingerprint != (
        "0cd0e215f3fa0fe6d0de2798098fbc57b283caa35e946c68160690406c2b4d1f")


def test_furniture_facing_rejects_equal_horizontal_components() -> None:
    content = document(relative_item())
    content["speakers"] = {"left": Point(4, 4, 1.2), "right": Point(6, 6, 1.2)}
    with pytest.raises(ValueError, match="x、y 一樣大"):
        pair(Scheme.model_validate(content))
