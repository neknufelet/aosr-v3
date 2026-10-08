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


def _validation_document(case: str) -> dict[str, object]:
    if case == "valid":
        return document(relative_item())
    if case == "outside":
        return document(relative_item(placement={"forward_m": 1, "left_m": 4, "bottom_height_m": 0, "yaw_deg": 0}))
    if case == "facing":
        return document(relative_item()) | {"speakers": {
            "left": {"x": 4, "y": 4, "z": 1.2}, "right": {"x": 6, "y": 6, "z": 1.2}}}
    if case == "two_block":
        # 0.3 m 見方的兩張桌：z-first 在 (2.5,4)，左喇叭兩條線在 y=4 經過 x=2.5 與 2.55；
        # a-second 在 (2.75,3.5)，兩條線在 y=3.5 經過 x=2.75 與 2.825。右喇叭兩條線都不碰。
        def table(furniture_id: str, forward: float, left: float) -> dict[str, object]:
            return relative_item(furniture_id=furniture_id, kind="desk", material="wood",
                                 width_m=0.3, depth_m=0.3, height_m=0.5,
                                 placement={"forward_m": forward, "left_m": left, "bottom_height_m": 1, "yaw_deg": 0})
        return document(table("z-first", 1.0, 0.5), table("a-second", 0.5, 0.25))
    width = 0.3 if case == "both" else 0.02
    left = 0.45 if case == "side" else 0.5
    return document(relative_item(furniture_id="desk", kind="desk", material="wood",
                                   width_m=width, depth_m=width, height_m=0.5,
                                   placement={"forward_m": 1, "left_m": left, "bottom_height_m": 1, "yaw_deg": 0}))


@pytest.mark.parametrize("case,messages", [
    ("outside", ("家具 seat 超出房間接觸界線",)),
    ("facing", ("專案方案的兩喇叭中點相對主位沒有主要方向（x、y 一樣大），定不出前牆",)),
    # 主位線在 y=4 時 x=2.5，周圍線 x=2.55；兩個 2 cm 小桌分別只擋一對。
    ("main", ("不符合擺位要求：喇叭 left 到座位 main 的直達路徑被家具 desk 擋住",)),
    ("side", ("不符合擺位要求：喇叭 left 到座位 side 的直達路徑被家具 desk 擋住",)),
    ("both", ("不符合擺位要求：喇叭 left 到座位 main 的直達路徑被家具 desk 擋住",
              "不符合擺位要求：喇叭 left 到座位 side 的直達路徑被家具 desk 擋住")),
    ("two_block", ("不符合擺位要求：喇叭 left 到座位 main 的直達路徑被家具 a-second、z-first 擋住",
                   "不符合擺位要求：喇叭 left 到座位 side 的直達路徑被家具 a-second、z-first 擋住")),
    ("valid", (GATE_MESSAGE,)),
])
def test_checked_inputs_rejects_furniture_before_building_pairs(
    monkeypatch: pytest.MonkeyPatch, case: str, messages: tuple[str, ...],
) -> None:
    def forbidden_pair(*args: object, **kwargs: object) -> dict[str, object]:
        raise AssertionError("輸入關應在組報表文件之前")

    monkeypatch.setattr("aosr.reporting.validation.pair_input_document", forbidden_pair)
    with pytest.raises(SchemeValidationError) as caught:
        checked_inputs(_validation_document(case), capabilities=CAPABILITIES, directivity=DIRECTIVITY)
    assert [(problem.path, problem.message) for problem in caught.value.problems] == [
        ("furniture", message) for message in messages]


def test_furniture_validation_reads_registry_independent_of_working_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    # 換到暫存目錄後仍走到輸入關那句，表示登記簿路徑不跟工作目錄走（不是找不到檔案）。
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SchemeValidationError) as caught:
        checked_inputs(_validation_document("valid"), capabilities=CAPABILITIES, directivity=DIRECTIVITY)
    assert [(problem.path, problem.message) for problem in caught.value.problems] == [("furniture", GATE_MESSAGE)]


@pytest.mark.parametrize("furniture", [None, []])
def test_no_furniture_does_not_read_contact_registry(monkeypatch: pytest.MonkeyPatch, furniture: object) -> None:
    def forbidden_read(path: object) -> float:
        raise AssertionError("沒有家具不得讀登記簿")

    monkeypatch.setattr("aosr.reporting.validation.furniture_contact_rel", forbidden_read)
    scheme, pairs = checked_inputs(reference_document() | {"furniture": furniture},
                                   capabilities=CAPABILITIES, directivity=DIRECTIVITY)
    assert scheme.furniture is None and pairs


def test_solver_inputs_preserve_normalized_furniture() -> None:
    from aosr.physics import report_io

    with_furniture = report_io.load_input_document(
        pair(Scheme.model_validate(document(relative_item()))), CAPABILITIES, DIRECTIVITY)
    assert report_io.solver_inputs(with_furniture).furniture is with_furniture.furniture
    plain = report_io.load_input_document(pair(Scheme.model_validate(reference_document())), CAPABILITIES, DIRECTIVITY)
    assert report_io.solver_inputs(plain).room == plain.room_m
    assert report_io.solver_inputs(plain).furniture is None


def test_reflection_screen_keeps_wall_values_with_furniture() -> None:
    from aosr.physics import report_io
    from aosr.physics.reflection_screen import build_reflection_screen

    with_furniture = report_io.load_input_document(
        pair(Scheme.model_validate(document(relative_item()))), CAPABILITIES, DIRECTIVITY)
    plain = with_furniture.model_copy(update={"furniture": None})
    furnished = build_reflection_screen(with_furniture, (500.0,))
    empty = build_reflection_screen(plain, (500.0,))
    assert furnished.pairs == empty.pairs
    assert furnished.next_order_earliest_delay_s == empty.next_order_earliest_delay_s
    assert furnished.source_m == empty.source_m
    assert furnished.receiver_m == empty.receiver_m
    assert furnished.frequencies_hz == empty.frequencies_hz
    assert furnished.scene_fingerprint != empty.scene_fingerprint


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
