"""家具方案的第二層驗證：全部幾何判斷使用第二支與登記簿界線。"""
import pytest

from aosr.geometry.furniture import FurnitureBox
from aosr.reporting.scheme import Scheme
from tests.engine._furniture_cases import cloud_item, document, relative_item
from tests.engine._precision_contracts import contract_value


def _boxes(*items: dict[str, object]) -> tuple[FurnitureBox, ...]:
    from aosr.reporting.furniture_layout import furniture_boxes

    return furniture_boxes(Scheme.model_validate(document(*items)),
                           contact_rel=contract_value("furniture_geometry_contact"))


def _seat(furniture_id: str = "seat", *, forward: float = -1.0, left: float = 1.0,
          bottom: float = 0.0, height: float = 0.6) -> dict[str, object]:
    return relative_item(furniture_id=furniture_id, depth_m=1.0, height_m=height,
                         placement={"forward_m": forward, "left_m": left,
                                    "bottom_height_m": bottom, "yaw_deg": 0})


def test_valid_furniture_boxes_use_absolute_coordinates_and_sorted_ids() -> None:
    boxes = _boxes(_seat("z-seat"), cloud_item())
    assert [(box.bottom_center_m, box.yaw_deg) for box in boxes] == [
        ((2.0, 2.0, 2.5), 90.0), ((2.0, 2.0, 0.0), 0.0)]


def test_absent_furniture_needs_no_geometry() -> None:
    assert _boxes() == ()


@pytest.mark.parametrize("items,reason", [
    ((_seat(left=4.0),), "seat.*超出房間"),
    ((_seat("first"), _seat("second", left=0.0)), "first.*second.*間隙"),
    ((_seat("first"), _seat("second", forward=0.0, left=0.0)), "first.*second.*間隙"),
    ((_seat("first"), _seat("second", left=0.5)), "first.*second.*間隙"),
    ((relative_item(kind="desk", material="wood", height_m=0.5,
                    placement={"forward_m": 2, "left_m": 1, "bottom_height_m": 1, "yaw_deg": 0}),),
     "喇叭 left.*seat.*內部"),
    ((_seat(forward=0.0, left=0.0, height=2.0),), "座位 main.*seat.*內部"),
    ((relative_item(width_m=0.1, depth_m=0.1, height_m=2.0,
                    placement={"forward_m": 0, "left_m": -0.1, "bottom_height_m": 0, "yaw_deg": 0}),),
     "座位 side.*seat.*內部"),
    ((_seat(bottom=0.1),), "seat.*貼地家具.*底面"),
    ((cloud_item(placement={"bottom_center_m": [2, 2, 0], "yaw_deg": 0}),), "cloud.*懸空家具.*底面"),
])
def test_invalid_furniture_layout_names_the_object_and_rule(items: tuple[dict[str, object], ...], reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        _boxes(*items)


def test_room_boundary_absorbs_only_the_registered_margin() -> None:
    margin = 8.0 * contract_value("furniture_geometry_contact")
    _boxes(_seat(left=2.5 + margin / 2.0))  # x 下緣在牆外半份界線。
    with pytest.raises(ValueError, match="seat.*超出房間"):
        _boxes(_seat(left=2.5 + 2.0 * margin))


def test_two_boxes_need_more_than_the_registered_gap() -> None:
    margin = 8.0 * contract_value("furniture_geometry_contact")
    with pytest.raises(ValueError, match="first.*second.*間隙"):
        _boxes(_seat("first"), _seat("second", left=-margin / 2.0))
    _boxes(_seat("first"), _seat("second", left=-2.0 * margin))


def test_grounded_and_suspended_bottoms_use_the_registered_margin() -> None:
    margin = 8.0 * contract_value("furniture_geometry_contact")
    _boxes(_seat(bottom=margin / 2.0))
    with pytest.raises(ValueError, match="seat.*貼地家具"):
        _boxes(_seat(bottom=2.0 * margin))
    with pytest.raises(ValueError, match="cloud.*懸空家具"):
        _boxes(cloud_item(placement={"bottom_center_m": [2, 2, margin / 2.0], "yaw_deg": 0}))
    _boxes(cloud_item(placement={"bottom_center_m": [2, 2, 2.0 * margin], "yaw_deg": 0}))


def test_furniture_layout_is_outside_modal_identity_closure() -> None:
    from aosr.reporting.modal_diagnosis import modal_identity, modal_import_closure

    assert modal_identity() == "modal-v1:5dcb7e682c04552785515d72567623f0bd8c42ae82a742ac4b7f5be015955e9e"
    assert "aosr.reporting.furniture_layout" not in modal_import_closure().modules
