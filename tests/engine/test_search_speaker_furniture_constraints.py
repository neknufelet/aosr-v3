"""四條搜尋施工限制的手算答案，接觸尾差與原因組合。"""
import pytest

from aosr.reporting.scheme import Scheme
from aosr.search import constraints, furniture_prefilter
from aosr.search.layout_settings import Box, LayoutSettings
from tests.engine._furniture_cases import cloud_item, relative_item
from tests.engine._search_speaker_setup_cases import geometric

CONTACT = 2.0 ** -40


def checked(scheme: Scheme, settings: LayoutSettings) -> dict[str, float]:
    return {v.reason.value: v.amount_m for v in furniture_prefilter.check_candidate(
        scheme, settings, contact_rel=CONTACT)}


def test_cabinet_off_table_hand_calculation() -> None:
    scheme, settings = geometric("desk")
    # f=(.8,.6)，左箱最小 x=1-.25*.8-.1*.6=.74；桌緣 x=2-.7-.4=.9。
    assert checked(scheme, settings) == {"cabinet_off_table": pytest.approx(0.16)}


@pytest.mark.parametrize("forward", [0.9, 0.86])
def test_table_fits_and_contact_roundoff_is_legal(forward: float) -> None:
    scheme, settings = geometric("desk", forward=forward)
    assert checked(scheme, settings) == {}


def test_supporting_table_is_excluded_and_bottom_is_structural() -> None:
    scheme, settings = geometric("desk", forward=0.9, bottom=0.41, thickness=0.035, center=0.175)
    assert checked(scheme, settings) == {}
    # 人打高度與加總只差最後一位，箱底不拿 z-center 核承托。
    doc = scheme.model_dump()
    doc["speakers"]["left"]["z"] = doc["speakers"]["right"]["z"] = 0.62
    assert checked(Scheme.model_validate(doc), settings) == {}


def test_cabinet_in_furniture_hand_calculation() -> None:
    scheme, settings = geometric("floor", center=0.9)
    # 分離軸退出距離 x=.16、y=.33、f=.23、s=.72，取 .16；高度重疊 .04。
    assert checked(scheme, settings) == {"cabinet_in_furniture": pytest.approx(0.16)}


@pytest.mark.parametrize("mount,forward", [("floor", 0.54), ("stand", 0.54)])
def test_cabinet_and_stand_contact_is_legal(mount: str, forward: float) -> None:
    scheme, settings = geometric(mount, forward=forward, center=0.9 if mount == "floor" else 0.2)
    # 桌緣 x=2-.54-.4=1.06，箱體最大 x=1.06。
    assert checked(scheme, settings) == {}


def test_stand_space_hand_calculation_and_free_space() -> None:
    scheme, settings = geometric("stand", forward=0.6)
    assert checked(scheme, settings) == {"stand_space_occupied": pytest.approx(0.06)}
    scheme, settings = geometric("stand", forward=0.3)
    assert checked(scheme, settings) == {}


def test_same_furniture_hits_cabinet_and_stand() -> None:
    scheme, settings = geometric("stand", forward=0.6, bottom=0.95, thickness=0.1)
    assert checked(scheme, settings) == {"cabinet_in_furniture": pytest.approx(0.06),
                                       "stand_space_occupied": pytest.approx(0.06)}
    illegal = constraints.to_illegal(furniture_prefilter.check_candidate(scheme, settings, contact_rel=CONTACT))
    assert illegal.reason == "cabinet_in_furniture+stand_space_occupied"
    assert illegal.violation == pytest.approx(0.12)


@pytest.mark.parametrize("upper,expected", [(1.3, {"furniture_in_keep_out": pytest.approx(0.2)}), (1.1, {})])
def test_moving_furniture_keep_out_hand_calculation(upper: float, expected: dict[str, object]) -> None:
    scheme, settings = geometric("stand", forward=0.3)
    sofa = relative_item(furniture_id="sofa", width_m=1.8, depth_m=0.75, height_m=0.65,
        placement={"forward_m": -1.0, "left_m": 0.0, "bottom_height_m": 0.0, "yaw_deg": 0})
    scheme = Scheme.model_validate(scheme.model_dump() | {"furniture": [sofa]})
    settings = settings.model_copy(update={"keep_out": (Box.model_validate({
        "x": {"low": 2.0, "high": 3.0}, "y": {"low": 0.0, "high": upper},
        "z": {"low": 0.0, "high": 2.1}}),)})
    assert checked(scheme, settings) == expected


def test_fixed_cloud_is_not_checked_against_keep_out() -> None:
    scheme, settings = geometric("stand", forward=0.3)
    scheme = Scheme.model_validate(scheme.model_dump() | {"furniture": [cloud_item()]})
    settings = settings.model_copy(update={"keep_out": (Box.model_validate({
        "x": {"low": 0.0, "high": 6.0}, "y": {"low": 0.0, "high": 4.0},
        "z": {"low": 2.0, "high": 3.0}}),)})
    assert checked(scheme, settings) == {}


def test_supporting_table_never_enters_collision_check(monkeypatch: pytest.MonkeyPatch) -> None:
    scheme, settings = geometric("desk", forward=0.9)
    def forbidden(first: object, second: object, margin: float) -> float:
        raise AssertionError("承托桌不准判穿家具")
    monkeypatch.setattr(furniture_prefilter, "_contact_penetration", forbidden)
    assert checked(scheme, settings) == {}


def test_stand_column_stops_at_bottom_not_cabinet_top() -> None:
    scheme, settings = geometric("stand", forward=0.6, bottom=1.1)
    assert checked(scheme, settings) == {"cabinet_in_furniture": pytest.approx(0.06)}


@pytest.mark.parametrize("mount,bottom", [("floor", 1.1 - 1e-16), ("stand", 1.0 - 1e-16)])
def test_vertical_contact_roundoff_is_legal(mount: str, bottom: float) -> None:
    scheme, settings = geometric(mount, forward=0.6, bottom=bottom, center=0.9 if mount == "floor" else 0.2)
    result = checked(scheme, settings)
    assert "stand_space_occupied" not in result
    if mount == "floor":
        assert result == {}


def test_keep_out_vertical_contact_is_legal() -> None:
    scheme, settings = geometric("stand", forward=0.3)
    sofa = relative_item(furniture_id="sofa", width_m=1.8, depth_m=0.75, height_m=0.65,
        placement={"forward_m": -1.0, "left_m": 0.0, "bottom_height_m": 0.0, "yaw_deg": 0})
    scheme = Scheme.model_validate(scheme.model_dump() | {"furniture": [sofa]})
    settings = settings.model_copy(update={"keep_out": (Box.model_validate({
        "x": {"low": 2.0, "high": 3.0}, "y": {"low": 0.0, "high": 1.3},
        "z": {"low": 0.65 - 1e-16, "high": 2.1}}),)})
    assert checked(scheme, settings) == {}
