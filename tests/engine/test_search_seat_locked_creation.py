"""建資料夾前拒收矛盾的鎖定輸入；答案是手算訊息原文與沒有資料夾。"""
import argparse
from pathlib import Path

import pytest

from aosr.reporting.scheme import Scheme
from aosr.geometry.shoebox import Point
from aosr.reporting.validation import SchemeValidationError
from aosr.search import cli
from aosr.search.store import SearchIdentity, SearchStore
from tests.engine._seat_locked_cases import locked_settings
from tests.engine._search_store_cases import reference_project, purpose_settings
from tests.engine._search_speaker_setup_cases import geometric


def create(root: Path, project: Scheme, **changes: object) -> SearchStore:
    settings = locked_settings(**changes)
    settings = settings.model_copy(update={"purpose": project.purpose})
    return SearchStore.create(root, project=project, settings=settings,
        identity=SearchIdentity("phys-test", "calc-test", purpose_settings(project.purpose)),
        versions={"python": "test", "optuna": "test", "numpy": "test"})


@pytest.mark.parametrize("changes,message", [
    ({"front_wall": "y0"}, "座位鎖定時前牆要是原方案面向的 x0；設定寫 y0"),
    ({"axis_offset_m": 0.0}, "座位鎖定時中軸要通過主位：主位y 1.9 m，推出偏移 -0.10000000000000009 m；設定寫 0.0 m"),
    ({"ear_height_m": 1.25}, "座位鎖定時耳高要等於主位高度 1.2 m；設定寫 1.25 m"),
    ({"listening_distance_m": {"low": 0.5, "high": 3.0}},
     "座位鎖定時不准寫聆聽距離範圍；聆聽距離由座位推出，範圍 1.2000000000000002～2.95 m"),
    ({"front_distance_m": {"low": 0.25, "high": 3.2}},
     "座位鎖定時離前牆上限 3.2 m 要小於主位到前牆的距離 3.2 m（聆聽距離由座位推出，要永遠為正）"),
])
def test_creation_rejects_each_geometric_contradiction_before_mkdir(
    tmp_path: Path, changes: dict[str, object], message: str,
) -> None:
    project = reference_project(tmp_path)
    root = tmp_path / "searches"
    with pytest.raises(SchemeValidationError) as caught:
        create(root, project, **changes)
    assert caught.value.problems[0].message == message
    assert not root.exists()


@pytest.mark.parametrize("error,accepted", [(0.0, True), (4e-12, True), (1e-11, False)])
def test_axis_contact_margin_accepts_tail_error_rejects_outside(
    tmp_path: Path, error: float, accepted: bool,
) -> None:
    project = reference_project(tmp_path)
    receivers = project.receiver_set.model_dump()
    for point in receivers["points"]:
        point["position_m"] = (point["position_m"][0], 0.999, point["position_m"][2])
    project = Scheme.model_validate(project.model_dump() | {"receiver_set": receivers})
    offset = -1.001 + error  # 2+(-1.001) 的最後一位不等於 .999。
    if accepted:
        store = create(tmp_path / "searches", project, axis_offset_m=offset)
        assert SearchStore.open(store.path).project == project
    else:
        with pytest.raises(SchemeValidationError) as caught:
            create(tmp_path / "searches", project, axis_offset_m=offset)
        assert caught.value.problems[0].message == (
            f"座位鎖定時中軸要通過主位：主位y 0.999 m，推出偏移 -1.001 m；設定寫 {offset!r} m")
        assert not (tmp_path / "searches").exists()


def test_cli_supplies_omitted_ear_height(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = reference_project(tmp_path)
    project_path, settings_path = tmp_path / "project", tmp_path / "settings"
    project_path.write_text(project.model_dump_json())
    settings_path.write_text(locked_settings().model_copy(update={"purpose": project.purpose}).model_dump_json(
        exclude={"layout": {"ear_height_m"}}))
    identity = SearchIdentity("phys-test", "calc-test", purpose_settings(project.purpose))
    monkeypatch.setattr(cli, "_identity", lambda purpose, capabilities: identity)
    store = cli._create(argparse.Namespace(project=project_path, settings=settings_path,
        root=tmp_path / "searches", capabilities=tmp_path / "unused"))
    assert store.settings.layout.ear_height_m.hex() == 1.2.hex()
    assert SearchStore.open(store.path).settings.fingerprint == store.settings.fingerprint


def test_locked_seat_in_keep_out_is_an_input_error(tmp_path: Path) -> None:
    project = reference_project(tmp_path)
    keep_out = [{"x": {"low": 3.0, "high": 3.5}, "y": {"low": 1.5, "high": 2.25},
                 "z": {"low": 1.0, "high": 1.4}}]
    with pytest.raises(SchemeValidationError) as caught:
        create(tmp_path / "searches", project, keep_out=keep_out)
    assert caught.value.problems[0].message == "座位鎖定時座位 main 在門或走道禁區內：0.20000000000000018 m"
    assert not (tmp_path / "searches").exists()


def test_locked_following_furniture_in_keep_out_is_an_input_error(tmp_path: Path) -> None:
    from tests.engine._furniture_cases import relative_item

    project = reference_project(tmp_path)
    # 底面中心 (4.2,1.9,0)，盒 x=[3.95,4.45]、y=[1.4,2.4]；禁區向 x 穿入 .2。
    item = relative_item(furniture_id="seat", placement={"forward_m": -1.0, "left_m": 0.0,
        "bottom_height_m": 0.0, "yaw_deg": 0})
    project = Scheme.model_validate(project.model_dump() | {"furniture": [item]})
    keep_out = [{"x": {"low": 4.25, "high": 4.5}, "y": {"low": 1.0, "high": 3.0},
                 "z": {"low": 0.0, "high": 1.0}}]
    with pytest.raises(SchemeValidationError) as caught:
        create(tmp_path / "searches", project, keep_out=keep_out)
    assert caught.value.problems[0].message == "座位鎖定時家具 seat 在門或走道禁區內：0.20000000000000018 m"
    assert not (tmp_path / "searches").exists()


@pytest.mark.parametrize("front,spacing", [({"low": 0.25, "high": 0.8}, {"low": 0.5, "high": 2.0}),
                                           ({"low": 1.0, "high": 1.5}, {"low": 2.1, "high": 2.5})])
def test_desk_with_no_possible_pair_of_acoustic_centers_is_rejected(
    tmp_path: Path, front: dict[str, float], spacing: dict[str, float],
) -> None:
    project, settings = geometric("desk")  # 固定桌面 x=[.9,1.7]、y=[1,3]。
    project = project.model_copy(update={"speakers": {key: Point(1.3, 1.4 if key == "left" else 2.6, point.z)
        for key, point in project.speakers.items()}})
    changes = {"axis_offset_m": 0.0, "front_distance_m": front, "spacing_m": spacing,
               "speaker_height_m": settings.speaker_height_m, "cabinet": settings.cabinet}
    with pytest.raises(SchemeValidationError) as caught:
        create(tmp_path / "searches", project, **changes)
    assert caught.value.problems[0].message == "座位鎖定時離前牆與間距範圍內，沒有任何一組能讓兩支喇叭的聲學中心落在桌面頂矩形上方"
    assert not (tmp_path / "searches").exists()


def test_desk_possible_centers_do_not_trim_search_ranges(tmp_path: Path) -> None:
    project, settings = geometric("desk")
    project = project.model_copy(update={"speakers": {key: Point(1.3, 1.4 if key == "left" else 2.6, point.z)
        for key, point in project.speakers.items()}})
    store = create(tmp_path / "searches", project, axis_offset_m=0.0,
        front_distance_m={"low": 0.25, "high": 1.95}, speaker_height_m=settings.speaker_height_m,
        cabinet=settings.cabinet)
    assert store.settings.layout.front_distance_m.low == 0.25
    assert store.settings.layout.front_distance_m.high == 1.95


def test_front_upper_bound_checks_placement_rounding_not_distance_comparison(tmp_path: Path) -> None:
    project = reference_project(tmp_path)
    receivers = project.receiver_set.model_dump()
    for point in receivers["points"]:
        point["position_m"] = (5.0, point["position_m"][1], point["position_m"][2])
    project = Scheme.model_validate(project.model_dump() | {"receiver_set": receivers,
        "speakers": {"left": Point(5.5, 1.3, 1.2), "right": Point(5.5, 2.5, 1.2)}})
    front = 0.9999999999999999  # F<6-5=1，但 fl(6-F)=5，實際推出 L=0。
    assert front < 1.0 and 6.0 - front == 5.0
    with pytest.raises(SchemeValidationError) as caught:
        create(tmp_path / "searches", project, front_wall="xL", front_distance_m={"low": 0.25, "high": front})
    assert caught.value.problems[0].message == (
        "座位鎖定時離前牆上限 0.9999999999999999 m 要小於主位到前牆的距離 1.0 m（聆聽距離由座位推出，要永遠為正）")
    assert not (tmp_path / "searches").exists()
