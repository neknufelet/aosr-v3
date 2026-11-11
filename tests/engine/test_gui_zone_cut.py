"""立面圖畫框的伺服器那一半（#1091）：補滿、材料跟著誰、邊的浮點誤差、只核被切的那一面、拒收的原因。

一律用 109 號房的長寬高考：牆寬 4.41988… 不在 5 公分格上，整數房間會把貼牆邊的問題藏起來。
"""
from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError
from starlette.testclient import TestClient

from aosr.gui.app import GuiSettings, create_app
from aosr.gui.zone_edit import MIN_SIDE_M, ROOM_ERROR, SHAPE_ERROR, CutBox, WallName, ZoneCutError, cut_wall, zone_cut_defaults
from aosr.reporting.scheme import Scheme
from tests.engine._furniture_cases import reference_document
from tests.engine._surface_zone_cases import NEW, OLD, ROOM_109, room109_document, room109_quarters

LY, LZ = ROOM_109["Ly"], ROOM_109["Lz"]


def _zones(document: dict[str, object]) -> dict[str, list[dict[str, object]]]:
    return cast(dict[str, list[dict[str, object]]], cast(dict[str, object], document["scene"])[NEW])


def _cut(document: dict[str, object], wall: WallName, boxes: list[dict[str, object]]) -> dict[str, object]:
    return cut_wall(document, wall, [CutBox.model_validate(box) for box in boxes])


def _applied(document: dict[str, object], wall: str, result: dict[str, object]) -> dict[str, object]:
    changed = deepcopy(document)
    _zones(changed)[wall] = cast(list[dict[str, object]], result["zones"])
    return changed


def _by_id(result: dict[str, object]) -> dict[str, dict[str, object]]:
    return {str(zone["id"]): zone for zone in cast(list[dict[str, object]], result["zones"])}


def _impedance(zone: dict[str, object]) -> float:
    return cast(float, cast(dict[str, object], zone["material"])["impedance_pa_s_m"])


def test_one_box_on_a_whole_wall_fills_the_rest_with_the_old_material() -> None:
    document = room109_document()
    material = deepcopy(_zones(document)["x0"][0]["material"])
    result = _cut(document, "x0", [{"id": "z1", "u_m": [0.0, 1.0], "v_m": [0.0, 1.2]}])
    zones = _by_id(result)
    assert list(zones) == ["z1", "auto1", "auto2"] and result["filled_ids"] == ["auto1", "auto2"]
    assert zones["auto1"]["u_m"] == [1.0, LY] and zones["auto1"]["v_m"] == [0.0, LZ]
    assert zones["auto2"]["u_m"] == [0.0, 1.0] and zones["auto2"]["v_m"] == [1.2, LZ]
    assert all(zone["material"] == material and zone["area"] == "rectangle" for zone in zones.values())
    assert result["inherited"] == {"z1": "whole_wall", "auto1": "whole_wall", "auto2": "whole_wall"}
    Scheme.model_validate(_applied(document, "x0", result))


def test_edges_a_hair_apart_are_pulled_together_instead_of_leaving_a_sliver() -> None:
    """網頁打 0.4 加寬 0.8 得 1.2000000000000002，貼著另一框的 1.2；貼牆的框多出 1e-9 也算貼牆。"""
    document = room109_document()
    lx = ROOM_109["Lx"]
    result = _cut(document, "y0", [{"id": "a", "u_m": [0.4, 0.4 + 0.8], "v_m": [0.0, 1.0]},
                                   {"id": "b", "u_m": [1.2, 2.0], "v_m": [0.0, 1.0]},
                                   {"id": "c", "u_m": [5.0, lx + 1e-9], "v_m": [2.0, LZ - 1e-9]}])
    edges = {key: (cast(list[float], zone["u_m"]), cast(list[float], zone["v_m"])) for key, zone in _by_id(result).items()}
    assert edges["a"][0][1] == edges["b"][0][0]
    assert edges["c"][0][1] == lx and edges["c"][1][1] == LZ
    Scheme.model_validate(_applied(document, "y0", result))


def test_a_side_that_is_min_side_up_to_float_error_is_not_too_small() -> None:
    """0.7 − 0.5 在浮點是 0.19999999999999996：畫的框與補出來的縫都不能因此被說成不到最小邊。"""
    assert 0.7 - 0.5 < MIN_SIDE_M
    document = room109_document()
    for boxes in ([{"id": "a", "u_m": [0.5, 0.7], "v_m": [0.0, LZ]}],
                  [{"id": "a", "u_m": [0.0, 0.5], "v_m": [0.0, LZ]}, {"id": "b", "u_m": [0.7, 2.0], "v_m": [0.0, LZ]}]):
        Scheme.model_validate(_applied(document, "x0", _cut(document, "x0", cast(list[dict[str, object]], boxes))))


def test_materials_follow_the_same_id_else_the_old_zone_under_the_centre() -> None:
    document = room109_document()
    room109_quarters(document, "x0", (100.0, 200.0, 300.0, 400.0))
    old = {str(zone["id"]): zone for zone in _zones(document)["x0"]}
    # u0v0 搬到 u1v1 那一角也還是它自己的材料；新框 z1 的中心落在 u0v1；補出來的每一格看自己的中心。
    result = _cut(document, "x0", [{"id": "u0v0", "u_m": [3.0, 4.0], "v_m": [2.0, LZ]},
                                   {"id": "z1", "u_m": [0.0, 1.0], "v_m": [2.0, LZ]}])
    zones = _by_id(result)
    inherited = cast(dict[str, str], result["inherited"])
    assert _impedance(zones["u0v0"]) == 100.0 and inherited["u0v0"] == "u0v0"
    assert _impedance(zones["z1"]) == 200.0 and inherited["z1"] == "u0v1"
    for zone_id in cast(list[str], result["filled_ids"]):
        zone = zones[zone_id]
        u, v = sum(cast(list[float], zone["u_m"])) / 2, sum(cast(list[float], zone["v_m"])) / 2
        under = next(key for key, entry in old.items()
                     if cast(list[float], entry["u_m"])[0] <= u <= cast(list[float], entry["u_m"])[1]
                     and cast(list[float], entry["v_m"])[0] <= v <= cast(list[float], entry["v_m"])[1])
        assert inherited[zone_id] == under and zone["material"] == old[under]["material"]
    Scheme.model_validate(_applied(document, "x0", result))


def test_a_wall_can_be_recut_while_another_wall_no_longer_fits_the_room() -> None:
    """房間改矮以後 x 起點牆與 x 終點牆的 2×2 都超出牆面：先切哪一面都要切得動（只核被切的那一面），
    清掉一面也做得到（中心點的材料照舊找得到）；兩面都修好整份方案才過。"""
    document = room109_document()
    room109_quarters(document, "x0", (100.0, 200.0, 300.0, 400.0))
    room109_quarters(document, "xL", (500.0, 600.0, 700.0, 800.0))
    cast(dict[str, float], cast(dict[str, object], document["scene"])["room_m"])["Lz"] = 2.5
    with pytest.raises(ValidationError):
        Scheme.model_validate(document)
    cleared = _cut(document, "xL", [])
    assert cleared["zones"] == [{"area": "whole_wall", "material": _zones(document)["xL"][0]["material"]}]
    assert cleared["inherited"] == {"whole_wall": "u0v0"}
    document = _applied(document, "xL", cleared)
    recut = _cut(document, "x0", [{"id": "z1", "u_m": [0.0, 1.0], "v_m": [0.0, 1.0]}])
    assert all(cast(list[float], zone["v_m"])[1] <= 2.5 for zone in cast(list[dict[str, object]], recut["zones"]))
    Scheme.model_validate(_applied(document, "x0", recut))


def test_clearing_a_wall_whose_zones_miss_the_centre_takes_the_nearest_zone() -> None:
    """房間改高到 8 公尺：牆中心（z＝4）沒有任何舊分區蓋到，取最近的上排第一區 u0v1。"""
    document = room109_document()
    room109_quarters(document, "x0", (100.0, 200.0, 300.0, 400.0))
    cast(dict[str, float], cast(dict[str, object], document["scene"])["room_m"])["Lz"] = 8.0
    cleared = _cut(document, "x0", [])
    assert cleared["inherited"] == {"whole_wall": "u0v1"}
    assert cleared["zones"] == [{"area": "whole_wall", "material": {"kind": "constant_impedance", "impedance_pa_s_m": 200.0}}]


def _room_without_height() -> dict[str, object]:
    document = room109_document()
    cast(dict[str, object], cast(dict[str, object], document["scene"])["room_m"])["Lz"] = None
    return document


def _six_numbers() -> dict[str, object]:
    return reference_document() | {"scene": cast(dict[str, object], reference_document()["scene"])}


REFUSED = {
    "auto id": ([{"id": "auto1", "u_m": [0.0, 1.0], "v_m": [0.0, 1.0]}], "auto 開頭"),
    "overlap": ([{"id": "a", "u_m": [0.0, 1.0], "v_m": [0.0, 1.0]}, {"id": "b", "u_m": [0.5, 1.5], "v_m": [0.5, 1.5]}], "矩形重疊"),
    "outside": ([{"id": "a", "u_m": [0.0, LY + 0.5], "v_m": [0.0, 1.0]}], "超出牆面範圍"),
    "duplicate": ([{"id": "a", "u_m": [0.0, 1.0], "v_m": [0.0, 1.0]}, {"id": "a", "u_m": [2.0, 3.0], "v_m": [0.0, 1.0]}], "代號重複"),
    "too small": ([{"id": "a", "u_m": [0.0, 0.15], "v_m": [0.0, 1.0]}], "最小邊"),
}


@pytest.mark.parametrize("case", sorted(REFUSED))
def test_refused_cuts_say_why(case: str) -> None:
    boxes, words = REFUSED[case]
    with pytest.raises(ZoneCutError, match=words):
        _cut(room109_document(), "x0", cast(list[dict[str, object]], boxes))


def test_a_thin_strip_left_by_the_fill_is_refused_with_its_place() -> None:
    with pytest.raises(ZoneCutError, match="窄條") as caught:
        _cut(room109_document(), "x0", [{"id": "a", "u_m": [0.1, 1.0], "v_m": [0.0, LZ]}])
    assert [(box.u_m, box.v_m) for box in caught.value.thin] == [((0.0, 0.1), (0.0, LZ))]


@pytest.mark.parametrize("document,message", [(_room_without_height(), ROOM_ERROR), (_six_numbers(), SHAPE_ERROR)],
                         ids=["room half filled", "six numbers not converted"])
def test_refused_documents_say_why(document: dict[str, object], message: str) -> None:
    assert OLD in cast(dict[str, object], _six_numbers()["scene"])
    with pytest.raises(ZoneCutError) as caught:
        _cut(document, "x0", [{"id": "a", "u_m": [0.0, 1.0], "v_m": [0.0, 1.0]}])
    assert str(caught.value) == message


def test_route_cuts_refuses_and_publishes_its_numbers(tmp_path: Path) -> None:
    with TestClient(create_app(GuiSettings(engine_commit="a" * 40, data_dir=tmp_path)), base_url="http://127.0.0.1") as client:
        assert client.get("/api/input-defaults").json()["zone_cut"] == zone_cut_defaults()
        document = room109_document()
        body = {"scheme": document, "wall": "xL", "boxes": [{"id": "z1", "u_m": [3.4, LY], "v_m": [0.0, 1.2]}]}
        response = client.post("/api/zone-cut", json=body)
        assert response.status_code == 200, response.text
        assert response.json() == _cut(document, "xL", [{"id": "z1", "u_m": [3.4, LY], "v_m": [0.0, 1.2]}])
        thin = client.post("/api/zone-cut", json=body | {"boxes": [{"id": "a", "u_m": [0.1, 1.0], "v_m": [0.0, LZ]}]})
        assert thin.status_code == 400 and "窄條" in thin.json()["error"], thin.text
        assert thin.json()["thin"] == [{"u_m": [0.0, 0.1], "v_m": [0.0, LZ]}]
        assert [row["text"] for row in thin.json()["problems"]] == [thin.json()["error"]]
        for bad in ({"scheme": document, "boxes": []}, {**body, "wall": "door"},
                    {**body, "boxes": [{"id": "a.b", "u_m": [0.0, 1.0], "v_m": [0.0, 1.0]}]}):
            refused = client.post("/api/zone-cut", json=bad)
            assert refused.status_code == 400 and "格式不完整" in refused.json()["error"], refused.text
        assert client.post("/api/zone-cut", content=b"not json", headers={"Content-Type": "application/json"}).status_code == 400
