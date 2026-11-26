"""輸入頁第 1 段右卡「這間房現在」四句與姿勢（#1265）：走正式的 /api/input-preview，答案逐句手寫。

房間用 cases.document 的四種面向（喇叭靠哪一面牆），尺寸改成長寬不同，才分得出「寬」「深」各取哪一軸。"""
import re
from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest
from starlette.testclient import TestClient

import aosr.gui
from aosr.gui.room_facts import metres
from tests.engine import _furniture_cases as cases
from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine.test_gui_app import _app
from tests.engine.test_gui_input_setup import edit

Doc = dict[str, object]


def _room(facing: str, primary: tuple[float, float, float] = (3.0, 3.0, 1.2)) -> Doc:
    document = cases.document(facing=facing, primary=primary)
    cast(Doc, document["scene"])["room_m"] = {"Lx": 7.0, "Ly": 6.0, "Lz": 3.2}
    return document


def _facts(client: TestClient, document: Doc) -> dict[str, str | None]:
    response = client.post("/api/input-preview", json=document)
    assert response.status_code == 200, response.text
    return cast(dict[str, str | None], response.json()["room_facts"])


@pytest.mark.parametrize("facing,primary,room,speakers,seat", [
    # 喇叭靠 x 起點牆：深是 x 那一軸（長 7）、寬是 y（6）；每個數後面寫是哪一格；主位 y=3 正好在兩支 y=2、4 的中間。
    ("west", (3.0, 3.0, 1.2), "寬 6（Ly）× 深 7（Lx）× 高 3.2 公尺", "兩支在前牆前 1 公尺，相距 2 公尺，高 1.2 公尺",
     "離前牆 3 公尺，在兩支喇叭的中線上，耳朵高 1.2 公尺"),
    # 靠 x 終點牆（x=7）：離前牆從另一頭量（7−5＝2、7−3＝4）；面向 +x，左手邊是 +y，主位 y=2.5 比中線 3 小＝偏右。
    ("east", (3.0, 2.5, 1.2), "寬 6（Ly）× 深 7（Lx）× 高 3.2 公尺", "兩支在前牆前 2 公尺，相距 2 公尺，高 1.2 公尺",
     "離前牆 4 公尺，在兩支喇叭的中線右邊 0.5 公尺，耳朵高 1.2 公尺"),
    # 靠 y 起點牆：深是 y（6）、寬是 x（7）；面向 −y，左手邊是 +x，主位 x=3.25 比中線 3 大＝偏左。
    ("south", (3.25, 3.0, 1.05), "寬 7（Lx）× 深 6（Ly）× 高 3.2 公尺", "兩支在前牆前 1 公尺，相距 2 公尺，高 1.2 公尺",
     "離前牆 3 公尺，在兩支喇叭的中線左邊 0.25 公尺，耳朵高 1.05 公尺"),
    # 靠 y 終點牆（y=6）：6−5＝1、6−3＝3。
    ("north", (3.0, 3.0, 1.2), "寬 7（Lx）× 深 6（Ly）× 高 3.2 公尺", "兩支在前牆前 1 公尺，相距 2 公尺，高 1.2 公尺",
     "離前牆 3 公尺，在兩支喇叭的中線上，耳朵高 1.2 公尺"),
])
def test_four_lines_follow_the_front_wall(tmp_path: Path, facing: str, primary: tuple[float, float, float],
                                         room: str, speakers: str, seat: str) -> None:
    with _app(tmp_path) as client:
        facts = _facts(client, _room(facing, primary))
    assert facts == {"room": room, "speakers": speakers, "primary": seat, "furniture": "沒有家具", "posture": None}


def test_speakers_at_different_distances_and_heights_name_each_one(tmp_path: Path) -> None:
    document = _room("west")
    cast(dict[str, Doc], document["speakers"])["right"] |= {"x": 1.25, "z": 0.95}
    with _app(tmp_path) as client:
        facts = _facts(client, document)
    # 相距照水平距離（√(0.25²+2²)≈2.02）；離前牆、高度不一樣就逐支寫名字。
    assert facts["speakers"] == ("兩支在前牆前 左聲道喇叭 1、右聲道喇叭 1.25 公尺，相距 2.02 公尺，"
                                 "高 左聲道喇叭 1.2、右聲道喇叭 0.95 公尺")


def test_moving_a_speaker_or_the_seat_changes_its_line(tmp_path: Path) -> None:
    """右卡的字從資料來：同一份方案只改喇叭的 x 或主位的 x，那一行跟著變、其他行不動。"""
    document = _room("west")
    with _app(tmp_path) as client:
        before = _facts(client, document)
        moved = deepcopy(document)
        for point in cast(dict[str, Doc], moved["speakers"]).values():
            point["x"] = 0.75
        after_speakers = _facts(client, moved)
        seated = deepcopy(document)
        cast(list[Doc], cast(Doc, seated["receiver_set"])["points"])[0]["position_m"] = [3.5, 3.0, 1.2]
        after_seat = _facts(client, seated)
    assert after_speakers["speakers"] == "兩支在前牆前 0.75 公尺，相距 2 公尺，高 1.2 公尺"
    assert {key: after_speakers[key] for key in ("room", "primary")} == {key: before[key] for key in ("room", "primary")}
    assert after_seat["primary"] == "離前牆 3.5 公尺，在兩支喇叭的中線上，耳朵高 1.2 公尺"
    assert after_seat["speakers"] == before["speakers"]


def test_an_unfilled_room_says_so_and_keeps_the_furniture_line(tmp_path: Path) -> None:
    document = _room("west")
    cast(Doc, cast(Doc, document["scene"])["room_m"])["Ly"] = None
    with _app(tmp_path) as client:
        facts = _facts(client, document)
    assert facts == {"room": None, "speakers": None, "primary": None, "furniture": "沒有家具", "posture": None}


def test_a_seat_not_facing_the_speaker_wall_falls_back_to_coordinates(tmp_path: Path) -> None:
    """喇叭靠 x 起點牆、主位卻面向 y 終點牆（喇叭中點在主位前方偏 y）：定不出前牆就不猜，房間寫長寬、主位寫座標。"""
    with _app(tmp_path) as client:
        facts = _facts(client, _room("west", primary=(1.5, 0.5, 1.2)))
    assert facts["room"] == "長 7 × 寬 6 × 高 3.2 公尺"
    assert facts["speakers"] == "兩支相距 2 公尺，高 1.2 公尺（分不出前牆，位置看「座標數字」）"
    assert facts["primary"] == "在 x 1.5、y 0.5，耳朵高 1.2 公尺（分不出前牆）"


@pytest.mark.parametrize("name,furniture", [("working", "書桌"), ("listening", "沙發、茶几")])
def test_a_shortcut_marks_its_posture_and_names_the_furniture(tmp_path: Path, name: str, furniture: str) -> None:
    with _app(tmp_path) as client:
        document = edit(client, _room("west"), "shortcut", name)
        facts = _facts(client, document)
        assert facts["posture"] == name and facts["furniture"] == furniture
        # 主位高度跟快捷不一樣（手改過）：哪一個都不算按下。
        primary = cast(list[Doc], cast(Doc, document["receiver_set"])["points"])[0]
        primary["position_m"] = [*cast(list[float], primary["position_m"])[:2], 1.15]
        assert _facts(client, document)["posture"] is None


def test_metres_round_to_centimetres_and_drop_trailing_zeros() -> None:
    assert [metres(value) for value in (3.0, 0.5, 1.206, 2.0049, -0.004)] == ["3", "0.5", "1.21", "2", "0"]


# ---- #1265 單元 2 審查修補（build-brief-4b-u2fix.md）----


@pytest.mark.parametrize("size", [0, -1.5, True, "abc"])
def test_a_half_typed_or_non_number_room_size_is_not_filled(tmp_path: Path, size: object) -> None:
    """審查第 6 條留下的那一格：長寬高打到一半（0、負數；表單送得出來）或不是數字（真假值、字串；表單送不出來，
    /api/input-preview 收得到，入口不核這三格的型別）：房間那一句寫還沒填完，不把真假值當成 1 公尺。"""
    document = _room("west")
    cast(Doc, cast(Doc, document["scene"])["room_m"])["Ly"] = size
    with _app(tmp_path) as client:
        facts = _facts(client, document)
    assert facts["room"] is None


def test_the_finished_run_states_are_written_in_one_place() -> None:
    """審查設計洞 1 前半：結果「算好了」那一份狀態（done、none）整個網頁程式只寫一次（app.js 的 runFinished），
    第 1 段右卡結果那一行與結果清單、頁底條鎖不鎖都叫它；不准哪一支自己再寫一份，改狀態時只改一處兩邊就不會對不上。"""
    static = Path(aosr.gui.__file__).parent / "static"
    sources = {path.name: path.read_text(encoding="utf-8") for path in sorted(static.glob("*.js"))}
    written = [(name, line.strip()) for name, text in sources.items() for line in text.splitlines()
               if re.search(r'"done",\s*"none"', line)]
    assert len(written) == 1, written
    assert "function runFinished(" in sources["app.js"]
    for name in ("app.js", "room_facts.js"):
        assert "runFinished(" in sources[name].replace("function runFinished(", ""), name
