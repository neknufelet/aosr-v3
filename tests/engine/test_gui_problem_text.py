"""方案檢查不過的問題：表單中文欄名、白話、同一句合併、原路徑留著；檢查、存檔、開算、重算都一樣。"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import cast

import pytest
from starlette.testclient import TestClient

from tests.engine._gui_cache import gui_startup_identity_memo
from aosr.gui.app import STATIC, GuiSettings, create_app
from aosr.gui.labels import LISTENING_POINTS, ROOM_LENGTHS, SPEAKERS, WALLS
from aosr.gui.problem_text import plain_problems
from aosr.reporting.validation import SchemeProblem


COMMIT = "a" * 40
RUN_ID = "b" * 32
KEYS = {"text", "message", "fields", "paths", "details"}
# 英文字連三個以上＝欄位路徑或英文訊息漏到畫面上（長 Lx、x 座標這種單位與軸名只有一兩個字母）。
ENGLISH = re.compile(r"[A-Za-z_]{3,}")


def _client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(GuiSettings(engine_commit=COMMIT, data_dir=tmp_path)),
                      base_url="http://localhost")


def _example(client: TestClient, scheme_id: str = "demo") -> dict[str, object]:
    document: dict[str, object] = client.get("/api/example").json()["scheme"]
    document["scheme_id"] = scheme_id
    return document


def _cell(document: dict[str, object], *keys: str) -> dict[str, object]:
    for key in keys:
        document = cast(dict[str, object], document[key])
    return document


def _points(document: dict[str, object]) -> list[dict[str, object]]:
    return cast(list[dict[str, object]], _cell(document, "receiver_set")["points"])


def _speaker_on_primary(document: dict[str, object]) -> None:
    primary = next(point for point in _points(document) if point["role"] == "primary")
    x, y, z = cast(list[float], primary["position_m"])
    _cell(document, "speakers")["left"] = {"x": x, "y": y, "z": z}


def _blank_cells(document: dict[str, object]) -> int:
    """清空表單上四種格子各一格；回傳被清空 y 座標的那個周圍點（主位前方）排第幾。"""
    _cell(document, "scene", "room_m")["Ly"] = None
    _cell(document, "scene", "impedance_pa_s_per_m_by_wall")["floor"] = None
    _cell(document, "speakers", "right")["y"] = None
    front = next(index for index, point in enumerate(_points(document)) if point["receiver_id"] == "front")
    cast(list[object], _points(document)[front]["position_m"])[1] = None
    return front


def _plain_shape(problems: list[dict[str, object]]) -> None:
    assert problems and all(set(item) == KEYS for item in problems)
    messages = [item["message"] for item in problems]
    assert messages == list(dict.fromkeys(messages))
    assert not [item["text"] for item in problems if ENGLISH.search(cast(str, item["text"]))]


def test_blank_cells_share_one_line_with_the_form_names(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        document = _example(client)
        front = _blank_cells(document)
        problems = client.post("/api/validate", json=document).json()["problems"]
    _plain_shape(problems)
    assert [(item["message"], set(item["fields"])) for item in problems] == [
        ("空著沒填", {ROOM_LENGTHS["Ly"], f"{WALLS['floor']}阻抗", f"{SPEAKERS['right']} y 座標",
                   f"{LISTENING_POINTS['front']} y 座標"})]
    assert set(problems[0]["paths"]) == {"scene.room_m.Ly", "scene.impedance_pa_s_per_m_by_wall.floor",
                                         "speakers.right.y", f"receiver_set.points.{front}.position_m.1"}
    assert all(detail.endswith("：必填") for detail in problems[0]["details"])


def test_same_problem_for_every_pair_is_said_once(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        document = _example(client)
        _speaker_on_primary(document)
        problems = client.post("/api/validate", json=document).json()["problems"]
        seats = {point["receiver_id"] for point in _points(document)}
    _plain_shape(problems)
    assert [(item["fields"], item["text"]) for item in problems] == [
        ([SPEAKERS["left"]], f"{SPEAKERS['left']}：跟主位放在同一點（產品預設指向讓喇叭對準主位，兩者不能重合）")]
    # 原本每一對各一條：路徑全留著，給考卷與技術細節。
    assert set(problems[0]["paths"]) == {f"pairs.left.{seat}.source_model.aim_m" for seat in seats}


def test_check_save_run_and_rerun_all_send_the_plain_form(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        document = _example(client)
        assert client.put("/api/schemes/demo", json=document).status_code == 200
        _speaker_on_primary(document)
        # 表單上的檢查與存檔：還沒存的表單內容。
        for response in (client.post("/api/plan", json=document),
                         client.put("/api/schemes/demo", json=document),
                         client.put("/api/schemes/other", json={**document, "scheme_id": "other"},
                                    headers={"If-None-Match": "*"})):
            assert response.status_code == 422
            _plain_shape(response.json()["problems"])
            assert response.json()["problems"][0]["fields"] == [SPEAKERS["left"]]
        # 開算與重算：磁碟上的方案（開算讀方案檔、重算讀結果裡的方案快照）。
        (tmp_path / "schemes" / "demo.json").write_text(json.dumps(document), encoding="utf-8")
        (tmp_path / "results" / f"{RUN_ID}.json").write_text(json.dumps({"scheme": document}),
                                                              encoding="utf-8")
        for response in (client.post("/api/runs", json={"scheme_id": "demo"}),
                         client.post(f"/api/results/{RUN_ID}/rerun", json={})):
            assert response.status_code == 422
            _plain_shape(response.json()["problems"])
            assert response.json()["problems"][0]["fields"] == [SPEAKERS["left"]]


def test_scheme_that_fails_the_first_check_still_gets_form_names(tmp_path: Path) -> None:
    # 方案本身就驗不過（讀檔那一步就被擋）：一樣寫表單中文欄名，座位用順序號查回是哪一個座位。
    with _client(tmp_path) as client:
        document = _example(client)
        _blank_cells(document)
        (tmp_path / "schemes" / "demo.json").write_text(json.dumps(document), encoding="utf-8")
        (tmp_path / "results" / f"{RUN_ID}.json").write_text(json.dumps({"scheme": document}),
                                                              encoding="utf-8")
        for response in (client.get("/api/schemes/demo"), client.get("/api/plan/demo"),
                         client.post("/api/runs", json={"scheme_id": "demo"}),
                         client.post(f"/api/results/{RUN_ID}/rerun", json={})):
            assert response.status_code == 422
            _plain_shape(response.json()["problems"])
            assert f"{LISTENING_POINTS['front']} y 座標" in response.json()["problems"][0]["fields"]


def test_speaker_outside_room_names_the_speaker_not_the_whole_scheme(tmp_path: Path) -> None:
    # 檢查把「喇叭跑出房間」掛在整份方案上、喇叭寫在原文裡：欄名改寫那支喇叭，訊息說它比的是座標和房間長寬高。
    with _client(tmp_path) as client:
        document = _example(client)
        _cell(document, "speakers", "left")["x"] = 99
        problems = client.post("/api/validate", json=document).json()["problems"]
    _plain_shape(problems)
    assert [(item["fields"], item["paths"]) for item in problems] == [([SPEAKERS["left"]], ["scheme"])]
    assert "房間長寬高" in problems[0]["message"]


def test_unknown_paths_and_messages_are_kept_readable() -> None:
    problems = plain_problems((SchemeProblem("mystery.shelf.0", "某個新的檢查訊息"),
                               SchemeProblem("scheme", "某個新的檢查訊息"),
                               SchemeProblem("scheme", "另一個整份方案的訊息")), None)
    assert [(item["message"], item["fields"], item["paths"]) for item in problems] == [
        ("某個新的檢查訊息", ["mystery › shelf › 第 1 個", "整份方案"], ["mystery.shelf.0", "scheme"]),
        ("另一個整份方案的訊息", [], ["scheme"])]
    assert problems[1]["text"] == "另一個整份方案的訊息"


def test_without_the_document_seats_fall_back_to_their_order() -> None:
    problems = plain_problems((SchemeProblem("receiver_set.points.3.position_m.2", "必填"),), None)
    assert problems[0]["text"] == "第 4 個座位 z 座標：空著沒填"


@pytest.mark.parametrize("case,ids", [("both", "desk"), ("two_block", "a-second、z-first")])
def test_furniture_problem_uses_pair_fields_and_preserves_furniture_ids(case: str, ids: str) -> None:
    from aosr.reporting.validation import validate_scheme
    from tests.engine._furniture_cases import CAPABILITIES
    from tests.engine._directivity import DIRECTIVITY
    from tests.engine.test_scheme_furniture import _validation_document

    document = _validation_document(case)
    written = validate_scheme(document, capabilities=CAPABILITIES, directivity=DIRECTIVITY)
    problems = plain_problems(written, document)
    message = f"不符合擺位要求：直達路徑被家具 {ids} 擋住"
    assert [(item["fields"], item["paths"], item["message"], item["text"], item["details"])
            for item in problems] == [
        ([label], [problem.path], message, f"{label}：{message}", [str(problem)])
        for label, problem in zip(("左聲道喇叭 → 主位", "左聲道喇叭 → 座位 side"), written, strict=True)]


def test_field_names_are_the_words_on_the_input_form() -> None:
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    assert all(name in script for name in ROOM_LENGTHS.values())
    assert all(f'{wall}: "{name}"' in script for wall, name in WALLS.items())


def test_pages_print_the_server_line_not_the_english_path() -> None:
    # 三頁（方案輸入頁的檢查與存檔、結果頁與比較頁的重算）都只印伺服器寫好的那一行。
    for name in ("app.js", "results.js", "compare.js"):
        script = (STATIC / name).read_text(encoding="utf-8")
        assert re.search(r"problems \|\| \[\]\)\.map\(\((\w+)\) => \1\.text\)", script), name
        assert not re.search(r"\b(item|problem)\.path\b", script), name


def test_blocked_pair_field_comes_from_the_message_when_the_seat_id_has_a_dot() -> None:
    # 座位代號可含點：pairs.left.side.a 用點切路徑會變成「座位 side」，欄名要從原句的代號取。
    from aosr.reporting.validation import validate_scheme
    from tests.engine._furniture_cases import CAPABILITIES
    from tests.engine._directivity import DIRECTIVITY
    from tests.engine.test_scheme_furniture import _validation_document

    document = _validation_document("both")
    receiver_set = cast(dict[str, object], document["receiver_set"])
    for point in cast(list[dict[str, object]], receiver_set["points"]):
        if point["receiver_id"] == "side":
            point["receiver_id"] = "side.a"
    written = validate_scheme(document, capabilities=CAPABILITIES, directivity=DIRECTIVITY)
    assert [problem.path for problem in written] == ["pairs.left.main", "pairs.left.side.a"]
    assert [item["text"] for item in plain_problems(written, document)] == [
        "左聲道喇叭 → 主位：不符合擺位要求：直達路徑被家具 desk 擋住",
        "左聲道喇叭 → 座位 side.a：不符合擺位要求：直達路徑被家具 desk 擋住"]
