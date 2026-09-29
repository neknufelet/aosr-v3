"""比較端點的成功與拒收路徑。"""
from __future__ import annotations

import csv
import io
import json
from http import HTTPStatus
from pathlib import Path

import pytest
from httpx import Response
from starlette.testclient import TestClient

from aosr.gui.app import GuiSettings, create_app
from aosr.gui.app import STATIC
from aosr.reporting.result import SchemeResult, save_result
from tests.engine.test_gui_compare_view import shorter_room_result
from tests.engine.test_scheme_pipeline import shared_control_result


@pytest.fixture(scope="module")
def pair(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> tuple[SchemeResult, SchemeResult]:
    return (shared_control_result(tmp_path_factory, worker_id, "wall-1"),
            shared_control_result(tmp_path_factory, worker_id, "wall-2"))


def _files(tmp_path: Path, result: SchemeResult, run_id: str) -> None:
    (tmp_path / "schemes" / f"{result.scheme.scheme_id}.json").write_text(result.scheme.model_dump_json())
    save_result(result, tmp_path / "results" / f"{run_id}.json")
    (tmp_path / "runs" / f"{run_id}.json").write_text(json.dumps({
        "run_id": run_id, "scheme_id": result.scheme.scheme_id,
        "status": "done", "started_at": 0, "pid": 0, "exit_code": 0,
        "result_path": str(tmp_path / "results" / f"{run_id}.json"),
        "stderr_path": str(tmp_path / "runs" / f"{run_id}.stderr"),
    }))


def _client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(GuiSettings(engine_commit="a" * 40, data_dir=tmp_path)),
                      base_url="http://localhost", raise_server_exceptions=False)


def _strings(value: object, field: str = "") -> bool:
    if isinstance(value, dict):
        return all(isinstance(key, str) and _strings(item, key) for key, item in value.items())
    if isinstance(value, list):
        return all(_strings(item, field) for item in value)
    if field.endswith("_text") or field in {"run_id", "scheme_id", "path", "label", "role",
                                              "speaker_id", "receiver_id", "receiver_role", "key",
                                              "category", "side", "run_date"}:
        return isinstance(value, str)
    return True


def test_compare_page_and_script_are_served(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        page = client.get("/compare/a/b")
        script = client.get(f"/static/{'compare' + '.js'}")
        unknown = client.get(f"/static/{'unknown' + '.js'}")
    assert page.status_code == HTTPStatus.OK
    assert "正在讀取並核對兩份結果" in page.text
    assert script.status_code == HTTPStatus.OK
    # 兩邊軸不同時聯集軸有空格，不跨空格連線會斷成點；要的是開著，不只是寫了這個字。
    assert "spanGaps: true" in script.text
    assert unknown.status_code == HTTPStatus.NOT_FOUND


def test_compare_script_only_renders_server_values() -> None:
    script = (STATIC / "compare.js").read_text(encoding="utf-8")
    assert all(forbidden not in script for forbidden in ("innerHTML", "Math.log", "Math.pow"))
    assert "spanGaps: true" in script
    assert "/api/compare/" in script
    assert "toBlob" in script


def _csv_rows(response: Response) -> list[list[str]]:
    content = response.content
    assert content.startswith(b"\xef\xbb\xbf")
    return list(csv.reader(io.StringIO(content.decode("utf-8-sig"))))


def test_curves_csv_matches_overlay_values(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult]) -> None:
    a_id, b_id = "b" * 32, "c" * 32
    with _client(tmp_path) as client:
        _files(tmp_path, pair[0], a_id)
        _files(tmp_path, pair[1], b_id)
        data = client.get(f"/api/compare/{a_id}/{b_id}").json()
        response = client.get(f"/api/compare/{a_id}/{b_id}/export/curves")
    assert response.status_code == HTTPStatus.OK
    assert response.headers["content-type"] == "text/csv; charset=utf-8"
    assert response.headers["content-disposition"] == (
        'attachment; filename="compare-bbbbbbbb-cccccccc-curves.csv"')
    rows = _csv_rows(response)
    overlay = data["overlay"]
    assert rows[0] == ["頻率 (Hz)", *(f"{item['legend_text']} (dB)" for item in overlay["series"])]
    assert len(rows[1:]) == len(overlay["frequency_hz"])
    for index, row in enumerate(rows[1:]):
        assert len(row) == len(rows[0])
        assert float(row[0]) == overlay["frequency_hz"][index]
        for cell, series in zip(row[1:], overlay["series"], strict=True):
            level = series["levels_db"][index]
            assert cell == "" if level is None else float(cell) == level


def test_summary_csv_has_three_sections_in_server_words(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult]) -> None:
    a_id, b_id = "b" * 32, "c" * 32
    with _client(tmp_path) as client:
        _files(tmp_path, pair[0], a_id)
        _files(tmp_path, pair[1], b_id)
        data = client.get(f"/api/compare/{a_id}/{b_id}").json()
        response = client.get(f"/api/compare/{a_id}/{b_id}/export/summary")
    assert response.status_code == HTTPStatus.OK
    assert response.headers["content-type"] == "text/csv; charset=utf-8"
    assert response.headers["content-disposition"] == (
        'attachment; filename="compare-bbbbbbbb-cccccccc-summary.csv"')
    sections: list[list[list[str]]] = [[]]
    rows = _csv_rows(response)
    for row in rows:
        if row:
            sections[-1].append(row)
        else:
            sections.append([])
    # 每一段的表頭都說清楚每一欄是什麼：說明句、指紋核對不放在「A」那一欄底下。
    first, overall, changes, fingerprints, categories, notes = sections
    # 總代價那一列：列名寫一次「總代價（越低越好）」，格子只放數字，不再「總代價｜總代價 1.367…」。
    # 計算指紋與程式提交代號（頁面收在技術細節）照樣進 CSV，名字跟結果頁、方案輸入頁同一套；程式不叫引擎。
    assert first == [
        ["欄位", "A", "B"],
        *([label, data["a"][field], data["b"][field]] for label, field in (
            ("方案代號", "scheme_id"), ("計算日期", "run_date"), ("計算時間（全程）", "total_text"))),
        ["總代價（越低越好）", data["table"]["a_cell"], data["table"]["b_cell"]],
        *([label, data["a"][field], data["b"][field]] for label, field in (
            ("計算指紋前 12 碼", "fingerprint_text"), ("程式提交代號", "engine_text"))),
    ]
    assert [row for row in rows if sum("總代價" in cell for cell in row) > 1] == []
    assert data["version_text"] == "兩份相同"
    assert not [row for row in rows if any("引擎" in cell for cell in row)]
    # 摘要那幾句跟頁面一樣：計算版本、能不能直接比、哪一份比較好、複核警戒與還不是最終推薦、兩份都尚未評估的類、校準白話。
    assert overall == [
        ["欄位", "內容"],
        ["計算版本", data["version_text"]],
        ["摘要句", data["summary_text"]],
        ["能不能直接比", data["table"]["reason_text"]],
        ["哪一份比較好", data["table"]["verdict_text"]],
        ["複核與推薦", data["table"]["review_text"]],
        ["尚未評估", data["pending_text"]],
        ["校準說明", data["table"]["calibration_text"]],
        ["音量基準", data["level_note"]],
    ]
    assert "比較好" in data["table"]["verdict_text"] and "尚未評估" in data["pending_text"]
    assert "不能當最終推薦" in data["table"]["review_text"]
    assert changes == [
        ["項目", "A", "B"],
        *([item["label"], item["a_text"], item["b_text"]] for item in data["changes"]),
    ]
    # 指紋前 7 碼只在 CSV（技術細節），頁面那張核對表不印。
    assert fingerprints == [["指紋", "核對", "指紋前 7 碼（A／B）"],
                            *([item["label"], item["text"], item["code_text"]]
                              for item in data["fingerprints"])]
    assert notes == [["說明"], *([note] for note in data["notes"])]
    assert categories == [
        ["類別", "A 狀態", "A 代價", "B 狀態", "B 代價", "哪一份較好", "說明"],
        *([item["label"], item["a"]["state_label"], item["a"]["cost_text"],
           item["b"]["state_label"], item["b"]["cost_text"], item["better_text"], item["note_text"]]
          for item in data["categories"]),
    ]


def test_export_rejects_like_compare(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult]) -> None:
    a_id, b_id = "b" * 32, "c" * 32
    altered = pair[0].model_copy(update={"calculation_fingerprint": "calc-v1:" + "1" * 64})
    with _client(tmp_path) as client:
        _files(tmp_path, pair[0], a_id)
        for left, right in ((a_id, a_id), (a_id, b_id), ("invalid", a_id)):
            normal = client.get(f"/api/compare/{left}/{right}")
            exported = client.get(f"/api/compare/{left}/{right}/export/curves")
            assert (exported.status_code, exported.json()) == (normal.status_code, normal.json())
        _files(tmp_path, altered, b_id)
        normal = client.get(f"/api/compare/{a_id}/{b_id}")
        exported = client.get(f"/api/compare/{a_id}/{b_id}/export/summary")
        assert (exported.status_code, exported.json()) == (normal.status_code, normal.json())
        assert client.get(f"/api/compare/{a_id}/{b_id}/export/unknown").status_code == HTTPStatus.NOT_FOUND


def _nulls_only_in_levels(value: object, field: str = "") -> bool:
    if isinstance(value, dict):
        return all(_nulls_only_in_levels(item, key) for key, item in value.items())
    if isinstance(value, list):
        return all(_nulls_only_in_levels(item, field) for item in value)
    # 圖面沿用 /api/plan：全向點源的 aim 為 null。
    return value is not None or field in {"levels_db", "aim"}


def test_compare_returns_both_sides(tmp_path: Path, pair: tuple[SchemeResult, SchemeResult]) -> None:
    a_id, b_id = "b" * 32, "c" * 32
    with _client(tmp_path) as client:
        _files(tmp_path, pair[0], a_id)
        _files(tmp_path, pair[1], b_id)
        # 結果快照是唯一圖面來源；即使方案存檔損壞也不影響比較。
        (tmp_path / "schemes" / "wall-1.json").write_text("{}")
        (tmp_path / "schemes" / "wall-2.json").write_text("{}")
        response = client.get(f"/api/compare/{a_id}/{b_id}")
    assert response.status_code == HTTPStatus.OK
    data = response.json()
    assert _strings(data)
    assert _nulls_only_in_levels(data)
    assert data["a"]["scheme_id"] == "wall-1" and data["b"]["scheme_id"] == "wall-2"
    selected = [next(row for row in data["overlay"]["series"] if row["key"] == key)
                for key in data["overlay"]["default_keys"]]
    assert {(row["role"], row["receiver_role"]) for row in selected} == {("left", "primary")}
    assert "server-timing" in response.headers


def test_compare_returns_both_plans_on_one_scale(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult],
        tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> None:
    # B 的房間各邊短 10%：共用比例要逐軸取較大那間，兩間一樣大時取大取小都一樣、考不出來。
    pair = (pair[0], shorter_room_result(tmp_path_factory, worker_id))
    a_id, b_id = "b" * 32, "c" * 32
    with _client(tmp_path) as client:
        _files(tmp_path, pair[0], a_id)
        _files(tmp_path, pair[1], b_id)
        # 圖要從結果檔裡的方案快照畫，不是 schemes/ 底下的存檔：存檔拿掉照樣要畫得出來。
        for result in pair:
            (tmp_path / "schemes" / f"{result.scheme.scheme_id}.json").unlink()
        response = client.get(f"/api/compare/{a_id}/{b_id}")
    assert response.status_code == HTTPStatus.OK
    data = response.json()
    for side, result in (("a", pair[0]), ("b", pair[1])):
        plan = data["plans"][side]
        assert {item["key"] for item in plan["speakers"] + plan["receivers"]} == {
            *(f"speaker:{name}" for name in result.scheme.speakers),
            *(f"receiver:{point.receiver_id}" for point in result.scheme.receiver_set.points)}
        room = result.scheme.scene.room_m
        assert plan["room"] == {"Lx": room.Lx, "Ly": room.Ly, "Lz": room.Lz}
    assert data["plan_scale_room"] == {
        axis: max(getattr(result.scheme.scene.room_m, axis) for result in pair)
        for axis in ("Lx", "Ly", "Lz")}


def test_same_scheme_different_fingerprint_lists_both_problems(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult],
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("aosr.gui.app.calculation_fingerprint", lambda **kwargs:
                        pair[0].calculation_fingerprint)
    a_id, b_id = "b" * 32, "c" * 32
    altered = pair[0].model_copy(update={"calculation_fingerprint": "calc-v1:" + "1" * 64})
    with _client(tmp_path) as client:
        _files(tmp_path, altered, a_id)
        _files(tmp_path, pair[0], b_id)
        response = client.get(f"/api/compare/{a_id}/{b_id}")
    assert response.status_code == HTTPStatus.CONFLICT
    assert any("候選代號重複" in item for item in response.json()["problems"])
    assert any("計算指紋" in item for item in response.json()["problems"])
    assert response.json()["rerun_urls"] == {}
    assert response.json()["outdated_sides"] == ["a"]
    assert response.json()["outdated_schemes"] == {"a": "wall-1"}


def test_duplicate_scheme_without_fingerprint_problem_offers_no_rerun(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult],
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("aosr.gui.app.calculation_fingerprint", lambda **kwargs:
                        pair[0].calculation_fingerprint)
    a_id, b_id = "b" * 32, "c" * 32
    with _client(tmp_path) as client:
        _files(tmp_path, pair[0], a_id)
        _files(tmp_path, pair[0], b_id)
        data = client.get(f"/api/compare/{a_id}/{b_id}").json()
    assert any("候選代號重複" in item for item in data["problems"])
    assert data["rerun_urls"] == {}
    assert data["outdated_schemes"] == {}


def test_fingerprint_problem_identifies_only_outdated_side(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult],
        monkeypatch: pytest.MonkeyPatch) -> None:
    current = pair[0].calculation_fingerprint
    monkeypatch.setattr("aosr.gui.app.calculation_fingerprint", lambda **kwargs: current)
    a_id, b_id = "b" * 32, "c" * 32
    different = pair[1].model_copy(update={"calculation_fingerprint": "calc-v1:" + "1" * 64})
    with _client(tmp_path) as client:
        _files(tmp_path, pair[0], a_id)
        _files(tmp_path, different, b_id)
        response = client.get(f"/api/compare/{a_id}/{b_id}")
    assert response.status_code == HTTPStatus.CONFLICT
    assert response.json()["outdated_sides"] == ["b"]
    assert response.json()["outdated_schemes"] == {"b": "wall-2"}
    assert set(response.json()["rerun_urls"]) == {"b"}
    assert response.json()["rerun_urls"]["b"].endswith("/rerun")


def test_two_old_results_name_both_sides_even_when_comparison_succeeds(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult],
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("aosr.gui.app.calculation_fingerprint", lambda **kwargs:
                        "calc-v1:" + "1" * 64)
    a_id, b_id = "b" * 32, "c" * 32
    with _client(tmp_path) as client:
        _files(tmp_path, pair[0], a_id)
        _files(tmp_path, pair[1], b_id)
        response = client.get(f"/api/compare/{a_id}/{b_id}")
    assert response.status_code == HTTPStatus.OK
    assert response.json()["outdated_schemes"] == {"a": "wall-1", "b": "wall-2"}


def test_one_side_rejected_names_side_and_reason(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult]) -> None:
    a_id, b_id = "b" * 32, "c" * 32
    with _client(tmp_path) as client:
        _files(tmp_path, pair[0], a_id)
        _files(tmp_path, pair[1], b_id)
        path = tmp_path / "results" / f"{b_id}.json"
        document = json.loads(path.read_text())
        document["engine_commit"] = "forged"
        path.write_text(json.dumps(document))
        response = client.get(f"/api/compare/{a_id}/{b_id}")
    assert response.status_code == HTTPStatus.CONFLICT
    assert response.json()["side"] == "b" and response.json()["reason"]
    # 重算網址要是被拒收的那一份，按了才解得了。
    assert response.json()["rerun_url"].split("/")[-2] == b_id
    assert "reason_kind" not in response.json() and "reason_text" not in response.json()


@pytest.mark.parametrize("old_side", ["a", "b"])
def test_old_format_side_rejection_names_side_in_plain_words(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult], old_side: str) -> None:
    ids = {"a": "b" * 32, "b": "c" * 32}
    with _client(tmp_path) as client:
        _files(tmp_path, pair[0], ids["a"])
        _files(tmp_path, pair[1], ids["b"])
        path = tmp_path / "results" / f"{ids[old_side]}.json"
        document = json.loads(path.read_text())
        document["schema_version"] = "aosr.scheme_result.v2"
        path.write_text(json.dumps(document))
        response = client.get(f"/api/compare/{ids['a']}/{ids['b']}")
    assert response.status_code == HTTPStatus.CONFLICT
    assert response.json()["side"] == old_side and response.json()["reason"]
    assert response.json()["rerun_url"].split("/")[-2] == ids[old_side]
    assert response.json()["reason_kind"] == "old_format"
    assert response.json()["reason_text"] == (
        f"{old_side.upper()} 那份是舊格式的結果（程式更新前算的），要用現在的程式重算才能比較")


def test_bad_missing_or_same_run_id(tmp_path: Path, pair: tuple[SchemeResult, SchemeResult]) -> None:
    a_id, b_id = "b" * 32, "c" * 32
    with _client(tmp_path) as client:
        _files(tmp_path, pair[0], a_id)
        assert client.get(f"/api/compare/invalid/{a_id}").status_code == HTTPStatus.BAD_REQUEST
        assert client.get(f"/api/compare/{a_id}/invalid").status_code == HTTPStatus.BAD_REQUEST
        # 同一份傳兩次要在讀檔前就擋，寫明是同一份；不是讀完才落到「代號重複」那一關。
        same = client.get(f"/api/compare/{a_id}/{a_id}")
        assert same.status_code == HTTPStatus.CONFLICT
        assert same.json() == {"error": "A 和 B 是同一份結果"}
        missing_b = client.get(f"/api/compare/{a_id}/{b_id}")
        missing_a = client.get(f"/api/compare/{b_id}/{a_id}")
    assert missing_b.status_code == missing_a.status_code == HTTPStatus.NOT_FOUND
    assert "B" in str(missing_b.json()) and "A" in str(missing_a.json())


def test_unreadable_file_is_not_offered_a_rerun(tmp_path: Path,
                                                pair: tuple[SchemeResult, SchemeResult]) -> None:
    # 讀不動檔（權限）不是結果被拒收，重算解不了：照結果頁回 404，不附重算網址。
    a_id, b_id = "b" * 32, "c" * 32
    with _client(tmp_path) as client:
        _files(tmp_path, pair[0], a_id)
        _files(tmp_path, pair[1], b_id)
        path = tmp_path / "results" / f"{b_id}.json"
        path.chmod(0)
        try:
            response = client.get(f"/api/compare/{a_id}/{b_id}")
        finally:
            path.chmod(0o600)
    assert response.status_code == HTTPStatus.NOT_FOUND
    assert "B" in response.json()["error"] and "rerun_url" not in response.json()


def test_compare_page_uses_the_shared_words() -> None:
    # 各頁同一套字：程式不叫引擎（重算按鈕寫「用現在的程式重算」），輸入頁叫方案輸入頁（回去的連結寫「回方案輸入頁」）。
    page = (STATIC / "compare.html").read_text(encoding="utf-8")
    script = (STATIC / "compare.js").read_text(encoding="utf-8")
    assert "引擎" not in page + script
    assert '<a href="/">回方案輸入頁</a>' in page
    assert "用現在的程式重算" in script
    # 拒收那一份不說「讀回被拒收」（老闆看不懂），瀏覽器的英文錯誤不直接印上主畫面。
    assert "讀回被拒收" not in script
    assert 'textContent = String(error)' not in script
