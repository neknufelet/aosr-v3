"""材料設計器讀取失敗、存檔拒收與另存重試的畫面考卷。"""
import json
from pathlib import Path

import pytest
from playwright.sync_api import Browser, Page, Response, Route, expect

from aosr.materials.library import SCHEMA_VERSION
from tests.engine._construction_cases import construction
from tests.engine._construction_gui_server import real_server
from tests.engine.test_gui_browser import _open, browser as browser
from tests.engine.test_gui_material_designer_browser import (
    _add, _advanced, _designer, _fields, _list_row, _open_sample, _save, _stored, _tuning_sample,
)


def _curve_from(response: Response, entry: dict[str, object], word: str) -> tuple[list[float], list[float]]:
    body = response.request.post_data_json
    assert isinstance(body, dict)
    for key in ("id", "layers", "backing", "placement", "product", "provenance"):
        assert body[key] == entry[key]
    reply = response.json()
    assert response.ok and (reply["status"], reply["provenance"]) == ("drawn", word)
    assert len(reply["frequencies_hz"]) == len(reply["absorption"]) > 1
    return reply["frequencies_hz"], reply["absorption"]


def _open_listed_source(page: Page, base: str, entry: dict[str, object], word: str) -> tuple[list[float], list[float]]:
    # 列表讀已存的材料。進入設計器後，預覽和兩個標籤都要跟這份草稿相符。
    def listed_preview(response: Response) -> bool:
        if not response.url.endswith("/api/material-curve?preview"):
            return False
        body = response.request.post_data_json
        return isinstance(body, dict) and body["id"] == entry["id"]

    with page.expect_response(listed_preview, timeout=60_000) as listed:
        row = _list_row(page, base, "fixed" if entry["product"] else "optimizable", str(entry["label"]))
    expect(row.locator(".source-badge")).to_have_text(word)
    expect(row.locator(f".preview svg.{entry['provenance']}")).to_be_visible()
    curve = _curve_from(listed.value, entry, word)
    with page.expect_response("**/api/material-curve", timeout=60_000) as preview:
        row.get_by_role("button", name="打開").click()
    draft = entry if entry["product"] else entry | {"provenance": "model"}
    shown = word if entry["product"] else "模型"
    assert _curve_from(preview.value, draft, shown) == curve
    expect(page.locator("#curve canvas").first).to_be_visible()
    page.get_by_text("出處與註記", exact=True).click()
    expect(page.locator("#source")).to_have_text(shown)
    expect(page.locator("#citation-provenance")).to_have_text(shown)
    return curve


@pytest.mark.parametrize(("origin", "word"), [("literature", "文獻"), ("measured", "實測")])
@pytest.mark.parametrize("rename", [False, True], ids=["untouched", "rename_only"])
def test_editable_construction_saves_the_displayed_source(tmp_path: Path, browser: Browser, origin: str, word: str, rename: bool) -> None:
    entry = construction(id="reed_lining", label="葦面試板", provenance=origin, source="葦面樣片簿")
    citation = f"原來源：{word}，出處：葦面樣片簿，網址：https://example.org/test"
    expected = entry | {"provenance": "model", "notes": [citation, "自編條件"]}
    with _designer(tmp_path, browser) as (page, base):
        url = f"{base}/api/materials/reed_lining"
        assert page.request.put(url, data=entry, headers={"If-None-Match": "*"}).ok
        library = tmp_path / "materials.json"
        assert json.loads(library.read_bytes())["materials"] == [entry]
        curve = _open_listed_source(page, base, entry, word)
        if rename:
            expected["label"] = "葦面試板乙"
            page.locator("#name").fill("葦面試板乙")
        for _ in range(1 if rename else 3):
            expect(page.locator("#source")).to_have_text("模型")
            expect(page.locator("#citation-provenance")).to_have_text("模型")
            with page.expect_response(lambda r: r.url == url and r.request.method == "PUT") as saved:
                _save(page, "參數構造")
            assert saved.value.ok
            stored, = json.loads(library.read_bytes())["materials"]
            assert saved.value.request.post_data_json == stored == expected
            expect(page.locator("#citation-notes")).to_have_value(citation + "\n自編條件")
        assert _open_listed_source(page, base, expected, "模型") == curve
        expect(page.locator("#citation-notes")).to_have_value(citation + "\n自編條件")


@pytest.mark.parametrize(("origin", "word"), [("literature", "文獻"), ("measured", "實測")])
def test_product_source_survives_view_and_reopen(tmp_path: Path, browser: Browser, origin: str, word: str) -> None:
    entry = construction(id="rush_tile", label="蘆面樣磚", product=True, provenance=origin)
    library = tmp_path / "materials.json"
    with _designer(tmp_path, browser) as (page, base):
        assert page.request.put(f"{base}/api/materials/rush_tile", data=entry, headers={"If-None-Match": "*"}).ok
        before = library.read_bytes()
        assert json.loads(before)["materials"] == [entry]
        puts: list[str] = []
        page.on("request", lambda r: puts.append(r.url) if r.method == "PUT" else None)
        for _ in range(2):
            _open_listed_source(page, base, entry, word)
            expect(page.locator("#save")).to_be_hidden()
            expect(page.locator("#name")).to_be_disabled()
            expect(page.locator("#citation-notes")).to_have_value("自編條件")
            page.evaluate("saveConstruction()")  # 即使直接觸發設計器的儲存動作也不送件。
        assert puts == []
        assert library.read_bytes() == before


@pytest.mark.parametrize("product", [False, True], ids=["unchanged_retry", "product_rename_retry"])
def test_server_saves_model_citation_once_after_lost_response(tmp_path: Path, browser: Browser, product: bool) -> None:
    entry = construction(id="rush_mat", label="藺面試板", provenance="measured", source="藺面樣片簿")
    with real_server(tmp_path) as base, _open(browser, "about:blank") as watched:
        page = watched.page
        url = f"{base}/api/materials/rush_mat"
        assert page.request.put(url, data=entry).ok
        page.goto(f"{base}/materials/designer?id=rush_mat")
        if product:
            page.locator("#product").check()

        def lose_response(route: Route) -> None:
            assert route.fetch().ok  # 真正送到後端寫檔，只讓前端收不到回覆。
            route.abort()

        page.route(url, lose_response, times=1)
        page.locator("#save").click()
        expect(page.locator("#save-message")).to_have_class("notice")
        library = tmp_path / "materials.json"
        first, = json.loads(library.read_bytes())["materials"]
        expected = first.copy()
        if product:
            # 回覆遺失時設計器仍在編輯同份草稿；改名重送，確認成功後才鎖住。
            page.locator("#name").fill("藺面試板乙")
            expected["label"] = "藺面試板乙"
        with page.expect_request(lambda r: r.method == "PUT") as retry:
            page.locator("#save").click()
        expect(page.locator("#save-message")).to_have_text(f"已存到「{'成品' if product else '參數構造'}」頁。")
        second, = json.loads(library.read_bytes())["materials"]
        assert (first["provenance"], second["provenance"]) == ("model", "model")
        assert retry.value.url == url and retry.value.post_data_json == expected == second
        assert first["product"] is second["product"] is product
        assert second["notes"] == ["原來源：實測，出處：藺面樣片簿，網址：https://example.org/test", "自編條件"]
        if product:
            expect(page.locator("#save")).to_be_hidden()


def test_blank_fixed_cavity_saves_untouched_but_blocks_an_edit(tmp_path: Path, browser: Browser) -> None:
    folder = tmp_path / "materials"
    folder.mkdir()
    library = folder / "materials.json"
    entry = _tuning_sample(False) | {"cavity": {"adjustable": False, "range_m": [0, 0], "initial_m": None}}
    library.write_text(json.dumps({"schema_version": SCHEMA_VERSION, "materials": [entry]}, indent=3), encoding="utf-8")
    before = library.read_bytes()
    with _designer(folder, browser) as (page, base):
        original = _stored(page, base)["woven_sample"]
        _open_sample(page, base)
        puts: list[str | None] = []
        page.on("request", lambda request: puts.append(request.post_data) if request.method == "PUT" else None)
        _save(page, "參數構造")
        expect(page.locator("#curve canvas").first).to_be_visible(timeout=60_000)
        expect(page.locator("#curve-state")).to_be_empty()
        assert library.read_bytes() == before
        page.locator("#name").fill("試織棉樣乙")
        page.locator("#save").click()
        expect(page.locator("#save-message")).to_have_text("第 2 層 空氣層的厚度還沒填")
        assert puts == [page.evaluate("entry => JSON.stringify(entry)", original)]
        assert library.read_bytes() == before


@pytest.mark.parametrize(("full", "origin"), [(False, "literature"), (True, "measured")])
@pytest.mark.parametrize("air", [
    pytest.param({}, id="omitted"),
    pytest.param({"cavity": None}, id="null"),
    pytest.param({"cavity": {"adjustable": False, "range_m": [0, 0], "initial_m": None}}, id="fixed_blank"),
    pytest.param({"cavity": {"adjustable": False, "range_m": [.02613, .02613], "initial_m": .02613}}, id="fixed"),
    pytest.param({"cavity": {"adjustable": True, "range_m": [0, .0847], "initial_m": None}}, id="adjustable_blank"),
    pytest.param({"cavity": {"adjustable": True, "range_m": [0, .0847], "initial_m": .02139}}, id="adjustable"),
])
def test_legacy_tuning_save_is_byte_identical(tmp_path: Path, browser: Browser, full: bool, origin: str, air: dict[str, object]) -> None:
    entry = _tuning_sample(full)
    entry.pop("cavity", None)
    if full:
        entry["suggested_total_depth_m"] = .0701234
    entry.update(air, provenance=origin)
    library = tmp_path / "materials.json"
    library.write_text(json.dumps({"schema_version": SCHEMA_VERSION, "materials": [entry]}, ensure_ascii=False, indent=3) + "\n")
    before = library.read_bytes()
    with _designer(tmp_path, browser) as (page, base):
        original = _stored(page, base)["woven_sample"]
        _open_sample(page, base)
        expect(page.locator("#source")).to_have_text("實測" if full else "文獻")
        with page.expect_request(lambda r: r.method == "PUT") as sent:
            _save(page, "參數構造")
        assert sent.value.post_data == page.evaluate("entry => JSON.stringify(entry)", original)
        assert library.read_bytes() == before


@pytest.mark.parametrize("copy", [False, True], ids=["in_place", "copy"])
def test_flow_resistivity_edit_records_literature_once(tmp_path: Path, browser: Browser, copy: bool) -> None:
    with _designer(tmp_path, browser) as (page, base):
        entry = _tuning_sample() | {"provenance": "literature"}
        assert page.request.put(f"{base}/api/materials/woven_sample", data=entry).ok
        _open_sample(page, base)
        if copy:
            page.get_by_role("button", name="複製一份").click()
        _fields(page, 0)
        expect(page.locator("#citation-provenance")).to_have_text("文獻")
        for flow in ("17800", "18200"):
            page.locator('[name="flow_resistivity_pa_s_m2"]').fill(flow)
            expect(page.locator("#citation-provenance")).to_have_text("模型")
            _save(page, "參數構造")
            saved = _stored(page, base)["woven_sample_copy" if copy else "woven_sample"]
            assert saved["provenance"] == "model"
            assert saved["notes"] == ["原來源：文獻，出處：試片紀錄，網址：https://example.org/woven", "自編試片\n第二面"]


def test_restoring_unspecified_flow_source_keeps_literature(tmp_path: Path, browser: Browser) -> None:
    with _designer(tmp_path, browser) as (page, base):
        entry = _tuning_sample(False) | {"provenance": "literature"}
        assert page.request.put(f"{base}/api/materials/woven_sample", data=entry).ok
        original = _stored(page, base)["woven_sample"]
        _open_sample(page, base)
        _fields(page, 0)
        _advanced(page)
        field = page.locator('[name="flow_resistivity_source"]')
        field.select_option("catalog")
        expect(page.locator("#source")).to_have_text("模型")
        field.select_option("unspecified")
        expect(page.locator("#source")).to_have_text("文獻")
        with page.expect_request(lambda r: r.method == "PUT") as sent:
            _save(page, "參數構造")
        assert sent.value.post_data == page.evaluate("entry => JSON.stringify(entry)", original)


@pytest.mark.parametrize("written", [False, True], ids=["refused", "reply_lost_after_write"])
def test_failed_kind_change_keeps_original_draft(tmp_path: Path, browser: Browser, written: bool) -> None:
    with real_server(tmp_path) as base, _open(browser, "about:blank") as watched:
        page = watched.page
        assert page.request.put(f"{base}/api/materials/woven_sample", data=_tuning_sample(False)).ok
        _open_sample(page, base)
        original_url = page.url
        page.get_by_role("switch").click()
        page.locator("#name").fill("試織修改")
        message = "材料庫寫入失敗，請稍後再試"
        def fail_once(route: Route) -> None:
            if written:
                assert route.fetch().ok
            route.fulfill(status=500, json={"error": message})

        page.route("**/api/materials/*", fail_once, times=1)
        page.once("dialog", lambda dialog: dialog.accept())
        with page.expect_request(lambda r: r.method == "PUT") as sent:
            page.locator("#save").click()
        assert sent.value.url.endswith("/woven_sample_copy")
        expect(page.locator("#save-message")).to_have_text(message)
        expect(page.locator("#name")).to_have_value("試織修改")
        expect(page.locator("#citation-source")).to_have_value("試片紀錄")
        expect(page.locator("#copy-designer")).to_be_visible()
        expect(page.get_by_role("switch")).not_to_be_checked()
        assert page.url == original_url
        assert set(_stored(page, base)) == ({"woven_sample", "woven_sample_copy"} if written else {"woven_sample"})
        page.once("dialog", lambda dialog: dialog.accept())
        _save(page, "參數構造")
        entries = _stored(page, base)
        assert set(entries) == {"woven_sample", "woven_sample_copy"}
        assert entries["woven_sample"]["label"] == "試織棉樣" and entries["woven_sample_copy"]["label"] == "試織修改（複本）"
        assert not watched.page_errors and all("500" in line for line in watched.console_errors)


def test_open_by_id_explains_broken_library(tmp_path: Path, browser: Browser) -> None:
    folder = tmp_path / "materials"
    folder.mkdir()
    (folder / "materials.json").write_text("{ broken", encoding="utf-8")
    with real_server(folder) as base, _open(browser, f"{base}/materials/designer?id=missing_sample") as watched:
        problem = watched.page.request.get(f"{base}/api/materials").json()["user_file_problem"]
        assert "讀不進來" in problem and "不是有效的 JSON" in problem
        expect(watched.page.locator("#entry-status")).to_have_text(problem)
        expect(watched.page.locator("#workspace")).to_be_hidden()


def test_only_air_adjustable_explains_both_frame_fields(tmp_path: Path, browser: Browser) -> None:
    with _designer(tmp_path, browser) as (page, base):
        _add(page, "porous", "air")
        page.get_by_role("switch").nth(1).click()
        _fields(page, 0)
        _advanced(page)
        page.locator('[name="frame_compression_modulus_pa"]').fill("15700")
        page.locator('[name="frame_loss_factor"]').fill("0.12")
        page.locator("#save").click()
        expect(page.locator("#save-message")).to_have_text("多孔層填了骨架兩格就不能給程式調")
        assert _stored(page, base) == {}


def test_nonpositive_total_depth_names_lower_bound(tmp_path: Path, browser: Browser) -> None:
    with _designer(tmp_path, browser) as (page, base):
        _add(page, "porous", "air")
        page.get_by_role("switch").nth(1).click()
        for depth in ("0", "-2"):
            page.get_by_label("總深度上限").fill(depth)
            page.locator("#save").click()
            expect(page.locator("#save-message")).to_have_text("總深度上限要大於 0")
        assert _stored(page, base) == {}


def test_kind_change_retries_with_unused_copy_id_after_two_collisions(tmp_path: Path, browser: Browser) -> None:
    with real_server(tmp_path) as base, _open(browser, "about:blank") as watched:
        page = watched.page
        assert page.request.put(f"{base}/api/materials/woven_sample", data=_tuning_sample(False)).ok
        _open_sample(page, base)
        page.get_by_role("switch").click()
        attempts: list[str] = []

        def collide(route: Route) -> None:
            material_id = route.request.url.rsplit("/", 1)[-1]
            attempts.append(material_id)
            if len(attempts) <= 2:
                rival = construction(id=material_id, label=f"試作牆面{len(attempts)}")
                assert page.request.put(route.request.url, data=rival).ok
            route.continue_()

        page.route("**/api/materials/*", collide)
        page.on("dialog", lambda dialog: dialog.accept())
        page.locator("#save").click()
        expect(page.locator("#save-message")).to_contain_text("自動編號又重複了")
        expect(page.locator("#name")).to_have_value("試織棉樣")
        _save(page, "參數構造")
        assert len(attempts) == len(set(attempts)) == 3
        assert _stored(page, base)["woven_sample"]["kind"] == "optimizable"


def test_both_routes_to_construction_copy_record_original_source(tmp_path: Path, browser: Browser) -> None:
    with _designer(tmp_path, browser) as (page, base):
        entry = _tuning_sample(False) | {"provenance": "literature", "links": ["https://example.org/recheck-weave"],
                                         "notes": ["試片側面"]}
        assert page.request.put(f"{base}/api/materials/woven_sample", data=entry).ok
        _open_sample(page, base)
        page.get_by_role("button", name="複製一份").click()
        page.get_by_role("switch").click()
        _save(page, "參數構造")
        _open_sample(page, base)
        page.get_by_role("switch").click()
        page.once("dialog", lambda dialog: dialog.accept())
        _save(page, "參數構造")
        copies = [row for key, row in _stored(page, base).items() if key != "woven_sample"]
        assert len(copies) == 2
        for row in copies:
            assert (row["kind"], row["provenance"], row["source"], row["links"]) == (
                "construction", "model", "照「試織棉樣」改的", [])
            assert row["notes"] == ["原來源：文獻，出處：試片紀錄，網址：https://example.org/recheck-weave", "試片側面"]


def test_blank_default_model_keeps_literature_and_notes(tmp_path: Path, browser: Browser) -> None:
    with _designer(tmp_path, browser) as (page, base):
        entry = _tuning_sample(False) | {"provenance": "literature", "notes": ["自編試片"]}
        assert page.request.put(f"{base}/api/materials/woven_sample", data=entry).ok
        _open_sample(page, base)
        _fields(page, 0)
        _advanced(page)
        page.locator('[name="model"]').select_option("")
        shown = page.locator("#source").inner_text()
        _save(page, "參數構造")
        saved = _stored(page, base)["woven_sample"]
        assert (shown, saved["provenance"], saved["notes"]) == ("文獻", "literature", ["自編試片"])


def test_zero_fixed_air_requires_positive_thickness_before_save(tmp_path: Path, browser: Browser) -> None:
    with _designer(tmp_path, browser) as (page, base):
        _add(page, "porous", "air")
        page.get_by_role("switch").first.click()
        depth = page.locator('[name="thickness_m"]').input_value()
        page.get_by_label("買得到的厚度（mm）").fill(depth)
        _fields(page, 1)
        thickness = page.locator('[name="thickness_m"]')
        label = page.locator("label").filter(has=thickness).inner_text()
        thickness.fill("0")
        puts: list[str] = []
        page.on("request", lambda request: puts.append(request.url) if request.method == "PUT" else None)
        page.locator("#save").click()
        expect(page.locator("#save-message")).not_to_be_empty()
        assert (page.locator("#save-message").inner_text(), label.endswith("＊"), puts, list(_stored(page, base))) == (
            "第 2 層 空氣層的厚度要大於 0；不要空氣就拿掉這一層", True, [], [])
        assert not (tmp_path / "materials.json").exists()
