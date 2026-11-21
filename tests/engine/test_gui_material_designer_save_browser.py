"""材料設計器讀取失敗、存檔拒收與另存重試的畫面考卷。"""
import json
from pathlib import Path

import pytest
from playwright.sync_api import Browser, Route, expect

from aosr.materials.library import SCHEMA_VERSION
from tests.engine._construction_cases import construction
from tests.engine._construction_gui_server import real_server
from tests.engine.test_gui_browser import _open, browser as browser
from tests.engine.test_gui_material_designer_browser import (
    _add, _advanced, _designer, _fields, _open_sample, _save, _stored, _tuning_sample,
)


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
