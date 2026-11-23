"""#1270 左右對稱畫框：真瀏覽器拖曳、選區、存檔；房間取自 zone-naming-sample。"""
from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import cast

import pytest
from playwright.sync_api import Browser, Page, Response, Route, expect

from aosr.gui.zone_design import DesignSettings, write_design
from aosr.gui.zone_edit import CutBox, WallName, cut_wall
from aosr.reporting.scheme import Scheme
from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine._zone_naming_room import ZONE_NAMING_SCHEME
from tests.engine.test_gui_browser import _open, _serve, browser as browser
from tests.engine.test_gui_zone_elevation_browser import _cut_by, _drag, _open_scheme, _ready_pixel, _write, _zones
from tests.engine.test_gui_zone_symmetry_browser import _settled

NAME = "zone-naming-sample"
FRONT = "yL"
W, H = 5.2, 3.15
START, END = (0.55, 0.02), (1.35, 1.8)
FIRST = CutBox(id="z1", u_m=(0.55, 1.35), v_m=(0, 1.8))
SECOND = CutBox(id="z2", u_m=(3.55, 4.35), v_m=(0, 1.8))
OTHER = CutBox(id="stay", u_m=(2.0, 2.8), v_m=(2.2, H))
BLOCK = CutBox(id="block", u_m=(3.9, 4.85), v_m=(0, 2.1))
Rect = dict[str, object]

# fe5b6bf9 真瀏覽器關著對稱畫、搬、刪：原始 HTTP 請求與回覆各自的 SHA-256。
OFF_DIGESTS = [
    ("068c882e4a5855c5be8920c140d8accafec87b5363c82b33bd5b21af9dc43ef4",
     "be05d91968b98338f1b8afb08f7dc82326651f7f455bed5415a527ef39eaf8d1"),
    ("b0e9f571535e50cd45647b206fd2c244aa3f8cc50bb9db2a45a9528ca4fe185b",
     "beaa9567574f0790c7f80da30aaff6d4758fce4b9921f7151cb40f6f0a009879"),
    ("701ef7e85c5521ad621affdf7cea6bf01dd1cd72825458b87c7f65789b4f2142",
     "f770b00bd34ed3503e9b9116b9e7a388e955243c8e51e882596995c0b7aaa824"),
]


@contextmanager
def _room(tmp_path: Path, browser: Browser, boxes: tuple[CutBox, ...] = (), *, centred: bool = False,
          wall: WallName = "yL", opposite: tuple[CutBox, ...] = (), search_floor: bool = False) -> Iterator[Page]:
    doc = deepcopy(ZONE_NAMING_SCHEME)
    scene = cast(Rect, doc["scene"])
    walls = cast(dict[str, list[Rect]], scene["surface_zones_by_wall"])
    for face in walls:
        walls[face] = [{"area": "whole_wall", "material": {"kind": "constant_impedance", "impedance_pa_s_m": 71500.0}}]
    walls["floor"] = cast(list[Rect], cut_wall(doc, "floor", (OTHER,))["zones"])
    walls["yL"] = cast(list[Rect], cut_wall(doc, "yL", boxes)["zones"])
    if opposite:
        walls["xL"] = cast(list[Rect], cut_wall(doc, "xL", opposite)["zones"])
    if centred:
        speakers = cast(dict[str, dict[str, float]], doc["speakers"])
        speakers["right"]["x"] = 4.1
        for point in cast(dict[str, list[Rect]], doc["receiver_set"])["points"]:
            cast(list[float], point["position_m"])[0] += 0.15
    _write(tmp_path, doc)
    if search_floor:
        (tmp_path / "designs").mkdir(exist_ok=True)
        design = DesignSettings.model_validate({"scheme_id": NAME, "schema_version": "aosr.zone_design.v1",
                                                "zones": {"floor": {"stay": {"mode": "search", "allowed": ["glass_wool_48k", "none"]}}}})
        write_design(tmp_path, design, Scheme.model_validate(doc))
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, NAME)
        page.locator("#zone-table-fold > summary").click()
        page.locator(f"#elevation-wall-{wall}").click()
        _settled(page)
        yield page
        assert not watched.page_errors


def _drawn(page: Page, wall: str = FRONT) -> dict[str, Rect]:
    return {str(z["id"]): z for z in _zones(page, wall) if z["area"] == "rectangle" and not str(z["id"]).startswith("auto")}


def _rect(page: Page, selector: str) -> list[float]:
    return cast(list[float], page.locator(selector).evaluate("r => ['x','y','width','height'].map(k => r[k].baseVal.value)"))


def _shot(page: Page, folder: Path, name: str) -> None:
    page.locator("#elevation-panel").screenshot(path=folder / f"{name}.png")


def _optimize(page: Page) -> None:
    page.locator("#zone-page-open").click()
    _settled(page)
    page.locator("#zone-map [data-wall=yL][data-zone=z1] .zv-fill").click()
    page.locator("#zone-panel-mode-search").check()
    page.locator("#zone-panel-apply").click()
    _settled(page)
    pairing = page.evaluate("zoneLimits.symmetry")
    assert {frozenset(p.values()) for p in pairing["pairs"]} >= {frozenset(("yL@z1", "yL@z2"))}
    assert not any({"yL@z1", "yL@z2"}.intersection((row["first"], row["second"])) for row in pairing["missed"])


def test_front_preview_and_search_pair(tmp_path: Path, browser: Browser) -> None:
    with _room(tmp_path, browser) as page:
        preview: list[float] = []

        def during() -> None:
            assert page.locator(".mirror-preview rect").count() == 1
            preview.extend(_rect(page, ".mirror-preview rect"))
            page.screenshot(path=tmp_path / "front-preview.png")

        _drag(page, W, H, START, END, during=during)
        _settled(page)
        first, second = (_drawn(page)[key] for key in ("z1", "z2"))
        axis = page.evaluate("zoneLimits.symmetry.axis.across_m")
        assert second["u_m"] == pytest.approx([2 * axis - x for x in reversed(cast(list[float], first["u_m"]))], abs=1e-9, rel=0)
        assert second["v_m"] == first["v_m"]
        assert _rect(page, "#elevation [data-zone-id=z2] > rect") == pytest.approx(preview, rel=0, abs=1e-9)
        expect(page.locator(".mirror-link text")).to_have_text("左右對稱")
        _shot(page, tmp_path, "front-pair")
        _optimize(page)


def test_move_resize_delete_preserve_other(tmp_path: Path, browser: Browser) -> None:
    with _room(tmp_path, browser, (FIRST, SECOND, OTHER)) as page:
        original = _drawn(page)
        _drag(page, W, H, (0.9, 0.75), (1.1, 0.75))
        moved = _drawn(page)
        assert moved["z2"]["u_m"] == pytest.approx([x - 0.2 for x in SECOND.u_m])
        assert moved["z1"]["u_m"] == pytest.approx([x + 0.2 for x in FIRST.u_m])
        page.locator("#elevation-width").fill("1.05")
        _cut_by(page, "press:Enter")
        adjusted = _drawn(page)
        assert adjusted["z2"]["u_m"] == pytest.approx([3.1, 4.15])
        assert adjusted["stay"] == moved["stay"] == original["stay"]
        _cut_by(page, "#elevation-delete")
        assert _drawn(page) == {"stay": original["stay"]}


def test_blocked_then_repair(tmp_path: Path, browser: Browser) -> None:
    with _room(tmp_path, browser, (BLOCK, OTHER)) as page:
        before = _drawn(page)
        _drag(page, W, H, START, END)
        _settled(page)
        found = _drawn(page)
        assert set(found) == {"z1", *before} and all(found[k] == v for k, v in before.items())
        label = page.evaluate("zoneCalled('yL', 'block')")
        expect(page.locator("#elevation-note")).to_contain_text(label)
        _cut_by(page, "#elevation-complete")
        _shot(page, tmp_path, "blocked")
        page.locator("#elevation [data-zone-id=block] > rect").click()
        page.locator("#elevation-height").fill("0.6")
        _cut_by(page, "press:Enter")
        page.locator("#elevation-bottom").fill("2.3")
        _cut_by(page, "press:Enter")
        page.locator("#elevation [data-zone-id=z1] > rect").click()
        _cut_by(page, "#elevation-complete")
        assert _drawn(page)["z2"]["u_m"] == pytest.approx(SECOND.u_m)


def test_outside_mirror(tmp_path: Path, browser: Browser) -> None:
    with _room(tmp_path, browser) as page:
        _drag(page, W, H, (4.4, 0.02), (5.19, 1.8))
        assert set(_drawn(page)) == {"z1"}
        expect(page.locator("#elevation-note")).to_contain_text("超出牆")


@pytest.mark.parametrize("centred", [True, False])
def test_side_walls(tmp_path: Path, browser: Browser, centred: bool) -> None:
    with _room(tmp_path, browser, centred=centred, wall="x0") as page:
        before = _zones(page, "xL")
        _drag(page, 5.5, H, START, END)
        if centred:
            assert _drawn(page, "xL") == _drawn(page, "x0")
            _shot(page, tmp_path, "left-wall")
            page.locator("#elevation-wall-xL").click()
            _shot(page, tmp_path, "right-wall")
        else:
            assert _zones(page, "xL") == before
            expect(page.locator("#elevation-note")).to_contain_text("不在房間正中")


def test_switch_save_and_single_wall_contract(tmp_path: Path, browser: Browser) -> None:
    with _room(tmp_path, browser, (OTHER,)) as page:
        assert set(_drawn(page)) == {"stay"}
        assert page.locator("#elevation-symmetry-off").count() == 1
        page.locator("#elevation-symmetry-off").click()
        page.locator("#zone-page-open").click()
        expect(page.locator("#zone-symmetry-off")).to_have_attribute("aria-pressed", "true")
        expect(page.locator("#zone-page-unsaved")).to_be_visible()
        page.locator("#zone-page-close").click()
        with page.expect_response(f"**/api/designs/{NAME}") as response:
            page.locator("#save").click()
        assert response.value.ok
        page.reload(wait_until="networkidle")
        _open_scheme(page, NAME)
        page.locator("#elevation-wall-yL").click()
        expect(page.locator("#elevation-symmetry-off")).to_have_attribute("aria-pressed", "true")
        original = _drawn(page)
        calls: list[tuple[dict[str, object], dict[str, object]]] = []
        digests: list[tuple[str, str]] = []

        def record(response: Response) -> None:
            if response.url.endswith("/api/zone-cut"):
                body = response.request.post_data_json
                assert isinstance(body, dict)
                calls.append((body, response.json()))
                raw = response.request.post_data_buffer
                assert raw is not None
                digests.append((sha256(raw).hexdigest(), sha256(response.body()).hexdigest()))

        page.on("response", record)
        _drag(page, W, H, START, END)
        _drag(page, W, H, (0.9, 0.75), (1.1, 0.75))
        _cut_by(page, "#elevation-delete")
        assert digests == OFF_DIGESTS
        assert len(calls) == 3 and _drawn(page) == original
        for request, reply in calls:
            assert set(request) == {"scheme", "wall", "boxes"} and request["wall"] == FRONT
            boxes = [CutBox.model_validate(b) for b in cast(list[Rect], request["boxes"])]
            assert reply == cut_wall(cast(Rect, request["scheme"]), "yL", boxes)


def test_overview_select_and_drop_pair(tmp_path: Path, browser: Browser) -> None:
    with _room(tmp_path, browser, (FIRST, SECOND)) as page:
        page.locator("#zone-page-open").click()
        _settled(page)
        first = page.locator("#zone-map [data-wall=yL][data-zone=z1] .zv-fill")
        first.click()
        expect(page.locator("#zone-panel")).to_have_attribute("data-zones", "yL/z1 yL/z2")
        expect(page.locator("#zone-panel")).to_contain_text("跟著一起選")
        page.screenshot(path=tmp_path / "overview-selection.png")
        page.locator("#zone-panel-drop-yL-z2").click()
        expect(page.locator("#zone-panel")).to_have_count(0)
        first.click()
        page.locator("#zone-panel-mode-search").check()
        page.locator("#zone-panel-apply").click()
        _settled(page)
        assert page.evaluate("['z1','z2'].map(k=>zoneSetting('yL',k).mode)") == ["search", "search"]


@pytest.mark.parametrize("send", [False, True], ids=["save", "send-to-search"])
def test_delayed_cut_then_save(tmp_path: Path, browser: Browser, send: bool) -> None:
    with _room(tmp_path, browser, centred=True, wall="x0", search_floor=send) as page:
        pending: list[Route] = []
        page.route("**/api/zone-cut", lambda route: pending.append(route))
        pixel = _ready_pixel(page, 5.5, H)
        page.mouse.move(*pixel(START))
        page.mouse.down()
        page.mouse.move(*pixel(END))
        with page.expect_request("**/api/zone-cut"):
            page.mouse.up()
        assert page.evaluate("inputBusy && document.querySelector('main').inert")
        assert not _drawn(page, "x0") and not _drawn(page, "xL")
        for button, note, refused in (("calculate", "messages", "沒有開算"), ("open-scheme", "open-note", "沒有打開"),
                                      ("load-template", "template-note", "沒有載入範本")):
            page.locator(f"#{button}").dispatch_event("click")
            expect(page.locator(f"#{note}")).to_contain_text(refused)
        page.locator("#zone-to-search" if send else "#save").dispatch_event("click")
        with page.expect_response(lambda r: r.url.endswith(f"/api/schemes/{NAME}") and r.request.method == "PUT", timeout=5000):
            pending[0].continue_()
        if send:
            page.wait_for_url(f"**/gradient?scheme={NAME}")
        stored = json.loads((tmp_path / "schemes" / f"{NAME}.json").read_text())
        Scheme.model_validate(stored)
        walls = stored["scene"]["surface_zones_by_wall"]
        assert walls["x0"] == walls["xL"] and walls["x0"][0]["area"] == "rectangle"


def test_corner_drag_resizes_both(tmp_path: Path, browser: Browser) -> None:
    with _room(tmp_path, browser, (FIRST, SECOND)) as page:
        page.locator("#elevation [data-zone-id=z1] > rect").click()
        handle = page.locator(".mirror-resize")
        assert handle.count() == 1
        pixel = _ready_pixel(page, W, H)
        handle.hover()
        page.mouse.down()
        page.mouse.move(*pixel((1.6, 2.0)))
        with page.expect_response("**/api/zone-cut"):
            page.mouse.up()
        _settled(page)
        zones = _drawn(page)
        assert zones["z1"]["u_m"] == pytest.approx([0.55, 1.6])
        assert zones["z2"]["u_m"] == pytest.approx([3.3, 4.35])
        assert zones["z1"]["v_m"] == zones["z2"]["v_m"] == [0, 2.0]


@pytest.mark.parametrize(("start", "end", "reason"), [
    ((1.9, 0.02), (3.0, 1.8), ""),
    ((1.9, 0.02), (2.75, 1.8), "跨在對稱軸上但沒有置中"),
    ((2.8, 0.02), (4.8, 1.8), "窄條"),
])
def test_centre_crossing_and_thin_remainder(tmp_path: Path, browser: Browser, start: tuple[float, float],
                                             end: tuple[float, float], reason: str) -> None:
    with _room(tmp_path, browser) as page:
        _drag(page, W, H, start, end)
        _settled(page)
        assert set(_drawn(page)) == {"z1"}
        line = page.locator("#elevation-note")
        if reason:
            expect(line).to_contain_text(reason)
            expect(page.locator("#elevation-complete")).to_be_visible()
        else:
            expect(page.locator(".mirror-axis text")).to_have_text("對稱軸")
            expect(line).not_to_contain_text("另一邊沒有跟著畫")
            expect(page.locator("#elevation-complete")).to_have_count(0)


def test_axis_unavailable_and_existing_boxes_stay_in_place(tmp_path: Path, browser: Browser) -> None:
    with _room(tmp_path, browser, (FIRST, SECOND)) as page:
        original = _drawn(page)
        page.locator("#receiver-main-x").fill("2.8")
        page.locator("#receiver-main-x").press("Tab")
        _settled(page)
        expect(page.locator("#elevation-axis-words")).to_contain_text("主位不在兩支喇叭中間")
        assert not page.locator(".mirror-axis").count()
        assert _drawn(page) == original

        def without_preview() -> None:
            assert not page.locator(".mirror-preview").count()

        _drag(page, W, H, (1.8, 2.1), (2.7, 2.8), during=without_preview)
        expect(page.locator("#elevation-note")).to_contain_text("定不出對稱軸")
        assert page.evaluate("zoneLimits.symmetry.partners") == {}
        assert set(_drawn(page)) == {"z1", "z2", "z3"}


def test_blocked_side_wall_keeps_every_fill_and_material(tmp_path: Path, browser: Browser) -> None:
    obstacle = CutBox(id="block", u_m=(0.8, 1.6), v_m=(0.5, 1.5))
    with _room(tmp_path, browser, centred=True, wall="x0", opposite=(obstacle, OTHER)) as page:
        old = {wall: _zones(page, wall) for wall in ("xL", "floor", "ceiling", "y0", "yL")}
        _drag(page, 5.5, H, START, END)
        assert set(_drawn(page, "x0")) == {"z1"}
        assert all(_zones(page, wall) == zones for wall, zones in old.items())
        expect(page.locator("#elevation-note")).to_contain_text(page.evaluate("zoneCalled('xL','block')"))
        _cut_by(page, "#elevation-complete")
        assert all(_zones(page, wall) == zones for wall, zones in old.items())
