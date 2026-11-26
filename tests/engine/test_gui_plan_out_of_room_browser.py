"""#1265：喇叭、主位或周圍點跑出房間時，平面圖照樣畫，跑出去的那幾個標紅（老闆看 07-calculate-stopped 問「這個又是甚麼」）。

以前座標一超出房間，伺服器重建方案時驗證丟錯、``/api/plan`` 的 422 回覆不帶圖，前端就把四張圖全部清空、只剩灰框。
現在 422 照樣帶 ``plan``：房間外框照新尺寸畫，每個點畫在它的座標上（可能在框外），超出的那幾個列在 ``outside_keys``，
圖上用跟擋路家具同一種紅（``#c62828``）畫；畫框撐大到容得下框外的點。只有長寬高本身不是數字這類畫不出來的情況才照舊清空。
方案的數字都是考卷自己編的（預設參考房改長度、喇叭搬到房外）。"""
from __future__ import annotations

import re
from pathlib import Path
from typing import cast

from playwright.sync_api import Browser, Page, expect
from starlette.testclient import TestClient

from aosr.gui.app import GuiSettings, create_app
from tests.engine._input_page_steps import open_example_scheme, reveal, show_view
from tests.engine.test_gui_browser import _assert_quiet_but_plan_refused, _open, _serve, browser as browser

RED = "#c62828"
PROBLEM = re.compile(r"^檢查沒過：")
VIEWS = ("plan-xy", "plan-xz", "zoom-xy", "zoom-xz")
# 預設參考房的七個聆聽點（主位加六個周圍點）x 都在 3.1～3.3 公尺，喇叭在 x 1.0：長度改 2.5 時只有喇叭還在房裡。
LISTENERS = {f"receiver:{name}" for name in ("main", "front", "back", "left", "right", "up", "down")}
SPEAKERS = {"speaker:left", "speaker:right"}


def _client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(GuiSettings(engine_commit="a" * 40, data_dir=tmp_path)), base_url="http://localhost")


def test_plan_post_out_of_room_still_returns_the_drawing_and_names_the_points_outside(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        document = client.get("/api/example").json()["scheme"]
        document["scheme_id"] = "short-room"
        passed = client.post("/api/plan", json=document)
        assert passed.status_code == 200 and "outside_keys" not in passed.json()
        document["scene"]["room_m"]["Lx"] = 2.5
        rejected = client.post("/api/plan", json=document)
    assert rejected.status_code == 422
    payload = rejected.json()
    # 拒收原因照舊（這一題不改字）；圖面照新尺寸，七個聆聽點都超出、兩支喇叭沒有。
    assert [item["text"] for item in payload["problems"]] == [
        "主位：座標超出房間（x、y、z 都要在 0 到房間的長、寬、高之間，貼牆也算）；請核對這組座標和房間長寬高"]
    plan = payload["plan"]
    assert plan["room"] == {"Lx": 2.5, "Ly": 4.0, "Lz": 3.0}
    assert set(plan["outside_keys"]) == LISTENERS
    drawn = {key for item in plan["views"]["plan"] for key in item["keys"]}
    assert SPEAKERS | LISTENERS <= drawn
    main = next(item for item in plan["receivers"] if item["key"] == "receiver:main")
    assert main["point"] == {"x": 3.2, "y": 2.0, "z": 1.2}


def test_plan_post_out_of_room_with_furniture_still_draws_and_never_breaks(tmp_path: Path) -> None:
    """有家具、主位又在房外：家具換算與擋路判定不准丟成伺服器錯（500）；照樣帶圖、主位標超出。"""
    from tests.engine.test_scheme_furniture import _validation_document

    document = _validation_document("both") | {"scheme_id": "sofa-outside"}
    room = cast(dict[str, float], cast(dict[str, object], document["scene"])["room_m"])
    receivers = cast(list[dict[str, object]], cast(dict[str, object], document["receiver_set"])["points"])
    main = next(point for point in receivers if point["role"] == "primary")
    main["position_m"] = [room["Lx"] + 0.7, *cast(list[float], main["position_m"])[1:]]
    with _client(tmp_path) as client:
        response = client.post("/api/plan", json=document)
    assert response.status_code == 422, response.text
    plan = response.json()["plan"]
    assert f"receiver:{main['receiver_id']}" in plan["outside_keys"]
    assert plan["furniture"]


def _commit(page: Page, selector: str, value: str) -> None:
    field = reveal(page, selector)
    field.fill(value)
    field.blur()


def _red_keys(page: Page, view: str) -> set[str]:
    return set(cast(list[str], page.locator(f"#{view} g[data-keys]").evaluate_all(
        f"nodes => nodes.filter(n => n.querySelector('circle:not(.changed-ring)').getAttribute('fill') === '{RED}')"
        ".flatMap(n => n.dataset.keys.split(' '))")))


def _drawn_keys(page: Page, view: str) -> set[str]:
    return set(cast(list[str], page.locator(f"#{view} g[data-keys]").evaluate_all(
        "nodes => nodes.flatMap(n => n.dataset.keys.split(' '))")))


def _room_frame(page: Page, view: str) -> tuple[float, float]:
    """房間外框（深藍那一個 rect）的寬、高，單位是畫框座標。"""
    box = cast(dict[str, float], page.locator(f"#{view} rect[stroke='#334b58']").evaluate(
        "n => ({w: Number(n.getAttribute('width')), h: Number(n.getAttribute('height'))})"))
    return box["w"], box["h"]


def _inside_viewbox(page: Page, view: str) -> list[str]:
    """畫框外面的點與字（回傳哪幾個露出去）：每一個實心點與它的字，外接框都要在 viewBox 裡。"""
    return cast(list[str], page.locator(f"#{view}").evaluate("""svg => {
      const [x0, y0, w, h] = svg.getAttribute('viewBox').split(' ').map(Number);
      const out = [];
      for (const node of svg.querySelectorAll('g[data-keys] circle:not(.changed-ring), g[data-keys] text')) {
        const b = node.getBBox();
        if (b.x < x0 || b.y < y0 || b.x + b.width > x0 + w || b.y + b.height > y0 + h)
          out.push(`${node.closest('g').dataset.keys} ${node.tagName}`);
      }
      return out;
    }"""))


def test_short_room_keeps_drawing_room_speakers_and_listener_with_the_outside_ones_red(tmp_path: Path, browser: Browser) -> None:
    """預設參考房長度改 2.5：四張圖都有畫；房間外框照 2.5 × 4 畫；喇叭有畫、沒紅；主位與周圍點畫在框外、標紅、沒被裁掉。
    改回 6：紅標消失、「檢查通過」。"""
    with _serve(tmp_path) as base, _open(browser, "about:blank", viewport_width=1440) as watched:
        page = watched.page
        page.set_viewport_size({"width": 1440, "height": 900})
        open_example_scheme(page, base, tmp_path)
        check = page.locator("#check-state")
        expect(check).to_have_text("檢查通過")
        _commit(page, "#room-Lx", "2.5")
        expect(check).to_have_text(PROBLEM)
        page.wait_for_function("document.querySelectorAll('#plan-xy g[data-keys]').length > 0")
        width, height = _room_frame(page, "plan-xy")
        assert abs(width / height - 2.5 / 4.0) < 1e-6, (width, height)
        side_width, side_height = _room_frame(page, "plan-xz")
        assert abs(side_width / side_height - 2.5 / 3.0) < 1e-6, (side_width, side_height)
        for view in VIEWS:
            show_view(page, view)
            drawn, red = _drawn_keys(page, view), _red_keys(page, view)
            assert "receiver:main" in drawn and "receiver:main" in red, (view, drawn, red)
            assert red <= LISTENERS, (view, red)
            if view.startswith("plan"):
                assert SPEAKERS <= drawn, (view, drawn)
            assert _inside_viewbox(page, view) == [], view
        legend = reveal(page, "#plan-legend li[data-id='receiver:main']")
        assert "x 3.20" in legend.inner_text()
        _commit(page, "#room-Lx", "6")
        expect(check).to_have_text("檢查通過")
        for view in VIEWS:
            assert _red_keys(page, view) == set(), view
        _assert_quiet_but_plan_refused(watched)


def test_a_speaker_far_outside_the_room_widens_the_drawing_instead_of_being_cut_off(tmp_path: Path, browser: Browser) -> None:
    """左喇叭搬到房外很遠（x 負、y 與 z 都超出）：平面、側面兩張都畫得到它、標紅，點與字都在畫框裡。"""
    with _serve(tmp_path) as base, _open(browser, "about:blank", viewport_width=1440) as watched:
        page = watched.page
        page.set_viewport_size({"width": 1440, "height": 900})
        open_example_scheme(page, base, tmp_path)
        check = page.locator("#check-state")
        for axis, value in (("x", "-3.5"), ("y", "9.25"), ("z", "7.75")):
            _commit(page, f"#speaker-left-{axis}", value)
        page.wait_for_function("document.getElementById('check-state').dataset.kind !== 'pending'")
        expect(check).to_have_text(PROBLEM)
        page.wait_for_function("shownPlan && shownPlan.speakers.some(s => s.id === 'left' && s.point.z === 7.75)")
        for view in ("plan-xy", "plan-xz"):
            show_view(page, view)
            assert _red_keys(page, view) == {"speaker:left"}, view
            assert _inside_viewbox(page, view) == [], view
        _assert_quiet_but_plan_refused(watched)
