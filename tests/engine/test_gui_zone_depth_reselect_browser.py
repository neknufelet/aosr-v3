"""真瀏覽器從厚度不明的固定區重選：每次看同一區的五公分限制，薄構造存好再開跑。"""
import json
from pathlib import Path

from playwright.sync_api import Browser, expect

from aosr.materials import library
from tests.engine.test_gui_browser import _open, browser as browser
from tests.engine.test_gui_gradient_browser import _serve, _shot
from tests.engine.test_gui_gradient_jobs import FakeLauncher, _place_surrogate
from tests.engine.test_gui_zone_depth_mismatch import _panel_files, _panel_trial
from tests.engine.test_gui_zone_elevation_browser import _open_scheme
from tests.engine.test_gui_zone_limits import _ENTRY, _scheme, _zones
from tests.engine.test_gui_zone_modes_browser import _save
from tests.engine.test_gui_zone_overview_tasks_browser import _apply, _click_zone


def test_reselect_rechecks_the_same_zone_cap(tmp_path: Path, browser: Browser) -> None:
    scheme_file, design_file, catalog = _panel_files(tmp_path, original="none")
    document, entry = _panel_trial()
    deep = entry | {"id": "woven_deep", "label": "自編深織板"}
    shallow = entry | {"id": "woven_slim", "label": "自編薄織板",
                       "layers": [{"kind": "air", "thickness_m": 0.034}]}
    catalog.write_text(library.dump_user_library(tuple(_ENTRY.validate_python(item) for item in (deep, shallow))))
    _zones(document)["x0"][0]["material"] = {"kind": "constant_impedance", "impedance_pa_s_m": 1847.0}
    scheme = _scheme(document)
    scheme_file.write_text(scheme.model_dump_json())
    _place_surrogate(tmp_path, scheme)
    launcher = FakeLauncher()
    with _serve(tmp_path, launcher) as base, _open(browser, base) as watched:
        page = watched.page
        name = scheme_file.stem
        page.goto(f"{base}/gradient?scheme={name}")
        expect(page.locator("#scheme-message")).to_contain_text("重選一筆材料")
        expect(page.locator("#start-block")).to_be_hidden()
        _shot(page, "l1-reselect-unknown")
        page.goto(base)
        _open_scheme(page, name)
        for identifier in ("woven_deep", "woven_slim"):
            page.locator("#zone-page-open").click()
            _click_zone(page, "x0/whole_wall")
            page.locator("#zone-panel-mode-fixed").check()
            page.locator("#zone-panel-material").select_option(identifier)
            _apply(page)
            page.locator("#zone-page-close").click()
            _save(page, name)
            if identifier == "woven_deep":
                page.goto(f"{base}/gradient?scheme={name}")
                expect(page.locator("#scheme-message")).to_contain_text("總深 0.081 公尺超過深度上限 0.05 公尺")
                expect(page.locator("#start-block")).to_be_hidden()
                _shot(page, "l1-reselect-over")
                page.goto(base)
                _open_scheme(page, name)
        saved = json.loads(design_file.read_bytes())["zones"]["x0"]["whole_wall"]
        assert saved == {"mode": "fixed", "picked_entry": "woven_slim", "max_depth_m": 0.05}
        page.goto(f"{base}/gradient?scheme={name}")
        expect(page.locator("#start-block")).to_be_visible()
        preview = page.request.post(f"{base}/api/gradient/schemes/{name}/plan", data={})
        assert preview.ok
        row = preview.json()["fixed_depths"]["x0"]
        assert row["cap"]["value_m"] == 0.05 and "總深 0.034 公尺，放得進" in row["lines"][0]["text"]
        _shot(page, "l1-reselect-fits")
        assert page.request.post(f"{base}/api/gradient/schemes/{name}/start", data={}).ok
        assert len(launcher.launched) == 1 and watched.page_errors == []
