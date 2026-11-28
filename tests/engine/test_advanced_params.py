"""進階參數清單（#1343）：登記檔的預設＝程式實際用的預設、API 與唯讀頁照清單吐、清單寫錯開機就炸。

「實際用的預設」從用值的那一端取：切區看 ``zone_cut_defaults()``（/api/input-defaults 交給網頁的那一份）、
第三階段看 ``GradientRequest()`` 的欄位預設（開跑命令列一律帶 ``--mount``／``--device``）與引擎設定檔載入器
（步數上限不帶 ``--max-steps`` 時命令列讀的那一份）、搜尋開跑頁看腳本讀的是清單哪幾格。
"""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError
from starlette.testclient import TestClient

from aosr.config.advanced_params import AdvancedParams, load_advanced_params, resolved_default
from aosr.config.fem_reduced import load_reduced_model_settings
from aosr.config.material_gradient import load_material_gradient_settings
from aosr.config.paths import CONFIG_DIR, config_path
from aosr.gui.app import STATIC_NAMES, GuiSettings, create_app
from aosr.gui.gradient_plan import GradientRequest
from aosr.gui.params_view import REGISTRY_FILE, params_listing
from aosr.gui.zone_edit import MIN_SIDE_M, SNAP_M, zone_cut_defaults

STATIC = Path(__file__).resolve().parents[2] / "src" / "aosr" / "gui" / "static"
PAGE_FILES = {"input": "index.html", "search_start": "search_start.html", "gradient": "gradient.html"}


def _registry() -> AdvancedParams:
    return load_advanced_params(config_path(REGISTRY_FILE))


def _default(key: str) -> object:
    return resolved_default(_registry().by_key(key), CONFIG_DIR)


def test_every_param_names_a_field_that_exists_on_its_page() -> None:
    registry = _registry()
    assert {page.id for page in registry.page} == set(PAGE_FILES)
    for param in registry.param:
        html = (STATIC / PAGE_FILES[param.page]).read_text(encoding="utf-8")
        assert f'id="{param.field}"' in html, f"{param.key} 指的 #{param.field} 不在 {PAGE_FILES[param.page]}"


def test_zone_cut_defaults_come_from_the_registry() -> None:
    assert SNAP_M == _default("zone_cut.snap_m") == zone_cut_defaults()["snap_m"]
    assert MIN_SIDE_M == _default("zone_cut.min_side_m") == zone_cut_defaults()["min_side_m"]


def test_gradient_request_defaults_come_from_the_registry() -> None:
    request = GradientRequest()
    assert request.mount == _default("gradient.mount")
    assert request.device == _default("gradient.device")
    # 沒填就不帶命令列參數：清單的「沒有固定值」對上請求的 None。
    assert request.frequency_batch is None and _default("gradient.frequency_batch") is None
    assert request.max_steps is None


def test_sourced_defaults_equal_the_engine_settings_and_keep_their_type() -> None:
    settings = load_material_gradient_settings(config_path("material_gradient.toml"))
    steps = _default("gradient.max_steps")
    assert steps == settings.descent.max_steps and type(steps) is int
    assert _default("gradient.family") == settings.family[0].id
    reduced = load_reduced_model_settings(config_path("fem_reduced.toml"))
    solves = _default("surrogate.max_full_solves")
    assert solves == reduced.max_full_solves and type(solves) is int
    assert _default("surrogate.stop_residual") == reduced.stop_residual


def test_sourced_params_keep_no_number_of_their_own() -> None:
    with config_path(REGISTRY_FILE).open("rb") as handle:
        raw = tomllib.load(handle)
    sourced = [param for param in raw["param"] if "source" in param]
    assert {param["key"] for param in sourced} >= {"gradient.max_steps", "surrogate.max_full_solves", "surrogate.stop_residual"}
    assert all("default" not in param for param in sourced)


def test_search_start_script_reads_its_numbers_from_the_registry() -> None:
    script = (STATIC / "search_start.js").read_text(encoding="utf-8")
    body = script[script.index("function defaultAdvanced("):]
    body = body[:body.index("\n}\n")]
    keys = set(re.findall(r'defaults\["([a-z_.]+)"\]', body))
    assert keys == {"search.n_startup_trials", "search.batch_size", "search.max_workers", "search.constant_liar"}
    assert all(key in {param.key for param in _registry().param} for key in keys)
    assert not re.search(r'put\("(startup|batch|workers)",\s*\d', script), "搜尋開跑頁又寫回一份數字"
    assert "/api/params" in script
    assert 'byId("liar").checked = true' not in script
    assert not re.search(r'id="liar"[^>]*checked', (STATIC / "search_start.html").read_text(encoding="utf-8"))


def test_gradient_page_takes_its_default_marks_from_the_server() -> None:
    html = (STATIC / "gradient.html").read_text(encoding="utf-8")
    assert "（預設）</option>" not in html, "裝法／裝置的預設改由伺服器標，選項字不另寫"
    script = (STATIC / "gradient.js").read_text(encoding="utf-8")
    assert 'markDefault("mount", defaults.mount)' in script and 'markDefault("device", defaults.device)' in script


def test_api_and_page_serve_the_registry(tmp_path: Path) -> None:
    with TestClient(create_app(GuiSettings(engine_commit="a" * 40, data_dir=tmp_path)), base_url="http://127.0.0.1") as client:
        reply = client.get("/api/params")
        assert reply.status_code == 200
        body = reply.json()
        assert body == params_listing()
        rows = {row["key"]: row for row in body["params"]}
        assert set(rows) == {param.key for param in _registry().param}
        for key, row in rows.items():
            assert row["default"] == _default(key)
            assert {"name", "unit", "kind", "min", "max", "choices", "blurb", "page", "default_note"} <= set(row)
        assert rows["gradient.max_steps"]["default_from"] == "material_gradient.toml 的 descent.max_steps"
        assert rows["zone_cut.snap_m"]["default_from"] == REGISTRY_FILE
        assert rows["search.seed"]["default_from"] is None and rows["search.seed"]["default_note"]
        page = client.get("/params")
        assert page.status_code == 200 and page.headers["content-type"].startswith("text/html")
        assert "params.js" in page.text
        for name in ("params.js", "params.css"):
            assert name in STATIC_NAMES
            assert client.get(f"/static/{name}").status_code == 200


def _write(tmp_path: Path, param: str, extra: str = "") -> Path:
    (tmp_path / "other.toml").write_text("[a]\nb = 3\nc = 1.5\n" + extra, encoding="utf-8")
    path = tmp_path / "params.toml"
    path.write_text('[[page]]\nid = "p"\nname = "頁"\npath = "/"\n\n[[param]]\nkey = "x.y"\npage = "p"\nfield = "f"\n'
                    f'name = "名"\nunit = ""\nblurb = "說明"\n{param}', encoding="utf-8")
    return path


@pytest.mark.parametrize("param", [
    'kind = "integer"\ndefault = 3\nsource = { file = "other.toml", key = "a.b" }\n',
    'kind = "integer"\n',
    'kind = "integer"\ndefault = 0\nmin = 1\n',
    'kind = "number"\ndefault = 0\nmin = 0\nmin_open = true\n',
    'kind = "integer"\ndefault = 1.5\n',
    'kind = "integer"\ndefault = true\n',
    'kind = "choice"\ndefault = "c"\nchoices = [{ value = "a", label = "甲" }]\n',
    'kind = "text"\ndefault = "a"\nchoices = [{ value = "a", label = "甲" }]\n',
], ids=["two homes", "no default at all", "below min", "at open min", "float for integer", "bool for integer",
        "choice not listed", "choices on text"])
def test_bad_registry_refuses_to_load(tmp_path: Path, param: str) -> None:
    with pytest.raises(ValidationError):
        load_advanced_params(_write(tmp_path, param))


@pytest.mark.parametrize("source,kind", [("a.missing", "integer"), ("a.c", "integer"), ("a.b", "integer")],
                         ids=["missing key", "wrong type", "out of range"])
def test_bad_source_refuses_to_resolve(tmp_path: Path, source: str, kind: str) -> None:
    limit = "max = 2\n" if source == "a.b" else ""
    registry = load_advanced_params(_write(tmp_path, f'kind = "{kind}"\nsource = {{ file = "other.toml", key = "{source}" }}\n{limit}'))
    with pytest.raises(ValueError):
        resolved_default(registry.by_key("x.y"), tmp_path)


def test_source_reads_the_other_file_not_a_copy(tmp_path: Path) -> None:
    registry = load_advanced_params(_write(tmp_path, 'kind = "integer"\nsource = { file = "other.toml", key = "a.b" }\n'))
    assert resolved_default(registry.by_key("x.y"), tmp_path) == 3
    (tmp_path / "other.toml").write_text("[a]\nb = 7\n", encoding="utf-8")
    assert resolved_default(registry.by_key("x.y"), tmp_path) == 7
