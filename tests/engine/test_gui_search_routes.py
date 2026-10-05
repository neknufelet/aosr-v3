"""搜尋端點只用代號找 data_dir/searches；坏塊仍回可讀的其餘資料。"""
from __future__ import annotations

from pathlib import Path

import pytest
from starlette.testclient import TestClient

from aosr.gui.app import GuiSettings, create_app
from tests.engine._gui_cache import gui_startup_identity_memo
from tests.engine._search_run_cases import make_store
from tests.engine._gui_best_cases import best_result, best_store
from aosr.reporting.result import SchemeResult


@pytest.mark.parametrize("name", ["../outside", "BAD", "a" * 31, "A" * 32, "g" * 32, "a" * 33])
def test_search_routes_reject_invalid_id(tmp_path: Path, name: str) -> None:
    with TestClient(create_app(GuiSettings(engine_commit="a" * 40, data_dir=tmp_path)), base_url="http://localhost") as client:
        for prefix in ("/api/searches", "/searches"):
            assert client.get(f"{prefix}/{name}").status_code in {400, 404}


def test_search_routes_missing_and_readable_folder(tmp_path: Path) -> None:
    store, _ = make_store(tmp_path)
    with TestClient(create_app(GuiSettings(engine_commit="a" * 40, data_dir=tmp_path)), base_url="http://localhost") as client:
        assert client.get(f"/api/searches/{'f' * 32}").status_code == 404
        assert client.get(f"/searches/{'f' * 32}").status_code == 404
        response = client.get(f"/api/searches/{store.search_id}")
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        data = response.json()
        assert data["search_id"] == store.search_id
        assert {block["key"] for block in data["blocks"]} == {
            "stage", "counts", "timings", "updated", "search-best", "refine-best", "reasons", "identity"}
        assert client.get(f"/searches/{store.search_id}").status_code == 200
        listing = client.get("/api/searches").json()
        assert {item["search_id"] for item in listing["searches"]} == {store.search_id}
        assert client.get("/searches").status_code == 200
        for asset_name in ("searches.js", "searches.css", "capabilities.js"):
            assert client.get(f"/static/{asset_name}").status_code == 200
        assert '/searches' in client.get("/").text


def test_search_listing_ignores_unusable_names_and_symlinks(tmp_path: Path) -> None:
    store, _ = make_store(tmp_path)
    (tmp_path / "searches" / "unrelated").mkdir()
    (tmp_path / "searches" / ("d" * 32)).symlink_to(store.path, target_is_directory=True)
    with TestClient(create_app(GuiSettings(engine_commit="a" * 40, data_dir=tmp_path)), base_url="http://localhost") as client:
        assert {item["search_id"] for item in client.get("/api/searches").json()["searches"]} == {store.search_id}
        assert client.get(f"/api/searches/{'d' * 32}").status_code == 404


def test_best_route_and_invalid_selection(tmp_path: Path, best_result: SchemeResult) -> None:
    store = best_store(tmp_path, best_result, "baseline")
    with TestClient(create_app(GuiSettings(engine_commit="a" * 40, data_dir=tmp_path)), base_url="http://localhost") as client:
        endpoint = f"/api/searches/{store.search_id}/best"
        assert client.get(endpoint).json()["which"] == "refine"
        assert client.get(f"{endpoint}?which=invalid").status_code == 400
        assert client.get(f"/api/searches/{'f' * 32}/best?which=search").status_code == 404
        for which in ("search", "refine"):
            response = client.get(f"{endpoint}?which={which}")
            assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
            data = response.json()
            assert data["which"] == which
            assert all(not data[key]["error"] for key in ("frequency", "rfz", "plan"))
        versions = client.get(f"/api/searches/{store.search_id}").json()["best_versions"]
        assert data["version"] == versions["refine"]["version"]
        store.refine_result_path(None).unlink()
        missing = client.get(f"{endpoint}?which=refine").json()
        assert all(missing[key]["error"].startswith("讀不到：") for key in ("frequency", "rfz", "plan"))


def test_best_rejects_result_symlink(tmp_path: Path, best_result: SchemeResult) -> None:
    store = best_store(tmp_path, best_result)
    target = store.candidate_path(0)
    outside = tmp_path / "outside.json"
    target.rename(outside)
    target.symlink_to(outside)
    with TestClient(create_app(GuiSettings(engine_commit="a" * 40, data_dir=tmp_path)), base_url="http://localhost") as client:
        view = client.get(f"/api/searches/{store.search_id}/best?which=search").json()
        assert all("符號連結" in view[key]["error"] for key in ("frequency", "rfz", "plan"))
