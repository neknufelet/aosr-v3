"""#1181：網頁與靜態檔都帶 Cache-Control: no-cache（合併重開後瀏覽器不再沿用舊的 js；這一版 Starlette 不回 304，每次重抓整檔，檔都小）；本來就是 no-store 的不被蓋掉。"""
from __future__ import annotations

from pathlib import Path

from starlette.testclient import TestClient

from aosr.gui.app import GuiSettings, create_app


def test_pages_and_static_files_ask_the_server_every_time(tmp_path: Path) -> None:
    with TestClient(create_app(GuiSettings(engine_commit="a" * 40, data_dir=tmp_path)), base_url="http://localhost") as client:
        for path in ("/", "/static/zone_modes.js", "/static/app.js", "/static/home.css", "/gradient", "/static/gradient.js"):
            response = client.get(path)
            assert response.status_code == 200, path
            assert response.headers["cache-control"] == "no-cache", path
            assert response.headers.get("etag"), path


def test_no_store_answers_keep_no_store(tmp_path: Path) -> None:
    with TestClient(create_app(GuiSettings(engine_commit="a" * 40, data_dir=tmp_path)), base_url="http://localhost") as client:
        response = client.get(f"/api/searches/{'f' * 32}")
        assert response.status_code == 404
        listing = client.get("/api/materials")
        assert listing.status_code == 200 and listing.headers["cache-control"] == "no-cache"
        assert client.get("/api/searches").headers["cache-control"] == "no-store"
