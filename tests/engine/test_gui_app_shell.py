"""全站新外框（#1342）：共用樣式與頂列／左功能列。

一頁換新外框＝連 app.css、載 app_shell.js、body 有 class="app"；功能列只列伺服器真的有的頁。
"""
from __future__ import annotations

import re
from pathlib import Path

from starlette.testclient import TestClient

from aosr.gui.app import STATIC, GuiSettings, create_app

ADOPTED = {"/room-model": "room_model", "/gradient": "gradient", "/search-start": "search_start",
           "/materials": "materials", "/materials/designer": "material-designer"}  # 已換新外框的頁 → 它的腳本前綴


def _app(tmp_path: Path) -> TestClient:
    return TestClient(create_app(GuiSettings(engine_commit="a" * 40, data_dir=tmp_path)), base_url="http://localhost")


def _nav() -> list[str]:
    return re.findall(r'\["(/[^"]*)", "[^"]+", "[^"]+"\]', (STATIC / "app_shell.js").read_text(encoding="utf-8"))


def test_every_nav_entry_is_a_real_page(tmp_path: Path) -> None:
    pages = _nav()
    assert len(pages) >= 5 and pages[0] == "/"
    with _app(tmp_path) as client:
        for path in pages:
            response = client.get(path)
            assert response.status_code == 200 and "text/html" in response.headers["content-type"], path


def test_shared_files_are_served(tmp_path: Path) -> None:
    with _app(tmp_path) as client:
        for name in ("app.css", "app_shell.js"):
            assert client.get(f"/static/{name}").status_code == 200


def test_adopted_pages_use_the_new_frame_and_keep_every_script_id(tmp_path: Path) -> None:
    with _app(tmp_path) as client:
        for path, prefix in ADOPTED.items():
            html = client.get(path).text
            links = re.findall(r'<link rel="stylesheet" href="/static/([^"]+)"', html)
            assert links[0] == "app.css" and "style.css" not in links
            scripts = re.findall(r'<script src="/static/([^"]+)"', html)
            assert "app_shell.js" in scripts and '<body class="app">' in html
            ids = set(re.findall(r'id="([^"]+)"', html))
            for script in STATIC.glob(f"{prefix}*.js"):
                code = script.read_text(encoding="utf-8")
                wanted = set(re.findall(r'(?:\$|\bbyId)\("([\w-]+)"\)', code))
                wanted -= set(re.findall(r'if \((?:\$|byId)\("([\w-]+)"\)\)', code))  # 先問在不在的是腳本自己生的節點
                assert wanted <= ids, (script.name, wanted - ids)


def _luminance(hex_color: str) -> float:
    channels = [int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def test_text_colors_are_readable_on_every_background() -> None:
    css = (STATIC / "app.css").read_text(encoding="utf-8")
    token = dict(re.findall(r"--([a-z-]+):(#[0-9a-f]{6})", css))
    for text in ("ink", "soft", "acc", "good", "warn", "bad"):
        for ground in ("bg", "panel", "sunk"):
            light, dark = sorted((_luminance(token[text]), _luminance(token[ground])), reverse=True)
            assert (light + 0.05) / (dark + 0.05) >= 4.5, (text, ground)
