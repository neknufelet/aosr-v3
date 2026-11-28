"""全站新外框（#1342）：共用樣式與頂列／左功能列。

一頁換新外框＝連 app.css、載 app_shell.js、body 有 class="app"；功能列只列伺服器真的有的頁。
"""
from __future__ import annotations

import re
from pathlib import Path

from starlette.testclient import TestClient

from aosr.gui.app import STATIC, GuiSettings, create_app

# 方案輸入頁（#1360）的腳本檔名不共用一個前綴（而且 app*.js 會掃到外框自己的 app_shell.js），照 index.html 列整份。
INPUT_PAGE_SCRIPTS = ("plan.js", "measured.js", "walls.js", "zone_names.js", "elevation.js", "zone_modes.js", "zone_limits.js",
                      "check_state.js", "zone_rules.js", "material-kind.js", "zone_overview.js", "wall_section.js",
                      "room_facts.js", "home.js", "app.js")
ADOPTED: dict[str, str | tuple[str, ...]] = {  # 已換新外框的頁 → 它的腳本前綴（或整份腳本檔名）
    "/": INPUT_PAGE_SCRIPTS,
    "/room-model": "room_model",
    "/gradient": "gradient",
    "/search-start": "search_start",
    "/materials": "materials",
    "/materials/designer": "material-designer",
    "/results/" + "b" * 32: "results",
    "/compare/a/b": "compare",
    "/searches": "searches",
}


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
            # body 上可以還有別的屬性（方案輸入頁的 data-input-start 是開頁那道門），class 要是 app。
            assert "app_shell.js" in scripts and re.search(r'<body class="app"[ >]', html), path
            ids = set(re.findall(r'id="([^"]+)"', html))
            files = [STATIC / name for name in prefix] if isinstance(prefix, tuple) else sorted(STATIC.glob(f"{prefix}*.js"))
            # 列整份的頁：一支腳本取的節點可能是同一頁另一支腳本生的，「是不是腳本自己生的」照整頁的腳本一起判。
            page_code = "\n".join(script.read_text(encoding="utf-8") for script in files)
            for script in files:
                code = script.read_text(encoding="utf-8")
                wanted = set(re.findall(r'(?:\$|\bbyId)\("([\w-]+)"\)', code))
                code = page_code if isinstance(prefix, tuple) else code
                wanted -= set(re.findall(r'if \((?:\$|byId)\("([\w-]+)"\)\)', code))  # 先問在不在的是腳本自己生的節點
                # 字串在取元素以外的地方也出現（例如 toggle("x", …)、.id = "x"）：那是腳本自己生的節點
                wanted = {name for name in wanted
                          if code.count(f'"{name}"') == len(re.findall(rf'(?:\$|\bbyId)\("{re.escape(name)}"\)', code))}
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
