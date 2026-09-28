"""從自己的手機看網頁：另外准許的網址只認明列的那幾個，伺服器照舊只聽 127.0.0.1。"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

import pytest
import uvicorn
from starlette.applications import Starlette
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.testclient import TestClient

from aosr.gui import __main__ as gui_main
from aosr.gui.app import LOCAL_HOSTS, GuiSettings, create_app

COMMIT = "a" * 40
TAILNET = "florian-coder.example.ts.net"


def _status(tmp_path: Path, host: str, extra_hosts: tuple[str, ...]) -> int:
    app = create_app(GuiSettings(engine_commit=COMMIT, data_dir=tmp_path, extra_hosts=extra_hosts))
    with TestClient(app, base_url="http://localhost") as client:
        status: int = client.get("/api/schemes", headers={"host": host}).status_code
    return status


def test_listed_host_is_accepted_and_others_stay_rejected(tmp_path: Path) -> None:
    assert _status(tmp_path, TAILNET, (TAILNET,)) == 200
    assert _status(tmp_path, "localhost", (TAILNET,)) == 200
    assert _status(tmp_path, "other.example.ts.net", (TAILNET,)) == 400
    # 沒列就跟以前一樣只認本機。
    assert _status(tmp_path, TAILNET, ()) == 400


@pytest.mark.parametrize("host", ["*", "*.ts.net", "Florian-coder.example.ts.net",
                                  "florian-coder.example.ts.net:443", "florian-coder",
                                  "a..ts.net", "-a.ts.net", "a b.ts.net", ""])
def test_wildcards_ports_and_malformed_hosts_are_refused_at_start(tmp_path: Path, host: str) -> None:
    with pytest.raises(ValueError, match="小寫點分主機名"):
        create_app(GuiSettings(engine_commit=COMMIT, data_dir=tmp_path, extra_hosts=(host,)))


def test_command_line_still_binds_loopback_and_passes_listed_hosts(tmp_path: Path,
                                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(app: Starlette, *, host: str, port: int) -> None:
        captured.update(app=app, host=host, port=port)

    monkeypatch.setattr(uvicorn, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["python -m aosr.gui", "--engine-commit", COMMIT,
                                      "--data-dir", str(tmp_path), "--port", "8765",
                                      "--allowed-host", TAILNET])
    gui_main.main()
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8765
    app = captured["app"]
    assert isinstance(app, Starlette)
    hosts = [item.kwargs["allowed_hosts"] for item in app.user_middleware
             if cast(object, item.cls) is TrustedHostMiddleware]
    assert hosts == [[*LOCAL_HOSTS, TAILNET]]
