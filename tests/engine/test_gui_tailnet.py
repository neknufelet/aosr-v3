"""從自己其他 Tailscale 裝置直接連：只准聽本機或 Tailscale 位址，另外准許的網址只認明列的那幾個。"""
from __future__ import annotations

import sys
from http import HTTPStatus
from pathlib import Path
from typing import cast

import pytest
import uvicorn
from starlette.applications import Starlette
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.testclient import TestClient

from aosr.gui import __main__ as gui_main
from aosr.gui.app import LOCAL_HOSTS, GuiSettings, create_app, listen_address

COMMIT = "a" * 40
SHORT = "florian-coder"
FULL = "florian-coder.example.ts.net"


def _status(tmp_path: Path, host: str, extra_hosts: tuple[str, ...]) -> HTTPStatus:
    app = create_app(GuiSettings(engine_commit=COMMIT, data_dir=tmp_path, extra_hosts=extra_hosts))
    with TestClient(app, base_url="http://localhost") as client:
        return HTTPStatus(client.get("/api/schemes", headers={"host": host}).status_code)


def test_listed_names_are_accepted_and_others_stay_rejected(tmp_path: Path) -> None:
    listed = (SHORT, FULL, "100.71.26.77")
    # 瀏覽器帶的 Host 會有埠號（florian-coder:8765），把關比對時去掉埠號。
    for host in (SHORT, f"{SHORT}:8765", FULL, "100.71.26.77:8765", "localhost"):
        assert _status(tmp_path, host, listed) is HTTPStatus.OK, host
    assert _status(tmp_path, "other.example.ts.net", listed) is HTTPStatus.BAD_REQUEST
    # 沒列就跟以前一樣只認本機。
    assert _status(tmp_path, SHORT, ()) is HTTPStatus.BAD_REQUEST


@pytest.mark.parametrize("host", ["*", "*.ts.net", "Florian-coder", "florian-coder:8765",
                                  "a..ts.net", "-a.ts.net", "a b", ""])
def test_wildcards_ports_and_malformed_hosts_are_refused_at_start(tmp_path: Path, host: str) -> None:
    with pytest.raises(ValueError, match="小寫主機名"):
        create_app(GuiSettings(engine_commit=COMMIT, data_dir=tmp_path, extra_hosts=(host,)))


@pytest.mark.parametrize("address", ["127.0.0.1", "100.71.26.77", "100.64.0.1", "100.127.255.254"])
def test_listen_accepts_loopback_and_tailscale_addresses(address: str) -> None:
    assert listen_address(address) == address


@pytest.mark.parametrize("address", ["0.0.0.0", "192.168.1.10", "10.0.0.5", "100.63.255.255",
                                     "100.128.0.1", "127.0.0.2", "::", "fd7a:115c:a1e0::1",
                                     "localhost", "100.71.26.77:8765", ""])
def test_listen_refuses_every_other_address(address: str) -> None:
    with pytest.raises(ValueError, match="聽"):
        listen_address(address)


def _run_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *extra: str) -> dict[str, object]:
    captured: dict[str, object] = {}

    def fake_run(app: Starlette, *, host: str, port: int) -> None:
        captured.update(app=app, host=host, port=port)

    monkeypatch.setattr(uvicorn, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["python -m aosr.gui", "--engine-commit", COMMIT,
                                      "--data-dir", str(tmp_path), "--port", "8765", *extra])
    gui_main.main()
    return captured


def test_command_line_defaults_to_loopback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert _run_main(tmp_path, monkeypatch)["host"] == "127.0.0.1"


def test_command_line_listens_on_tailscale_and_passes_listed_names(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _run_main(tmp_path, monkeypatch, "--listen", "100.71.26.77",
                         "--allowed-host", SHORT, "--allowed-host", FULL)
    assert captured["host"] == "100.71.26.77"
    assert captured["port"] == 8765
    app = captured["app"]
    assert isinstance(app, Starlette)
    hosts = [item.kwargs["allowed_hosts"] for item in app.user_middleware
             if cast(object, item.cls) is TrustedHostMiddleware]
    assert hosts == [[*LOCAL_HOSTS, SHORT, FULL]]


def test_command_line_refuses_to_listen_on_every_interface(tmp_path: Path,
                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="不准 0.0.0.0"):
        _run_main(tmp_path, monkeypatch, "--listen", "0.0.0.0")
