"""從自己其他 Tailscale 裝置直接連：只准聽本機或 Tailscale 位址，另外准許的網址只認明列的那幾個。"""
from __future__ import annotations

import asyncio
import sys
from http import HTTPStatus
from pathlib import Path
from typing import cast

import pytest
import uvicorn
from starlette.applications import Starlette
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.testclient import TestClient
from starlette.types import Message, Scope

from aosr.gui import __main__ as gui_main
from aosr.gui.app import (LOCAL_HOSTS, TAILNET_CLIENTS, GuiSettings, client_allowed, create_app,
                          listen_address)

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
                                     "localhost", "100.71.26.77:8765", "",
                                     "100.071.026.077", "1681267277"])
def test_listen_refuses_every_other_address(address: str) -> None:
    with pytest.raises(ValueError, match="聽"):
        listen_address(address)


@pytest.mark.parametrize(("host", "allowed"), [
    ("127.0.0.1", True), ("100.69.223.68", True), ("100.64.0.1", True),
    ("192.168.0.5", False), ("172.17.0.2", False), ("100.128.0.1", False),
    ("100.115.92.5", False), ("100.115.93.254", False), ("100.115.94.1", True),
    ("testclient", False), ("", False), (None, False)])
def test_client_address_must_be_loopback_or_tailscale(host: str | None, allowed: bool) -> None:
    assert client_allowed(host, TAILNET_CLIENTS) is allowed


def _client_status(app: Starlette, client: str, path: str = "/api/schemes") -> HTTPStatus:
    with TestClient(app, base_url="http://localhost", client=(client, 50000)) as test_client:
        return HTTPStatus(test_client.get(path, headers={"host": SHORT}).status_code)


def _status_without_client(app: Starlette) -> HTTPStatus:
    # 伺服器讀不出連線來源（ASGI 的 client 是 None）時也要擋；TestClient 造不出這種請求，直接送原始請求。
    sent: list[Message] = []

    async def receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: Message) -> None:
        sent.append(message)

    scope: Scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "GET",
        "scheme": "http", "path": "/api/schemes", "raw_path": b"/api/schemes", "query_string": b"",
        "root_path": "", "headers": [(b"host", SHORT.encode())], "client": None,
        "server": ("100.71.26.77", 8765)}
    asyncio.run(app(scope, receive, send))
    start = next(message for message in sent if message["type"] == "http.response.start")
    return HTTPStatus(cast(int, start["status"]))


def _run_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *extra: str) -> dict[str, object]:
    captured: dict[str, object] = {}

    def fake_run(app: Starlette, *, host: str, port: int, proxy_headers: bool) -> None:
        captured.update(app=app, host=host, port=port, proxy_headers=proxy_headers)

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
    assert captured["proxy_headers"] is False
    app = captured["app"]
    assert isinstance(app, Starlette)
    hosts = [item.kwargs["allowed_hosts"] for item in app.user_middleware
             if cast(object, item.cls) is TrustedHostMiddleware]
    assert hosts == [[*LOCAL_HOSTS, SHORT, FULL]]
    # 區網與容器照樣送得到 Tailscale 位址：來源不是本機或 Tailscale 就 403。
    assert _client_status(app, "100.69.223.68") is HTTPStatus.OK
    # 首頁、靜態檔、不存在的網址也先經過這一道，不是只有 API。
    for path in ("/api/schemes", "/", "/static/app.js", "/no-such-page"):
        assert _client_status(app, "192.168.0.5", path) is HTTPStatus.FORBIDDEN, path
    assert _client_status(app, "172.17.0.2") is HTTPStatus.FORBIDDEN
    assert _status_without_client(app) is HTTPStatus.FORBIDDEN


def test_command_line_on_loopback_does_not_filter_clients(tmp_path: Path,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    app = _run_main(tmp_path, monkeypatch, "--allowed-host", SHORT)["app"]
    assert isinstance(app, Starlette)
    assert _client_status(app, "testclient") is HTTPStatus.OK


def test_command_line_refuses_to_listen_on_every_interface(tmp_path: Path,
                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="不准 0.0.0.0"):
        _run_main(tmp_path, monkeypatch, "--listen", "0.0.0.0")
