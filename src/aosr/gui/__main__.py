"""啟動本機網頁：預設只聽本機，可改聽這台的 Tailscale 位址。"""
from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from aosr.gui.app import LOOPBACK, TAILNET_CLIENTS, GuiSettings, create_app, listen_address


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m aosr.gui")
    parser.add_argument("--engine-commit", required=True)
    parser.add_argument("--data-dir", type=Path, default=GuiSettings.data_dir)
    parser.add_argument("--port", type=int, default=8000)
    # 從自己其他 Tailscale 裝置直接連：--listen 給這台的 Tailscale 位址（只准它或 127.0.0.1），
    # --allowed-host 列出瀏覽器會帶的名字（機器短名、全名、位址），沒列的一律拒收。可以給好幾次。
    parser.add_argument("--listen", default="127.0.0.1")
    parser.add_argument("--allowed-host", action="append", default=[])
    args = parser.parse_args()
    host = listen_address(args.listen)
    # 聽 Tailscale 位址時只收本機與 Tailscale 來的連線（區網、容器照樣送得到這個位址）。
    clients = () if host == str(LOOPBACK) else TAILNET_CLIENTS
    app = create_app(GuiSettings(engine_commit=args.engine_commit, data_dir=args.data_dir,
                                 extra_hosts=tuple(args.allowed_host), client_networks=clients))
    # 前面沒有反向代理：不信任 X-Forwarded-For，免得環境變數 FORWARDED_ALLOW_IPS 被放寬時，來源可以被標頭冒充。
    uvicorn.run(app, host=host, port=args.port, proxy_headers=False)


if __name__ == "__main__":
    main()
