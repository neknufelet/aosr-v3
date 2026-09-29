"""啟動本機網頁：預設只聽本機，可改聽這台的 Tailscale 位址。"""
from __future__ import annotations

import argparse
from pathlib import Path

from aosr.reporting.calculation_fingerprint import calculation_fingerprint


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m aosr.gui")
    parser.add_argument("--engine-commit", required=True)
    # 預設值住在 GuiSettings；這裡不寫第二份，等 app 載入後再補（量指紋前不准載入 app）。
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--port", type=int, default=8000)
    # 從自己其他 Tailscale 裝置直接連：--listen 給這台的 Tailscale 位址（只准它或 127.0.0.1），
    # --allowed-host 列出瀏覽器會帶的名字（機器短名、全名、位址），沒列的一律拒收。可以給好幾次。
    parser.add_argument("--listen", default="127.0.0.1")
    parser.add_argument("--allowed-host", action="append", default=[])
    args = parser.parse_args()
    from aosr.config.paths import config_path

    startup_fingerprint = calculation_fingerprint(capabilities_path=config_path("capabilities.toml"))
    import uvicorn

    from aosr.gui.app import LOOPBACK, TAILNET_CLIENTS, GuiSettings, create_app, listen_address

    host = listen_address(args.listen)
    # 聽 Tailscale 位址時只收本機與 Tailscale 來的連線（區網、容器照樣送得到這個位址）。
    clients = () if host == str(LOOPBACK) else TAILNET_CLIENTS
    data_dir = args.data_dir if args.data_dir is not None else GuiSettings.data_dir
    app = create_app(GuiSettings(engine_commit=args.engine_commit, data_dir=data_dir,
                                 extra_hosts=tuple(args.allowed_host), client_networks=clients,
                                 startup_fingerprint=startup_fingerprint))
    # 前面沒有反向代理：不信任 X-Forwarded-For，免得環境變數 FORWARDED_ALLOW_IPS 被放寬時，來源可以被標頭冒充。
    uvicorn.run(app, host=host, port=args.port, proxy_headers=False)


if __name__ == "__main__":
    main()
