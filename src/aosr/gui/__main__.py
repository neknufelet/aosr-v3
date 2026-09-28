"""只在迴圈位址啟動本機網頁。"""
from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from aosr.gui.app import GuiSettings, create_app


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m aosr.gui")
    parser.add_argument("--engine-commit", required=True)
    parser.add_argument("--data-dir", type=Path, default=GuiSettings.data_dir)
    parser.add_argument("--port", type=int, default=8000)
    # 從自己的手機、平板看：用 Tailscale Serve 把 https 網址轉到這裡，再把那個網址列在這裡；
    # 伺服器照舊只聽 127.0.0.1，沒列的網址一律拒收。可以給好幾次。
    parser.add_argument("--allowed-host", action="append", default=[])
    args = parser.parse_args()
    app = create_app(GuiSettings(engine_commit=args.engine_commit, data_dir=args.data_dir,
                                 extra_hosts=tuple(args.allowed_host)))
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
