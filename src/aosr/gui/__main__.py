"""啟動本機網頁：預設只聽本機，可改聽這台的 Tailscale 位址。"""
from __future__ import annotations

import argparse
import fcntl
import os
import sys
from pathlib import Path

from aosr.reporting.calculation_fingerprint import calculation_fingerprint

# 收到停止訊號後最多等幾秒讓在途連線收尾（uvicorn 預設不設上限）。一條上傳到一半停住、連線沒關的
# 請求（例如手機上傳到一半睡著），沒上限時會一直等，舊服務就一直拿著資料夾鎖、新服務起不來（#687 審查實測）。
SHUTDOWN_GRACE_SECONDS = 5


def hold_data_folder(path: Path) -> int:
    """鎖資料夾本身；回傳不可繼承的描述子。正常結束由入口關閉；被訊號停掉時由核心放掉。"""
    folder = path.expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    # os.open 的描述子預設不可繼承，計算子行程不會延長網頁的鎖。
    descriptor = os.open(folder, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(descriptor)
        raise
    return descriptor


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
    from aosr.config.capabilities import load_capabilities
    from aosr.config.directivity_defaults import load_directivity_defaults
    from aosr.reporting.physics_identity import physics_identity

    startup_physics_identity = physics_identity(
        capabilities=load_capabilities(config_path("capabilities.toml")),
        directivity=load_directivity_defaults(config_path("directivity_defaults.toml")))
    import uvicorn

    from aosr.gui.app import LOOPBACK, TAILNET_CLIENTS, GuiSettings, create_app, listen_address

    host = listen_address(args.listen)
    # 聽 Tailscale 位址時只收本機與 Tailscale 來的連線（區網、容器照樣送得到這個位址）。
    clients = () if host == str(LOOPBACK) else TAILNET_CLIENTS
    data_dir = args.data_dir if args.data_dir is not None else GuiSettings.data_dir
    try:
        descriptor = hold_data_folder(data_dir)
    except BlockingIOError:
        folder = data_dir.expanduser().resolve()
        sys.stderr.write(f"這個資料夾（{folder}）已經有一份網頁服務在用，不再啟動第二份；"
                         "要重開請先停掉舊的那一份\n")
        raise SystemExit(1) from None
    try:
        app = create_app(GuiSettings(engine_commit=args.engine_commit, data_dir=data_dir,
                                     extra_hosts=tuple(args.allowed_host), client_networks=clients,
                                     startup_fingerprint=startup_fingerprint,
                                     startup_physics_identity=startup_physics_identity))
        # 前面沒有反向代理：不信任 X-Forwarded-For，免得環境變數 FORWARDED_ALLOW_IPS 被放寬時，來源可以被標頭冒充。
        uvicorn.run(app, host=host, port=args.port, proxy_headers=False,
                    timeout_graceful_shutdown=SHUTDOWN_GRACE_SECONDS)
    finally:
        os.close(descriptor)


if __name__ == "__main__":
    main()
