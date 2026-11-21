"""本機網頁服務接受的主機名稱、監聽位址與用戶端網段。"""
from __future__ import annotations

import ipaddress
import re


LOCAL_HOSTS = ("127.0.0.1", "localhost")
# 另外准許的網址主機名：只收小寫的主機名（機器短名、點分全名或 IPv4 位址，例如 Tailscale 給這台的名字），
# 不收萬用字元、埠號或大寫——TrustedHost（只認登記網址的把關）遇到 * 就等於不把關。
HOST_NAME = re.compile(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)*\Z")
# 伺服器可以聽的位址：本機，或 Tailscale（私人網路）發給這台的位址（100.64.0.0/10）。不收 0.0.0.0 與區網位址。
# 聽在 Tailscale 位址也擋不住區網與容器：Linux 會把任何網卡收到、寄給本機任一位址的封包照收
# （弱主機模型），這台的防火牆也沒開。所以聽 Tailscale 位址時，另外只接受下面兩個網段來的連線。
LOOPBACK = ipaddress.IPv4Address("127.0.0.1")
TAILSCALE_NET = ipaddress.IPv4Network("100.64.0.0/10")
# 扣掉 100.115.92.0/23：Tailscale 不會把這段發給裝置，它自己的防火牆規則對這段也不擋（留給 ChromeOS 虛擬機），
# 區網裡收得到閘道流量的人可以拿這段當來源、完成連線。
TAILNET_CLIENTS = (ipaddress.IPv4Network("127.0.0.1/32"),
                   *TAILSCALE_NET.address_exclude(ipaddress.IPv4Network("100.115.92.0/23")))


def client_allowed(host: str | None, networks: tuple[ipaddress.IPv4Network, ...]) -> bool:
    """連線來源位址是否落在准許的網段；讀不出位址一律不准。"""
    try:
        address = ipaddress.IPv4Address(host or "")
    except ValueError:
        return False
    return any(address in network for network in networks)


def listen_address(value: str) -> str:
    """伺服器要聽的位址：只准本機或 Tailscale 位址，其他一律拒絕。"""
    try:
        address = ipaddress.IPv4Address(value)
    except ValueError as exc:
        raise ValueError(f"聽的位址要是 IPv4：{value!r}") from exc
    if address != LOOPBACK and address not in TAILSCALE_NET:
        raise ValueError(f"只准聽 127.0.0.1 或 Tailscale 位址（100.64.0.0/10），不准 {value}")
    return str(address)


def allowed_hosts(extra_hosts: tuple[str, ...]) -> list[str]:
    for host in extra_hosts:
        if not HOST_NAME.fullmatch(host):
            raise ValueError(f"另外准許的網址只收小寫主機名，不收萬用字元或埠號：{host!r}")
    return [*LOCAL_HOSTS, *extra_hosts]
