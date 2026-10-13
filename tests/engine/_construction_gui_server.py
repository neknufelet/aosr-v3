"""構造結果讀回需雙精度：用正式網頁入口起獨立本機行程，不改 pytest 的算數全域設定。"""
from __future__ import annotations

import signal
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx

from aosr.runtime import child_process_env


@contextmanager
def real_server(folder: Path) -> Iterator[str]:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    log = folder / "gui.stderr"
    with log.open("wb") as stream:
        child = subprocess.Popen([sys.executable, "-B", "-m", "aosr.gui", "--engine-commit", "a" * 40,
            "--data-dir", str(folder), "--port", str(port)], cwd=folder,
            env=child_process_env(threads=1), stdout=subprocess.DEVNULL, stderr=stream)
        try:
            deadline = time.monotonic() + 30
            with httpx.Client(trust_env=False) as client:
                while True:
                    if child.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError(f"正式網頁沒有起來：{log.read_text()}")
                    try:
                        response = client.get(f"{base}/api/input-defaults", timeout=1)
                    except httpx.TransportError:
                        time.sleep(.05)
                        continue
                    response.raise_for_status()
                    break
            yield base
        finally:
            child.send_signal(signal.SIGINT)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
