"""構造結果讀回需雙精度：用正式網頁入口起獨立本機行程，不改 pytest 的算數全域設定。"""
from __future__ import annotations

import signal
import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx

from aosr.runtime import child_process_env


def _stop_calculations(folder: Path, base: str) -> None:
    """只按本題資料夾的紀錄停整組；讓網頁收屍後再確認行程組消失。"""
    with httpx.Client(trust_env=False) as client:
        for path in (folder / "runs").glob("*.json"):
            state = json.loads(path.read_text())
            pid = state.get("pid")
            if not isinstance(pid, int) or pid <= 0:
                continue
            # 正式停止端點核對行程出生身分，避免已結束的紀錄碰到重用的行程號。
            response = client.post(f"{base}/api/runs/{path.stem}/stop", json={}, timeout=15)
            response.raise_for_status()
            try:
                os.killpg(pid, 0)
            except ProcessLookupError:
                continue
            raise RuntimeError(f"計算行程組收尾後仍存在：{pid}")


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
            try:
                _stop_calculations(folder, base)
            finally:
                child.send_signal(signal.SIGINT)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
