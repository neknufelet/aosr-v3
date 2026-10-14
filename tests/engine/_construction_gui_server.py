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

from aosr.gui.jobs import _proc_stat
from aosr.runtime import child_process_env


def calculation_group_exists(pid: int, start_ticks: object) -> bool:
    """那一筆計算的行程組還在不在。組還有成員時核心不會把這個行程號再發出去，所以：
    領頭在就比開始時脈（不同＝行程號已被重用、原組早沒了）；領頭不在、組號還探得到，就是原組留下的成員。"""
    stat = _proc_stat(pid)
    if stat is not None:
        return stat[1] == pid and stat[2] == start_ticks
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _stop_calculations(folder: Path, base: str) -> None:
    """只按本題資料夾的紀錄停整組；停止端點自己核出生身分、停不掉回錯，每一筆都停過才彙報。"""
    left = []
    with httpx.Client(trust_env=False) as client:
        for path in sorted((folder / "runs").glob("*.json")):
            state = json.loads(path.read_text())
            try:
                response = client.post(f"{base}/api/runs/{path.stem}/stop", json={}, timeout=15)
                body = response.json() if response.status_code == 200 else None
            except (httpx.HTTPError, ValueError) as exc:
                # 斷線、逾時、回的不是 JSON：記下這一筆，照樣去停下一筆。
                left.append(f"{path.stem}：停止請求失敗 {type(exc).__name__}")
                continue
            if not isinstance(body, dict) or body.get("status") in (None, "running"):
                left.append(f"{path.stem}：{response.status_code} {response.text}")
            elif isinstance(pid := state.get("pid"), int) and pid > 0 and calculation_group_exists(pid, state.get("proc_start_ticks")):
                left.append(f"{path.stem}：行程組 {pid} 收尾後仍存在")
    if left:
        raise RuntimeError("計算收尾未確認：" + "；".join(left))


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
