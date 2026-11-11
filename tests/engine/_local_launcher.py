"""真跑網頁工作的考卷共用：本機行程代替使用者層服務（考卷不開真的 systemd 服務）。"""
from __future__ import annotations

import signal
import subprocess
import threading
from collections.abc import Sequence
from contextlib import AbstractContextManager
from pathlib import Path

from tests.engine._cpu_pins import forward_cpu_pins


class LocalLauncher:
    """本機行程代替服務：新的行程群、停止只送給最外層（同 ``KillMode=mixed``）。"""

    def __init__(self, tmp: Path) -> None:
        self.tmp = tmp
        self.processes: dict[str, subprocess.Popen[bytes]] = {}
        self.start_lock = threading.Lock()

    def starting(self) -> AbstractContextManager[object]:
        return self.start_lock

    def launch(self, unit: str, argv: Sequence[str], *, env: dict[str, str], cwd: Path) -> None:
        child = forward_cpu_pins(env)
        child["TMPDIR"] = str(self.tmp)
        self.processes[unit] = subprocess.Popen(list(argv), env=child, cwd=cwd, start_new_session=True,
                                                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                                stderr=subprocess.DEVNULL)

    def is_active(self, unit: str) -> bool:
        process = self.processes.get(unit)
        return process is not None and process.poll() is None

    def active_units(self) -> set[str]:
        return {unit for unit in self.processes if self.is_active(unit)}

    def stop(self, unit: str) -> None:
        self.processes[unit].send_signal(signal.SIGTERM)
