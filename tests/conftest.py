"""開跑前先產一份真的 junit 收據，並且不寫 .pyc。

**為什麼要有這一段。** `green-must-be-real-green` 那張卡判的是「pytest 產的 junit 收據」。
後設測試的第一回合要求每張卡在乾淨樹回 0，可是收據不進版控（見 `.gitignore`），
剛 clone 的樹裡它根本不存在。三條路：

1. 檢查程式「檔不存在就回 0」——這正是 v2 假綠的病根，不准。
2. 檢查程式缺檔就自己補一份——尺自己造證據，更不准。
3. 後設測試在第一回合之前，先跑一次**真的全套**，把收據產出來。

選 3。收據是那一跑的真實結果，不是這裡編的。產收據那一跑帶環境變數
`AOSR_GREEN_RECEIPT_SEED=1`，它自己不會再往下套一層；而且在那一跑裡，宣告了 `[junit]`
的卡的第一回合斷言的是**「收據不在的時候檢查必須回 1」**——比平常那一回合更凶，不是放水。
順序因此是：丟掉舊收據 → 產收據那一跑（此時收據不在，檢查必須紅）→ 外層這一跑
（收據在了，檢查必須綠）。舊收據一律丟掉，不准沿用上一跑的檔。
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# 不寫 .pyc：上一張卡撞過「改了程式，子程序卻拿 __pycache__ 裡的舊位元碼」的坑。
sys.dont_write_bytecode = True

import pytest  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from governance.loader import load_all_cards  # noqa: E402

SEED_ENV = "AOSR_GREEN_RECEIPT_SEED"
SEED_TIMEOUT = 900


def junit_paths() -> list[str]:
    """所有卡宣告的 junit 收據路徑（去重）。"""
    return sorted({card.junit_path for card in load_all_cards(REPO) if card.junit})


def _tail(proc: subprocess.CompletedProcess[str], rows: int = 15) -> str:
    return "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-rows:])


def pytest_sessionstart(session: pytest.Session) -> None:
    """外層這一跑開跑前，先跑一次真的全套把收據產出來。"""
    if os.environ.get(SEED_ENV):
        return  # 我就是產收據那一跑，不再往下套一層

    paths = junit_paths()
    if not paths:
        return
    if len(paths) > 1:
        raise pytest.UsageError(
            f"有卡各自宣告了不同的 junit 收據路徑 {paths}——一跑 pytest 只產得出一份 junit。"
            "要嘛統一成同一個路徑，要嘛這裡改成一條路徑跑一次"
        )

    target = REPO / paths[0]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)  # 舊收據一律丟掉：不准拿上一跑的檔當這一跑的綠

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env[SEED_ENV] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", f"--junitxml={target}"],
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
            timeout=SEED_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        raise pytest.UsageError(f"產收據那一跑超過 {SEED_TIMEOUT} 秒沒跑完") from exc
    if proc.returncode != 0 or not target.is_file():
        raise pytest.UsageError(
            f"產收據那一跑自己紅了（離開碼 {proc.returncode}），先修它——"
            f"收據是那一跑的真實結果，這裡不會替它補一份。最後幾行：\n{_tail(proc)}"
        )
