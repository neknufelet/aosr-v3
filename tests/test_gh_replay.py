"""快取只給後設測試：沒設那個環境變數的時候，檢查真的會去叫 `gh`（GitHub 的命令列工具）。

`governance/gh_replay.py` 把 gh 的回應在一跑之內錄一次、重播給後設測試的其餘回合，省下的是
牆上時間與 API 額度，不是判準。這一份就盯著那句話的兩半：

1. **沒開快取就真的上網**——把一支假的 `gh` 外殼（shim，冒充那支工具的小腳本）擺到 `PATH`
   最前面、把錄音目錄那一格從子程序的環境裡拿掉，跑一次 `merge-gate-read-back` 那支檢查，
   然後看那支外殼有沒有被叫到。雲端那一跑身上就是這個形狀（沒有那一格），所以它不會走快取。
2. **開了快取也不准放水**——重播之前一定要求那支工具真的在 `PATH` 上。後設測試第 4 回合
   （把卡宣告的外部工具抽掉，必須回 2）靠的就是這一條；抽掉工具還能從錄音裡拿到答案，
   那一回合就會從 2 變 0，是不折不扣的假綠。
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from governance import gh_replay
from governance.exit_codes import TOOL_BROKEN, ToolBroken

REPO = Path(__file__).resolve().parents[1]

# 拿來當實驗對象的那支檢查：兩張准上網的卡之一，它問伺服器的那一句走 gh_replay。
CHECK_MODULE = "governance.checks.merge_gate_read_back"
FAKE_TOOL = "gh"
# 假外殼把自己被叫到的那一次寫進這一格指到的檔；shell 腳本讀環境變數，測試讀那個檔。
FAKE_LOG_ENV = "AOSR_FAKE_GH_LOG"
FAKE_SHIM = """#!/bin/sh
printf '%s\\n' "$*" >> "$AOSR_FAKE_GH_LOG"
exit 1
"""


def _shim_dir(tmp_path: Path, log: Path) -> Path:
    """造一個只裝著假 gh 的目錄，回傳它。log 那個檔由外殼自己寫。"""
    bin_dir = tmp_path / "fake-bin"
    bin_dir.mkdir()
    shim = bin_dir / FAKE_TOOL
    shim.write_text(FAKE_SHIM, encoding="utf-8")
    shim.chmod(0o755)
    log.write_text("", encoding="utf-8")
    return bin_dir


def test_without_the_replay_dir_the_check_really_calls_gh(tmp_path: Path) -> None:
    """沒設錄音目錄那一格，檢查真的會叫 gh——雲端那一跑就是這個形狀。"""
    log = tmp_path / "calls.txt"
    bin_dir = _shim_dir(tmp_path, log)

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop(gh_replay.REPLAY_DIR_ENV, None)  # 這一格就是快取的開關，這裡刻意拿掉
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env[FAKE_LOG_ENV] = str(log)
    env["PATH"] = os.pathsep.join([str(bin_dir), env.get("PATH", "")])

    proc = subprocess.run(
        [sys.executable, "-m", CHECK_MODULE, "--scan-root", str(REPO)],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
    )
    called = [line for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert called, (
        f"沒設 {gh_replay.REPLAY_DIR_ENV} 的時候 {CHECK_MODULE} 沒有叫到 {FAKE_TOOL}"
        "——快取只准給後設測試用，真樹上的那一跑必須真的去問伺服器；"
        f"檢查回 {proc.returncode}，它說：{proc.stderr.strip()[-300:]}"
    )
    assert proc.returncode == TOOL_BROKEN, (
        f"假外殼回非零的時候 {CHECK_MODULE} 回 {proc.returncode}，應為 {TOOL_BROKEN}"
        "——問不到伺服器就不出結論"
    )


def test_recorded_answer_comes_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """錄一次、重播一次：同一句 argv 拿回同一份答案，不同的 argv 拿不到。"""
    monkeypatch.setenv(gh_replay.REPLAY_DIR_ENV, str(tmp_path))
    argv = [FAKE_TOOL, "api", "repos/owner/name/rulesets"]
    gh_replay.record(argv, '{"ok": true}')
    assert gh_replay.replay(argv, "測試用的一句") == '{"ok": true}'
    assert gh_replay.replay([*argv, "--per-page=100"], "另一句") is None


def test_replay_is_off_when_the_env_var_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """環境變數不在，錄音就不存在也讀不到——這一層預設是關的。"""
    argv = [FAKE_TOOL, "api", "repos/owner/name/rulesets"]
    monkeypatch.setenv(gh_replay.REPLAY_DIR_ENV, str(tmp_path))
    gh_replay.record(argv, '{"ok": true}')
    monkeypatch.delenv(gh_replay.REPLAY_DIR_ENV)
    assert gh_replay.replay_dir() is None
    assert gh_replay.replay(argv, "測試用的一句") is None


def test_a_cache_hit_still_needs_the_tool_on_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """錄音在、工具不在，一樣是「這一跑不算數」——後設測試第 4 回合靠的就是這一條。"""
    monkeypatch.setenv(gh_replay.REPLAY_DIR_ENV, str(tmp_path))
    empty = tmp_path / "no-tools-here"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    argv = [FAKE_TOOL, "api", "repos/owner/name/rulesets"]
    gh_replay.record(argv, '{"ok": true}')
    with pytest.raises(ToolBroken):
        gh_replay.replay(argv, "測試用的一句")
