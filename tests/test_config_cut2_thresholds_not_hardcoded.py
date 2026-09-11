"""門檻不准硬編碼在別的地方——上一代那條守衛改寫成盯新家，住治理層那一籃。

**上一代那條長什麼樣（上一代那一棵樹的 config 考卷裡的 ``test_perceptual_thresholds_are_not_hardcoded_in_lib``）。**
它掃 ``(REPO_ROOT / "lib").rglob("*.py")``，禁四個字面數字（``-10.0``／``-25.0``／``15.0``／
``0.9``）、豁免清單指名上一代 config 層的 art_lane 與 scoring 層的 perceptual。那條守衛
在新家**是空的**：上一代那一層不在這裡，路徑全部對不上。

**這一刀照樣把它帶過來，只是換瞄準的對象。** 掃描面改成這一層（新家第一塊
自己家的那一層，也就是這四個數字**應該**住的地方）；豁免清單只留新家真的需要的那一筆
（``art_lane.py`` 的 ``-25.0`` 是它的 WLS 視窗下限，不是感知門檻）。

**這條守衛住在 ``tests/``（治理層那一籃），不在 ``tests/engine/``。** 它一個 ``aosr`` 的
import 都沒有——它讀的是原始碼文字，判的是「這一塊有沒有把門檻漏到別的地方」。掃原始碼的
守衛屬治理層。

**照實說：這一條在新家比上一代窄。** 上一代掃的是**整個 lib/**（26 支 config 加上別的層），
今天新家只有 ``config`` 這一層長出來，所以掃描面就是它。等到別的層長出來，這一條的
掃描面（``src/aosr/``）要跟著長——那時它會咬到更多地方，也會需要重新量一次豁免清單。
在那之前，這條守衛蓋得到的是「這一塊沒有把四個門檻寫在自己家裡」；蓋不到的是
「別的層有沒有各自再寫一份」——今天那些層還不存在。
"""
from __future__ import annotations

import re
from pathlib import Path

# 這個 repo 的根（`<root>/tests/` 的上一層）。跟 cwd 無關。
REPO_ROOT = Path(__file__).resolve().parents[1]

# 這一塊的家：新家第 2 層。
CONFIG_ROOT = REPO_ROOT / "src" / "aosr" / "config"

# 這四個字面數字不准出現在這一層的程式碼裡；它們該住的地方是這一層 `data` 目錄底下那幾個 .toml。
PROHIBITED = ("-10.0", "-25.0", "15.0", "0.9")

# 豁免清單：**具名一筆，寫理由**（不是整個目錄放掉）。
# `art_lane.py` 的 `ART_WLS_T20_LO_DB = -25.0` 是 T20 積分視窗的下限（art_lane 那一支自己的
# 結構常數），不是感知門檻。上一代的豁免清單裡還有一筆 scoring 那一層的 perceptual 模組——那一支
# 搬去 scoring 那一塊（#135），新家還沒有它，所以這裡只有一筆。真的多一筆就要在這裡
# 寫出理由（`exemptions-need-expiry` 那張卡管到期日）。
ALLOWED = frozenset({CONFIG_ROOT / "art_lane.py"})


def _pattern(literal: str) -> re.Pattern[str]:
    """比**整個數字 token**，不是比子字串。

    裸 ``0.9`` 咬得到，``0.999``（Nyquist 邊緣的護欄）／``0.95``／``10.9``／``0.9e5``
    咬不到。上一代有一段註解記著這條：原本的 ``literal in text`` 子字串版誤咬過
    ``auralize.py`` 的 ``nyquist_hz * 0.999``。
    """
    return re.compile(rf"(?<![\d.]){re.escape(literal)}(?![\d.eE])")


def test_no_perceptual_threshold_is_hardcoded_in_the_config_layer() -> None:
    """這一層的每一支 .py 都不准把那四個門檻寫死（豁免那一筆除外）。"""
    patterns = {literal: _pattern(literal) for literal in PROHIBITED}
    scanned = 0
    offenders: list[str] = []
    for path in sorted(CONFIG_ROOT.rglob("*.py")):
        scanned += 1
        if path in ALLOWED:
            continue
        text = path.read_text(encoding="utf-8")
        for literal, pattern in patterns.items():
            if pattern.search(text):
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {literal}")
    assert scanned > 0, "一支 .py 都沒掃到——掃描面算錯了，這一跑不算數"
    assert offenders == []


def test_the_guard_would_actually_bite_a_hardcoded_threshold(tmp_path: Path) -> None:
    """這一條守衛真的會咬：把一個門檻寫進一支臨時檔，同一個判準要抓到它。

    沒有這一條，「上面那一條綠」只證明「今天沒人寫」——分不出「乾淨」與「判準壞掉」。
    這一段刻意**複製**上面那一條的判準（同一個 `_pattern`），餵一支必定違規的檔。
    """
    probe = tmp_path / "pretend.py"
    probe.write_text("THRESHOLD_DB = -10.0\n", encoding="utf-8")
    pattern = _pattern("-10.0")
    assert pattern.search(probe.read_text(encoding="utf-8"))
    # 反向控制：子字串那一版會誤咬的那一種寫法，這一版不准咬。
    assert not pattern.search("nyquist_hz * 0.999\n")
