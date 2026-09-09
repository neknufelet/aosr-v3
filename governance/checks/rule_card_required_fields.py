#!/usr/bin/env python3
"""規矩卡必填欄位 ＋ 附的樣本真的會咬。

兩關，缺一不可：

**第一關（欄位）**
每張卡的必填欄位、列舉值、掛載點四段都由 :mod:`governance.loader` 驗。
這一關再多一道交叉驗證，不然「執行者」只是換個地方申報假的：

1. 卡宣告的 ``job`` 必須真的出現在 ``.github/workflows/*.yml`` 的某個 job；
2. 那個 job 的步驟裡必須真的呼叫這張卡宣告的 check 模組；
3. 那個 job 必須列在版控裡的 ``governance/required-status-checks.txt``。

三者任一對不上就紅。v2 的 L01 就是「CLAUDE.md 申報執行牙是 pre-commit doc-topology，
而 .git/hooks 底下只有 .sample」——申報了執行者，它其實沒在跑。

**第二關（會咬）**
欄位填滿 ≠ 守衛會咬。對每張卡，把它宣告的每一份必紅樣本（含控制樣本）餵給它自己
宣告的檢查模組，退出碼必須是 1。回 0 就代表這張卡附的是不會咬的樣本，等於沒有守衛
（v2 的五支守衛四支假綠就是這樣過關的）。

掃描面只限 ``governance/`` 底下的結構化宣告與 ``.github/workflows/``，**不掃全樹散文**：
找碴席實測今天的乾淨樹裡 ``v2-audit/lessons.json`` 自己就含「執行牙」「會擋下」字樣，
掃全樹自然語言的寫法會在還沒有任何規矩卡之前就把乾淨樹判紅。
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from governance.exit_codes import ToolBroken, VIOLATION, run  # noqa: E402
from governance.loader import RULES_DIR, Card, card_problems, load_card  # noqa: E402

WORKFLOW_DIR = ".github/workflows"
REQUIRED_CHECKS_FILE = "governance/required-status-checks.txt"

# 第二關會遞迴（卡的樣本裡也有卡）。深度到 2 就不再往下，不然會沒完沒了。
MAX_BITE_DEPTH = 2
DEPTH_ENV = "AOSR_BITE_DEPTH"
BITE_TIMEOUT = 300


def _job_blocks(text: str, source: str) -> dict[str, str]:
    """從一份 workflow yml 裡挖出 {job 名: 那個 job 的原文}。

    刻意只認 YAML 的一個子集（``jobs:`` 底下同一層縮排的鍵），不引入 yaml 依賴。
    看不懂的檔一律 raise ToolBroken——寧可回 2 說「我沒看懂」，也不要假裝乾淨回 0。
    """
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if re.match(r"^jobs\s*:\s*$", line):
            start = i + 1
            break
    if start is None:
        raise ToolBroken(f"{source} 裡找不到頂層的 jobs:，這份 workflow 我看不懂")

    body = lines[start:]
    indent = None
    for line in body:
        if line.strip() and not line.lstrip().startswith("#"):
            indent = len(line) - len(line.lstrip())
            break
    if not indent:
        raise ToolBroken(f"{source} 的 jobs: 底下沒有任何 job")

    jobs: dict[str, list[str]] = {}
    current: str | None = None
    for line in body:
        if not line.strip() or line.lstrip().startswith("#"):
            if current:
                jobs[current].append(line)
            continue
        here = len(line) - len(line.lstrip())
        if here < indent:
            break
        match = re.match(rf"^ {{{indent}}}([A-Za-z0-9_.\-]+)\s*:", line)
        if here == indent and match:
            current = match.group(1)
            jobs[current] = [line]
        elif current:
            jobs[current].append(line)
    if not jobs:
        raise ToolBroken(f"{source} 的 jobs: 底下解不出任何 job 名")
    return {name: "\n".join(rows) for name, rows in jobs.items()}


def _mount_problems(card: Card, scan_root: Path, files: list[Path]) -> list[str]:
    """交叉驗證：宣告的 job 有沒有真的在崗。"""
    bad: list[str] = []
    workflows = [f for f in files if f.parent == scan_root / WORKFLOW_DIR and f.suffix in (".yml", ".yaml")]
    if not workflows:
        return [f"卡 {card.id} 宣告 job={card.job!r}，但 {WORKFLOW_DIR} 底下一份 workflow 都沒有"]

    found: dict[str, str] = {}
    for wf in workflows:
        for name, block in _job_blocks(wf.read_text(encoding="utf-8"), str(wf.relative_to(scan_root))).items():
            found.setdefault(name, block)

    if card.job not in found:
        bad.append(
            f"卡 {card.id} 宣告 job={card.job!r}，但 {WORKFLOW_DIR} 裡沒有這個 job"
            f"（實際有 {sorted(found)}）——申報了執行者，它其實不在崗"
        )
    elif card.check_module not in found[card.job] and card.check not in found[card.job]:
        bad.append(
            f"卡 {card.id} 宣告 job={card.job!r}，那個 job 存在，"
            f"但它的步驟裡沒有呼叫這張卡宣告的檢查模組 {card.check_module}"
        )

    expect = scan_root / REQUIRED_CHECKS_FILE
    if expect not in files:
        bad.append(f"卡 {card.id} 宣告 job={card.job!r}，但版控裡沒有 {REQUIRED_CHECKS_FILE}，無從確認它會卡住合併")
    else:
        listed = [
            ln.strip()
            for ln in expect.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")
        ]
        if card.job not in listed:
            bad.append(
                f"卡 {card.id} 宣告 job={card.job!r}，但它沒被列在 {REQUIRED_CHECKS_FILE}"
                f"（實際列了 {listed}）——不是 required check 就擋不住合併"
            )
    return bad


def _bite_problems(card: Card, scan_root: Path, depth: int) -> list[str]:
    """第二關：卡宣告的每一份必紅樣本，餵給它自己的檢查都必須回 1。"""
    if depth >= MAX_BITE_DEPTH:
        print(f"NOTE: 第二關（會咬）在深度 {depth} 停止遞迴，{card.id} 這一層只驗欄位", file=sys.stderr)
        return []

    cases = card.negative_cases(scan_root)
    control = card.control_path(scan_root)
    if control.is_dir() and control not in cases:
        cases.append(control)
    if not cases:
        return [f"卡 {card.id} 宣告的必紅樣本目錄底下沒有任何樣本——第二關無從證明它會咬"]

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env[DEPTH_ENV] = str(depth + 1)

    bad: list[str] = []
    for case in cases:
        try:
            proc = subprocess.run(
                [sys.executable, "-m", card.check_module, "--scan-root", str(case)],
                cwd=scan_root,
                env=env,
                capture_output=True,
                text=True,
                timeout=BITE_TIMEOUT,
            )
        except subprocess.TimeoutExpired as exc:
            raise ToolBroken(f"跑 {card.check_module} 餵 {case} 超過 {BITE_TIMEOUT} 秒") from exc
        if proc.returncode != VIOLATION:
            bad.append(
                f"卡 {card.id} 的樣本 {case.relative_to(scan_root)} 餵給 {card.check_module} 回 "
                f"{proc.returncode}，應為 1——填滿欄位不算有牙，附的樣本必須真的讓檢查回 1"
            )
    return bad


def check(scan_root: Path, files: list[Path]) -> list[str]:
    depth = int(os.environ.get(DEPTH_ENV, "0"))
    rules = sorted(f for f in files if f.parent == scan_root / RULES_DIR and f.suffix == ".toml")
    if not rules:
        raise ToolBroken(f"{scan_root}/{RULES_DIR} 底下一張版控裡的規矩卡都沒有——這一跑沒掃到東西")

    bad: list[str] = []
    for path in rules:
        problems = card_problems(path, scan_root)
        if problems:
            bad += [f"{path.relative_to(scan_root)}：{p}" for p in problems]
            continue
        card = load_card(path, scan_root)
        bad += _mount_problems(card, scan_root, files)
        bad += _bite_problems(card, scan_root, depth)
    return bad


if __name__ == "__main__":
    sys.exit(run(check, description="規矩卡必填欄位，且附的必紅樣本真的會讓檢查回 1"))
