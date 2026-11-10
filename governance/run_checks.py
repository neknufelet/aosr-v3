"""推之前一行跑完全部規矩檢查（#1071）：雲端才紅的那種，先在本機抓到。

要跑哪些不另寫清單：``governance/rules/`` 每一張卡的 ``check`` 欄就是那張卡的檢查程式，照雲端的叫法
``python -m governance.checks.<名> --scan-root .`` 一支一個子行程並行跑，外加雲端 shard-a 那一步的
``ruff check``。加一張卡、拿掉一張卡，這裡自動跟著變。並行之前先照雲端 shard-a 的順序跑一次
``governance.status.mirror_receipts``（把雲端收據鏡像到 ``governance/receipts/cloud``）：讀收據的那幾支
沒有鏡像就回 2。

離開碼照 :mod:`governance.exit_codes` 的約定：有任何一支回 1 就回 1；沒有 1、但有一支不是 0（2、被訊號
殺掉）就回 2——那一支這一跑沒量到東西；全部 0 才回 0。不設逾時：卡住就 Ctrl-C，雲端每個 job 有自己的上限。

讀全套考卷收據的那張卡（卡上 ``[junit] path``）：收據不在，代表本機沒跑全套，那一支照跑、照印，
但標「本機沒有全套收據，雲端 verify 才算」、不進離開碼；收據在就照算。先跑
``uv run pytest --junitxml=<那個路徑>`` 再跑這一支，那一格才是本機真的驗過。

**這一支不是規矩卡**：本機跑出來的只是宣稱，雲端 ``verify`` 與 ``acceptance`` 兩個必要檢查才算數；
它只是讓「本機就紅得出來的」不必等雲端 15 分鐘才知道。

輸出全走 :func:`governance.exit_codes.note`（stderr，一支一行，不是 0 的附整段輸出）。
用法：``uv run python -m governance.run_checks``（``--jobs N`` 調並行數）。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from governance.exit_codes import CLEAN, TOOL_BROKEN, VIOLATION, ToolBroken, note, repo_root
from governance.loader import Card, CardError, load_all_cards

MIRROR_MODULE = "governance.status.mirror_receipts"


@dataclass(frozen=True)
class Step:
    name: str
    argv: tuple[str, ...]
    # 讀全套考卷收據的卡才有：收據相對掃描根的路徑。
    receipt: str = ""


@dataclass(frozen=True)
class Outcome:
    step: Step
    code: int
    seconds: float
    output: str


type Call = Callable[[Step, Path], tuple[int, str]]


def mirror_step(python: str = sys.executable) -> Step:
    return Step("mirror_receipts", (python, "-m", MIRROR_MODULE))


def plan(cards: Sequence[Card], python: str = sys.executable) -> list[Step]:
    """每張卡一步（照卡的 id 排），最後加 ruff。"""
    if not cards:
        raise ToolBroken("governance/rules/ 底下一張卡都沒有，沒有東西可跑")
    steps = [Step(card.id, (python, "-m", card.check_module, "--scan-root", "."), card.junit_path)
             for card in sorted(cards, key=lambda card: card.id)]
    return [*steps, Step("ruff", (python, "-m", "ruff", "check"))]


def _call(step: Step, root: Path) -> tuple[int, str]:
    done = subprocess.run(step.argv, cwd=root, capture_output=True, text=True)
    return done.returncode, done.stdout + done.stderr


def run(steps: Sequence[Step], root: Path, *, jobs: int, call: Call = _call) -> list[Outcome]:
    """並行跑，結果照 ``steps`` 的順序回。"""
    def timed(step: Step) -> Outcome:
        start = time.perf_counter()
        code, output = call(step, root)
        return Outcome(step, code, time.perf_counter() - start, output)

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(timed, steps))


def counted(outcome: Outcome, root: Path) -> bool:
    """收據不在的那一支不進離開碼（本機沒跑全套）；其餘都算。"""
    return not outcome.step.receipt or (root / outcome.step.receipt).is_file()


def verdict(outcomes: Sequence[Outcome], root: Path) -> int:
    codes = [outcome.code for outcome in outcomes if counted(outcome, root)]
    if VIOLATION in codes:
        return VIOLATION
    return CLEAN if all(code == CLEAN for code in codes) else TOOL_BROKEN


def report(outcomes: Sequence[Outcome], root: Path) -> list[str]:
    """一支一行（離開碼、秒數、名字、標記），不是 0 的附整段輸出。"""
    lines: list[str] = []
    for outcome in outcomes:
        label = "" if counted(outcome, root) else f"  （本機沒有全套收據 {outcome.step.receipt}，雲端 verify 才算）"
        lines.append(f"{outcome.code:>3}  {outcome.seconds:7.1f} 秒  {outcome.step.name}{label}")
        if outcome.code != CLEAN:
            lines.extend(f"       {text}" for text in outcome.output.strip().splitlines())
    return lines


def tally(outcomes: Sequence[Outcome], root: Path, code: int) -> str:
    kept = [outcome for outcome in outcomes if counted(outcome, root)]
    clean = sum(1 for outcome in kept if outcome.code == CLEAN)
    hits = sum(1 for outcome in kept if outcome.code == VIOLATION)
    return (f"共 {len(outcomes)} 支：0 有 {clean} 支、1 有 {hits} 支、其他 {len(kept) - clean - hits} 支、"
            f"本機不算 {len(outcomes) - len(kept)} 支；離開碼 {code}")


def main(argv: Sequence[str] | None = None, *, call: Call = _call) -> int:
    parser = argparse.ArgumentParser(prog="python -m governance.run_checks", description="推之前一行跑完全部規矩檢查")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2), help="同時跑幾支")
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs 至少 1")
    try:
        root = repo_root()
        steps = plan(load_all_cards(root))
    except (ToolBroken, CardError) as exc:
        note(str(exc))
        return TOOL_BROKEN
    outcomes = [*run([mirror_step()], root, jobs=1, call=call), *run(steps, root, jobs=args.jobs, call=call)]
    for line in report(outcomes, root):
        note(line)
    code = verdict(outcomes, root)
    note(tally(outcomes, root, code))
    return code


if __name__ == "__main__":
    sys.exit(main())
