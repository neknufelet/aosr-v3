"""#595 實驗：兩片考卷的收尾核對（草稿合併請求，不合併進主線；只量時間與驗證收尾的寫法）。

核三件事：①兩份清單互不重疊、合起來剛好是版控裡全部考卷檔；②每個考卷檔在它那一片的 junit 裡至少有一題；
③兩片都沒有失敗、錯誤、跳過。任何一件不成立就以 1 結束；結果寫進 GitHub 的步驟摘要檔，不印到終端。
"""
from __future__ import annotations

import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

LISTS = {"a": Path(".github/shard-trial/shard-a.txt"), "b": Path(".github/shard-trial/shard-b.txt")}


def _listed(name: str) -> list[str]:
    return [line.strip() for line in LISTS[name].read_text(encoding="utf-8").splitlines() if line.strip()]


def _module(path: str) -> str:
    return path.removesuffix(".py").replace("/", ".")


def _junit_modules(path: Path) -> tuple[set[str], int, int]:
    root = ET.parse(path).getroot()
    modules: set[str] = set()
    cases = bad = 0
    for case in root.iter("testcase"):
        cases += 1
        parts = (case.get("classname") or "").split(".")
        modules.add(".".join(part for part in parts if not part[:1].isupper()))
        if any(case.find(tag) is not None for tag in ("failure", "error", "skipped")):
            bad += 1
    return modules, cases, bad


def main(junit_dir: Path) -> int:
    problems: list[str] = []
    tracked = set(subprocess.run(["git", "ls-files", "tests/test_*.py", "tests/engine/test_*.py"],
                                 capture_output=True, text=True, check=True).stdout.split())
    a, b = _listed("a"), _listed("b")
    if set(a) & set(b):
        problems.append(f"兩片重疊：{sorted(set(a) & set(b))}")
    if set(a) | set(b) != tracked:
        problems.append(f"兩片合起來不等於全部考卷檔：少 {sorted(tracked - set(a) - set(b))}，多 {sorted(set(a) | set(b) - tracked)}")
    summary = [f"版控考卷檔 {len(tracked)}"]
    for name, listed in (("a", a), ("b", b)):
        files = sorted(junit_dir.rglob(f"shard-{name}.junit.xml"))
        if not files:
            problems.append(f"第 {name} 片沒有 junit")
            continue
        modules, cases, bad = _junit_modules(files[0])
        missing = [path for path in listed if _module(path) not in modules]
        if missing:
            problems.append(f"第 {name} 片有考卷檔沒跑到：{missing}")
        if bad:
            problems.append(f"第 {name} 片有 {bad} 題失敗、錯誤或跳過")
        summary.append(f"第 {name} 片：考卷檔 {len(listed)}、題數 {cases}")
    report = "\n".join(["## 分片收尾核對", *summary, *(f"- 不通過：{item}" for item in problems)]) + "\n"
    target = os.environ.get("GITHUB_STEP_SUMMARY")
    if target:
        with open(target, "a", encoding="utf-8") as handle:
            handle.write(report)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
