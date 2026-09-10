"""把 CI 裡一步的真實結果記成一片收據（片段），然後**原封不動**回那一步的離開碼。

跑法（雲端 `verify` 那個 job 的每一步都包一層）：

    uv run python -m governance.status.record_step --name <這一步的代號> \\
        --out-dir governance/receipts/steps -- uv run python -m governance.checks.<x> --scan-root .

**它不是裁判，是抄寫員。** 子程序回幾就回幾：0 就 0、1 就 1、2 就 2；被訊號殺掉就照 shell
的慣例回 128 加訊號號碼——沒有任何一條路會把非零翻成零。理由在決策紙
`docs/decisions/receipts-two-layers-cloud-on-status-branch.md`：收據要寫得出「每支檢查的離開碼
與報告行」，而 GitHub 只記每一步是綠是紅，0／1／2 那三種結局（乾淨／違規／工具自壞）在
GitHub 的紀錄裡分不開。這一層只補那個細節；紅綠的權威仍是 GitHub 記的那一步結論，
`build_receipt` 會拿兩邊交叉比對，片段說 0 而 GitHub 說紅（或反過來）都會被寫進收據。

**子程序的輸出原樣轉出去。** 它 stdout 印什麼、stderr 印什麼，這裡一個位元組都不改地寫回
同一條線上，CI 的 log 看起來跟沒包一樣；片段裡另存去掉終端機控制碼之後的最後幾行，
還有那一行 `scan_root=… files=… hits=…` 的判決收據（在**整段** stdout 裡從後往前找，
找不到才記 null，不編一行——只看最後幾行的話，輸出長一點那一格就變成 null，
而 null 的意思是「這一步本來就不印那一行」，兩件事會混在一起）。
這裡刻意不用 print：規矩卡 style-guard 只准輸出層 print，而這一段不是這支程式自己要說的話，
是子程序說的話——原樣轉發，不是紀錄。

**片段寫在版控樹裡一個被忽略的目錄。** `governance/receipts/` 在 `.gitignore` 裡（pytest 的
junit 收據早就住那裡），不會被任何檢查列舉到，也不會被推進主線；雲端那一跑跑完把整個目錄
上傳成 artifact，`status.yml` 的 receipt job 再把它跟 GitHub 記的結論合成一份收據推到 `status` 分支。
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from governance.exit_codes import TOOL_BROKEN, ToolBroken, note

# 判決收據那一行的形狀（governance/exit_codes.py 的 report() 印的）。
REPORT_RE = re.compile(r"^scan_root=.* files=\d+ hits=\d+$")
# 終端機控制碼（顏色、游標）。片段裡存的是去掉它們之後的字，log 上原樣轉出去的不動。
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
# 被訊號殺掉時回的離開碼：shell 的慣例是 128 加訊號號碼。
SIGNAL_EXIT_BASE = 128
FRAGMENT_SCHEMA = 1


@dataclass(frozen=True)
class Fragment:
    """一步的真實結果。欄位全是這一跑量到的，沒有一格是推測。"""

    schema: int
    name: str
    argv: tuple[str, ...]
    started: str  # UTC ISO 8601
    seconds: float
    exit_code: int  # 回給 CI 的那個數（被訊號殺掉時是 128 加訊號號碼）
    signal: int | None  # 被哪個訊號殺掉；正常結束是 None
    report: str | None  # `scan_root=… files=… hits=…` 那一行；子程序沒印就是 None
    stdout_tail: tuple[str, ...]
    stderr_tail: tuple[str, ...]


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """命令列參數。`--` 後面整段是要跑的子程序，一個字都不解讀。"""
    parser = argparse.ArgumentParser(
        description="跑一步、把它的真實結果記成一片收據、原封不動回它的離開碼",
    )
    parser.add_argument("--name", required=True, help="這一步的代號（片段檔名，一步一個）")
    parser.add_argument("--out-dir", required=True, help="片段寫到哪個目錄")
    parser.add_argument(
        "--tail-lines",
        type=int,
        default=30,
        help="片段裡各存 stdout／stderr 的最後幾行（去掉終端機控制碼）",
    )
    parser.add_argument("child", nargs=argparse.REMAINDER, help="-- 後面：要跑的子程序")
    args = parser.parse_args(argv)
    # argparse 的 REMAINDER 會把那個 `--` 一起留下來，這裡把它剝掉（子程序的第一個字才是命令）。
    if args.child and args.child[0] == "--":
        args.child = args.child[1:]
    if not args.child:
        parser.error("-- 後面要接一個子程序")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.name):
        parser.error(f"--name 只准 ASCII 字母、數字、_ . -：{args.name!r}")
    return args


def clean_lines(raw: bytes) -> tuple[str, ...]:
    """去掉終端機控制碼之後的每一行。解不開的位元組用替代字，不炸。"""
    text = ANSI_RE.sub("", raw.decode("utf-8", "replace"))
    return tuple(ln.rstrip() for ln in text.splitlines())


def clean_tail(raw: bytes, rows: int) -> tuple[str, ...]:
    """去掉終端機控制碼之後的最後幾行。"""
    lines = clean_lines(raw)
    return lines[-rows:] if rows > 0 else ()


def find_report(stdout_lines: Sequence[str]) -> str | None:
    """找判決收據那一行（最後一個符合形狀的）。沒有就 None。

    餵進來的是**整段** stdout，不是最後幾行：判決那一行印在哪裡由被包的那支程式決定，
    後面接幾行輸出也由它決定。原本只看最後 200 行，那個數字是猜的——輸出比它長一點，
    收據上那一格就變成 null，而 null 的意思是「這一步本來就不印那一行」，兩件事混在一起。
    從後往前找，第一個符合形狀的就是它。
    """
    for line in reversed(stdout_lines):
        if REPORT_RE.match(line):
            return line
    return None


def exit_code_of(returncode: int) -> tuple[int, int | None]:
    """子程序的 returncode 翻成（回給 CI 的離開碼, 訊號號碼）。負數就是被訊號殺掉。"""
    if returncode < 0:
        return SIGNAL_EXIT_BASE - returncode, -returncode
    return returncode, None


def run_child(child: Sequence[str], name: str, tail_lines: int) -> Fragment:
    """跑子程序、把輸出原樣轉出去、量出片段。子程序叫不動就 ToolBroken。"""
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    clock = time.monotonic()
    try:
        proc = subprocess.run([*child], capture_output=True)
    except FileNotFoundError as exc:
        raise ToolBroken(f"這一步的子程序叫不動：{child[0]}（{exc}）") from exc
    seconds = round(time.monotonic() - clock, 3)
    # 原樣轉發：這是子程序說的話，不是這支程式的紀錄（見檔頭）。
    sys.stdout.buffer.write(proc.stdout)
    sys.stdout.flush()
    sys.stderr.buffer.write(proc.stderr)
    sys.stderr.flush()
    exit_code, signal = exit_code_of(proc.returncode)
    stdout_tail = clean_tail(proc.stdout, tail_lines)
    return Fragment(
        schema=FRAGMENT_SCHEMA,
        name=name,
        argv=tuple(child),
        started=started,
        seconds=seconds,
        exit_code=exit_code,
        signal=signal,
        report=find_report(clean_lines(proc.stdout)),
        stdout_tail=stdout_tail,
        stderr_tail=clean_tail(proc.stderr, tail_lines),
    )


def write_fragment(out_dir: Path, fragment: Fragment) -> Path:
    """一步一片，同名第二片就是 workflow 寫錯了，不覆蓋、回 2。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{fragment.name}.json"
    if target.exists():
        raise ToolBroken(
            f"片段 {target} 已經存在——同一跑裡兩步用了同一個 --name {fragment.name!r}，"
            "分不出哪一片是哪一步，這一跑的收據不算數"
        )
    target.write_text(json.dumps(asdict(fragment), ensure_ascii=False, indent=1), encoding="utf-8")
    return target


def main(argv: Sequence[str] | None = None) -> int:
    """入口。回子程序的離開碼；只有「記不下來」這一種情況回 2。"""
    args = parse_args(argv)
    try:
        fragment = run_child(args.child, args.name, args.tail_lines)
        target = write_fragment(Path(args.out_dir), fragment)
    except ToolBroken as exc:
        note(f"這一步的收據記不下來，離開碼 2（工具自壞）：{exc}")
        return TOOL_BROKEN
    note(f"片段 {target.name}：離開碼 {fragment.exit_code}，{fragment.seconds} 秒")
    return fragment.exit_code


if __name__ == "__main__":
    sys.exit(main())
