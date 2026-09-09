"""退出碼約定 ＋ 每支檢查共用的外殼。

約定（v2 的頭號死因是「只會回綠的檢查」，所以三種結局要分得開）：

* ``0`` 乾淨——真的掃過東西，而且沒有違規。
* ``1`` 有違規。
* ``2`` 工具自壞——**沒有掃到東西**，所以「沒找到違規」這句話不算數。
  以下一律 2：掃描根解析不到、列舉檔案的子程序非零退出、列舉出來是空集合、
  掃描根落在 repo 外面。

每支檢查在回傳前一定印一行 ``scan_root=<路徑> files=<整數> hits=<整數>``，
連 2 的路徑也要印——沒有這一行看不出「回綠是因為乾淨，還是因為什麼都沒掃到」。

**列舉模式（``--list-files``）。** 報告行只有一個 ``files=<整數>``，是計數不是清單：
兩棵樹檔案數剛好一樣、內容完全不同也看不出來。所以外殼多收一個 ``targets`` 函式
（``targets(掃描根, 全部檔案) -> 這支檢查真的會讀／會判的那些檔``），加上 ``--list-files``
旗標：帶了它就**只跑列舉、不下判斷**，把那些檔一行一個印到 stdout（相對掃描根，
前後各一行界線），沒帶就跟以前一模一樣（既有的探針與後設測試不受影響）。

列舉模式刻意不印報告行——那一行是判決的收據，這一跑沒有判決。規矩卡
``scan-scope-has-no-holes`` 拿這個清單跟卡上宣告的 ``scope`` 比集合。
兩件事寫成一個函式而不是兩份宣告：``check()`` 自己就是呼叫同一支 ``targets()`` 拿掃描面，
所以「印出來的清單」與「真的被掃的檔」不是兩份會各自漂的宣告。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

CLEAN = 0
VIOLATION = 1
TOOL_BROKEN = 2

# 列舉版控裡的檔案用的子程序。只認本機 git，不連網。
#
# ``-c core.quotePath=false`` 不是裝飾：git 預設把非 ASCII 檔名印成 ``"\344\270\255..."``
# 這種八進位跳脫碼，那串東西本身是純 ASCII，管路徑字元的檢查就瞎了（實測拿掉旗標後
# file-placement-allowlist 的「純 ASCII」那一條抓不到中文檔名，只剩「名字被 git 加引號」
# 那道保險在咬，訊息還看不懂）。加了旗標才拿得到真正的名字。
# 出處：決策紙 docs/decisions/ascii-filenames.md。
ENUMERATE_ARGV = (
    "git",
    "-c",
    "core.quotePath=false",
    "ls-files",
    "--cached",
    "--others",
    "--exclude-standard",
)


class ToolBroken(Exception):
    """工具自壞：這一跑沒有真的掃到東西，結論不算數（退出碼 2）。"""


def repo_root() -> Path:
    """這個 repo 的根（往上找 .git）。檢查程式不准讀這個範圍外的路徑。"""
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / ".git").exists():
            return parent
    raise ToolBroken(f"從 {here} 往上找不到 .git，無法確定 repo 根")


def resolve_scan_root(raw: str | Path) -> Path:
    """把 --scan-root 解析成一個真的存在、而且在 repo 裡面的目錄。"""
    root = Path(raw).expanduser()
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()
    else:
        root = root.resolve()
    if not root.is_dir():
        raise ToolBroken(f"掃描根不存在或不是目錄：{root}")
    inside = repo_root()
    if root != inside and inside not in root.parents:
        raise ToolBroken(f"掃描根 {root} 落在 repo（{inside}）外面，檢查程式不准讀")
    return root


def enumerate_files(root: Path) -> list[Path]:
    """列舉掃描根底下「進得了版控」的檔案。

    刻意走 ``git ls-files`` 而不是自己走檔案系統：規矩只管版控裡的東西，
    而且這樣「外部工具被抽掉」是一個真的會發生的失敗，不是假設。
    """
    try:
        proc = subprocess.run(
            [*ENUMERATE_ARGV],
            cwd=root,
            capture_output=True,
        )
    except FileNotFoundError as exc:
        raise ToolBroken(f"列舉用的外部工具不在 PATH：{ENUMERATE_ARGV[0]}（{exc}）") from exc
    if proc.returncode != 0:
        raise ToolBroken(
            f"列舉子程序非零退出（{proc.returncode}）：{' '.join(ENUMERATE_ARGV)}"
            f"／{proc.stderr.decode('utf-8', 'replace').strip()[:200]}"
        )
    # 刻意自己解碼：檔名不是合法 UTF-8 的時候要回 2（我沒看懂），不是讓 Python 炸出
    # 追蹤訊息然後以離開碼 1 收場——那會被 CI 讀成「抓到違規」。
    try:
        out = proc.stdout.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ToolBroken(f"列舉出來的名字不是合法 UTF-8（{exc}），我沒看懂就不出結論") from exc
    names = [ln for ln in out.splitlines() if ln.strip()]
    if not names:
        raise ToolBroken(f"列舉出來是空集合：{root} 底下一個版控檔案都沒有")
    return [root / n for n in names]


def report(scan_root: str | Path, files: int, hits: int) -> None:
    """印那一行報告。每支檢查回傳前都要叫一次。"""
    print(f"scan_root={scan_root} files={files} hits={hits}")


def note(message: str) -> None:
    """檢查程式要對人多說的一句話（放行清單用到幾條、探針停在哪一層、量到幾個對象）。

    **為什麼要有這一支。** 這些話本來是各支檢查自己 print 到 stderr 的，寫法一支一份：
    有的帶 ``NOTE:`` 前綴、有的不帶，有的印到 stdout 混在報告行旁邊。規矩卡
    ``style-guard`` 的第①條把「print 只准出現在輸出層」立成規矩，輸出層就是這個檔——
    判決收據（:func:`report`）、列舉清單（:func:`list_files`）、HIT 與 FAIL（:func:`run`）
    本來都在這裡，這一支把「多說的一句話」也收進來。

    走同一支函式的好處不是省行數，是**看得出哪一行是這一跑的產品**：治理層對外說的每一句話
    都從這個檔出去，之後誰在別的地方 print，就是他在那支程式裡自己記 log，那條規矩咬得到。

    一律印到 stderr：stdout 那一條線留給機器讀的東西（報告行、列舉清單），
    別支檢查與後設測試都在剖析它。
    """
    print(f"NOTE: {message}", file=sys.stderr)


# 列舉模式的界線。有界線才分得出「清單是空的」與「這支檢查根本沒印清單」。
LIST_BEGIN = "list_files_begin"
LIST_END = "list_files_end"


def list_files(scan_root: Path, picked: Iterable[Path]) -> int:
    """列舉模式印的東西：一行一個路徑（相對掃描根、posix 寫法），前後各一行界線。

    回傳印出幾個。空集合由呼叫端判成工具自壞——列舉不出東西的時候，「掃描面就是這樣」
    這句話不算數。
    """
    names = sorted({p.relative_to(scan_root).as_posix() for p in picked})
    print(f"{LIST_BEGIN}={len(names)}")
    for name in names:
        print(name)
    print(LIST_END)
    return len(names)


CheckFn = Callable[[Path, list[Path]], list[str]]
# ``targets(掃描根, 全部檔案)`` 回這支檢查真的會讀／會判的檔。``check()`` 自己也用它。
TargetFn = Callable[[Path, list[Path]], list[Path]]


def _list_only(root: Path, files: list[Path], targets: TargetFn | None) -> int:
    """``--list-files``：只列舉掃描面，不下判斷。"""
    if targets is None:
        raise ToolBroken(
            "這支檢查沒有交出列舉函式（run(..., targets=...)）"
            "——量不到它的掃描面，就沒有人能證明它掃的跟卡上宣告的是同一組檔"
        )
    picked = targets(root, files)
    outside = sorted(str(p) for p in picked if not p.is_relative_to(root))
    if outside:
        raise ToolBroken(f"列舉函式交出掃描根外面的路徑：{outside}——檢查程式不准讀那裡")
    if list_files(root, picked) == 0:
        raise ToolBroken(
            f"列舉函式在 {root} 上交出空集合——這支檢查在這棵樹上一個檔都不掃，"
            "「掃描面就是這樣」這句話不算數"
        )
    return CLEAN


def run(
    check: CheckFn,
    argv: Sequence[str] | None = None,
    *,
    description: str = "",
    targets: TargetFn | None = None,
) -> int:
    """每支檢查的外殼：解析參數、列舉檔案、印報告行、決定 0／1／2。

    ``check(scan_root, files)`` 回傳違規訊息的 list，空 list 就是乾淨。
    它可以 raise ToolBroken 表示「這一跑不算數」。

    ``targets`` 是那支檢查的掃描面（它真的會讀／會判的檔）。帶 ``--list-files`` 跑的時候
    只叫它、不叫 ``check``——列舉模式不下判斷，所以也不會跑動態探針、不會遞迴。
    沒交 ``targets`` 的檢查在列舉模式回 2（量不到它的掃描面）。
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--scan-root", required=True, help="要掃的樹根（必須在 repo 裡）")
    parser.add_argument(
        "--list-files",
        action="store_true",
        help="只列舉這支檢查的掃描面（一行一個路徑），不下判斷、不印報告行",
    )
    args = parser.parse_args(argv)

    files: list[Path] = []
    try:
        root = resolve_scan_root(args.scan_root)
        files = enumerate_files(root)
        if args.list_files:
            return _list_only(root, files, targets)
        hits = check(root, files)
    except ToolBroken as exc:
        # 列舉模式不印報告行（那一行是判決的收據，這一跑沒有判決），其餘照舊。
        if not args.list_files:
            report(args.scan_root, len(files), 0)
        print(f"FAIL(2) 工具自壞：{exc}", file=sys.stderr)
        return TOOL_BROKEN

    for hit in hits:
        print(f"HIT: {hit}", file=sys.stderr)
    report(root, len(files), len(hits))
    if hits:
        print(f"FAIL(1) 有 {len(hits)} 筆違規", file=sys.stderr)
        return VIOLATION
    return CLEAN
