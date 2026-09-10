"""把 `status` 分支上的機器收據鏡到版控樹裡一個被忽略的目錄，給三張收據卡讀。

跑法：

    uv run python -m governance.status.mirror_receipts            # 讀 origin/status，寫到 governance/receipts/cloud/

**為什麼要鏡。** 收據住在機器分支 `status`（決策紙 `docs/decisions/receipts-two-layers-cloud-on-status-branch.md`），
而 `verify` 檢出的是 PR 的樹。三張收據卡的檢查程式跟其他檢查一樣只讀掃描根底下的檔，不碰 git 物件庫、
不上網；所以要有一層先把那條分支上的 `receipts/` 抄成檔案。抄到 `governance/receipts/cloud/`：
那個目錄在 `.gitignore` 裡（pytest 的 junit、每一步的片段早就住在旁邊），不會被列舉、不會進主線。

**不上網。** 只讀本機已經有的 ref（預設 `origin/status`）。`actions/checkout` 帶 `fetch-depth: 0` 會把所有
分支抓下來，本機 `git pull` 之後也有；ref 不在就回 2 並說「先 git fetch」，這裡不替你抓——決策紙
merge-gate-check-may-read-github 只放行那一支檢查上網，這一支不是。

**順手記來源。** 每一份收據是 `status` 分支上哪一筆提交寫的、author 與 committer 是誰，寫進同一層的
`provenance.json`（跟收據放在不同目錄，收據目錄底下「每一個 json 檔」的 glob 才不會把它當成一份收據）。
四席那張卡靠這一份判「收據是不是機器寫的」。git 的 author 是自報的，這一份只證明「宣稱的身分」，
卡上要照實寫。

**離開碼只有兩種。** 0 鏡好了、2 鏡不成（git 不在、ref 不在、那條分支上一份收據都沒有、抄到一半失敗）。
鏡不成就把目錄清掉，不留半份——半份鏡像看起來像「收據就這幾份」。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from governance.exit_codes import CLEAN, TOOL_BROKEN, ToolBroken, note, repo_root

VCS = "git"
DEFAULT_REF = "origin/status"
# 相對 repo 根。這個目錄在 .gitignore 裡（/governance/receipts/），所以鏡像不會被列舉、不會進主線。
DEFAULT_OUT = "governance/receipts/cloud"
BRANCH_DIR = "receipts"
PROVENANCE_FILE = "provenance.json"
SEP = "\x1f"


@dataclass(frozen=True)
class Origin:
    """一份收據在 status 分支上的來源：最後一筆動到它的提交，以及那筆提交宣稱的身分。"""

    commit: str
    author_email: str
    committer_email: str


def _git(root: Path, argv: Sequence[str], what: str, timeout: int) -> str:
    """跑一個只讀的版控指令。叫不動、逾時、非零，一律 ToolBroken。"""
    try:
        proc = subprocess.run(
            [VCS, *argv],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        )
    except FileNotFoundError as exc:
        raise ToolBroken(f"{VCS} 不在 PATH 上（{what}）") from exc
    except subprocess.TimeoutExpired as exc:
        raise ToolBroken(f"{VCS} {' '.join(argv)} 超過 {timeout} 秒沒回話（{what}）") from exc
    if proc.returncode != 0:
        raise ToolBroken(f"{VCS} {' '.join(argv)} 回 {proc.returncode}（{what}）：{proc.stderr.strip()[:300]}")
    return proc.stdout


def resolve_ref(root: Path, ref: str, timeout: int) -> str:
    """ref 指到哪一筆提交。不在就 ToolBroken，這裡不替你 fetch。"""
    try:
        return _git(root, ["rev-parse", "--verify", f"{ref}^{{commit}}"], f"解析 {ref}", timeout).strip()
    except ToolBroken as exc:
        raise ToolBroken(f"{exc}——本機沒有 {ref} 這個 ref，先 git fetch（這一支刻意不上網）") from exc


def list_receipts(root: Path, ref: str, timeout: int) -> list[str]:
    """那條分支上 receipts/ 底下的每一個路徑（相對分支根）。一份都沒有就 ToolBroken。"""
    out = _git(root, ["ls-tree", "-r", "--name-only", ref, "--", f"{BRANCH_DIR}/"], f"列 {ref} 的收據", timeout)
    names = [ln.strip() for ln in out.splitlines() if ln.strip().endswith(".json")]
    if not names:
        raise ToolBroken(f"{ref} 上 {BRANCH_DIR}/ 底下一份收據都沒有——沒有收據就沒有東西可判，鏡像不算數")
    return names


def origin_of(root: Path, ref: str, path: str, timeout: int) -> Origin:
    """最後一筆動到這份收據的提交，與它宣稱的 author／committer。"""
    out = _git(root, ["log", "-1", f"--format=%H{SEP}%ae{SEP}%ce", ref, "--", path], f"查 {path} 的來源", timeout)
    parts = out.strip().split(SEP)
    if len(parts) != 3 or not all(parts):
        raise ToolBroken(f"{path} 在 {ref} 上查不到提交來源（git log 回：{out.strip()[:100]!r}）")
    return Origin(commit=parts[0], author_email=parts[1], committer_email=parts[2])


def mirror(root: Path, ref: str, out: Path, timeout: int) -> Path:
    """整份抄過來。先清掉舊鏡像；抄到一半失敗就整個清掉，不留半份。回 provenance 檔的路徑。"""
    commit = resolve_ref(root, ref, timeout)
    names = list_receipts(root, ref, timeout)
    if out.exists():
        shutil.rmtree(out)
    target_dir = out / BRANCH_DIR
    target_dir.mkdir(parents=True)
    files: dict[str, dict[str, str]] = {}
    try:
        for path in names:
            body = _git(root, ["show", f"{ref}:{path}"], f"讀 {path}", timeout)
            (target_dir / Path(path).name).write_text(body, encoding="utf-8")
            files[Path(path).name] = asdict(origin_of(root, ref, path, timeout))
    except ToolBroken:
        shutil.rmtree(out, ignore_errors=True)
        raise
    provenance = out / PROVENANCE_FILE
    provenance.write_text(
        json.dumps({"ref": ref, "commit": commit, "files": files}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    return provenance


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="把 status 分支上的機器收據鏡到 governance/receipts/cloud/（鏡不成回 2）")
    parser.add_argument("--ref", default=DEFAULT_REF, help="讀哪個 ref（本機要先有；這一支不上網）")
    parser.add_argument("--out", default=DEFAULT_OUT, help="鏡到哪個目錄（相對 repo 根；預設在 .gitignore 裡）")
    parser.add_argument("--timeout", type=int, default=60, help="每一個版控指令的看門狗秒數")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        root = repo_root()
        provenance = mirror(root, args.ref, root / args.out, args.timeout)
    except ToolBroken as exc:
        note(f"鏡不成，離開碼 2（工具自壞）：{exc}")
        return TOOL_BROKEN
    count = len(json.loads(provenance.read_text(encoding="utf-8"))["files"])
    note(f"鏡好了：{args.ref} 上 {count} 份收據 → {provenance.parent}")
    return CLEAN


if __name__ == "__main__":
    sys.exit(main())
