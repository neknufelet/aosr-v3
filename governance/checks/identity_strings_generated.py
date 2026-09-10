"""規矩卡 identity-strings-generated：版控裡的文件出現的 commit sha 與雲端 run id 必須解析得到。

v2 事故 hand-copied-identity-strings-drift：sha 手抄錯、diff 數字抄錯，每次都在最貴的終審站才被抓。
這張卡守 repo 裡的那一半：決策紙、handoff、方向檔、入口檔裡出現的身分字串，機器對一次。兩條：

1. **commit sha**：完整的（四十位十六進位）與夠長的短 sha（長度下限登記在卡上，而且同時有數字與字母）
   必須在本機物件庫解析得到（``git cat-file -e``，不上網）。解析不到就紅——抄錯了，或是清空前舊歷史的 sha，
   那種引用該改寫成「清空前的歷史」而不是留一個死 sha。
2. **雲端 run id**：卡上登記位數的純數字串，必須是收據鏡像裡某一份的 ``run.run_id`` 或 ``written_by.run_id``
   （``governance/cloud_receipts.py`` 說明鏡像）。找不到就紅——抄錯了，或引用了一個沒有收據的跑。

先塗掉不算數的東西：網址（裡面的十六進位段是路徑不是 sha）、UUID（八-四-四-四-十二那種形狀）。
判準只在卡上：長度、位數、要塗的樣式。今天樹裡零個真 sha、零個 run id，第一回合是「掃了文件、沒有可對的」的綠；
文件開始引用之後它才真的在咬。誤咬一個不是 sha 的十六進位字串時，改寫法或在卡上具名放行（帶理由與到期日），
不放寬判準。git 不在回 2；文件裡有 run id 而鏡像不在回 2。
"""
from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from governance.cloud_receipts import Mirror, allowed_names, card_settings, read_mirror, table
from governance.exit_codes import ToolBroken, note, run
from governance.loader import setting_int, setting_strings

CHECK_REL = "governance/checks/identity_strings_generated.py"
VCS = "git"


@dataclass(frozen=True)
class Rules:
    """卡上的判準。"""

    full_len: int
    short_min: int
    run_id_digits: int
    mask: tuple[re.Pattern[str], ...]
    doc_suffixes: tuple[str, ...]
    doc_dirs: tuple[str, ...]
    doc_files: tuple[str, ...]


def read_rules(settings: Mapping[str, object]) -> Rules:
    try:
        masks = tuple(re.compile(p) for p in setting_strings(settings, "mask_patterns"))
    except re.error as exc:
        raise ToolBroken(f"卡上的 mask_patterns 有一條不是合法的正則：{exc}") from exc
    return Rules(
        full_len=setting_int(settings, "sha_full_len"),
        short_min=setting_int(settings, "sha_short_min_len"),
        run_id_digits=setting_int(settings, "run_id_digits"),
        mask=masks,
        doc_suffixes=tuple(setting_strings(settings, "doc_suffixes")),
        doc_dirs=tuple(d.rstrip("/") + "/" for d in setting_strings(settings, "doc_dirs")),
        doc_files=tuple(setting_strings(settings, "doc_files")),
    )


def targets_for(scan_root: Path, files: list[Path], rules: Rules) -> list[Path]:
    picked: list[Path] = []
    for f in files:
        rel = f.relative_to(scan_root).as_posix()
        in_dir = any(rel.startswith(d) for d in rules.doc_dirs) and f.suffix in rules.doc_suffixes
        if in_dir or rel in rules.doc_files:
            picked.append(f)
    return sorted(picked)


# ── 找身分字串 ────────────────────────────────────────────────────────────────


def masked(text: str, rules: Rules) -> str:
    """把網址與 UUID 塗掉（換成等長的空白，行號與欄位不動）。"""
    for pattern in rules.mask:
        text = pattern.sub(lambda m: " " * len(m.group(0)), text)
    return text


def sha_candidates(text: str, rules: Rules) -> list[str]:
    """完整 sha，加上夠長、同時有數字與字母的短 sha。"""
    out: list[str] = []
    for token in re.findall(r"(?<![0-9a-zA-Z_])[0-9a-f]+(?![0-9a-zA-Z_])", text):
        if len(token) == rules.full_len or (
            rules.short_min <= len(token) < rules.full_len and re.search(r"[0-9]", token) and re.search(r"[a-f]", token)
        ):
            out.append(token)
    return out


def run_id_candidates(text: str, rules: Rules) -> list[str]:
    return re.findall(rf"(?<![0-9a-zA-Z_.-])[0-9]{{{rules.run_id_digits}}}(?![0-9a-zA-Z_.-])", text)


# ── 對回去 ────────────────────────────────────────────────────────────────────


def sha_resolves(scan_root: Path, sha: str) -> bool:
    """本機物件庫裡有沒有這個物件（任何型別）。git 不在或炸掉就 ToolBroken。"""
    try:
        proc = subprocess.run([VCS, "cat-file", "-e", sha], cwd=scan_root, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise ToolBroken(f"{VCS} 不在 PATH 上——對不到物件庫就不出結論") from exc
    if proc.returncode == 0:
        return True
    err = proc.stderr.strip()
    if "Not a valid object name" in err or "could not get object info" in err or not err:
        return False
    raise ToolBroken(f"{VCS} cat-file -e {sha} 回 {proc.returncode}：{err[:200]}")


def receipt_run_ids(mirror: Mirror) -> set[str]:
    ids: set[str] = set()
    for receipt in mirror.receipts:
        run_block = table(receipt.body.get("run"), f"{receipt.name} 的 run")
        writer = table(receipt.body.get("written_by"), f"{receipt.name} 的 written_by")
        for value in (run_block.get("run_id"), writer.get("run_id")):
            if value is not None:
                ids.add(str(value))
    return ids


def judge(scan_root: Path, docs: list[Path], rules: Rules, settings: Mapping[str, object]) -> list[str]:
    hits: list[str] = []
    skip = allowed_names(settings, date.today())
    mirror: Mirror | None = None
    known: set[str] = set()
    checked_sha: dict[str, bool] = {}
    for doc in docs:
        rel = doc.relative_to(scan_root).as_posix()
        text = masked(doc.read_text(encoding="utf-8"), rules)
        for sha in sha_candidates(text, rules):
            if sha in skip:
                continue
            if sha not in checked_sha:
                checked_sha[sha] = sha_resolves(scan_root, sha)
            if not checked_sha[sha]:
                hits.append(f"{rel} 寫了 sha {sha}，本機物件庫裡沒有這個物件——抄錯了，或是清空前舊歷史的 sha")
        run_ids = run_id_candidates(text, rules)
        if run_ids and mirror is None:
            mirror = read_mirror(scan_root, settings, "identity-strings-generated")
            known = receipt_run_ids(mirror)
        for run_id in run_ids:
            if run_id not in skip and run_id not in known:
                hits.append(f"{rel} 寫了 run id {run_id}，收據鏡像裡沒有這一跑——抄錯了，或引用了一個沒有收據的跑")
    note(f"對過 {len(checked_sha)} 個 sha、鏡像裡 {len(known)} 個 run id（掃了 {len(docs)} 份文件）")
    return hits


def targets(scan_root: Path, files: list[Path]) -> list[Path]:
    """這支檢查真的會讀的檔：卡上登記的文件目錄與檔名，加上自己那張卡。"""
    card, settings = card_settings(scan_root, files, CHECK_REL)
    return sorted({card, *targets_for(scan_root, files, read_rules(settings))})


def check(scan_root: Path, files: list[Path]) -> list[str]:
    _, settings = card_settings(scan_root, files, CHECK_REL)
    rules = read_rules(settings)
    docs = targets_for(scan_root, files, rules)
    if not docs:
        raise ToolBroken("掃描面上一份文件都沒有——沒有東西可對就不出結論")
    return judge(scan_root, docs, rules, settings)


if __name__ == "__main__":
    sys.exit(
        run(
            check,
            description="文件裡的 commit sha 與雲端 run id 必須解析得到；git 不在或需要鏡像而鏡像不在回 2",
            targets=targets,
        )
    )
