"""規矩卡 receipt-schema-complete：雲端收據的欄位要齊全，每個數字要有證據。

讀的是 status 分支鏡過來的機器收據（`governance/cloud_receipts.py` 說明鏡像長什麼樣、為什麼不在列舉集合裡）。
六條，欄位名全部住在卡的 ``[settings]``，按收據的 schema 版本挑：

1. 頂層、``run``、``written_by`` 的必填欄位都在；``checks`` 是非空清單，每一筆的必填欄位都在。
2. ``pytest`` 不是 null 時它的必填欄位都在；登記了 ``evidence`` 的版本，evidence 裡要有 junit 的雜湊與大小
   ——數字是從那一份算的，沒有雜湊就是一個沒有證據的數字（v2 事故 verified-command-with-falsified-number）。
3. 每一支檢查的 ``exit_code`` 是整數；``report``（判決收據那一行）是字串，只有卡上登記「本來就不印那一行」
   的那幾支准是 null——離開碼是數字，那一行是它的證據。
4. 檔名 ``<run id>-<attempt>.json`` 要等於收據裡的 ``run.run_id`` 與 ``run.attempt``：檔名是索引，內文是宣稱，
   兩邊要一致。
5. ``run.head_sha`` 要是完整的十六進位 commit id（長度登記在卡上）——被截短的 sha 對不回那棵樹。
6. 收據的 schema 比卡登記的還新就回 2（沒有尺）；比卡舊的按它自己那版判。

刻意沒管（藍圖上原本有、雲端收據上沒有對象的）：空跑收據、窄修回原 session——那是派工收據的概念，
本機派工收據在 repo 外、算宣稱（決策紙 receipts-two-layers-cloud-on-status-branch）。
"""
from __future__ import annotations

import re
import sys
from collections.abc import Mapping
from pathlib import Path

from governance.cloud_receipts import (
    Mirror,
    Receipt,
    allowed_names,
    by_schema,
    card_settings,
    is_int,
    read_mirror,
    rows,
    table,
)
from governance.exit_codes import note, run
from governance.loader import setting_int, setting_strings

CHECK_REL = "governance/checks/receipt_schema_complete.py"


def _missing(body: Mapping[str, object], required: list[str], where: str) -> list[str]:
    return [f"{where} 缺欄位 {key!r}" for key in required if key not in body]


def _check_rows(receipt: Receipt, settings: Mapping[str, object], where: str) -> list[str]:
    """第①條的 checks[] 半段與第③條。"""
    hits: list[str] = []
    checks = rows(receipt.body.get("checks"), f"{receipt.name} 的 checks")
    if not checks:
        return [f"{receipt.name} 的 checks 是空的——一支檢查都沒記的收據證明不了任何事"]
    required = by_schema(settings, "required_check", receipt.schema, where)
    optional_report = set(setting_strings(settings, "report_optional_names"))
    for index, item in enumerate(checks):
        row = table(item, f"{receipt.name} 的 checks[{index}]")
        label = f"{receipt.name} 的 checks[{index}]（{row.get('name', '?')}）"
        hits += _missing(row, required, label)
        if "exit_code" in row and not is_int(row["exit_code"]):
            hits.append(f"{label} 的 exit_code 不是整數：{row['exit_code']!r}")
        report = row.get("report")
        if "report" in row and report is None and str(row.get("name")) not in optional_report:
            hits.append(f"{label} 的 report 是 null——離開碼是數字，判決收據那一行是它的證據；本來就不印那一行的要登記在卡上")
        elif report is not None and not isinstance(report, str):
            hits.append(f"{label} 的 report 不是字串：{report!r}")
    return hits


def _pytest_block(receipt: Receipt, settings: Mapping[str, object], where: str) -> list[str]:
    """第②條。"""
    block = receipt.body.get("pytest")
    if block is None:
        return []
    summary = table(block, f"{receipt.name} 的 pytest")
    required = by_schema(settings, "required_pytest", receipt.schema, where)
    hits = _missing(summary, required, f"{receipt.name} 的 pytest")
    for key in required:
        if key in summary and key != "evidence" and not is_int(summary[key]):
            hits.append(f"{receipt.name} 的 pytest.{key} 不是整數：{summary[key]!r}")
    if "evidence" in required and "evidence" in summary:
        evidence = table(summary["evidence"], f"{receipt.name} 的 pytest.evidence")
        hex_len = setting_int(settings, "sha256_hex_len")
        digest = evidence.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(rf"[0-9a-f]{{{hex_len}}}", digest):
            hits.append(f"{receipt.name} 的 pytest.evidence.sha256 不是一個十六進位雜湊：{digest!r}——四個數是從 junit 算的，沒有雜湊就沒有證據")
        if not is_int(evidence.get("bytes")):
            hits.append(f"{receipt.name} 的 pytest.evidence.bytes 不是整數：{evidence.get('bytes')!r}")
    return hits


def _identity(receipt: Receipt, settings: Mapping[str, object], where: str) -> list[str]:
    """第①條的 run／written_by 半段、第④條、第⑤條。"""
    hits: list[str] = []
    run_block = table(receipt.body.get("run"), f"{receipt.name} 的 run")
    hits += _missing(run_block, by_schema(settings, "required_run", receipt.schema, where), f"{receipt.name} 的 run")
    hits += _missing(
        table(receipt.body.get("written_by"), f"{receipt.name} 的 written_by"),
        by_schema(settings, "required_written_by", receipt.schema, where),
        f"{receipt.name} 的 written_by",
    )
    run_id, attempt = run_block.get("run_id"), run_block.get("attempt")
    expected_name = f"{run_id}-{attempt}.json"
    if not is_int(run_id) or not is_int(attempt):
        hits.append(f"{receipt.name} 的 run.run_id／run.attempt 不是整數：{run_id!r}／{attempt!r}")
    elif receipt.name != expected_name:
        hits.append(f"{receipt.name} 的檔名跟內文對不上：內文說它是 {expected_name}——檔名是索引，內文是宣稱，兩邊要一致")
    sha = run_block.get("head_sha")
    sha_len = setting_int(settings, "sha_hex_len")
    if not isinstance(sha, str) or not re.fullmatch(rf"[0-9a-f]{{{sha_len}}}", sha):
        hits.append(f"{receipt.name} 的 run.head_sha 不是完整的十六進位 commit id：{sha!r}——被截短的 sha 對不回那棵樹")
    return hits


def receipt_hits(receipt: Receipt, settings: Mapping[str, object], where: str) -> list[str]:
    """一份收據的全部違規。"""
    hits = _missing(receipt.body, by_schema(settings, "required_top", receipt.schema, where), receipt.name)
    if "run" in receipt.body and "written_by" in receipt.body:
        hits += _identity(receipt, settings, where)
    if "checks" in receipt.body:
        hits += _check_rows(receipt, settings, where)
    if "pytest" in receipt.body:
        hits += _pytest_block(receipt, settings, where)
    return hits


def judge(mirror: Mirror, settings: Mapping[str, object], where: str) -> list[str]:
    skip = allowed_names(settings)
    hits: list[str] = []
    for receipt in mirror.receipts:
        if receipt.name in skip:
            note(f"放行（卡上具名）：{receipt.name}")
            continue
        hits += receipt_hits(receipt, settings, where)
    return hits


def targets(scan_root: Path, files: list[Path]) -> list[Path]:
    """這支檢查在列舉集合裡真的會讀的檔：只有自己那張卡。鏡像不在集合裡（見 cloud_receipts 檔頭）。"""
    return [card_settings(scan_root, files, CHECK_REL)[0]]


def check(scan_root: Path, files: list[Path]) -> list[str]:
    card, settings = card_settings(scan_root, files, CHECK_REL)
    where = str(card.relative_to(scan_root))
    return judge(read_mirror(scan_root, settings, where), settings, where)


if __name__ == "__main__":
    sys.exit(
        run(
            check,
            description="雲端收據的欄位要齊全、每個數字有證據；鏡像不在或版本比卡新回 2",
            targets=targets,
        )
    )
