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
from datetime import date
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
    job_names = _job_names(receipt) if "job" in required else set()
    for index, item in enumerate(checks):
        row = table(item, f"{receipt.name} 的 checks[{index}]")
        label = f"{receipt.name} 的 checks[{index}]（{row.get('name', '?')}）"
        hits += _missing(row, required, label)
        if "job" in required and (not isinstance(row.get("job"), str) or row["job"] not in job_names):
            hits.append(f"{label} 的 job 不在 jobs 名單裡：{row.get('job')!r}")
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
    required = by_schema(settings, "required_pytest", receipt.schema, where)
    if block is None:
        return [f"{receipt.name} 的 pytest 缺 parts（不可為 null）"] if "parts" in required else []
    summary = table(block, f"{receipt.name} 的 pytest")
    hits = _missing(summary, required, f"{receipt.name} 的 pytest")
    if "parts" in required:
        return hits + (_pytest_parts(receipt, summary, settings, where) if "parts" in summary else [])
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


def _job_names(receipt: Receipt) -> set[str]:
    """缺 jobs 由頂層必填判紅；形狀解不開沿用共用收窄的回 2。"""
    if "jobs" not in receipt.body:
        return set()
    names: set[str] = set()
    for item in rows(receipt.body["jobs"], f"{receipt.name} 的 jobs"):
        name = table(item, "一個 job").get("name")
        if isinstance(name, str) and name.strip():
            names.add(name)
    return names


def _jobs_block(receipt: Receipt, settings: Mapping[str, object], where: str) -> list[str]:
    """schema 3 job 清單的欄位齊全與可用名字；不加跨 job 審計。"""
    jobs = rows(receipt.body["jobs"], f"{receipt.name} 的 jobs")
    if not jobs:
        return [f"{receipt.name} 的 jobs 是空的"]
    required = by_schema(settings, "required_job", receipt.schema, where)
    hits: list[str] = []
    for index, item in enumerate(jobs):
        job = table(item, "一個 job")
        label = f"{receipt.name} 的 jobs[{index}]"
        hits += _missing(job, required, label)
        if not isinstance(job.get("name"), str) or not str(job["name"]).strip():
            hits.append(f"{label} 的 name 不是非空字串")
    return hits


def _pytest_parts(receipt: Receipt, summary: Mapping[str, object], settings: Mapping[str, object], where: str) -> list[str]:
    """每片四個數、junit 路徑、雜湊與大小都要有證據；空清單表示沒跑到 pytest。"""
    required = by_schema(settings, "required_pytest_part", receipt.schema, where)
    counts = by_schema(settings, "pytest_count_fields", receipt.schema, where)
    job_names = _job_names(receipt)
    hex_len = setting_int(settings, "sha256_hex_len")
    hits: list[str] = []
    for index, item in enumerate(rows(summary["parts"], f"{receipt.name} 的 pytest.parts")):
        part = table(item, f"pytest.parts[{index}]")
        label = f"{receipt.name} 的 pytest.parts[{index}]"
        hits += _missing(part, required, label)
        for key in counts:
            if not is_int(part.get(key)):
                hits.append(f"{label}.{key} 不是整數：{part.get(key)!r}")
        digest = part.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(rf"[0-9a-f]{{{hex_len}}}", digest):
            hits.append(f"{label}.sha256 不是一個十六進位雜湊：{digest!r}——沒有雜湊就沒有證據")
        if not is_int(part.get("bytes")):
            hits.append(f"{label}.bytes 不是整數：{part.get('bytes')!r}")
        if not isinstance(part.get("junit"), str) or not str(part["junit"]).strip():
            hits.append(f"{label}.junit 不是非空字串")
        if not isinstance(part.get("job"), str) or part["job"] not in job_names:
            hits.append(f"{label}.job 不在 jobs 名單裡：{part.get('job')!r}")
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
    if "jobs" in by_schema(settings, "required_top", receipt.schema, where) and "jobs" in receipt.body:
        hits += _jobs_block(receipt, settings, where)
    if "run" in receipt.body and "written_by" in receipt.body:
        hits += _identity(receipt, settings, where)
    if "checks" in receipt.body:
        hits += _check_rows(receipt, settings, where)
    if "pytest" in receipt.body:
        hits += _pytest_block(receipt, settings, where)
    return hits


def judge(mirror: Mirror, settings: Mapping[str, object], where: str) -> list[str]:
    skip = allowed_names(settings, date.today())
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
