"""規矩卡 receipt-authority-is-the-cloud-run：算數的收據來自雲端那一跑，自報的離開碼不算。

讀的是 status 分支鏡過來的機器收據（`governance/cloud_receipts.py` 說明鏡像與按版本判）。四條：

1. ``authority`` 必須是卡上登記的那個值（``cloud-run``），``run.run_id`` 與 ``written_by.run_id`` 是整數，
   ``run.url`` 裡要帶著 run id——收據要機器可回查，不是一句「雲端跑過了」。
2. **從 raw 重算，不信收據自己的 consistency 欄**：GitHub 記的 job 結論是綠（``job.conclusion`` 等於卡上登記的
   成功值）而 ``checks[]`` 裡有任何一支的 ``exit_code`` 非零，就紅——自報的離開碼跟 GitHub 記的紅綠對不上，
   中間有一層把它吞了（v2 事故 verifier-trusts-self-reported-fields：驗證器信自報欄位、不從 raw 重算，九起假綠）。
3. 逾時與真退出碼分開記：每一支檢查有 ``signal`` 欄；``signal`` 不是 null 的（被訊號殺掉、逾時）``exit_code``
   必須是卡上登記的基底加訊號號碼，被記成 0 就紅——被殺掉不是完成。
4. 新規矩不准回頭把舊收據判成無效：收據的 schema 比卡登記的最新版新就回 2（沒有尺），比卡舊的按它自己那版判
   ——這一條由共用零件的 ``newest_schema`` 與 ``by_schema`` 落實，這裡不重寫。

一份出過事的收據會讓主線永遠紅（status 分支不改寫歷史）：卡上 ``[[settings.allow]]`` 具名放過那一份，帶理由與到期日。
"""
from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path

from governance.cloud_receipts import (
    Mirror,
    Receipt,
    allowed_names,
    card_settings,
    is_int,
    read_mirror,
    rows,
    table,
)
from governance.exit_codes import note, run
from governance.loader import setting_int, setting_text

CHECK_REL = "governance/checks/receipt_authority_is_the_cloud_run.py"


def _authority(receipt: Receipt, settings: Mapping[str, object]) -> list[str]:
    """第①條。"""
    hits: list[str] = []
    want = setting_text(settings, "authority_value")
    got = receipt.body.get("authority")
    if got != want:
        hits.append(f"{receipt.name} 的 authority 是 {got!r}，算數的只有 {want!r}——本機任何人寫的都是宣稱，不走這條路")
    run_block = table(receipt.body.get("run"), f"{receipt.name} 的 run")
    writer = table(receipt.body.get("written_by"), f"{receipt.name} 的 written_by")
    run_id = run_block.get("run_id")
    if not is_int(run_id):
        hits.append(f"{receipt.name} 的 run.run_id 不是整數：{run_id!r}——沒有 run id 就回查不到那一跑")
    elif str(run_id) not in str(run_block.get("url", "")):
        hits.append(f"{receipt.name} 的 run.url 裡沒有 run id {run_id}：{run_block.get('url')!r}——收據要機器可回查")
    if not is_int(writer.get("run_id")) and not (isinstance(writer.get("run_id"), str) and str(writer.get("run_id")).isdigit()):
        hits.append(f"{receipt.name} 的 written_by.run_id 不是一個 run id：{writer.get('run_id')!r}——不知道是哪一跑寫的收據")
    return hits


def _recomputed(receipt: Receipt, settings: Mapping[str, object]) -> list[str]:
    """第②條：GitHub 說綠、片段說非零，中間有一層吞了離開碼。"""
    job = table(receipt.body.get("job"), f"{receipt.name} 的 job")
    conclusion = job.get("conclusion")
    checks = [table(item, f"{receipt.name} 的 checks[]") for item in rows(receipt.body.get("checks"), f"{receipt.name} 的 checks")]
    nonzero = [f"{row.get('name')}={row.get('exit_code')}" for row in checks if is_int(row.get("exit_code")) and row["exit_code"] != 0]
    if conclusion == setting_text(settings, "success_conclusion") and nonzero:
        return [
            f"{receipt.name}：GitHub 記 job 結論是 {conclusion!r}，但收據裡這幾支的離開碼非零：{nonzero}"
            "——自報的離開碼跟 GitHub 記的紅綠對不上，中間有一層把它吞了"
        ]
    return []


def _signals(receipt: Receipt, settings: Mapping[str, object]) -> list[str]:
    """第③條：被訊號殺掉（逾時）的不准被記成 0。"""
    hits: list[str] = []
    base = setting_int(settings, "signal_exit_base")
    for index, item in enumerate(rows(receipt.body.get("checks"), f"{receipt.name} 的 checks")):
        row = table(item, f"{receipt.name} 的 checks[{index}]")
        label = f"{receipt.name} 的 checks[{index}]（{row.get('name', '?')}）"
        if "signal" not in row:
            hits.append(f"{label} 沒有 signal 欄——逾時與真退出碼要分開記，沒有這一格就分不開")
            continue
        signal = row["signal"]
        if signal is None:
            continue
        if not is_int(signal):
            hits.append(f"{label} 的 signal 不是整數也不是 null：{signal!r}")
        elif row.get("exit_code") != base + signal:
            hits.append(f"{label} 被訊號 {signal} 殺掉，exit_code 卻是 {row.get('exit_code')!r}（該是 {base + signal}）——被殺掉不是完成")
    return hits


def judge(mirror: Mirror, settings: Mapping[str, object]) -> list[str]:
    skip = allowed_names(settings)
    hits: list[str] = []
    for receipt in mirror.receipts:
        if receipt.name in skip:
            note(f"放行（卡上具名）：{receipt.name}")
            continue
        hits += _authority(receipt, settings) + _recomputed(receipt, settings) + _signals(receipt, settings)
    return hits


def targets(scan_root: Path, files: list[Path]) -> list[Path]:
    """這支檢查在列舉集合裡真的會讀的檔：只有自己那張卡。鏡像不在集合裡（見 cloud_receipts 檔頭）。"""
    return [card_settings(scan_root, files, CHECK_REL)[0]]


def check(scan_root: Path, files: list[Path]) -> list[str]:
    card, settings = card_settings(scan_root, files, CHECK_REL)
    return judge(read_mirror(scan_root, settings, str(card.relative_to(scan_root))), settings)


if __name__ == "__main__":
    sys.exit(
        run(
            check,
            description="算數的收據來自雲端那一跑：自報離開碼跟 GitHub 記的紅綠對不上就紅；鏡像不在或版本比卡新回 2",
            targets=targets,
        )
    )
