"""規矩卡 four-roles-different-actors：收據不准由被查的那一方寫，也不准由人寫。

藍圖上這張卡講的是派工的四席（工人、找碴、驗收、老闆）不得同一人；本機派工收據在 repo 外、算宣稱
（決策紙 receipts-two-layers-cloud-on-status-branch），雲端收據能證明的只有兩件，這張卡就守這兩件：

1. **寫收據的不是被查的那一跑**：``written_by.run_id`` 不准等於 ``run.run_id``。收據由另一跑（status.yml 的
   receipt job）合成，被判的那一跑自己寫自己的收據就是 v2 事故 builder-grades-own-work-edits-ruler 的形狀
   （施工者自審自報）。
2. **收據只有機器寫得進去**：來源檔（鏡像的 provenance）裡每一份收據最後一筆提交的 author 與 committer 都必須是
   卡上登記的機器身分。人手推上 status 分支的收據，紅。

照實記的洞：git 的 author／committer 是自報的（任何人都能把 email 設成機器那個），這一條證明的是「宣稱的身分」，
不是簽過名的身分；擋人手推的另一半是 status 分支只有 workflow 的 token 推得動（權限那一層）。「找碴席要換一家模型」
從雲端收據看不出來，不在這張卡裡。收據的 schema 比卡登記的還新就回 2；鏡像不在、來源檔缺了哪一份，一律回 2。
"""
from __future__ import annotations

import sys
from collections.abc import Mapping
from datetime import date
from pathlib import Path

from governance.cloud_receipts import (
    Mirror,
    Receipt,
    allowed_names,
    card_settings,
    is_int,
    read_mirror,
    table,
)
from governance.exit_codes import ToolBroken, note, run
from governance.loader import setting_strings

CHECK_REL = "governance/checks/four_roles_different_actors.py"


def _writer_is_not_the_judged(receipt: Receipt) -> list[str]:
    """第①條。"""
    run_id = table(receipt.body.get("run"), f"{receipt.name} 的 run").get("run_id")
    writer = table(receipt.body.get("written_by"), f"{receipt.name} 的 written_by").get("run_id")
    if not is_int(run_id) or writer is None:
        return [f"{receipt.name} 的 run.run_id／written_by.run_id 讀不成 run id：{run_id!r}／{writer!r}"]
    if str(writer) == str(run_id):
        return [f"{receipt.name}：寫收據的就是被查的那一跑（run {run_id}）——施工者自審自報，收據不算數"]
    return []


def _committed_by_machine(receipt: Receipt, mirror: Mirror, machines: set[str]) -> list[str]:
    """第②條。來源檔缺這一份就 ToolBroken：不知道是誰寫的，不出結論。"""
    files = table(mirror.provenance.get("files"), "來源檔的 files")
    if receipt.name not in files:
        raise ToolBroken(f"來源檔裡沒有 {receipt.name} 這一份——不知道它是誰寫的，這一跑不算數（鏡像不完整）")
    origin = table(files[receipt.name], f"來源檔裡 {receipt.name} 那一筆")
    hits: list[str] = []
    for role in ("author_email", "committer_email"):
        who = origin.get(role)
        if not isinstance(who, str) or not who.strip():
            hits.append(f"{receipt.name} 的來源 {role} 是空的：{who!r}——不知道是誰寫的")
        elif who.strip().lower() not in machines:
            hits.append(f"{receipt.name} 是 {who} 寫進 status 分支的（{role}），不是登記的機器身分——收據只有機器寫得進去")
    return hits


def judge(mirror: Mirror, settings: Mapping[str, object]) -> list[str]:
    machines = {m.strip().lower() for m in setting_strings(settings, "machine_emails")}
    if not machines:
        raise ToolBroken("卡上 machine_emails 是空的——沒有登記的機器身分就分不出誰是機器")
    skip = allowed_names(settings, date.today())
    hits: list[str] = []
    for receipt in mirror.receipts:
        if receipt.name in skip:
            note(f"放行（卡上具名）：{receipt.name}")
            continue
        hits += _writer_is_not_the_judged(receipt) + _committed_by_machine(receipt, mirror, machines)
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
            description="收據不准由被查的那一跑寫、不准由人寫；鏡像不在或來源不全回 2",
            targets=targets,
        )
    )
