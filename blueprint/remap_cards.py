#!/usr/bin/env python3
"""把 38 張合併卡的 v2 教訓 id 機器重對，產出 blueprint/cards-38.json。

背景：38 張合併卡不在版控裡，只在備份的 convergence-journal.jsonl 裡。
合併那一步自己手寫的 lesson_ids 有一部分是編的（不存在於 lessons.json），
所以這支腳本整組丟掉，改用「沿 covers 回查候選規則的 lesson_ids」重建。

可重跑。跑完自己驗一次，任何 lesson id 對不回 lessons.json 就 exit 1。
用法：python3 blueprint/remap_cards.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# ---- 來源（常數）----
REPO = Path(__file__).resolve().parents[1]
JOURNAL = Path("/home/florian/aosr-v3-blueprint-2026-09-09/convergence-journal.jsonl")
JOURNAL_ROW = 8  # 第 9 行（index 8）的 result.cards 是那 38 張
LESSONS = REPO / "v2-audit/lessons.json"
BATCH1 = REPO / "blueprint/batch1-127.json"
RULES436 = REPO / "blueprint/rules-436.json"
CONVERGENCE = REPO / "blueprint/convergence-result.json"
OUT = REPO / "blueprint/cards-38.json"

CARD_FIELDS = ("id", "human", "check_idea", "fixture_idea", "covers", "depends_on")


def die(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    if not JOURNAL.exists():
        die(f"備份不在：{JOURNAL}（38 張卡只存在那裡，沒有它無法重跑）")

    lessons = json.loads(LESSONS.read_text(encoding="utf-8"))
    incident_ids = {i["id"] for i in lessons["incidents"]}
    if len(incident_ids) != 66:
        die(f"lessons.json 的 incidents 應為 66 筆，實際 {len(incident_ids)}")

    # 候選規則 id -> lesson_ids。batch1 優先，查不到再查 436。
    batch1 = {c["id"]: list(c.get("lesson_ids") or []) for c in json.loads(BATCH1.read_text(encoding="utf-8"))}
    rules436 = {c["id"]: list(c.get("lesson_ids") or []) for c in json.loads(RULES436.read_text(encoding="utf-8"))}

    rows = [json.loads(line) for line in JOURNAL.read_text(encoding="utf-8").splitlines() if line.strip()]
    journal_result = rows[JOURNAL_ROW]["result"]
    raw_cards = journal_result["cards"]
    if len(raw_cards) != 38:
        die(f"journal 第 {JOURNAL_ROW + 1} 行應有 38 張卡，實際 {len(raw_cards)}（行號跑掉了？）")

    conv = json.loads(CONVERGENCE.read_text(encoding="utf-8"))["result"]
    ranked = {c["id"]: c for c in conv["排序後的卡"]}
    deferred = {c["id"]: c for c in conv["被刷掉的卡"]}
    card_ids = {c["id"] for c in raw_cards}
    if set(ranked) | set(deferred) != card_ids or set(ranked) & set(deferred):
        die("convergence-result 的 12＋26 沒有剛好切開 38 張卡")

    cards = []
    for raw in raw_cards:
        covers = list(raw.get("covers") or [])
        provenance: dict[str, list[str]] = {}
        for cand in covers:
            src = batch1[cand] if cand in batch1 else rules436.get(cand)
            if src is None:
                die(f"候選規則 {cand} 在 batch1-127 與 rules-436 都查不到")
            for lid in src:
                if lid in incident_ids:
                    provenance.setdefault(lid, [])
                    if cand not in provenance[lid]:
                        provenance[lid].append(cand)
        stated = list(raw.get("lesson_ids") or [])
        card = {
            "id": raw["id"],
            "human": raw["human"],
            "check_idea": raw["check_idea"],
            "fixture_idea": raw["fixture_idea"],
            "covers": covers,
            "lesson_ids": sorted(provenance),
            "lesson_provenance": {k: provenance[k] for k in sorted(provenance)},
            "dropped_ghost_ids": [x for x in stated if x not in incident_ids],
            "feasibility": "pass" if raw["id"] in ranked else "deferred",
            "depends_on": list(raw.get("depends_on") or []),
        }
        if raw["id"] in ranked:
            card["rank"] = ranked[raw["id"]]["rank"]
        else:
            d = deferred[raw["id"]]
            # 原文照抄，不改寫
            card["defer_reason"] = {"stuck_at": d["卡在"], "critic_reason": d["找碴理由"]}
        cards.append(card)

    cards.sort(key=lambda c: (c["feasibility"] != "pass", c.get("rank", 0), c["id"]))

    all_stated = {x for c in raw_cards for x in (c.get("lesson_ids") or [])}
    ghosts = {x for x in all_stated if x not in incident_ids}
    covered = {lid for c in cards for lid in c["lesson_ids"]}
    with_debt = [c for c in cards if c["lesson_ids"]]
    pass_cards = [c for c in cards if c["feasibility"] == "pass"]

    # 上游乾淨度：候選規則檔自己引用的 lesson id 有沒有幽靈
    upstream_refs = {x for v in batch1.values() for x in v} | {x for v in rules436.values() for x in v}
    upstream_ghosts = sorted(upstream_refs - incident_ids)

    # 合併那步原本寫對、但沿 covers 回查不到的真 id（照規格丟掉，但要留紀錄）
    raw_by_id = {r["id"]: r for r in raw_cards}
    unreachable = {}
    for c in cards:
        stated_real = {x for x in (raw_by_id[c["id"]].get("lesson_ids") or []) if x in incident_ids}
        extra = sorted(stated_real - set(c["lesson_ids"]))
        if extra:
            unreachable[c["id"]] = extra
    # 丟掉這些之後才變成 0 血債的卡（若有，就是重對真的弄丟了血債）
    cards_by_id = {c["id"]: c for c in cards}
    lost_debt = sorted(k for k in unreachable if not cards_by_id[k]["lesson_ids"])

    meta = {
        "產生方式": "丟掉合併那步手寫的 lesson_ids，改成沿每張卡的 covers 回查候選規則檔的 lesson_ids 取聯集，"
        "只留存在於 v2-audit/lessons.json 的 id；由 blueprint/remap_cards.py 產生，可重跑。",
        "產生腳本": "blueprint/remap_cards.py",
        "來源": {
            "38 張卡（不在版控裡）": f"{JOURNAL}（第 {JOURNAL_ROW + 1} 行 result.cards）",
            "教訓": "v2-audit/lessons.json",
            "候選規則（covers 回查，優先）": "blueprint/batch1-127.json",
            "候選規則（查不到才用）": "blueprint/rules-436.json",
            "可行性與排名": "blueprint/convergence-result.json（排序後的卡 12／被刷掉的卡 26）",
        },
        "統計": {
            "卡數": len(cards),
            "通過可行性": len(pass_cards),
            "被刷掉": len(cards) - len(pass_cards),
            "covers 去重": len({x for c in cards for x in c["covers"]}),
            "合併那步自稱引用的教訓 id 去重": len(all_stated),
            "其中真的存在": len(all_stated) - len(ghosts),
            "其中是幽靈": len(ghosts),
            "重對後有血債的卡": len(with_debt),
            "12 張裡有血債的卡": sum(1 for c in pass_cards if c["lesson_ids"]),
            "重對後覆蓋的教訓筆數": len(covered),
            "教訓總筆數": len(incident_ids),
            "上游候選規則檔的幽靈教訓 id": len(upstream_ghosts),
            "原本寫對但沿 covers 查不到、照規格丟掉的教訓 id（卡數）": len(unreachable),
            "丟掉它們之後才變成 0 血債的卡": len(lost_debt),
        },
        "註": [
            "幽靈只在合併那一步產生：上游 batch1-127.json 與 rules-436.json 引用的教訓 id 全部存在於 lessons.json。",
            "整體數字上，「只沿 covers 重建」與「covers 再聯集合併那步原本寫對的 id」一樣："
            f"血債 {len(with_debt)}/38、12 張裡 {sum(1 for c in pass_cards if c['lesson_ids'])}/12、覆蓋 {len(covered)}/66 都不變。"
            "逐張看則有幾張卡的某個真 id 沿 covers 回查不到（見「原本寫對但查不到的」），"
            "照規格丟掉，沒有任何一張卡因此從有血債變成 0 血債。",
            "重對後 0 血債的卡不代表沒有 v2 血債，只代表它的 covers 那幾條候選規則本來就沒掛教訓；"
            "要按內容再對一次，見 blueprint/first-batch-review.json。",
        ],
        "原本寫對但沿 covers 查不到的": unreachable,
        "重對後 0 血債的卡": [c["id"] for c in cards if not c["lesson_ids"]],
    }

    OUT.write_text(
        json.dumps({"meta": meta, "cards": cards}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # ---- 自檢：重讀寫出去的檔 ----
    written = json.loads(OUT.read_text(encoding="utf-8"))
    wcards = written["cards"]
    if len(wcards) != 38:
        die(f"寫出的卡數 {len(wcards)} != 38")
    if {c["id"] for c in wcards} != card_ids:
        die("寫出的卡 id 集合跟來源不一致")
    bad = []
    for c in wcards:
        for f in CARD_FIELDS:
            if f not in c:
                bad.append(f"{c['id']} 缺欄位 {f}")
        for lid in c["lesson_ids"]:
            if lid not in incident_ids:
                bad.append(f"{c['id']} 的 lesson_id {lid} 不在 lessons.json")
        for lid, srcs in c["lesson_provenance"].items():
            if lid not in incident_ids:
                bad.append(f"{c['id']} 的 provenance key {lid} 不在 lessons.json")
            if not srcs:
                bad.append(f"{c['id']} 的 provenance {lid} 沒有來源候選規則")
            for s in srcs:
                if s not in c["covers"]:
                    bad.append(f"{c['id']} 的 provenance 來源 {s} 不在它的 covers 裡")
        if sorted(c["lesson_provenance"]) != c["lesson_ids"]:
            bad.append(f"{c['id']} 的 lesson_ids 與 lesson_provenance 的 key 不一致")
        if c["feasibility"] == "pass" and "rank" not in c:
            bad.append(f"{c['id']} 通過可行性卻沒有 rank")
        if c["feasibility"] == "deferred" and "defer_reason" not in c:
            bad.append(f"{c['id']} 被刷掉卻沒有 defer_reason")
    if lost_debt:
        bad.append(f"丟掉「原本寫對但查不到」之後這些卡變成 0 血債，重對弄丟了血債：{lost_debt}")
    if upstream_ghosts:
        bad.append(f"上游候選規則檔竟然有幽靈教訓 id：{upstream_ghosts}")
    if bad:
        for b in bad:
            print(f"FAIL: {b}", file=sys.stderr)
        sys.exit(1)

    s = meta["統計"]
    print(f"OK 寫出 {OUT.relative_to(REPO)}")
    print(f"  卡 {s['卡數']} 張（通過 {s['通過可行性']}／被刷掉 {s['被刷掉']}），covers 去重 {s['covers 去重']} 條")
    print(f"  合併那步自稱引用 {s['合併那步自稱引用的教訓 id 去重']} 個教訓 id，真 {s['其中真的存在']}／幽靈 {s['其中是幽靈']}")
    print(f"  重對後有血債 {s['重對後有血債的卡']}/{s['卡數']}，12 張裡 {s['12 張裡有血債的卡']}/12")
    print(f"  覆蓋教訓 {s['重對後覆蓋的教訓筆數']}/{s['教訓總筆數']}；上游幽靈 {s['上游候選規則檔的幽靈教訓 id']}")
    print(f"  重對後 0 血債：{meta['重對後 0 血債的卡']}")
    print("  自檢通過：所有 lesson id 都對回 lessons.json")


if __name__ == "__main__":
    main()
