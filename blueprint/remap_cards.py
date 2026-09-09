#!/usr/bin/env python3
"""把 38 張合併卡的 v2 教訓 id 機器重對，產出 blueprint/cards-38.json。

背景：38 張合併卡不在版控裡，只在備份的 convergence-journal.jsonl 裡。
合併那一步自己手寫的 lesson_ids 有一部分是編的（不存在於 lessons.json），
所以這支腳本整組丟掉，改用「沿 covers 回查候選規則的 lesson_ids」重建。

第 2 版（revision 2）：第二輪人寫的欄位（check_idea_v2、feasibility_v2、blocked_on…）
不是這支腳本產生的，而是寫在 blueprint/cards-38.json 裡。重跑時這支腳本先讀舊檔、
把那些欄位原樣搬過來，**絕不覆蓋、絕不清掉**，然後重算統計並重驗一次。
所以「改 v2 欄位」的做法是直接編輯 cards-38.json，再跑一次這支腳本讓它重算統計與自驗。

可重跑，且對同一份輸入是冪等的（跑兩次結果一樣）。
跑完自己驗一次，任何 lesson id 對不回 lessons.json 就 exit 1。
用法：python3 blueprint/remap_cards.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, NoReturn

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

# 第二輪人寫的欄位。這支腳本只搬運、不生成、不覆蓋。
V2_FIELDS = (
    "check_idea_v2",
    "fixture_idea_v2",
    "requires_shared_parts",
    "feasibility_v2",
    "feasibility_v2_reason",
    "blocked_on",
    "depends_on_v2",
    "narrowed",
    "recommend_drop",
    "recommend_drop_reason",
    "still_leaky",
    "still_leaky_reason",
    "disposition",
    "merge_suggestion",
    "critic_v2",
    "lesson_ids_dropped_by_review",
    "lesson_ids_dropped_reason",
    "lesson_ids_added_by_review",
    "lesson_ids_added_reason",
)

# blocked_on 的合法值。同一種等待用同一個 slug，才排得出「一組一張 issue」。
BLOCKED_ON_SLUGS = {
    "entry-files-exist",
    "docs-layout-decided",
    "src-tree-in-repo",
    "dispatch-tool-in-repo",
    "receipts-exist",
    "shell-scripts-in-repo",
    "design-report-exists",
    "boss-decision",
}

# requires_shared_parts 的合法值。第一張卡要把這些一起帶進來。
SHARED_PARTS = {
    "card-loader",
    "exit-code-convention",
    "fixture-runner",
    "ci-workflow",
    "threshold-in-card",
}


def die(msg: str) -> NoReturn:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def load_inputs() -> dict[str, Any]:
    """讀所有輸入檔並驗它們的形狀。回傳後面每一步共用的那幾份資料。"""
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

    return {
        "incident_ids": incident_ids,
        "batch1": batch1,
        "rules436": rules436,
        "raw_cards": raw_cards,
        "ranked": ranked,
        "deferred": deferred,
        "card_ids": card_ids,
    }


def carried_v2_fields() -> dict[str, dict[str, object]]:
    """讀舊檔，把第二輪人寫的欄位原樣搬過來（不覆蓋、不生成）。"""
    carried: dict[str, dict[str, object]] = {}
    if OUT.exists():
        old = json.loads(OUT.read_text(encoding="utf-8"))
        for oc in old.get("cards", []):
            carried[oc["id"]] = {f: oc[f] for f in V2_FIELDS if f in oc}
    return carried


def build_cards(inputs: dict[str, Any], carried: dict[str, dict[str, object]]) -> list[dict[str, Any]]:
    """沿每張卡的 covers 回查候選規則的 lesson_ids，重建 38 張卡。"""
    incident_ids = inputs["incident_ids"]
    batch1 = inputs["batch1"]
    rules436 = inputs["rules436"]
    ranked = inputs["ranked"]
    deferred = inputs["deferred"]

    cards = []
    for raw in inputs["raw_cards"]:
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

        # 第二輪欄位原樣接回
        card.update(carried.get(raw["id"], {}))

        # lesson_ids 是機器沿 covers 算的，保持原樣；找碴席的增刪另存一欄
        dropped = list(card.get("lesson_ids_dropped_by_review") or [])
        added = list(card.get("lesson_ids_added_by_review") or [])
        if dropped or added:
            card["lesson_ids_v2"] = sorted(
                (set(card["lesson_ids"]) - set(dropped)) | set(added)
            )
        else:
            card["lesson_ids_v2"] = list(card["lesson_ids"])

        cards.append(card)

    cards.sort(key=lambda c: (c["feasibility"] != "pass", c.get("rank", 0), c["id"]))
    return cards


def compute_derived(inputs: dict[str, Any], cards: list[dict[str, Any]]) -> dict[str, Any]:
    """算第 1 版統計要用的那幾組集合（全部由機器算，不手寫）。"""
    incident_ids = inputs["incident_ids"]
    raw_cards = inputs["raw_cards"]

    all_stated = {x for c in raw_cards for x in (c.get("lesson_ids") or [])}
    ghosts = {x for x in all_stated if x not in incident_ids}
    covered = {lid for c in cards for lid in c["lesson_ids"]}
    with_debt = [c for c in cards if c["lesson_ids"]]
    pass_cards = [c for c in cards if c["feasibility"] == "pass"]

    # 上游乾淨度：候選規則檔自己引用的 lesson id 有沒有幽靈
    upstream_refs = {x for v in inputs["batch1"].values() for x in v} | {
        x for v in inputs["rules436"].values() for x in v
    }
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

    return {
        "all_stated": all_stated,
        "ghosts": ghosts,
        "covered": covered,
        "with_debt": with_debt,
        "pass_cards": pass_cards,
        "upstream_ghosts": upstream_ghosts,
        "unreachable": unreachable,
        "lost_debt": lost_debt,
    }


def compute_stats_v2(inputs: dict[str, Any], cards: list[dict[str, Any]]) -> dict[str, Any]:
    """第 2 版統計（全部由這裡算，不手寫）。"""
    pass_v2 = [c for c in cards if c.get("feasibility_v2") == "pass"]
    deferred_v2 = [c for c in cards if c.get("feasibility_v2") == "deferred"]
    dropped_v2 = [c for c in cards if c.get("feasibility_v2") == "dropped"]
    blocked_hist: dict[str, list[str]] = {}
    for c in deferred_v2:
        blocked_hist.setdefault(str(c.get("blocked_on")), []).append(c["id"])
    covered_v2 = {lid for c in cards for lid in c["lesson_ids_v2"]}
    with_debt_v2 = [c for c in cards if c["lesson_ids_v2"]]
    return {
        "cards": len(cards),
        "feasibility_v2_pass": len(pass_v2),
        "feasibility_v2_deferred": len(deferred_v2),
        "feasibility_v2_dropped": sorted(c["id"] for c in dropped_v2),
        "pass_with_blood_debt": sum(1 for c in pass_v2 if c["lesson_ids_v2"]),
        "still_leaky": sorted(c["id"] for c in cards if c.get("still_leaky")),
        "narrowed": sorted(c["id"] for c in cards if c.get("narrowed")),
        "recommend_drop": sorted(c["id"] for c in cards if c.get("recommend_drop")),
        "merge_suggested": sorted(c["id"] for c in cards if c.get("merge_suggestion")),
        "cards_with_blood_debt_v2": len(with_debt_v2),
        "lessons_covered_v2": len(covered_v2),
        "lessons_total": len(inputs["incident_ids"]),
        "blocked_on": {k: sorted(v) for k, v in sorted(blocked_hist.items())},
        "shared_parts_used": sorted(
            {p for c in cards for p in (c.get("requires_shared_parts") or [])}
        ),
    }


def build_meta(
    inputs: dict[str, Any],
    cards: list[dict[str, Any]],
    derived: dict[str, Any],
    stats_v2: dict[str, Any],
) -> dict[str, Any]:
    incident_ids = inputs["incident_ids"]
    all_stated = derived["all_stated"]
    ghosts = derived["ghosts"]
    covered = derived["covered"]
    with_debt = derived["with_debt"]
    pass_cards = derived["pass_cards"]

    return {
        "revision": 2,
        "revision_2_說明": "第一版只做「教訓 id 機器重對」。第 2 版對著清空後的空 repo 重判可行性，"
        "並把 12 張通過的卡的「怎麼查／必紅樣本」重寫成 check_idea_v2／fixture_idea_v2，"
        "每張經一個獨立找碴子代理審過（結果在 critic_v2）。舊欄位全部保留，v2 欄位由人寫、腳本只搬運。",
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
            "上游候選規則檔的幽靈教訓 id": len(derived["upstream_ghosts"]),
            "原本寫對但沿 covers 查不到、照規格丟掉的教訓 id（卡數）": len(derived["unreachable"]),
            "丟掉它們之後才變成 0 血債的卡": len(derived["lost_debt"]),
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
        "原本寫對但沿 covers 查不到的": derived["unreachable"],
        "重對後 0 血債的卡": [c["id"] for c in cards if not c["lesson_ids"]],
        "stats_v2": stats_v2,
    }


def write_out(meta: dict[str, Any], cards: list[dict[str, Any]]) -> None:
    OUT.write_text(
        json.dumps({"meta": meta, "cards": cards}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def check_card_shape(c: dict[str, Any], incident_ids: set[str]) -> list[str]:
    """一張卡的欄位自檢（第 1 版欄位）。"""
    bad = []
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
    return bad


def check_card_v2(
    c: dict[str, Any],
    incident_ids: set[str],
    card_ids: set[str],
    wcards_by_id: dict[str, dict[str, Any]],
) -> list[str]:
    """一張卡的第 2 版欄位自檢（可行性、共用零件、併卡對稱、找碴席增刪）。"""
    bad = []
    fv2 = c.get("feasibility_v2")
    if fv2 not in ("pass", "deferred", "dropped"):
        bad.append(f"{c['id']} 的 feasibility_v2 是 {fv2!r}，只准 pass／deferred／dropped")
    if fv2 == "dropped" and "docs/decisions/" not in str(c.get("feasibility_v2_reason", "")):
        bad.append(f"{c['id']} 是 dropped 但 feasibility_v2_reason 沒有指向 docs/decisions/ 的決策紙")
    if fv2 == "deferred":
        if c.get("blocked_on") not in BLOCKED_ON_SLUGS:
            bad.append(f"{c['id']} 是 deferred 但 blocked_on={c.get('blocked_on')!r} 不在 slug 清單裡")
    elif "blocked_on" in c:
        bad.append(f"{c['id']} 是 pass 卻填了 blocked_on")
    for part in c.get("requires_shared_parts") or []:
        if part not in SHARED_PARTS:
            bad.append(f"{c['id']} 的 requires_shared_parts 有未知零件 {part}")
    ms = c.get("merge_suggestion") or {}
    for key in ("merge_into",):
        if key in ms and ms[key] not in card_ids:
            bad.append(f"{c['id']} 的 {key} 指向不存在的卡 {ms[key]}")
    for tgt in ms.get("merge_in") or []:
        if tgt not in card_ids:
            bad.append(f"{c['id']} 的 merge_in 指向不存在的卡 {tgt}")
    # 併卡建議必須雙向對稱：A 說 merge_in=[B]，B 就要說 merge_into=A
    for tgt in ms.get("merge_in") or []:
        other = wcards_by_id.get(tgt, {}).get("merge_suggestion") or {}
        if other.get("merge_into") != c["id"]:
            bad.append(f"{c['id']} 說要併入 {tgt}，但 {tgt} 沒有回填 merge_into={c['id']}")
    into = ms.get("merge_into")
    if into:
        other = wcards_by_id.get(into, {}).get("merge_suggestion") or {}
        if c["id"] not in (other.get("merge_in") or []):
            bad.append(f"{c['id']} 說要併進 {into}，但 {into} 的 merge_in 沒有列它")
    for lid in (c.get("lesson_ids_dropped_by_review") or []):
        if lid not in incident_ids:
            bad.append(f"{c['id']} 要拿掉的 lesson_id {lid} 不在 lessons.json")
        if lid not in c["lesson_ids"]:
            bad.append(f"{c['id']} 要拿掉的 lesson_id {lid} 本來就不在它的 lesson_ids 裡")
    for lid in (c.get("lesson_ids_added_by_review") or []):
        if lid not in incident_ids:
            bad.append(f"{c['id']} 要加上的 lesson_id {lid} 不在 lessons.json")
    for lid in c["lesson_ids_v2"]:
        if lid not in incident_ids:
            bad.append(f"{c['id']} 的 lesson_ids_v2 有 {lid} 不在 lessons.json")
    if c.get("still_leaky") and not c.get("still_leaky_reason"):
        bad.append(f"{c['id']} 標了 still_leaky 卻沒寫原因")
    if c.get("recommend_drop") and not c.get("recommend_drop_reason"):
        bad.append(f"{c['id']} 標了 recommend_drop 卻沒寫原因")
    return bad


def self_check(inputs: dict[str, Any], derived: dict[str, Any]) -> None:
    """重讀寫出去的檔驗一次。任何一條對不上就 exit 1。"""
    incident_ids = inputs["incident_ids"]
    card_ids = inputs["card_ids"]

    written = json.loads(OUT.read_text(encoding="utf-8"))
    wcards = written["cards"]
    if len(wcards) != 38:
        die(f"寫出的卡數 {len(wcards)} != 38")
    if {c["id"] for c in wcards} != card_ids:
        die("寫出的卡 id 集合跟來源不一致")
    bad = []
    wcards_by_id = {c["id"]: c for c in wcards}
    for c in wcards:
        bad += check_card_shape(c, incident_ids)
        bad += check_card_v2(c, incident_ids, card_ids, wcards_by_id)
    if derived["lost_debt"]:
        bad.append(
            f"丟掉「原本寫對但查不到」之後這些卡變成 0 血債，重對弄丟了血債：{derived['lost_debt']}"
        )
    if derived["upstream_ghosts"]:
        bad.append(f"上游候選規則檔竟然有幽靈教訓 id：{derived['upstream_ghosts']}")
    if bad:
        for b in bad:
            print(f"FAIL: {b}", file=sys.stderr)
        sys.exit(1)


def print_summary(meta: dict[str, Any]) -> None:
    s = meta["統計"]
    print(f"OK 寫出 {OUT.relative_to(REPO)}")
    print(f"  卡 {s['卡數']} 張（通過 {s['通過可行性']}／被刷掉 {s['被刷掉']}），covers 去重 {s['covers 去重']} 條")
    print(f"  合併那步自稱引用 {s['合併那步自稱引用的教訓 id 去重']} 個教訓 id，真 {s['其中真的存在']}／幽靈 {s['其中是幽靈']}")
    print(f"  重對後有血債 {s['重對後有血債的卡']}/{s['卡數']}，12 張裡 {s['12 張裡有血債的卡']}/12")
    print(f"  覆蓋教訓 {s['重對後覆蓋的教訓筆數']}/{s['教訓總筆數']}；上游幽靈 {s['上游候選規則檔的幽靈教訓 id']}")
    print(f"  重對後 0 血債：{meta['重對後 0 血債的卡']}")
    print("  自檢通過：所有 lesson id 都對回 lessons.json")
    t = meta["stats_v2"]
    print(f"  第 2 版：對空 repo 重判 pass {t['feasibility_v2_pass']}／暫緩 {t['feasibility_v2_deferred']}，"
          f"pass 裡有血債 {t['pass_with_blood_debt']}")
    print(f"  仍漏 {t['still_leaky']}；收窄 {len(t['narrowed'])} 張；建議砍 {t['recommend_drop']}；"
          f"建議併卡 {t['merge_suggested']}")
    for slug, ids in t["blocked_on"].items():
        print(f"    等 {slug}：{len(ids)} 張 {ids}")


def main() -> None:
    inputs = load_inputs()
    cards = build_cards(inputs, carried_v2_fields())
    derived = compute_derived(inputs, cards)
    meta = build_meta(inputs, cards, derived, compute_stats_v2(inputs, cards))
    write_out(meta, cards)
    self_check(inputs, derived)
    print_summary(meta)


if __name__ == "__main__":
    main()
