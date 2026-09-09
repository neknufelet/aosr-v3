---
title: 清空重來，規矩從 v2 事故資料重新長
date_created: 2026-09-09
date_modified: 2026-09-09
status: accepted
kind: governance
supersedes: ""
superseded_by: ""
summary: "2026-09-09 把 v3 治理層整個清空，只留 v2 事故資料與規則藍圖；規矩從 v2-audit/ 重新長，不遷就舊半成品。"
---

# 清空重來，規矩從 v2 事故資料重新長

## 問題

清空前的 repo 累積了 13 張規矩卡、15 支檢查、8 張決策紙與一堆半成品，每一份都要問一次「這份怎麼辦」。工人為了保留舊東西不斷遷就，新規矩的形狀被舊半成品牽著走。

## 選項

1. 逐份整理舊半成品，能留的留。
2. 全部清掉，只留 v2 的事故資料（`v2-audit/`）與從設計報告抽出的規則藍圖（`blueprint/`），規矩重新長。

## 決定

選項 2。清空後 repo 只剩 `v2-audit/`、`blueprint/`、`DIRECTION.md`、`handoff.md`。之後每一張規矩卡都要對回 `v2-audit/lessons.json` 的某筆事故，或者明白標示「沒有 v2 血債」。

## 為什麼

老闆原話：「一點點就東西，然後 agent 為了要保留一直遷就被影響。」半成品的維護成本已經超過它的價值。

## 代價

舊的 13 張卡連同檢查程式一起消失，只剩規格描述留在 `blueprint/convergence-result.json` 與 repo 外的備份（`~/aosr-v3-blueprint-2026-09-09/`，含完整 git 歷史 bundle）。

## 拍板

老闆，2026-09-09 對話。
