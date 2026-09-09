---
title: v3 自建規矩本，跑順再上提
date_created: 2026-09-09
date_modified: 2026-09-09
status: accepted
kind: governance
supersedes: ""
superseded_by: ""
summary: "v3 在 governance/ 自建規矩本從 3.0.0 起，不動舊 BASELINE 2.1.1；skills 留在 AI_TOOLS 不複製；跑順後 governance/ 搬到 standards 成新版；各專案釘版本、手動升。"
---

# v3 自建規矩本，跑順再上提

## 問題

v3 的新做法與共用的 BASELINE、PLATFORM、專案範本、九個流程 skill 不同，現在改總本還是先在 v3 試。

## 選項

1. 現在改總本升 3.0.0，四個專案一起跟。
2. v3 先自建、自用、自測；跑順再上提；其他專案釘舊版、想升再升。

## 決定

選項 2。v3 repo 開 `governance/`，只放規矩卡、平台慣例、範本三區，內容全新，版號從 3.0.0 起。舊 BASELINE 2.1.1 與其他專案不動。
九個 skill 留在 AI_TOOLS 當上層唯一來源，v3 直接用、不複製、不放副本進 repo；要改就改 AI_TOOLS。派工工具 calling-other-models 照用；其他流程 skill v3 先不呼叫，以分工五席取代，但不搬不動。
跑順後整個資料夾搬到 standards repo 成為新版；每個專案寫明自己釘哪一版，升版是一個 PR，不自動跟。

## 為什麼

現在連一支檢查都沒寫，改總本是紙上談兵。總本自己沒有雲端檢查；同步靠各專案自抄偵測程式，抄三份漂三份。釘版本、手動升讓總本可以一直更新而專案不會被動變。

## 代價

試驗期 v3 與其他專案規矩不一致；上提時要做一次遷移。skill 不在 repo 內，換機器要先裝 AI_TOOLS。

## 拍板

老闆，2026-09-09 對話。

清空時被清掉，2026-09-10 照原文補回（檔名改英文）。
