---
title: 精度契約卡的第三顆牙：「跑過且過」由收集考卷加綠卡合起來守，卡只守存在
date_created: 2026-09-15
date_modified: 2026-09-15
status: accepted
kind: governance
supersedes: ""
superseded_by: ""
summary: "補強 precision-contract-thresholds-live-in-one-registry 第 5 條、不取代：規矩卡 precision-contracts-live-in-one-registry 的第三顆牙只判登記簿每條指名的紙找得到、變異考卷的函式真的定義了（用程式結構找）；「指名的變異考卷跑過且過」由考卷 test_every_registered_mutant_node_collects（每條指名的節點收集得到）加綠卡（收據沒有 failures、沒有 skip）合起來守，不由卡去讀 junit 收據。"
---

# 精度契約卡的第三顆牙：「跑過且過」由收集考卷加綠卡合起來守，卡只守存在

## 問題

`precision-contract-thresholds-live-in-one-registry.md` 第 5 條寫規矩卡的第三顆牙是「指名的變異考卷要在這一跑的 junit 收據裡跑過且過」。
立卡時發現卡自己去讀 junit 收據做不到：後設測試第 1 回（乾淨樹）在 pytest 裡面跑，收據要等整份 pytest 跑完才寫出來，
卡在那一回讀不到收據，回 2 就讓乾淨樹紅、回 0 就是沒收據也放行。綠卡走的是另一條特例（宣告 `[junit]`，第 1 回改判「沒收據必須回 1」），
那條特例為收據卡設計，登記簿卡套上去得帶一整組籃子表。本紙補強第 5 條，不取代原紙。

## 選項

- **A：卡只守存在，「跑過且過」由兩件既有的機器合起來守。** 卡判每條指名的紙找得到、變異考卷的函式真的定義了（用程式結構找，註解裡寫一行不算）；
  考卷 `tests/engine/test_precision_contracts.py::test_every_registered_mutant_node_collects` 證明每條指名的節點 pytest 收集得到；
  綠卡證明整份收據沒有 failures、沒有 skip。三件合起來：指名的考卷存在、被收集、整份沒紅沒跳，就是跑過且過。
- **B：卡讀 junit 收據。** 卡宣告 `[junit]` 走綠卡那條特例，verify 那一步移到 pytest 之後。代價是登記簿卡帶一整組籃子表，兩張卡讀同一份收據各判一次。

## 決定

採選項 A。

1. 規矩卡 `precision-contracts-live-in-one-registry` 第三顆牙：登記簿每條的 `decision_paper` 在決策目錄或封存區找得到；`mutant_test` 寫成「檔案::測試函式」，
   那支檔在版控裡而且用程式結構（ast）找得到同名的測試函式。
2. 「跑過且過」的鏈：存在（這張卡）→ 收集得到（`test_every_registered_mutant_node_collects`）→ 整份沒紅沒跳（綠卡）。
   把真的變異考卷改名成不會被收集的名字、留一行註解冒充，第一環用程式結構找不到、第二環收集不到，都紅。
3. 原紙第 5 條的「在這一跑的 junit 收據裡跑過且過」照本紙的鏈讀，不由卡去讀收據。

## 為什麼

- 卡在 pytest 裡面被叫時收據還沒寫出來，讀收據那條路在乾淨樹上不是紅就是假綠。
- 收集考卷已經在（#320 第一刀帶進來的），綠卡已經在；再讓卡讀一次收據是兩張卡各判一次同一份檔。
- 存在用程式結構找而不比字串：找碴席第一輪示範過「註解裡寫一行 def」可以騙過字串比對。

## 代價

- 「跑過且過」不在一張卡上，散在三個地方；讀卡面的人要知道另外兩環在哪（卡面與這張紙都寫了）。
- 收集考卷本身若被刪掉，第二環就沒了；綠卡的收集數地板會掉一題，但抓不出是哪一題。這一格靠 PR 審。

## 拍板

2026-09-15 深夜，助理在立卡時發現原紙第 5 條做不到、改成本紙的做法；老闆同日拍過原紙（A）與立卡順序，本紙是做法層的補強，不動老闆拍的邊界。找碴席第一輪點名「卡沒照第 5 條做」，本紙是回答。
