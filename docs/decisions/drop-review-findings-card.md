---
title: 砍掉「審查發現可追溯」卡，改開 GitHub 的審查意見必須處理完才能合
date_created: 2026-09-10
date_modified: 2026-09-10
status: accepted
kind: governance
supersedes: ""
superseded_by: ""
summary: "review-findings-traceable 不立卡：審查紀錄照 docs 三類決策紙走 issue／PR 不進 repo，卡沒有對象；改在主線 ruleset 開 required_review_thread_resolution，審查意見沒處理完不准合。"
---

# 砍掉「審查發現可追溯」卡，改開 GitHub 的審查意見必須處理完才能合

## 問題

候選卡 `review-findings-traceable`（審查發現要有編號、處置與重現收據）原本等 `docs/` 結構拍板。結構拍了：`docs/` 只准 decisions／design／cairn 三類，審查紀錄是過程紀錄，走 issue 與 PR，不進 repo。卡要掃的 `docs/reviews/` 依決策不會存在。

## 選項

1. 砍掉這張卡，改成慣例：PR 上的審查意見逐條處理。
2. 重寫成看 GitHub 的卡：合併前每條審查意見都要 resolved——要上網讀 GitHub，違反「檢查不上網」；同一件事 GitHub 自己有設定可做。
3. 選項 2 的設定版：主線 ruleset（GitHub 上鎖主線的設定）開 `required_review_thread_resolution`（審查意見的討論串沒標「已處理」不准合），不寫程式；卡砍掉，資料檔記理由。

## 決定

選項 3。2026-09-10 已在 ruleset `22615925` 的 `pull_request` 規則裡打開。

## 為什麼

v2 事故 `review-findings-not-tracked-redone-twice`：審查沒編號、沒處置、沒進版控，同一個問題重修兩次。病根是「處置沒有落點」，GitHub 的討論串 resolve 就是落點，而且由平台強制，不需要自己再寫一支會上網的檢查。

## 代價

只管 PR 上的審查意見；在對話裡口頭講的審查不算數——那本來就該落成 PR 意見或 issue（「fork 結論先開 issue」那張紙）。

## 拍板

老闆，2026-09-10 對話：「照你的意見」（issue #52）。
