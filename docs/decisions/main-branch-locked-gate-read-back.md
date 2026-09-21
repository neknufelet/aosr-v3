---
title: 主線鎖死，合併門口由機器回讀平台設定
date_created: 2026-09-21
date_modified: 2026-09-21
status: accepted
kind: governance
supersedes: "main-branch-locked.md, merge-gate-check-may-read-github.md"
superseded_by: ""
summary: "公開版本庫的主線只收全綠 PR、禁刪與改寫且無人可繞；merge-gate-read-back 只讀回查平台規則組，放行具名且有到期日。"
---

# 主線鎖死，合併門口由機器回讀平台設定

## 問題

第一張舊紙決定主線必須鎖到無人可繞；第二張補上可執行的回讀方式，避免版控裡只寫期望、GitHub 上的 ruleset（規則組）卻漂掉。這兩張是同一扇合併門的政策與驗證方式。

## 選項

逐張的完整選項在 `docs/archive/` 的舊紙。鎖主線紙比較不鎖、鎖但管理員可繞、完全鎖死；回讀紙比較讓 `merge-gate-read-back` 具名、只讀地上網查真設定，與不上網、只比對版控裡的一份期望檔。

## 決定

**主線鎖死（2026-09-09，拍板在 2026-09-08）。** 採第三個選項。repo（版本庫）公開；GitHub rulesets 設成主線只收 PR（合併請求）、必要檢查全綠才能合併、禁止 force push（強制改寫歷史），管理員 bypass（繞過名單）關閉。改公開是第一個動作。

**合併門口檢查准回讀 GitHub ruleset（2026-09-10，issue #64）。** 這是 issue #9 那組最後一張，採第一個選項。`merge-gate-read-back` 在 CI（持續整合）裡可用 `gh`（GitHub 命令列工具）唯讀查主線規則組，把 repo、ruleset id、規則、必要檢查與空的 bypass 名單逐格對照卡上 `[settings]`；讀不到、工具不在、未登入或平台無回應時回 2，不回 0。2026-09-10 已先驗過公開 repo 的規則組與主線規則兩個讀取端點，不帶 token（存取憑證）也回 HTTP 200，內容完整。

放行寫在 `governance/rules/merge-gate-read-back.toml` 的 `[[settings.allow]]`，含 `reason` 與 `expires`，由 `exemptions-need-expiry` 守；老闆拍板「90 天」並要求和第一批放行同日審，現行登記到期日是 2026-12-08。缺席或過期時檢查回 2 且不上網。檢查只打讀取端點，不改平台設定；使用 CI 自帶 `GITHUB_TOKEN`，權限是 `contents: read`。期望值全住在卡的 `[settings]`，改平台設定也要開 PR 改卡。

准上網不是通則。2026-09-10 後，具名例外從一支變兩支：本支與 `issues-closed-only-by-merged-pr`；第二支的權限、到期日與三層分工見 `docs/decisions/issues-closed-only-by-merged-pr-hand-closed-blocks.md`。規則讀得到就比；管理員 bypass 名單若因平台只給管理員而在雲端看不到，輸出必須明說「這一格這一跑沒守」，不得假裝通過。

## 為什麼

上一代有 796 個提交從未推出、528 個只在一顆硬碟、旁支領先主線 228 個沒有合回，還曾在推出後改寫歷史，根因是主線沒有真正鎖住。私有 repo 的分支保護當時要付費，實測回 403；公開後才能免費使用。事故 `enforcer-declared-but-never-installed` 與 `amended-a-pushed-commit` 又證明版控裡的期望和伺服器實際設定會分開漂；只比對期望檔守不到出事的那一端，所以必須讀回平台真值。

## 代價

- 鎖主線紙的代價：repo 必須公開；緊急繞過前要先寫決策紙解鎖，解鎖期間不得宣稱主線受保護。
- 回讀紙的代價：檢查與後設測試六回合依賴網路與 token，本機斷網時第一回合會因回 2 而紅；`verify` 工作要把 CI token 交給 `gh`。改 GitHub 設定者必須同步開 PR 改卡。必紅樣本用「卡上期望與伺服器不一致」，控制樣本用「登記的 ruleset id 不存在」，兩者都要上網。雲端看不到管理員 bypass 時只有老闆用管理員帳號本機跑的那次能守到。到期日要重新審，續期時卡與各樣本樹的放行都要一起改。
- 合併帶來的代價：鎖門政策與回讀方式在機器眼裡成為同一題；之後其中一題要翻案，得取代整張並重帶另一題仍有效的決定。

## 拍板

- 主線鎖死：老闆，2026-09-08 對話；舊紙未記原話與票號。清空時被清掉，2026-09-10 照原文補回。
- 回讀平台：老闆，2026-09-10 對話，原話「選 1，90 天，和之前好像有另一個也是 90 天，一起審」，issue #64。
- 兩張合成一張：老闆 2026-09-21 對話「我想要B」（票 #412），以及同日「決策紙收得太少」的要求。
