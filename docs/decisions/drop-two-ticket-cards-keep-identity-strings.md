---
title: 砍掉兩張票務卡，身分字串那張改寫成「文件裡的 sha 與 run id 必須解析得到」
date_created: 2026-09-10
date_modified: 2026-09-10
status: accepted
kind: governance
supersedes: ""
superseded_by: ""
summary: "ticket-open-preflight 與 ticket-stage-gate 不立卡：對象（票務工具、工作樹、關票）全在 repo 外，改寫成看收據也沒有東西可看。identity-strings-generated 收窄改寫：版控裡的文件出現 commit sha 與雲端 run id，必須在本機物件庫與收據鏡像裡解析得到。兩顆值得留的牙各歸 delivery skill（停止條件）與狀態頁（關掉的票對不對得到綠收據）。"
---

# 砍掉兩張票務卡，身分字串那張改寫成「文件裡的 sha 與 run id 必須解析得到」

## 問題

issue #11 那一組三張卡在等「派工／票務工具搬進 repo」。老闆 2026-09-10 拍板工具留在 AI_TOOLS 不搬（決策紙「派工工具留在 AI_TOOLS」那一題記在 issue #11 的留言），所以這三張卡的對象永遠不在 repo 樹裡。收據那條線做完之後（決策紙「收據分兩層」、三張收據卡），要判：改寫成看雲端收據，還是承認由 skill 自己的測試守、把卡砍掉。

三張卡的對象在哪：
- ticket-open-preflight（一票一棵乾淨工作樹、身分先驗、重票機器判級）：工人本機的做法，雲端收據看不到工作樹。
- ticket-stage-gate（沒收據不准關票、CI 沒綠不准刪工作樹、停止條件）：v3 的「票」是 GitHub issue，關票發生在 GitHub 上，要判就得上網；停止條件住在 delivery skill 的帳本。
- identity-strings-generated（手抄的 sha／run id 會漂，每次在最貴的終審才被抓）：手抄現場是 PR 描述與 issue 留言（repo 外），但版控裡的文件（決策紙、handoff、方向檔）也會抄 sha 與 run id，那一半在 repo 裡。

## 選項

1. 三張全砍。停止條件記到 #59（delivery skill 的調整），「關掉的票對不對得到綠收據」交給狀態頁當一格。
2. 砍兩張，identity-strings-generated 收窄改寫：版控裡的文件出現的 commit sha 必須在本機物件庫解析得到、雲端 run id 必須在收據鏡像裡找得到。今天對象是零（文件裡沒有一個真 sha、沒有一個 run id），卡先立起來等對象出現。

## 決定

選項 2。老闆原話（2026-09-10）：「留一張改寫」。

具體：
- ticket-open-preflight、ticket-stage-gate 標 dropped，藍圖 `cards-38.json` 的 feasibility_v2_reason 指到這張紙。
- identity-strings-generated 立卡，人話改成「版控裡的文件出現的 commit sha 與雲端 run id 必須解析得到」：完整 sha 與夠長的短 sha 用本機物件庫查（不上網），run id 在收據鏡像裡找（`run.run_id` 或 `written_by.run_id`）。網址與 UUID 裡的十六進位段不算 sha（實測：決策紙裡的示範頁網址就含這種段）。清空前舊歷史的 sha 不會解析得到，那是對的——那些引用該改寫成「清空前的歷史」而不是留一個死 sha。
- 停止條件那顆牙：#59 加一條「現用的 v4 帳本要有停止條件的測試」（v2、v3 那幾版有，v4 沒看到）。
- 「關掉的票對不對得到綠收據」那顆牙：狀態頁加一格，另開 issue；那是給人看的紅字，不是擋合併的卡。

## 為什麼

一張卡的對象不在掃描面上，它就是守著空氣：真樹永遠回綠，卡面看起來有人守。這個 repo 的每一張卡都要有必紅樣本與血債，票務兩張連樣本都只能對 mock 寫。identity-strings-generated 不同：它的血債 hand-copied-identity-strings-drift（SHA 抄錯、diff 數字抄錯，每次在最貴的終審站才被抓）在版控文件裡有一半的現場，而且檢查不用上網——物件庫在本機、收據已經鏡到本機。

## 代價

- 兩張卡的牙分散到 repo 外（skill 的測試）與非擋合併的狀態頁，都不是這個 repo 的機器在守；要靠 #59 與狀態頁那張 issue 真的做掉。
- identity-strings-generated 今天零對象，第一回合是「掃了文件、沒有 sha」的綠，不是「有 sha 而且都對」的綠。等文件開始引用 run id 與 sha，它才真的在咬。
- 短 sha 的判準是「夠長、同時有數字與字母、不在網址與 UUID 裡」，仍可能把某些十六進位字串誤當 sha；誤咬時改文件寫法或在卡上具名放行，不放寬判準。
- PR 描述與 issue 留言裡手抄的 id 還是沒人守——那半在 repo 外，照實記。

## 拍板

老闆，2026-09-10 對話，記在 issue #11。
