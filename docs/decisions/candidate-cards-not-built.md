---
title: 不立的候選卡：五張砍掉、身分字串那張收窄改寫（四張舊紙合成一張）
date_created: 2026-09-21
date_modified: 2026-09-21
status: accepted
kind: governance
supersedes: "drop-handoff-replayable-card.md, drop-review-findings-card.md, drop-text-format-consistent-card.md, drop-two-ticket-cards-keep-identity-strings.md"
superseded_by: ""
summary: "五張候選卡不立（文字格式一致、交接單可重播、審查發現可追溯、兩張票務卡），各自的牙歸哪裡記在這裡；identity-strings-generated 收窄改寫後立卡；單字母標籤改為命名慣例。決定本身沒有變，只是四張紙合成一張騰位置。"
---

# 不立的候選卡：五張砍掉、身分字串那張收窄改寫

## 問題

2026-09-09 到 09-10 老闆分四次拍了「哪幾張候選卡不立」，一次一張紙。2026-09-21 決策紙份數滿了（票 #412），老闆拍板讓一張新紙合併取代好幾張舊紙（`docs/decisions/one-decision-one-paper-may-merge-many.md`）。這四張同一族：都在回答「這張候選卡有沒有牙、沒有的話它的牙歸誰」。這一張把四張仍然有效的決定搬過來，句子重新組過、**決定的內容沒有改**；四張舊紙在封存區，當時的原文、選項與完整理由看那裡。

## 選項

四張各有各的選項（整張砍掉、砍到只剩一條仍立卡、併進既有的卡當一條窄規則、改用 GitHub 平台的設定、收窄改寫後立卡），逐張的選項在封存區的舊紙，這裡不重抄。

## 決定

**一、`text-format-consistent`（文字格式一致）不立卡。**（2026-09-09）換行一律 LF、不准繁簡混用、不准字母加數字當代號，這三條在 v2 的 66 筆事故裡零血債，第三條還會誤咬正常文件（lint 規則編號、聲學指標 T20 這一類）。單字母標籤只剩慣例價值，寫成命名慣例、不當會回紅的檢查：**標籤、選項、席位不用單字母（A／B／C、E／F／G），用能自解釋的短詞。** 這一句慣例就住在這張紙，之後有寫作規範再搬。換行符號若真的出問題，用 `.gitattributes` 一行設定處理，不佔卡。

**二、`handoff-must-be-replayable`（交接單可重播）不立卡。**（2026-09-10，issue #54）它的三顆牙：「引用得到真的產物」已由 `refs-and-links-resolve` 在守（路徑要解析得到、絕對路徑與家目錄一律紅、備份路徑具名放行）；「不准長成帳本」已由 `status-page-computed-not-typed` 在守（`handoff.md` 放行但限 60 行、不准像待辦檔）；「run id 要解析得到」當時要上網、違反檢查不上網（後來由下面第四條的卡用本機收據鏡像守住）。通則：兩支檢查掃同一個對象會互相遮蔽（v2 事故 `guard-teeth-shadow-each-other`），沒有新牙的卡不立——立卡判準（`docs/decisions/card-admission-threshold.md`）就是從這一條歸納出來的。

**三、`review-findings-traceable`（審查發現可追溯）不立卡。**（2026-09-10，issue #52）審查紀錄是過程紀錄，走 issue 與 PR、不進 repo，卡沒有對象可掃。改用平台的設定：主線 ruleset `22615925` 的 `pull_request` 規則打開 `required_review_thread_resolution`（審查意見的討論串沒標「已處理」不准合），2026-09-10 已開。病根是 v2 事故 `review-findings-not-tracked-redone-twice`：處置沒有落點；GitHub 的討論串 resolve 就是落點，由平台強制。只管 PR 上的審查意見；對話裡口頭講的審查要落成 PR 意見或 issue 才算數（`docs/decisions/fork-results-return-via-issue.md`）。

**四、兩張票務卡不立，`identity-strings-generated` 收窄改寫後立卡。**（2026-09-10，issue #11，老闆原話「留一張改寫」）
- `ticket-open-preflight`（一票一棵乾淨工作樹、身分先驗）與 `ticket-stage-gate`（沒收據不准關票、停止條件）標 dropped：對象（票務工具、工作樹、關票）全在 repo 外，卡守的是空氣——真樹永遠回綠，卡面看起來有人守。
- `identity-strings-generated` 立卡，人話是「版控裡的文件出現的 commit sha 與雲端 run id 必須解析得到」：完整 sha 與夠長的短 sha 用本機物件庫查（不上網），run id 在收據鏡像裡找（`run.run_id` 或 `written_by.run_id`）；網址與 UUID 裡的十六進位段不算 sha。清空前舊歷史的 sha 解析不到是對的，那種引用改寫成「清空前的歷史」。血債是 `hand-copied-identity-strings-drift`。短 sha 誤咬時改文件寫法或在卡上具名放行，不放寬判準。立卡當天文件裡一個真的 sha、一個 run id 都沒有，卡先立起來等對象出現——沒有對象時的綠是「掃了文件、沒有 sha」，不是「有 sha 而且都對」。
- 停止條件那顆牙歸 delivery skill（派工流程那套工具）：issue #59 加一條「現用的 v4 帳本要有停止條件的測試」，由 skill 自己的測試守；「關掉的票對不對得到綠收據」歸狀態頁當一格，是給人看的紅字、不是擋合併的卡。
- 藍圖 `blueprint/cards-38.json` 裡這幾張卡的 feasibility_v2_reason 指到這張紙。

## 為什麼

一張卡要有牙、要有對象。沒東西可咬的卡立了就是 v2「規矩多但沒人守」的重演；對象不在掃描面上的卡連必紅樣本都只能對假的寫。各張的完整理由在封存區的舊紙。

四張合成一張的理由：份數上限的用意是逼舊事合併下沉，不是到了就加（票 #412）。

## 代價

- 照舊的：「交接單引用的雲端那一跑還在不在」沒人守，要看狀態頁；PR 描述與 issue 留言裡手抄的 id 在 repo 外、沒人守；兩張票務卡的牙分散到 repo 外與不擋合併的狀態頁，不是這個 repo 的機器在守。
- 合併帶來的：這五張卡在機器眼裡變成同一題，之後其中任何一張要翻案（例如某張候選卡重新要立），就得開一張新紙取代這整張、把其餘仍有效的決定再搬一次。

## 拍板

老闆：2026-09-09 對話「砍」（文字格式）；2026-09-10 對話「照你的意見」（issue #54、#52）；2026-09-10「留一張改寫」（issue #11）。四張合成一張：老闆 2026-09-21 對話「我想要B」（票 #412）。
