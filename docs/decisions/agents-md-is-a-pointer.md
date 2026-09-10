---
title: AGENTS.md 只是指路牌，內容住 CLAUDE.md
date_created: 2026-09-10
date_modified: 2026-09-10
status: accepted
kind: governance
supersedes: ""
superseded_by: ""
summary: "兩份入口檔不再逐字相同：CLAUDE.md 放內容（規矩節由卡生成、手寫段短），AGENTS.md 只放固定幾行指到 CLAUDE.md；指路牌文本登記在卡上，多一字少一字都紅。"
---

# AGENTS.md 只是指路牌，內容住 CLAUDE.md

## 問題

「規矩節生成其餘手寫」那張紙寫「CLAUDE.md 與 AGENTS.md 的規矩節由卡片生成」，入口檔第一版照做成兩份逐字相同的副本。老闆看了：太多太囉嗦，AGENTS.md 直接指到 CLAUDE.md 就好。兩份內容一樣也讓「開工前讀哪裡」有兩個答案。

## 選項

1. 維持兩份逐字相同的副本，由產生器同步。
2. AGENTS.md 只放指路牌：入口檔是 CLAUDE.md、開工先讀它、一行驗證指令；指路牌文本登記在卡 `entry-files-rendered-from-registry` 的 `[settings]`，內容不等於登記文本即紅。

## 決定

選項 2。這張紙補上「規矩節生成其餘手寫」沒說清楚的那半：規矩節只生成在 CLAUDE.md；AGENTS.md 由同一支產生器寫出指路牌。那張紙其餘內容不變。

## 為什麼

v2 的病是兩份入口檔內容不同、其中一份只服務一家工具還沒進版控，「開工前讀哪裡」各自漂。副本解了漂，但把「讀哪裡」變成兩個答案；指路牌只留一個答案，而且文本被卡釘住，漂不了。

## 代價

不讀 CLAUDE.md 的工具只看得到指路牌，看不到規矩節——它本來就該去讀 CLAUDE.md。

## 拍板

老闆，2026-09-10 對話：「agent.md 就直接指引到 claude.md 就好」「要補（決策紙）」。
