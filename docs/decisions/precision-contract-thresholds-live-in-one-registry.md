---
title: 精度契約的門檻只住一份登記簿，改值要新紙，公式與案例由變異考卷守
date_created: 2026-09-15
date_modified: 2026-09-15
status: accepted
kind: governance
supersedes: ""
superseded_by: ""
summary: "九個精度契約的門檻常數從產品程式搬到 blueprint/precision_contracts.toml 一份登記簿，每條指名設定它的決策紙；產品程式不准再定義契約常數；改值要同一支合併請求帶新紙；每個契約配一支變異考卷守比對公式與受驗案例。型錄吸音率紙第 7 條「常數住在換算模組」只換住處、物理不動。"
---

# 精度契約的門檻只住一份登記簿，改值要新紙，公式與案例由變異考卷守

## 問題

2026-09-15 治理層盤點查到：九個精度契約的門檻常數全部住在產品程式裡（晚期能量、晚期衰減 T20 與 T30 性質、剛性模態、FEniCS 凍結答案、
直達能量、反射能量地板、反射係數、型錄吸音率性質），考卷再從產品 import 進來當尺；比對公式（相對差怎麼算、分母取誰）與受驗案例也住產品裡。
規矩卡 `thresholds-live-only-in-registry` 只掃治理層的程式與卡，掃不到 `src/`。工人在改物理的同一支合併請求裡把門檻放寬、或把分母換掉，
今天全部檢查照樣綠。這是上一代事故 `builder-grades-own-work-edits-ruler`（施工者能同時改 production 與尺）與
`governance-numbers-changed-on-a-whim`（治理數字心證改）的形狀。

`docs/decisions/catalog-absorption-random-incidence-paris-inversion.md` 第 7 條拍過「常數住在換算模組」，其他八個是跟著習慣放的。

## 選項

- **A：搬出去。** 門檻只住一份登記簿 `blueprint/precision_contracts.toml`，每條指名設定它的決策紙；產品程式不准再定義契約常數；
  改值要同一支合併請求帶新紙；每個契約配一支變異考卷（期望值推到容差外一點點，考卷必須紅）守比對公式與受驗案例；
  之後由一張規矩卡守這三層。
- **B：維持住在產品程式裡**，把「施工者改尺沒有卡會紅」寫進卡面「刻意沒管」，靠驗收席看差異。

## 決定

採選項 A。

1. 登記簿 `blueprint/precision_contracts.toml`，一條一節：名字、值、單位（相對差、絕對差或 ULP）、設定它的決策紙檔名、
   對到哪份答案檔或哪類真值、變異考卷的完整測試 id。放 `blueprint/` 是為了跟它管的答案檔住一起、又在 `src/` 外面。
2. 九條的來源紙（值以登記簿為準，這裡只記出處）：晚期能量對上一代見 `precision-contract-art-late-energy-exact-solve.md`；
   晚期衰減 T20 對上一代見 `precision-contract-late-decay-t20-2pow20-corrected.md`；晚期衰減 T30 性質見
   `late-decay-t30-property-tolerance-2pow30.md`；剛性模態對解析解與 v3 對 FEniCS 凍結答案見 `fem-contract-fenics-frozen-answers.md`；
   直達能量見 `precision-contract-direct-energy-2pow20.md`；反射能量地板見 `precision-contract-totals-root-sum-square.md`；
   反射係數見 `precision-contract-amplitude-phase-scaled.md`；型錄吸音率性質見 `catalog-absorption-random-incidence-paris-inversion.md`。
3. 產品程式裡的九個常數刪掉；比對函式把容差當參數收；命令列的比對模式從登記簿讀（路徑必給，不設預設，照
   `config-loaders-keep-path-required.md`）；考卷改從登記簿讀，不准再從 `aosr` 拿容差；`blueprint/` 底下的獨立檢查程式也不准自己再抄一份。
4. 搬家那一支合併請求數值與判決結果不准變：全套考卷通過數不變、每份答案檔比對的最大差不變。
5. 每個契約配一支變異考卷，登記簿指名它的測試 id；規矩卡另立（候選票 #312），三顆牙：產品程式不准有契約常數的形狀、
   登記簿改值要同範圍新增指名的紙且紙上出現契約名與新值、指名的變異考卷要在這一跑的 junit 收據裡跑過且過。
6. 型錄吸音率紙第 7 條只換住處，Paris 反推與 2^-30 的物理與數值依據都不動，那張紙不取代。
7. 搬家與立卡兩支合併請求之間，不准穿插別的物理契約修改。

## 為什麼

- 尺有三層：容差值、比對公式、受驗案例。只搬常數只守第一層；容差不動、把相對差的分母換掉或乘一百，一樣放過該紅的結果。
  變異考卷把期望值推到容差外一點點，公式被改鬆它當場綠掉，這是機器看得到的訊號。
- 「登記簿改了要同範圍新增決策紙」若只查有沒有多一份紙，一份無關的紙就過得去；所以要求紙上出現契約名與新值字串。
- 獨立真值本身今天是乾淨的（Paris 積分用 scipy 另算、剛性模態解在 `blueprint/reference_fem_check.py` 另寫、不 import 引擎），
  髒的是尺，所以只搬尺、不動真值。
- 老闆選 A：治理層不能宣稱「尺跟施工者分開」而讓尺住在施工者能改的檔裡。

## 代價

- 容差改寫成函式內的區域變數繞得過名字掃描；比對公式改鬆卡本身看不到，靠變異考卷；變異考卷一起被改鬆靠驗收席看差異。
  規矩卡面要照實寫這三句，這張紙不宣稱「施工者改尺不可能」。
- 九個契約各加一支變異考卷，`tests/engine` 那一籃的最少題數要跟著調。
- 命令列比對模式多一個必給的登記簿路徑參數。
- 登記簿的路徑寫死在這張紙與卡上，搬檔要走新紙。

## 拍板

2026-09-15，老闆看過治理層盤點第二版後回 A，助理建議 A。證據：盤點頁的九條門檻住處表（逐檔實查）；候選票 #312。
