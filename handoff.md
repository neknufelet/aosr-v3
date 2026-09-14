# Handoff

給下一個對話。先讀這份，再讀 `DIRECTION.md`。這份分三層：**要老闆回的**（給人看，一題四格）、**座標**（給下一個對話看）、**備查**（不用動）。待辦不列，全在 GitHub issue（見 `docs/decisions/backlog-in-github-issues.md`）。

## 要老闆回的（一題）

沒有。

## 座標（給下一個對話）

現在在哪（2026-09-14，計算策略與有限元素契約都拍完、第七段可以開工）：
- 2026-09-09 清空重來，規矩從 `v2-audit/` 重新長；卡、票、考卷的數字全部看狀態頁（機器現算），這裡不抄。
- 引擎走到：鏡像法一到五段加第十段在主線；第六段有限元素、第八段晚期混響的參考答案與契約在主線，v3 還沒寫。
- 當天拍完的四張紙（都在 `docs/decisions/`）：`fem-numerical-tools-scikit-fem-p2-pydiso.md`（#247 數值工具）、`compute-strategy-three-stages-three-lanes.md`（#248 計算策略總表）、`legacy-answers-three-roles.md`（#251 上一代答案三種角色）、`fem-contract-fenics-frozen-answers.md`（#258，取代 #255 那張：正式路徑 gmsh＋P2 由剛性解析與凍結的 FEniCS 答案擋合併，repo 只放題目與答案，FEniCS 留本機；上一代 flat 那層降為紀錄）。
- **保留的工作樹**：只剩 `~/ghq/aosr-v3-134-loaders`（PR #173 草稿，別刪）；以 `git worktree list` 為準。
- **PR #173** 草稿、雲端紅在三個插值浮點比對，原因未定案，要動先實跑；票 #134／#135 暫停但開著，不准硬關。**PR #168** 跟主線衝突，舊綠不算數。

每一條標動詞，看了就知道要不要動：
- **要開票問老闆（一次一題，照這個順序）**：①晚期混響契約寫死雙精度，換成第三階段 GPU 單精度（開新紙取代）；②段序紙排入材料最佳化階段（repo 裡的「第三段」是高階反射，別撞名）。
- **可以開工**：第七段有限元素 v3。正式路徑 gmsh＋P2；v3 自己的網格定下來後，在本機用 FEniCS 對那張網格重算答案（`~/aosr-v3-work/fenics1/` 有腳本、代跑佇列 `queue-runner.sh`、映像備份），同時量自我收斂寫進證據；第一份答案進 `blueprint/` 時照 #259 立卡。依賴：`pyproject` 加 scikit-fem、pydiso、mkl、gmsh 與編譯依賴（mkl-devel、meson-python、meson、ninja、cython、setuptools_scm），pydiso 設 `no-build-isolation-package`，CI 設 `PKG_CONFIG_PATH=.venv/lib/pkgconfig`；先查這幾個套件的型別資訊（type-guard 沒有逐模組豁免）；考卷進雲端後 15 分鐘上限看 #120。
- **排著的量測（repo 外、還沒派工）**：#253 有吸音時有限元素 P2 在 GPU 單精度夠不夠，材料最佳化進 repo 前量完。
- **先別動**：#249（材料最佳化上 GPU 前要改的六格規矩），等材料最佳化真的開工；GPU 考卷要自架執行機，repo 是公開的，觸發權限要先拍。
- **老闆還沒決定、會卡後面的**：評分／目標函數（三個階段都要，最大）；考卷 2、3（FEniCS、I-Simpa、物理性質）用不用、擋不擋合併；材料連續阻抗還是型錄；每組候選的時間預算（P2 單核一組 29 點約 42 秒，推算）；晚期混響要不要比 6×6 更細；凹形房間是否不做。
- **自動跑的約定（2026-09-13 凌晨拍，仍有效）**：照段序做，只在五種情況停：要開新卡或改卡的邊界／門檻／白名單；某段契約要拿證據訂容差；雲端 verify 同一原因紅兩次；工人與找碴矛盾而驗不出誰對；要動 #173／#134／#135。
- **沒有退回機制**：CPU 只用 pydiso、GPU 用什麼就是什麼；出錯就報錯看原文，不准自動換解法；GPU 不能用時改 CPU 要人決定。
- **量測證據（repo 外）**：`~/aosr-v3-work/` 底下 `255-regular-mesh/`（R1 規則網格相容量測）、`247-papers/`（三張紙的票文與驗收）、`precision-remeasure/`、`compute-strategy/`、`fem-assembly-lab/`、`cudss-lab/`、`spineax-lab/`、`pydiso-ci-lab/`；GPU 套件版本與每次量測數字記在 `~/aosr-v3-work/` 那本版本帳本，每次測都追加。
- 先不做：外部模擬器交叉驗證、前端、脈衝響應。
- **機器驗證的三個缺口（重判後再立候選票，目前沒有票號）**：輸入檔與答案檔正式 schema；物理性質的獨立真值（互易、能量非負、整牆與同阻抗分格）；時紅時綠偵測。
- 共用零件：載入器 `governance/loader.py`、離開碼與輸出層 `governance/exit_codes.py`、後設測試 `tests/test_fixture_runner.py`、主線卡名單 `governance/mainline_cards.py`、CI `.github/workflows/verify.yml`；考卷分兩籃，治理層 `tests/`、引擎 `tests/engine/`。
- 主線 ruleset（合併門檻的設定，id `22615925`）：不准刪、不准改寫歷史、只能走 PR、`verify` 綠且跟上主線才准合。狀態頁由 `governance/status/` 推到機器分支 `status`、掛 GitHub Pages：https://neknufelet.github.io/aosr-v3/

## 備查

- 立卡：一張卡一個 PR，帶卡的 TOML、檢查程式、必紅樣本目錄（含控制樣本），後設測試八回合；加卡要順手把 `green-must-be-real-green.toml` 受影響那一籃的 `collected_floor` 調到實跑收集數。雲端 `verify` 綠了才算。
- 清空前的備份在 `~/aosr-v3-blueprint-2026-09-09/`。
- 做法：分工試驗記在 #245（監督／施工／獨立驗收與停止規則；驗收席位模型全名 `deepseek-v4-flash`）。外部模型走 calling-other-models 並留收據，主對話逐條重跑、終審與合併；實作在 repo **外面**的獨立工作樹；派工指示寫成檔案、只給路徑；工人沙箱看不到 GPU、沒網路，GPU 與要上網的步驟由主對話在沙箱外跑；PR 內文用 `Closes #n` 關票；指示裡寫「不要派別的模型」（工人會自己去叫驗收）。
- 踩過的坑：大包資料寫成檔案只給路徑；數字沒實跑過標「未驗，推測」；工人會編出像真的代號，交叉引用對回原始檔；「有幾筆」跟「每一筆都被比」是兩件事；裁判自己要有控制組；對照組要驗它會紅；GPU 行程一出 ILLEGAL_ADDRESS 整個行程就毀，每個設定各開一個行程；結果檔每算完一列就寫；驗收也會讀錯（#254 第一輪三條有兩條是它誤讀），它說的每一條都回原文對。
- 政策：既有相容性合約維持；改公開呼叫契約算行為、不算結構，要改就當一張決定做。

## 這份怎麼寫（下一個對話照抄）

- 第一節永遠只有一題、四格（發生什麼事／結果／我建議／你回什麼），每格兩三句，最多兩個選項並說選哪個；沒題就寫「沒有」。那一題同時要有一張標籤 `decision` 的票，這裡只抄票號。
- 座標每一條標動詞；給機器看的代號只出現在座標與備查；數得出來的數字指去狀態頁；每個英文名詞同一句配中文；全篇日常講法。
- 已落地的事併成一行；超過 60 行就砍，舊事沉進決策紙；repo 外的路徑只寫到目錄、不寫帶副檔名的檔案（refs 卡會咬）；不寫 commit 與雲端 run 編號（identity 卡會咬）。
