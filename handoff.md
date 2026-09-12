# Handoff

給下一個對話。先讀這份，再讀 `DIRECTION.md`。這份分三層：**要老闆回的**（給人看，一題四格）、**座標**（給下一個對話看）、**備查**（不用動）。待辦不列，全在 GitHub issue（見 `docs/decisions/backlog-in-github-issues.md`）。

## 要老闆回的（一題）

沒有。第五段拍了 A（決策紙 `docs/decisions/stage-five-ism-totals-and-energy.md`）；下一題會是總壓力與能量的契約，等第 1 段的證據量完再立 `decision` 票。

## 座標（給下一個對話）

現在在哪（2026-09-12 收工）：
- 2026-09-09 清空重來，規矩從 `v2-audit/` 重新長；決定與理由在 `docs/decisions/`。卡有幾張、藍圖 38 張各自的去向、票開著幾張，全部看狀態頁（機器現算），這裡不抄。
- **停的位置是主線**：git log 最新一筆（新卡 answer-files-carry-provenance 合了）。
- **PR #173（分支 `feat/134-materials-loaders`）已轉草稿、還沒合、尚未通過驗收**：雲端 `verify` 紅在三個插值浮點數比對案例。**原因還沒定案**——不准把它寫成某個猜測（硬體、環境、精度都只是待驗的假設），要動先實跑；程式與證據保留，不當成已完成成果。
- 票 #134／#135（材料、scoring（計分）那兩塊）**暫停但開著**，不准硬關——新單位下它們的形狀要重寫，那是以後的事。

每一條標動詞，看了就知道要不要動：
- 要老闆回：沒有（見第一節）。
- **已做完**：第一段參考答案（#175）；幾何契約逐位元；第二段一次反射（#181）；第三段二、三階（#187）；第四段振幅與材料吸收（契約 `precision-contract-amplitude-phase-scaled`；#194）；規矩層：新卡 `answer-files-carry-provenance`（答案檔要帶出身，#199）、`assertions-not-pinned-to-counts` 與 `refs-and-links-resolve` 各補一顆牙（#200、#201）；狀態頁改給人看；src/aosr 裝進 uv 環境；第五段拍板（決策紙 `stage-five-ism-totals-and-energy`）。命令列：`uv run python -m aosr.physics.room_paths <input.json> --compare blueprint/reference_amplitude_varied.json`。外部證據在 `~/aosr-v3-work/<票號>/evidence/`。
- **進行中**：第五段票 #206（總壓力與能量）。順序：第 1 段量證據（v3 雙精度加總 vs 上一代 totals、候選界線用到幾成）→ 開 `decision` 票拍契約 → 第 2 段 v3 與考卷。`room_paths.py` 快頂到 style-guard 的檔案行數上限，比對零件要先搬去同層新模組。派工單的教訓：控制組要寫明「用 monkeypatch 讓判契約函式真的吃到壞界線」；驗收指令不要 `| tail -1; echo $?`；工人常逾時不交回報，驗收一律主對話自己重跑。
- 先不做：最佳化、外部模擬器交叉驗證、前端。
- 共用零件：載入器 `governance/loader.py`、離開碼與輸出層 `governance/exit_codes.py`、後設測試 `tests/test_fixture_runner.py`、CI `.github/workflows/verify.yml`（一個 `verify` 工作跑全部檢查＋pytest＋ruff；mypy 由 `type-guard` 那張卡在 pytest 裡叫）；考卷分兩個籃子，治理層住 `tests/`、引擎住 `tests/engine/`。
- 主線 ruleset（合併門檻的設定，id `22615925`）四條：不准刪、不准改寫歷史、只能走 PR、`verify` 沒綠不准合。狀態頁由 `governance/status/` 推到機器分支 `status`、掛 GitHub Pages。
- 已拍板、已落地：新家一段一段長、下一段先做參考答案（決策紙 `engine-grows-by-room-workflow`，取代了「一塊＝一個子套件」那張）；第一塊 config 的形狀與載入器路徑必填（決策紙 `engine-first-block-config-shape`、`config-loaders-keep-path-required`）；新家分層卡與寫法卡。
- 只是看：狀態頁 https://neknufelet.github.io/aosr-v3/ ；要拍板的題永遠是標籤 `decision` 的開著票。

## 備查

- 立卡：一張卡一個 PR，帶卡的 TOML、檢查程式、必紅樣本目錄（含一份控制樣本）。後設測試對每張卡跑六回合（見 `tests/test_fixture_runner.py` 檔頭）。雲端 `verify` 綠了才算。立卡前先讀該卡在 `cards-38.json` 的規格。
- 加卡的 PR 要順手把 `green-must-be-real-green.toml` 裡**受影響那一籃**的 `collected_floor` 調到那一跑的實跑收集數（不調，雲端會判「地板過期」；籃子越小那個窗口越窄）。
- 清空前的備份在 `~/aosr-v3-blueprint-2026-09-09/`（完整 git 歷史的 bundle 檔）。
- 做法：分工三層——做工派另一家模型的 CLI（deepseek，走 calling-other-models 那支工具、帶收據）、找碴派 opus 子代理唯讀、主對話終審與排下一步；實作一律在**獨立工作樹**（`git worktree add` 到 repo **外面**，別放 `.claude/` 底下免得髒了主樹）做，主對話只判題、驗收、合併；派工指示寫成檔案、只給路徑；PR 內文用 `Closes #n` 關票，不准手動關。找碴那一道：派別家模型**唯讀**、題目固定，它講的**一條一條自己重跑**才採信——找碴的人沒有 Bash，「它說的」永遠只是「去哪裡驗」。
- 踩過的坑：大包資料寫成檔案只給路徑；數字沒實跑過標「未驗，推測」；工人會編出很像真的代號，交叉引用都要對回原始檔。第一塊多三筆：**「有幾筆」跟「每一筆都被拿去比」是兩件事**——要用逐筆刪的突變當證據；**裁判自己也要有控制組**——共用比對函式一鬆，幾百題一起假綠；**對照組本身要驗它會紅**——拿裁判自己報的字串去造的對照組是假的。
- 政策：既有相容性合約維持；新功能依工作票的範圍與精度契約驗收——**改公開呼叫契約**（把必填參數放寬成可選）算行為、不算結構；要改就當一張決定做。

## 這份怎麼寫（下一個對話照抄）

- 第一節永遠只有一題、四格（發生什麼事／結果／我建議／你回什麼），每格兩三句，最多兩個選項並說選哪個；沒題就寫「沒有」。
- 那一題同時要有一張標籤 `decision` 的票；這裡只抄票號，不抄內文。
- 座標那一節每一條都要標動詞。給機器看的代號只能出現在座標與備查。
- 數得出來的數字（幾張卡、幾張票、幾題考卷）不抄進來，指去狀態頁。
- 每個英文工具名旁邊同一句要有中文說它在做什麼。不用沒驗過的數字。全篇用日常講法。
- 已落地的事併成一行、只留卡名或決策紙名、不留票號；連著兩行以上「已落地」就是帳本，該併。
- 超過 60 行就砍：舊事沉進決策紙，不留在這裡。
