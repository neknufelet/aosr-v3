# Handoff

給下一個對話。先讀這份，再讀 `DIRECTION.md`。這份分三層：**要老闆回的**（給人看，一題四格）、**座標**（給下一個對話看）、**備查**（不用動）。待辦不列，全在 GitHub issue（見 `docs/decisions/backlog-in-github-issues.md`）。

## 要老闆回的（一題）

**發生什麼事**：今天砍了兩張卡、把牙併進別張、又改寫了一張，靠的是三條散在不同地方的判準加上老闆當場拍板。沒有一張紙寫「什麼東西夠格立卡」，判準只活在對話裡。
**結果**：把 `src/`（引擎程式碼那棵樹）搬進來之後會冒出一批新候選，沒有門檻就會長回上一代那幾百條規則的樣子。
**我建議**：把現有三條寫成決策紙——對得回 `v2-audit/` 裡的事故、四欄必填而且必紅樣本真的紅、在事故當下就會紅才算血債——再加一條「對象不在 repo 裡的卡一律不立」。新候選一律先暫緩一輪，找碴席（獨立審那一關）過了才立。
**你回什麼**：新對話談 `src/` 計畫時第一題就談這個，拍板後我派人寫決策紙，PR 內文用 Closes 關掉那張票。題在 #95。

排隊的工程題（不用老闆回）：#94 後設測試的下一個瓶頸（一支測試 19 秒、跟卡數成正比，記著沒動）、#10 #15 #16 等對象出現、#59 delivery skill（一張票從開工到收工的流程機）五件等引擎，第五件是 v4 帳本補停止條件的測試（見那張票的留言）、之後把 `src/`（引擎程式碼那棵樹）搬進來，那會一起解掉 #16 與 #59。`uv run pytest`（本機跑全套）今天實跑 55 秒。

## 座標（給下一個對話）

現在在哪（2026-09-11）：
- 2026-09-09 清空重來，規矩從 `v2-audit/` 重新長。為什麼、怎麼做，見 `docs/decisions/`。
- 主線 27 張卡（`governance/rules/`），藍圖 38 張全部有去向：哪些已立、併掉、暫緩，由 `blueprint/remap_cards.py` 算出，寫在 `cards-38.json` 的 `meta.establishment`，不手抄。
- 共用零件：載入器 `governance/loader.py`、離開碼與輸出層 `governance/exit_codes.py`、後設測試 `tests/test_fixture_runner.py`、CI `.github/workflows/verify.yml`（一個 `verify` job 跑全部檢查＋pytest＋ruff＋mypy）。
- 主線 ruleset（id `22615925`）四條：不准刪、不准改寫歷史、只能走 PR、`verify` 沒綠不准合。
- 狀態頁已上線：`governance/status/`、`.github/workflows/status.yml`，推到機器分支 `status`，掛 GitHub Pages。
- 入口檔已立：`CLAUDE.md`（規矩節由卡生成）、`AGENTS.md`（指路牌）。

每一條標動詞，看了就知道要不要動：
- 要老闆回：只有上面那一節那一題。
- 在等機器（不用回）：暫緩 4 張卡，一組一張 issue（三張，標籤 `deferred-cards`）——1 張等 v3 自己的設計報告、1 張等 repo 裡出現 shell 腳本、2 張等 `src/` 搬進來。張數由 `blueprint/remap_cards.py`（重算藍圖 38 張去向那支）算出，不手抄。
- 已拍板、已落地（細節在決策紙與 issue，這裡不留票號）：合併門口那一支檢查准上網讀 ruleset（卡 `merge-gate-read-back`）、人手關的票超過零張就擋合併（卡 `issues-closed-only-by-merged-pr`，第二支准上網的檢查），兩支的放行都到期 2026-12-08 跟第一批一起審；票務兩張卡砍掉、`identity-strings-generated` 改寫成文件裡的 sha 與 run id 必須解析得到（今天零對象）；票只准由合進主線的 PR 關（PR 內文寫 Closes #n），人手關的由機器當場重開，決策紙 `docs/decisions/issues-closed-only-by-merged-pr.md`；後設測試改成平行跑、那兩張會上網的卡在測試裡一跑只問 GitHub 一次（純工程調整，沒有決策紙）。
- 已拍板、已落地（收據線）：收據分兩層——生產者、鏡像、三張收據卡都在主線，紅的那條路雲端驗過一次，狀態頁那個 status job（重算頁面那一步）等收據推完才開始算，時間差消失；收據不再會無聲消失——推分支撞到就重疊到最新的再推，狀態頁多一格列出主線最近哪幾跑沒有收據；擋合併那個 verify job（雲端把全部檢查跑一遍那一跑）底下每一個步驟，要嘛包著抄寫員（把離開碼記進收據那支）要嘛是卡上登記的水管，認不出那一行是什麼一律紅，水管名單收窄到只剩裝依賴那一步——能跑任意命令的入口不准當水管。
- 已拍板、已落地（引擎樹）：產品程式的家是 `src/`（v2 根層的鏡像，今天還是空的），搬之前先讓七張卡在掃描面上宣告「這棵樹別看」——`style-guard`、`type-guard`、`exemptions-need-expiry`、`uv-single-entrypoint`、`refs-and-links-resolve`、`status-page-computed-not-typed`、`secrets-never-committed`，根層那一格由 `file-placement-allowlist` 的白名單放進來；排除沒有到期日，解除一張卡一個 PR，決策紙 `docs/decisions/engine-tree-lands-unwatched-then-rules-bite-one-by-one.md`。
- 已拍板、時候未到：派工工具留在 AI_TOOLS 不搬進 repo；delivery skill 的五件調整等引擎搬進來前做（#59）。
- 只是看：狀態頁 https://neknufelet.github.io/aosr-v3/ ；要拍板的題永遠是標籤 `decision` 的開著 issue。

## 備查

- 立卡：一張卡一個 PR，帶卡的 TOML、檢查程式、必紅樣本目錄（含一份控制樣本）。後設測試對每張卡跑六回合：乾淨樹 0、必紅樣本 1、掃描根不存在 2、抽掉外部工具 2、控制樣本 1、該回 2 的樣本 2（見 `tests/test_fixture_runner.py` 檔頭）。雲端 `verify` 綠了才算。立卡前先讀該卡在 `cards-38.json` 的 `check_idea_v2`／`fixture_idea_v2`／`still_leaky_reason`。
- 加卡的 PR 要順手把 `green-must-be-real-green.toml` 的 `collected_floor` 調到那一跑的實跑收集數（收據上的 `tests=`），不調 CI 判「地板過期」會紅。
- 清空前的備份：`~/aosr-v3-blueprint-2026-09-09/`（完整 git 歷史 bundle、四份設計報告、被清掉的 8 張決策紙）。還原：`git clone aosr-v3-full-history.bundle <目錄>`。
- 做法：實作一律派子代理（另一個較便宜的模型）在獨立 worktree（同一個 repo 的另一棵工作樹）做，主對話只判題、驗收（雲端 `verify` 綠、看狀態頁）、合併；派工指示寫成檔案、只給路徑；PR 內文用 `Closes #n` 關票，不准手動關。
- 踩過的坑：大包資料寫成檔案只給路徑；數字沒實跑過標「未驗，推測」；工人會編出很像真的代號，交叉引用都要對回原始檔；找碴那層要獨立而且要凶。

## 這份怎麼寫（下一個對話照抄）

- 第一節永遠只有一題、四格（發生什麼事／結果／我建議／你回什麼），每格兩三句，最多兩個選項並說選哪個。老闆回了就換下一題；沒題就寫「沒有」。
- 那一題同時要有一張標籤 `decision` 的 issue，狀態頁才算得出來；這裡只抄 issue 號碼，不抄內文。
- 座標那一節每一條都要標動詞（要老闆回／在等機器／已拍板／只是看）。給機器看的代號只能出現在座標與備查，不准出現在第一節。
- 每個英文工具名旁邊同一句要有中文說它在做什麼。不用沒驗過的數字。
- 已落地的事併成一行、只留卡名或決策紙名、不留票號與 PR 號；細節本來就在決策紙與 issue 裡。連著兩行以上「已落地」就是帳本，該併。
- 超過 60 行就砍：舊事沉進決策紙，不留在這裡。
