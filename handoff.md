# Handoff

給下一個對話。先讀這份，再讀 `DIRECTION.md`。這份分三層：**要老闆回的**（給人看，一題四格）、**座標**（給下一個對話看）、**備查**（不用動）。待辦不列，全在 GitHub issue（見 `docs/decisions/backlog-in-github-issues.md`）。

## 要老闆回的（一題）

**發生什麼事**：卡 `merge-gate-read-back`（合併門口的設定要從伺服器回讀比對：必要檢查是不是 `verify`、不准繞過、不准改主線歷史）得在 CI 裡上網讀 GitHub 自己的 ruleset 設定，跟「檢查不上網、只認雲端收據」的主張直接衝突。
**結果**：已測（記在 #64）：repo 是公開的，ruleset 不帶任何 token 就讀得到，所以「token 有沒有權限、要不要 admin」兩個問號都消掉了。剩下純規矩題：准不准這一支檢查在 CI 裡上網讀。
**我建議**：選項 1——准，但只准這一支、只准讀，卡上登記例外並寫到期日，讀不到回 2。選項 2 是不上網、把 ruleset 期望值寫成檔進版控、卡改成比對那份檔，代價是實際設定漂了抓不到。我選 1。
**你回什麼**：「選 1」或「選 2」。回了我就寫決策紙、立那張卡。

排隊的下一題：沒有。三張收據卡與 #11 三張票務卡是工程題，不用老闆回。

## 座標（給下一個對話）

現在在哪（2026-09-10）：
- 2026-09-09 清空重來，規矩從 `v2-audit/` 重新長。為什麼、怎麼做，見 `docs/decisions/`。
- 主線 21 張卡（`governance/rules/`），藍圖 38 張全部有去向：哪些已立、併掉、暫緩，由 `blueprint/remap_cards.py` 算出，寫在 `cards-38.json` 的 `meta.establishment`，不手抄。
- 共用零件：載入器 `governance/loader.py`、離開碼與輸出層 `governance/exit_codes.py`、後設測試 `tests/test_fixture_runner.py`、CI `.github/workflows/verify.yml`（一個 `verify` job 跑全部檢查＋pytest＋ruff＋mypy）。
- 主線 ruleset（id `22615925`）四條：不准刪、不准改寫歷史、只能走 PR、`verify` 沒綠不准合。
- 狀態頁已上線：`governance/status/`、`.github/workflows/status.yml`，推到機器分支 `status`，掛 GitHub Pages。
- 入口檔已立：`CLAUDE.md`（規矩節由卡生成）、`AGENTS.md`（指路牌）。

每一條標動詞，看了就知道要不要動：
- 要老闆回：只有上面那一節那一題。
- 在等機器（不用回）：暫緩 20 張卡，一組一張 issue，標籤 `deferred-cards`。它們在等收據、引擎、腳本這些對象出現。
- 已拍板、時候未到：收據分兩層（#63，決策紙 `docs/decisions/receipts-two-layers-cloud-on-status-branch.md`）。生產者已上線（PR #66）：`verify` 每一步包一層抄寫員記離開碼，`status.yml` 的 receipt job 合成收據推 `status` 分支的 `receipts/`，主線第一份是 run 34440216514；紅的那條路（verify 紅時 artifact 照傳、consistency 欄寫出來）只在本機驗過，雲端未驗；下一個工程題是立三張收據卡，改寫成掃 `status` 分支；派工工具留在 AI_TOOLS 不搬進 repo；delivery skill 的四件調整等引擎搬進來前做（#59）。
- 只是看：狀態頁 https://neknufelet.github.io/aosr-v3/ ；要拍板的題永遠是標籤 `decision` 的開著 issue。

## 備查

- 立卡：一張卡一個 PR，帶卡的 TOML、檢查程式、必紅樣本目錄（含一份控制樣本）。後設測試對每張卡跑五回合：乾淨樹 0、必紅樣本 1、掃描根不存在 2、抽掉外部工具 2、控制樣本 1。雲端 `verify` 綠了才算。立卡前先讀該卡在 `cards-38.json` 的 `check_idea_v2`／`fixture_idea_v2`／`still_leaky_reason`。
- 加卡的 PR 要順手把 `green-must-be-real-green.toml` 的 `collected_floor` 調到那一跑的實跑收集數（收據上的 `tests=`），不調 CI 判「地板過期」會紅。
- 清空前的備份：`~/aosr-v3-blueprint-2026-09-09/`（完整 git 歷史 bundle、四份設計報告、被清掉的 8 張決策紙）。還原：`git clone aosr-v3-full-history.bundle <目錄>`。
- 踩過的坑：大包資料寫成檔案只給路徑；數字沒實跑過標「未驗，推測」；工人會編出很像真的代號，交叉引用都要對回原始檔；找碴那層要獨立而且要凶。

## 這份怎麼寫（下一個對話照抄）

- 第一節永遠只有一題、四格（發生什麼事／結果／我建議／你回什麼），每格兩三句，最多兩個選項並說選哪個。老闆回了就換下一題；沒題就寫「沒有」。
- 那一題同時要有一張標籤 `decision` 的 issue，狀態頁才算得出來；這裡只抄 issue 號碼，不抄內文。
- 座標那一節每一條都要標動詞（要老闆回／在等機器／已拍板／只是看）。給機器看的代號只能出現在座標與備查，不准出現在第一節。
- 每個英文工具名旁邊同一句要有中文說它在做什麼。不用沒驗過的數字。
- 超過 60 行就砍：舊事沉進決策紙，不留在這裡。
