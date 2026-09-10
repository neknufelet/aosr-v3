# Handoff

給下一個對話。先讀這份，再讀 `DIRECTION.md`。這份分三層：**要老闆回的**（給人看，一題四格）、**座標**（給下一個對話看）、**備查**（不用動）。待辦不列，全在 GitHub issue（見 `docs/decisions/backlog-in-github-issues.md`）。

## 要老闆回的（一題）

沒有。標籤 `decision` 的開著 issue 是零張（2026-09-10）。下一題出現時照檔尾寫法補四格。

排隊的工程題（不用老闆回）：#11 三張票務卡要判改寫成看收據還是由 skill 自己的測試守、#10 #15 #16 等對象出現、#59 delivery skill 四件等引擎、之後搬 `src/`。

## 座標（給下一個對話）

現在在哪（2026-09-10）：
- 2026-09-09 清空重來，規矩從 `v2-audit/` 重新長。為什麼、怎麼做，見 `docs/decisions/`。
- 主線 25 張卡（`governance/rules/`），藍圖 38 張全部有去向：哪些已立、併掉、暫緩，由 `blueprint/remap_cards.py` 算出，寫在 `cards-38.json` 的 `meta.establishment`，不手抄。
- 共用零件：載入器 `governance/loader.py`、離開碼與輸出層 `governance/exit_codes.py`、後設測試 `tests/test_fixture_runner.py`、CI `.github/workflows/verify.yml`（一個 `verify` job 跑全部檢查＋pytest＋ruff＋mypy）。
- 主線 ruleset（id `22615925`）四條：不准刪、不准改寫歷史、只能走 PR、`verify` 沒綠不准合。
- 狀態頁已上線：`governance/status/`、`.github/workflows/status.yml`，推到機器分支 `status`，掛 GitHub Pages。
- 入口檔已立：`CLAUDE.md`（規矩節由卡生成）、`AGENTS.md`（指路牌）。

每一條標動詞，看了就知道要不要動：
- 要老闆回：只有上面那一節那一題。
- 在等機器（不用回）：暫緩 20 張卡，一組一張 issue，標籤 `deferred-cards`。它們在等收據、引擎、腳本這些對象出現。
- 已拍板、已落地：合併門口那一支檢查准上網讀 ruleset（#64，決策紙 `docs/decisions/merge-gate-check-may-read-github.md`，卡 `merge-gate-read-back`），放行到期 2026-12-08 跟第一批一起審；bypass 名單雲端看不到、只在輸出明說。
- 已拍板、已落地：三張收據卡（`receipt-schema-complete`、`receipt-authority-is-the-cloud-run`、`four-roles-different-actors`）讀 `status` 分支鏡到 `governance/receipts/cloud/` 的收據（鏡像工具 `governance/status/mirror_receipts.py`，不上網；conftest 開跑前先鏡）；#14 關了。
- 已拍板、已落地：收據分兩層（#63，決策紙 `docs/decisions/receipts-two-layers-cloud-on-status-branch.md`），生產者 #66、鏡像 #71；紅的那條路已在雲端驗過一次（run 34443108224 的收據落地、指出是哪一支紅）。
- 已拍板、時候未到：派工工具留在 AI_TOOLS 不搬進 repo；delivery skill 的四件調整等引擎搬進來前做（#59）。
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
