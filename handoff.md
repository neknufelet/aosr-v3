# Handoff

給下一個對話。先讀這份，再讀 `DIRECTION.md`。**這份只寫「現在在哪、去看哪」，不列待辦**——待辦全在 GitHub issue（見 `docs/decisions/backlog-in-github-issues.md`）。

## 現在在哪（2026-09-09）

- 2026-09-09 清空重來，規矩從 `v2-audit/` 重新長。為什麼、怎麼做，見 `docs/decisions/`。
- **第一張有牙齒的卡已立**（PR #19）：`governance/rules/rule-card-required-fields.toml`，連同四樣共用零件（卡的載入器 `governance/loader.py`、離開碼約定 `governance/exit_codes.py`、跑必紅樣本的後設測試 `tests/test_fixture_runner.py`、CI `.github/workflows/verify.yml`）。
- 主線 ruleset（id `22615925`）四條：不准刪、不准改寫歷史、只能走 PR、**`verify` 沒綠不准合**。
- 第一批要立的卡與順序：`blueprint/first-batch-order.json`（16 張，第一張已立）。每張卡的規格、對到的 v2 事故、找碴結果：`blueprint/cards-38.json`。

## 去看哪

- 要拍板的題：issue 標籤 `decision`。
- 暫緩的卡在等什麼：issue 標籤 `deferred-cards`（一組一張）。
- 第一批進度：milestone `batch-1`。
- 清空前的備份：`~/aosr-v3-blueprint-2026-09-09/`（完整 git 歷史 bundle、四份設計報告、被清掉的 8 張決策紙、工作流中間產物）。還原：`git clone aosr-v3-full-history.bundle <目錄>`。

## 立卡的規矩

一張卡一個 PR，帶三樣：卡的 TOML、檢查程式、必紅樣本目錄（含一份控制樣本）。後設測試會對每張卡跑五回合：乾淨樹 0、必紅樣本 1、掃描根不存在 2、抽掉外部工具 2、控制樣本 1。雲端 `verify` 綠了才算。立卡前先讀該卡在 `cards-38.json` 的 `check_idea_v2`／`fixture_idea_v2`／`still_leaky_reason`。

加卡的 PR 還要順手把 `green-must-be-real-green.toml` 的 `collected_floor` 調到那一跑的實跑收集數（收據上的 `tests=`）。不調 CI 會紅：實跑數超過地板 × `floor_stale_ratio` 就判「地板過期」。

## 踩過的坑（別再犯）

1. 不要把大包資料塞進工人的 prompt，資料寫成檔案、只給路徑。
2. 數字要帶證據；沒實跑過的一律標 `(未驗，推測)`。
3. 工人會編出很像真的 id，任何交叉引用都要對回原始檔驗一次。
4. 找碴那層要獨立而且要凶。
5. 手寫的狀態檔（包括這份）會長成帳本。這份超過 60 行就該砍。

## 跟老闆講話的方式

一次一題、四格白話（發生什麼事／結果／我建議／你回什麼），每格兩三句，最多一個例子。每個英文工具名旁邊同一句要有中文說它在做什麼。要決定的事最多兩個選項並說選哪個。不用沒驗過的數字。
