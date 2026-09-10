# 這個 repo 怎麼工作

開工先讀這一份。「規矩」那一節由規矩卡生成、被標記包住，手改雲端即紅；標記外四節手寫，改完跑 `uv run python -m governance.render_entry` 重生（它同時寫 `AGENTS.md` 那張指路牌）。這一份有行數上限，所以每一節都不留閒話：空行也算一行。
## 座標
- 這個 repo 是 v3 的治理層——規矩本身，不是產品程式；每一條規矩都從 v2 的事故長出來，不從舊 repo 搬。
- 一條規矩一張卡：卡在 `governance/rules/`、檢查程式在 `governance/checks/`、必紅樣本在 `governance/fixtures/`。
- 拍板過的決定一題一張紙，在 `docs/decisions/`：要動契約、裁判或門檻，先去那裡找那張紙。
- 來源資料：v2 的事故在 `v2-audit/lessons.json`，候選規則與每張卡的規格在 `blueprint/cards-38.json`。
## 驗證
本機一行跑完 `uv run pytest && uv run ruff check`；本機跑出來的只是宣稱，雲端 `verify`（GitHub Actions 上那個檢查工作）綠了才算數。
## 去哪裡看
- 機器算出來的那一頁（開著的票、每支檢查最近的結果）：https://neknufelet.github.io/aosr-v3/
- 拍板過的決定：`docs/decisions/`，一題一檔。
- 要拍板的題：GitHub issue 標籤 `decision`；還在等條件才立得起來的卡：標籤 `deferred-cards`。
## 七條習慣（沒有機器在守，踩過才寫下來的）
1. 大包資料不塞進派給工人的指示裡，寫成檔案只給路徑。
2. 每個數字都要帶證據，沒真的跑過就標「未驗，推測」。
3. 交叉引用的代號都要回原始檔對一次——工人會編出很像真的代號。
4. 找碴那一層要獨立、而且要凶：寫的人不驗自己寫的東西。
5. 手寫的狀態檔一定會過期，寫進版控前先問機器算不算得出來。
6. 跟老闆一次一題、四格白話（發生什麼事／結果／我建議／你回什麼）。
7. 每個英文名詞旁邊同一句要有中文說它在做什麼。
<!-- rules:begin generated from governance/rules - do not edit -->
## 規矩
一條規矩一張卡，卡住在 `governance/rules/`，每張卡自帶檢查程式與必紅樣本。下面一行是一張卡人話的第一句（整段住在卡上），括號裡是它擋不擋合併。

- **assertions-not-pinned-to-counts**（擋合併）：測試檔裡的斷言不准把數量鎖死。
- **check-exit-code-honest**（擋合併）：每支檢查的離開碼要誠實：0 是真的掃過而且乾淨、1 是抓到違規、2 是這一跑不算數。
- **ci-jobs-cannot-die-quietly**（擋合併）：雲端那一跑不准無聲死掉，四條：①任何 step 或 job 不准 `continue-on-error: true`——紅了不擋等於沒跑；②`run:` 裡不准把離開碼吞掉（`|| true`、`; exit 0`、單獨一行 `exit 0`、`set +e`、把 stderr 導進黑洞），也不准用沒有 pipefail 的多段管線（`cmd | tail` 只讀最後一段的離開碼），而且要遞迴進 `run:` 呼叫、進得了版控的腳本——藏進腳本就繞過去了；③job 不准用 `if: always()` 把紅漂成綠，也不准缺 `timeout-minutes`——沒有上限等於可以無聲卡死，上限數字登記在這張卡的 [settings]；④每個 job 至少要有一步真的在跑檢查或測試，只有 checkout／setup 的空 job 一律紅。
- **commit-author-allowlisted**（擋合併）：本次 PR 整段提交範圍（base..head，不只 HEAD）的每一筆，author 與 committer 兩個 email 都必須在 `governance/authors.txt` 名單裡，否則紅。
- **decision-paper-structure**（擋合併）：決策紙一題一檔，格式與取代關係由機器守。
- **doc-frontmatter-and-dates**（擋合併）：docs 底下的設計文件與知識文件要有齊全的標頭，份數逐類有上限、全部加起來另有一個總量上限，docs 三類與兩份入口檔的每一行都有字元上限，三類裡面不准再分層。
- **entry-files-rendered-from-registry**（擋合併）：入口檔的規矩節由登記簿生成，不是手寫的：標記之間的內容必須等於用 `governance/rules/` 底下每一張卡重生一次的結果，手改即紅、加了卡沒重生也紅；標記缺一個、多一個、或前後顛倒即紅（界線量不出來就不知道哪一段是產物）；卡宣告的每一份入口檔都要在，而且第二份起是**指路牌**——內容必須逐字等於卡上登記的 `pointer_text`，多一字少一字都紅（規矩與手寫段只住在原稿裡，指路牌只負責把人指去原稿）；標記之間出現卡上 `placeholder_markers` 列的那幾種佔位字樣即紅，標記外面的手寫段不管（那是人的地方）；任何一份的行數超過這張卡登記的上限即紅——歷史往決策紙與知識區沉，不往入口檔堆。
- **exemptions-need-expiry**（擋合併）：每一筆「放行」都要寫得出為什麼、以及什麼時候失效。
- **file-placement-allowlist**（擋合併）：檔案放哪裡走白名單，名字一律 ASCII，兩條：①repo 根層與 docs 根只准出現這張卡列出的檔名與目錄名，多一個檔、多一個目錄（含隱藏目錄）就紅——清單寫在這張卡自己的 [[allowlist]]，不寫死在檢查程式裡，改寬清單要走 PR；②git ls-files 全集（含未被忽略的未追蹤檔）的每一個路徑都必須是純 ASCII，非 ASCII 只准出現在卡宣告的必紅樣本樹底下，那是刻意的壞樣本、由後設測試單獨餵。
- **green-must-be-real-green**（擋合併）：測試那盞綠燈要是真的綠：pytest 產的 junit 收據裡 `skipped` 必須等於 0（資源缺席就 fail，不准 skip 成非紅）、收集數不准低於這張卡登記的地板、收集到零個測試（pytest 離開碼 5 那個形狀）一律紅不是綠、收據裡有 failures／errors 也不准當綠。
- **merge-gate-read-back**（擋合併）：合併門口的設定要從伺服器回讀比對：主線的 ruleset 必須是登記的那一個、還在生效、掛在預設分支上、沒有人能繞過（這一格 GitHub 只交給 admin 看，雲端的 token 看不到；看得到就比、看不到就在輸出明說「這一格這一跑沒守」），而且四條規則都在——不准刪、不准非快進、走 PR 且候選一改舊核准就作廢、必要檢查含 `verify` 而且分支要跟上主線才准合。
- **no-model-names-in-entry-files**（擋合併）：每次開工都會載入的檔——兩份入口檔，加上 `docs/decisions/` 底下每一份決策紙——**全文**不准出現卡上字典列的那些字：`model_words`（各家模型的名字）、`cli_words`（模型家那幾支 CLI 的名字）、`quota_words`（講「用得完／用不完」那類字樣）三張清單，比對不分大小寫、只咬整個詞——前後不准是英文字母或底線，所以名字後面接版本數字一樣咬，`.` 與 `-` 兩邊算斷開（名字被寫進網址、檔名、複合詞裡照樣咬）。
- **refs-and-links-resolve**（擋合併）：文件裡的路徑與連結都要解析得到，也不准指到 repo 外。
- **rule-card-required-fields**（擋合併）：規矩卡必填「怎麼查、壞樣本在哪、擋得住還是只會叫、裝在哪個 job」四欄，缺一欄、掛載點寫不滿四段、執行者填人或本機 hook、或宣告的 job 沒真的接到 CI，一律紅。
- **scan-scope-has-no-holes**（擋合併）：每張卡宣告的掃描面（scope）必須等於那支檢查實際列舉出來的檔案集合，兩者不等即紅——宣告了沒掃到是洞（卡面看起來有人守，那些檔實際上沒人看），掃了沒宣告是越權（卡面看不出它會咬到那裡）。
- **secrets-never-committed**（擋合併）：密碼、金鑰、`.env` 一進版控就紅，現在的樹與**全部歷史**都要掃。
- **status-page-computed-not-typed**（擋合併）：進度、待辦、狀態一律由機器現算，不准手寫進版控。
- **style-guard**（擋合併）：寫法警衛，掃版控裡的每一支 .py（扣掉必紅樣本樹與本機工具目錄），四條。
- **tests-isolated-from-real-env**（擋合併）：測試不准依賴環境現況，也不准寫進真的 repo。
- **thresholds-live-only-in-registry**（擋合併）：門檻數字只准住在卡的登記簿，檢查程式與卡的人話裡不准再寫一次。
- **type-guard**（擋合併）：型別警衛，掃版控裡的每一支 .py（扣掉必紅樣本樹與本機工具目錄），兩層，第二層刻意獨立於 mypy。
- **uv-single-entrypoint**（擋合併）：環境交給 uv 管，`uv run` 是唯一入口，三條。
<!-- rules:end -->
