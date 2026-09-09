# 這個 repo 怎麼工作

這一份是入口檔：開工先讀它。下面「規矩」那一節被兩個標記包住，是機器從規矩卡生成的產物，
手改會被雲端檢查擋下來；標記外面四節是手寫的。

## 座標

- 這個 repo 是 v3 的治理層——規矩本身，不是產品程式。每一條規矩都從 v2 的事故資料長出來，
  不從舊的 repo 搬。
- 一條規矩一張卡：卡在 `governance/rules/`，它的檢查程式在 `governance/checks/`，
  刻意寫壞、餵下去必須變紅的樣本在 `governance/fixtures/`，共用零件是
  `governance/loader.py`（讀卡）與 `governance/exit_codes.py`（離開碼與輸出）。
- 拍板過的決定一題一張紙，在 `docs/decisions/`。要動契約、裁判或門檻，先去那裡找那張紙。
- 規矩的來源資料：v2 的事故在 `v2-audit/lessons.json`，候選規則與每張卡的規格在
  `blueprint/cards-38.json`。
- 這份入口檔以兩個檔名各存一份、內容逐字相同，好讓不同家的工具都讀得到同一份東西。
  手寫段只改排在前面那一份，改完跑 `uv run python -m governance.render_entry`
  （重生入口檔的產生器）把另一份同步過去。

## 驗證

本機一行跑完：

```
uv run pytest && uv run ruff check
```

`uv`（管環境與執行的工具）是這個 repo 唯一的入口；`pytest`（測試跑者）會把每一張卡在乾淨樹
上各跑一次、再餵一次它的壞樣本；`ruff`（寫法檢查器）是寫法警衛的另一半。本機跑出來的結果
只是宣稱，雲端 `verify`（GitHub Actions 上那個檢查工作）綠了才算數。

## 去哪裡看

- 機器算出來的那一頁（開著的票、每支檢查最近的結果、主線領先落後）：
  https://neknufelet.github.io/aosr-v3/
- 拍板過的決定：`docs/decisions/`，一題一檔。
- 要拍板的題：GitHub issue（線上的票）標籤 `decision`；還在等條件才立得起來的卡：
  標籤 `deferred-cards`。

## 七條習慣

這七條沒有機器在守，是踩過才寫下來的。

1. 大包資料不塞進派給工人的指示裡：寫成檔案，只給路徑。
2. 每個數字都要帶證據；沒有真的跑過的一律標「未驗，推測」。
3. 任何交叉引用的代號都要回原始檔對一次——工人會編出很像真的代號。
4. 找碴那一層要獨立、而且要凶：寫的人不驗自己寫的東西。
5. 手寫的狀態檔一定會長成過期帳本；要寫進版控之前先問「這件事機器算不算得出來」。
6. 跟老闆一次一題、四格白話（發生什麼事／結果／我建議／你回什麼），每格兩三句、最多一個例子。
7. 每個英文名詞旁邊同一句要有中文說它在做什麼；要決定的事最多給兩個選項，並說自己選哪個。

<!-- rules:begin generated from governance/rules - do not edit -->

## 規矩

一條規矩一張卡，卡住在 `governance/rules/`，每張卡自帶檢查程式與必紅樣本。下面一行就是一張卡的人話，括號裡是它擋不擋合併。

- **assertions-not-pinned-to-counts**（擋合併）：測試檔裡的斷言不准把數量鎖死。AST 掃 `tests/` 底下的 .py，斷言裡出現「數量對死一個數字」就紅：`assert len(<任何東西>) == <整數>`、`assert <呼叫>() == <整數>`，`!=` 一樣算，藏在 `and`／`or`／`not` 底下一樣算，先把數字指給同一支檔裡的名字再比（`n = 3` 之後比 `n`）也一樣算——數字寫死在同一支檔裡，行為跟直接寫 3 沒差別。連 `== 0` 都算，它不會懲罰改善但一樣把「是哪幾筆」藏起來。合規寫法兩種：逐項具名比對（`assert names == {"a", "b"}`），或對照別處登記、import 進來的值（`assert len(x) == len(REGISTERED)`）。理由：「必須仍有 N 筆」不是單調安全的性質，多抓到一筆真缺陷就把閘弄紅，斷言變成在懲罰改善。
- **check-exit-code-honest**（擋合併）：每支檢查的離開碼要誠實：0 是真的掃過而且乾淨、1 是抓到違規、2 是這一跑不算數。掃描根不存在、外部工具缺席、列舉子程序非零退出、列舉出來是空集合，一律回 2 不准回 0；每條回傳路徑都要印一行 `scan_root= files= hits=`；每支檢查都要有一張卡宣告它並附一份已知會咬的控制樣本，餵下去必須回 1。靜態上離開碼只准 0／1／2 三個字面值，子程序的退出碼不准被布林化、不准把錯誤導進黑洞、離開碼約定自己不准被改寫。另外單獨戳共用外殼：拿一個空目錄餵它的列舉函式，必須 raise 讓外殼回 2——空集合回一個空 list 就是「空集合當乾淨」，這一針不能靠整支檢查去測，因為每支檢查自己的「沒東西可掃」守門會把外殼那道守門遮住。
- **ci-jobs-cannot-die-quietly**（擋合併）：雲端那一跑不准無聲死掉，四條：①任何 step 或 job 不准 `continue-on-error: true`——紅了不擋等於沒跑；②`run:` 裡不准把離開碼吞掉（`|| true`、`; exit 0`、單獨一行 `exit 0`、`set +e`、把 stderr 導進黑洞），也不准用沒有 pipefail 的多段管線（`cmd | tail` 只讀最後一段的離開碼），而且要遞迴進 `run:` 呼叫、進得了版控的腳本——藏進腳本就繞過去了；③job 不准用 `if: always()` 把紅漂成綠，也不准缺 `timeout-minutes`——沒有上限等於可以無聲卡死，上限數字登記在這張卡的 [settings]；④每個 job 至少要有一步真的在跑檢查或測試，只有 checkout／setup 的空 job 一律紅。掃描面是掃描根自己的 `.github/workflows/*.yml`／`*.yaml`，用真的 YAML 剖析（pyyaml），不用正則猜縮排；列舉到零個 workflow 檔、或某份 yaml 剖析不開，回 2 不回 0。
- **commit-author-allowlisted**（擋合併）：本次 PR 整段提交範圍（base..head，不只 HEAD）的每一筆，author 與 committer 兩個 email 都必須在 `governance/authors.txt` 名單裡，否則紅。名單裡的假身分也擋：email 的網域精確等於 `example.com`／`example.invalid`／`localhost`（或它們的子網域），或 local part 完整等於 `test`，一律紅——比對錨定在網域與 local part 的邊界上，所以 `x@example.com.tw`、`attest@corp.com` 不會被誤咬。名單檔本身有守衛：這段範圍新增的 email 必須有提交在用它（同一段範圍裡用，或這條歷史裡已經在用；後者是為了 GitHub 合併時的機器身分，那種 email 不可能跟同一個 PR 的提交同時出現）。名單檔讀不到、範圍算出來是空的、拿不到 base（例如 CI 淺 clone 沒有那顆物件）一律回 2，不准回 0。**前提寫在卡面上：這是 PR 範圍的閘，任何繞過 PR 直推主線的路徑都在它視野外**——那要靠主線 ruleset 擋（見 merge-gate-read-back）；它也只擋事故的下半場（污染身分的提交進主線），擋不住本機 hook 當下已造成的 .git 寫入，那半歸 tests-isolated-from-real-env 的 git_sandbox。
- **entry-files-rendered-from-registry**（擋合併）：入口檔的規矩節由登記簿生成，不是手寫的：標記之間的內容必須等於用 `governance/rules/` 底下每一張卡重生一次的結果，手改即紅、加了卡沒重生也紅；標記缺一個、多一個、或前後顛倒即紅（界線量不出來就不知道哪一段是產物）；卡宣告的每一份入口檔都要在，而且第二份起必須跟原稿逐字相同（手寫段只住在原稿裡）；任何一份的行數超過這張卡登記的上限即紅——歷史往決策紙與知識區沉，不往入口檔堆。入口檔清單與行數上限只寫在這張卡的 `[settings]`，檢查程式沒有預設值；讀不到那一段、某張卡讀不出人話或掛載點、掃描面上一張卡都沒有，一律回 2 不回 0。
- **exemptions-need-expiry**（擋合併）：每一筆「放行」都要寫得出為什麼、以及什麼時候失效。掃描面明確列舉，只有這幾類算放行條目：①規矩卡 toml 裡住在 `[settings]` 底下的表陣列（今天全叫 `[[settings.allow]]`），一筆就是一個具名放過；②測試檔裡的 skip 標記（`pytest.skip`、`pytest.mark.skip`／`skipif`／`xfail`）；③原始碼註解裡的抑制標記（`# noqa`、`# type: ignore`、`# pragma: no cover`）。每一筆都必須帶 `reason`（非空）與 `expires`（ISO 日期，`YYYY-MM-DD`），三件事會紅：缺任一欄、`expires` 早於今天（今天從系統時間讀，不寫死）、`reason` 正規化之後整串只由卡上登記的關鍵字組成（「暫時」「legacy」「TODO」這種字樣不是理由，是把問題往後推的說法）。①的欄位寫成 toml 的鍵；②③寫成標記那一行行尾的註解，形如 `expires=YYYY-MM-DD reason=<一句話>`——理由取到行尾，所以 `expires` 寫在前面。跑完在 stderr 印一行「目前有效的放行幾筆、最近到期的是哪一筆」，讓放行清單長大這件事看得見。**哪些算放行、哪些算資料**：判準是「這一條是在放過一個原本會紅的東西」。放行是具名指向一個個體、把它從已經算出來的違規裡撈出來（`[[settings.allow]]` 的每一筆）；資料是規矩本身的形狀，拿掉它規矩就沒有形狀可言——`file-placement-allowlist` 的 `[[allowlist]]`（那是「准出現什麼」的正面定義，不是放過某一筆違規）、`governance/authors.txt` 與 `governance/required-status-checks.txt` 這種名單檔、各卡的 `scan_exempt_prefixes`／`name_exempt_prefixes`（那是掃描面宣告，已經由卡的 `scope` 與 scan-scope-has-no-holes 那張卡在守）、`swallow_snippets` 這種樣式表。讀不到這張卡的 `[settings]`、掃描面上一個對象都沒有、某張卡或某支 .py 剖不開，一律回 2 不回 0。
- **file-placement-allowlist**（擋合併）：檔案放哪裡走白名單，名字一律 ASCII，兩條：①repo 根層與 docs 根只准出現這張卡列出的檔名與目錄名，多一個檔、多一個目錄（含隱藏目錄）就紅——清單寫在這張卡自己的 [[allowlist]]，不寫死在檢查程式裡，改寬清單要走 PR；②git ls-files 全集（含未被忽略的未追蹤檔）的每一個路徑都必須是純 ASCII，非 ASCII 只准出現在卡宣告的必紅樣本樹底下，那是刻意的壞樣本、由後設測試單獨餵。這張卡只看白名單宣告的那幾層（根層與 docs 根，深度 1）；已核准的目錄裡面長多少子目錄、堆多少檔案它看不到，那要另一張卡。
- **green-must-be-real-green**（擋合併）：測試那盞綠燈要是真的綠：pytest 產的 junit 收據裡 `skipped` 必須等於 0（資源缺席就 fail，不准 skip 成非紅）、收集數不准低於這張卡登記的地板、收集到零個測試（pytest 離開碼 5 那個形狀）一律紅不是綠、收據裡有 failures／errors 也不准當綠。收據不存在也紅——沒有收據就是沒有綠，不准「檔不存在就當乾淨」。另外綁死收據的來源：卡宣告的那個 job 裡必須真的有一步跑 pytest 並把 junit 寫到卡宣告的那個路徑，測試步驟整個被拿掉、pytest 沒帶 `--junitxml`、或路徑跟卡宣告的不一樣，一律紅。正式跑法只有一種：跑測試那一步（含它呼叫的腳本）不准帶 `--deselect`／`--ignore`／`-k` 排除清單、不准用 `--collect-only` 冒充跑過、不准把離開碼吞掉，pytest 設定只准寫在 pyproject.toml，不准另開 pytest.ini／tox.ini／setup.cfg 的第二套設定。地板與收據路徑只寫在這張卡的 `[junit]` 裡，檢查程式沒有預設值——沒有卡宣告 `[junit]` 就回 2 說「沒東西可判」。
- **refs-and-links-resolve**（擋合併）：文件裡的路徑與連結都要解析得到，也不准指到 repo 外。兩條：①版控的文字檔（副檔名由這張卡登記：md／toml／yaml／py／json；py 只看字串與註解、json 只看字串值，會跑的程式碼本身不看）裡形如 `目錄/…/檔名.副檔名` 的路徑 token，加上 md 的 markdown link 目標，都必須在 `git ls-files` 的集合裡解析得到——glob 算解析得到（`governance/checks/*.py` 有東西命中就過），markdown link 指到目錄也算（有檔住在那個前綴底下）；解不到就紅。②不准指到 repo 外：`../` 逃出樹、開頭是 `/` 的絕對路徑、開頭是 `~/` 的家目錄路徑，一律紅——這一條不去讀那些路徑，只判它的形狀。刻意要指到解析不到的東西（repo 外的備份座標、刻意不進版控的產出物、只活在必紅樣本樹裡的道具、v2 事故現場的路徑）走 `[[settings.allow]]`，一條一個路徑、每條必須寫理由與到期日（過期即紅那一關由 exemptions-need-expiry 判）；清單是資料、放在這張卡裡，不寫死在檢查程式，要放寬就得改卡走 PR。沒有副檔名的裸目錄 token（`docs/`、`.github/`）不當引用看，錨點（`#小節`）與 http(s) 連結這一版不管。
- **rule-card-required-fields**（擋合併）：規矩卡必填「怎麼查、壞樣本在哪、擋得住還是只會叫、裝在哪個 job」四欄，缺一欄、掛載點寫不滿四段、執行者填人或本機 hook、或宣告的 job 沒真的接到 CI，一律紅。附的壞樣本還必須真的讓那支檢查回 1，只填滿欄位不算有牙。
- **scan-scope-has-no-holes**（擋合併）：每張卡宣告的掃描面（scope）必須等於那支檢查實際列舉出來的檔案集合，兩者不等即紅——宣告了沒掃到是洞（卡面看起來有人守，那些檔實際上沒人看），掃了沒宣告是越權（卡面看不出它會咬到那裡）。怎麼量：卡的 scope 用版控的列舉集合展開成「宣告集合」，語法只有一份定義在載入器裡（`.` 是整棵樹、目錄前綴、單檔、glob，前面加 `!` 就是扣掉）；再用共用外殼的列舉模式跑那支檢查，拿它自己印出來的清單當「實際集合」；兩邊做集合差。掃描面不是檔案的卡在卡上明寫 `scope_kind = "commits"`（今天只有一張，它判的是提交身分不是路徑），這張卡對它只驗宣告在不在、宣告的名單在版控裡有沒有對象，不比集合。卡沒宣告 scope、scope_kind 打錯字、卡指的檢查跑不出清單，一律紅——量不到就不准當乾淨。宣告了一個今天還沒有檔案的前綴不算違規：那個前綴今天沒有對象，明天有檔進去兩邊會一起長。
- **secrets-never-committed**（擋合併）：密碼、金鑰、`.env` 一進版控就紅，現在的樹與**全部歷史**都要掃。三層。①存在性：進得了版控的 `.env` 與 `.env.*`（`.env.api`、`.env.local` 都算）一律紅，**不看內容**——v2 漏掉的就是這種檔，而且「裡面只是樣板值」正是當年放過它的理由。②賦值語境：掃到的文字裡形如 `名字=值`／`名字: 值`／`名字 = "值"`，名字帶到卡上登記的字樣（`PASSWORD`、`SECRET`、`TOKEN`、`API_KEY` 這些，完整清單在 `[settings]` 的 name_words），而值不是佔位符，就紅——**不管熵、不管長度、不管像不像金鑰**，這一層就是為 v2 那個八位十六進位樣板值寫的。佔位符只有三種：卡上登記的佔位字（`changeme`、`example`，比對值的字段）、卡上登記的包法（`${…}`、`<…>`），以及「整個值裡一個英數字元都沒有」（空值、`***` 這種）。③格式金鑰：卡上登記的已知前綴樣式（AWS 的 `AKIA…`、GitHub 的 `ghp_…`／`github_pat_…`、`sk-…`、PEM 的私鑰開頭）。三層都同時套在兩個集合上：`git ls-files` 的全集，以及 `git rev-list --objects --all` 走得到的**每一顆歷史 blob**——刪掉檔案不會讓歷史裡那一顆消失，所以歷史那一筆紅不掉，只能具名放行（主線 ruleset 不准改寫歷史）。**淺 clone 一律回 2**：`git rev-parse --is-shallow-repository` 是 true 就代表這一跑根本看不到舊提交，「掃過全歷史」是假話，所以 CI 的 checkout 必須帶 `fetch-depth: 0`。放行只有一種寫法——這張卡的 `[[settings.allow]]`，一條一個路徑前綴、必須寫理由與到期日；不掃的前綴（必紅樣本樹與本機工具目錄）走 `[settings]` 的 scan_exempt_prefixes，那是掃描面宣告、跟卡的 `scope` 對得上。讀不到這張卡的 `[settings]`、扣完前綴一個檔都不剩、拿不到歷史、或宣告檔的格式看不懂，一律回 2 不回 0。
- **status-page-computed-not-typed**（擋合併）：進度、待辦、狀態一律由機器現算，不准手寫進版控。`git ls-files` 裡出現檔名像進度／待辦／狀態檔（next、todo、task、status、progress、checkpoint、roadmap、backlog、tracker、handoff 開頭）、frontmatter 把自己標成 todo／progress／status 這類、md 裡勾選框超過卡上登記的個數、或出現「待辦／已完成／進度」這種段落標題，一律紅。明文放行的只有 handoff.md 一個檔，而且它有行數上限，超過照樣紅。決策紙照題目命名（docs/decisions/ 底下）不受檔名那一條管，但內容那兩條照管。
- **style-guard**（擋合併）：寫法警衛，掃版控裡的每一支 .py（扣掉必紅樣本樹與本機工具目錄），四條。①`print` 只准出現在輸出層——輸出層是這張卡 `[[settings.allow]]` 具名列出的那幾個檔，每筆帶理由與到期日（那兩格由 exemptions-need-expiry 那張卡守）；別的檔要對人說話就走 `governance/exit_codes.py` 的 `note()`，不要自己 print，因為散在各支程式裡的 print 混在判決輸出裡，看不出哪一行是收據哪一行是隨手記的。②字串拼路徑：`+` 拼出來的字串流進開檔／路徑／子程序參數（`open()`、`Path()`、`os.path.*` 系列、`subprocess` 的 argv）才紅，看到字串相加不會紅——判準刻意收窄成「拼接鏈裡有字串字面值，而且結果直接當參數、或先指給同一個作用域裡的一個名字再當參數」；乾淨樹裡 `blueprint/remap_cards.py` 就有一處 `json.dumps(...) + 換行` 的拼接，它流進的是檔案內容不是路徑，所以不咬。③開檔一律 `with`：`open(...)` 不在某個 `with` 的頭上就紅（`Path.read_text`／`write_text` 合法，它們自己關）。④函式的行數（不含文件字串）與分支數超過這張卡 `[settings]` 登記的門檻就紅；門檻怎麼量出來的寫在卡的註解裡，人話這裡不抄數字。這四條之外的寫法規則交給 ruff：規則集與它自己的數字住在 `pyproject.toml` 的 `[tool.ruff]`（只有一份、走 PR 看得到），CI 有獨立一步 `uv run ruff check`。ruff 裝不起來這支檢查回 2 不回 0——ruff 是這張卡的另一半，它不在的時候「沒問題」這句話不算數。讀不到這張卡的 `[settings]`、掃描面上一支 .py 都沒有、某支 .py 剖不開，一律回 2 不回 0。
- **tests-isolated-from-real-env**（擋合併）：測試不准依賴環境現況，也不准寫進真的 repo。兩段。①靜態（AST 掃 `tests/` 底下所有進得了版控的 .py，含 conftest 與輔助模組）：會 spawn 版控工具、或會往真樹寫檔的函式，必須經過唯一那支 sandbox fixture——判準是「這個函式（或包著它的那層）有沒有要那支 fixture」，不是「參數字串裡有沒有 git status」（實測後者抓不到：`subprocess.run(["git", "status", "--porcelain"])` 的字面值裡沒有任何一個含 "git status"）；不經 fixture 直接 spawn、參數陣列先指給名字再餵、直接寫 `REPO / ...`、同名 fixture 被定義第二次、要了一支這棵樹裡不存在的 fixture、拿 `Path.home()`／`expanduser("~")` 當答案來源，一律紅。②動態（這張卡帶進 `tests/conftest.py`）：session 起始若 `GIT_DIR`／`GIT_WORK_TREE`／`GIT_INDEX_FILE` 已經被注入就直接停跑，並記下真 repo 的 `git status --porcelain`；全部測試跑完後再量一次，多一筆未追蹤檔、少一筆、或有檔被改，收據上就是一筆 error，離開碼非零。寫入真樹的例外只准開在這張卡的 `[[settings.allow]]`（一條一格：哪支檔、哪個函式、寫到哪、為什麼、什麼時候失效），而且放行的路徑必須被掃描根的 `.gitignore` 蓋住——今天唯一一條是 pytest 的 junit 收據落在 `governance/receipts/`。門檻（那支 fixture 的名字、放行清單）只寫在這張卡上，檢查程式沒有預設值：讀不到就回 2 說「這一跑不算數」。
- **thresholds-live-only-in-registry**（擋合併）：門檻數字只准住在卡的登記簿，檢查程式與卡的人話裡不准再寫一次。兩條：①`governance/` 與 `governance/checks/` 底下的 .py，凡是模組載入時就算出來的數字都不准寫死——模組層級的常數賦值（`MAX_LINES = 200` 這種，藏在模組層的 `if`／`try` 底下一樣算）與函式簽章的預設值（`def check(root, max_lines=200)`）都算；函式跑起來才算的字面值不管，那是演算法不是門檻。放行只有兩種：值就是離開碼約定的那三個數（它們的家是 `governance/exit_codes.py`，那裡是唯一一份），或具名登記在這張卡的 `[[settings.allow]]` 並寫明理由；而名字長得像門檻的（樣式登記在這張卡的 [settings]，`MAX_*`、`*_TIMEOUT`、`*_DEPTH` 這些）就算值剛好等於離開碼那三個數，一樣要登記。②卡的 `human` 欄不准出現跟同一張卡 `[settings]`／`[junit]` 裡登記值相同的數字——人話重抄一次門檻，之後改的是卡上那一份，人話那一份不會跟著改，當場開始漂。判準一句：一個數字調寬了會讓紅變綠，它就是門檻，必須住在登記簿；調它只會讓紅更紅、或讓這一跑不算數，才准登記成例外。讀不到這張卡的 [settings]、掃描面裡一支 .py 都沒有、或某支 .py 剖不開，一律回 2 不回 0。
- **type-guard**（擋合併）：型別警衛，掃版控裡的每一支 .py（扣掉必紅樣本樹與本機工具目錄），兩層，第二層刻意獨立於 mypy。①`mypy` 嚴格模式跑那一整組檔：缺參數或回傳標註、型別對不上，一律紅。嚴格度只住在這張卡登記的那個設定檔的 `[tool.mypy]`（今天是 `pyproject.toml`），這支檢查自己不帶任何嚴格度旗標——設定檔不在、或 mypy 不在 PATH 上、或 mypy 自己崩掉（回一個不是「乾淨」也不是「有錯誤」的離開碼），一律回 2 不回 0；沒有尺、或量到一半死掉，「沒問題」這句話就不算數。②逃生門掃描，用 AST 判，**完全不看 mypy 的旗標與輸出**：標註位置（參數、回傳、變數標註、型別別名）出現 `Any` 就紅，巢狀的也算（`dict[str, Any]` 一樣咬）；`cast(Any, …)` 這個呼叫本身紅；`# type: ignore` 沒帶錯誤碼（方括號裡沒寫明是哪一種錯）紅；`from typing import *` 紅（那一行之後看不見綁定，量不到就不准當乾淨）。名字是**解析出來的、不是比字面**：先讀這份檔的 import 綁定，改名 import（`from typing import Any as 別的名字`）、屬性寫法（`import typing as t` 之後的 `t.Any`）、用賦值做出來的別名（接力幾手都追）全部算 `Any`，`cast` 的別名同一套判準。只比字面的版本會被「改個名字」整條繞過去，那是協調席實測戳出來的洞，迴歸在必紅樣本裡。為什麼第②層不能靠 mypy：嚴格模式那一包不含 `disallow_any_explicit`，所以 `-> Any` 在純嚴格模式底下合法通過；`warn_unused_ignores` 也只抓「多餘的抑制」，抓不到「真的在壓一個錯誤」的那種。**這張卡不管抑制註解的理由與到期日**——那兩格由 exemptions-need-expiry 守，兩張卡看同一行的不同格，刻意不重疊（兩張卡掃同一件事會互相遮蔽，那是 v2 事故 guard-teeth-shadow-each-other 的形狀）。真的需要 `Any`（讀進來的動態資料型別本來就未知）就在這張卡的 `[[settings.allow]]` 具名放行那個檔，寫理由與到期日。掃描面上一支 .py 都沒有、某支 .py 剖不開、讀不到這張卡的 `[settings]`，一律回 2 不回 0。
- **uv-single-entrypoint**（擋合併）：環境交給 uv 管，`uv run` 是唯一入口，三條。①命令位置不准直接叫直譯器與工具：`.github/workflows/*.yml` 的 `run:`（含 `run: |` 區塊）、版控裡的 `*.sh`／`Makefile`、`pyproject.toml` 裡放指令的表（表名結尾是 scripts／tasks），只要某個命令的第一個字是 `python`／`python3`／`pytest`／`mypy`／`ruff` 就紅；`uv run python …` 合法，`uv` 開頭的任何子命令（`uv sync`、`uv lock`）也合法。②Python 原始碼不准硬插模組搜尋路徑：AST 判 `sys.path.append`／`insert`／`extend` 與對 `sys.path` 本身的指派，不是字串比對。**一律紅，沒有放行的寫法**——摻進字串字面值、`os.environ`／`getenv`、cwd、`sys.argv`／`sys.prefix`，以及「路徑完全由 `__file__` 推出來」的自我定位（`Path(__file__).resolve().parents[2]`，經模組層的名字轉一手也算），四種都紅，只是訊息不同。要宣告模組搜尋路徑就寫進 `pyproject.toml` 的 `[tool.pytest.ini_options]` 的 `pythonpath`：那是設定，只有一份、走 PR 看得到，不是每支程式各自插一行。③鎖檔要在、同步要鎖死：`uv.lock` 必須進版控，掃描面裡至少要有一步 `uv sync`，而且每一處 `uv sync` 都得帶 `--locked`；只寫 `uv sync` 或改成 `--frozen` 都紅（`--frozen` 跳過鎖檔新鮮度檢查，鎖檔可以跟 pyproject 漂開，環境就又是變數了）。名單與門檻只寫在這張卡的 `[settings]`，讀不到、形狀不對、或 `.github/workflows` 底下一份 workflow 都沒有，一律回 2 不回 0。

<!-- rules:end -->
