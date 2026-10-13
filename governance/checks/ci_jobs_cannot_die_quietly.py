#!/usr/bin/env python3
"""雲端那一跑不准無聲死掉。

掃描面是掃描根自己那一層的 ``.github/workflows/*.yml``／``*.yaml``（不含樣本樹裡的道具），
加上 ``run:`` 呼叫、進得了版控的腳本，再加上版控裡的 ``governance/required-status-checks.txt``
（那份名單說哪幾個 job 擋得住合併）。門檻與名單全部只寫在卡的 ``[settings]`` 裡，
讀不到就回 2（工具自壞），不回 0。九條：

1. **紅了不准不擋**——任何 step 或 job 寫 ``continue-on-error: true`` 就紅。這是 GitHub 上
   把紅漂成綠最直接的一個鍵：那一步失敗了，job 照樣算成功，required check 照樣綠。
   ``${{ ... }}`` 那種算出來才知道的值也算紅——閘的死活不准取決於一個算出來的旗標。
2. **離開碼不准被吞掉**——``run:`` 的內容（以及它呼叫、進得了版控的腳本）出現卡上登記的
   ``swallow_snippets``（把離開碼或錯誤導掉的 shell 字樣）就紅；單獨一行 ``exit 0`` 也紅
   （效果跟接在分號後面一樣，只咬帶分號那種會漏）；沒有 pipefail 的多段管線也紅——GitHub
   在 Linux 上預設的 ``run:`` 是 ``bash -e {0}``，**沒有** pipefail，``cmd | tail`` 只回最後
   一段的離開碼，前面那段紅了看不見。作者自己寫了 pipefail、或把 shell 指成卡上登記的
   ``pipefail_shells``（``shell: bash`` 在 GitHub 上是 ``bash -eo pipefail {0}``）就放行。
   遞迴進腳本是找碴席點名的：只掃 yaml 字面的話，把 ``|| true`` 搬進版控裡的某支腳本
   就繞過去了。
3. **job 不准漂綠、不准沒有上限**——登記收尾的 job 層條件改以整串比對；其他 job 的 ``if:`` 出現卡上登記的 ``forbidden_job_ifs``
   （``always()``）就紅：前置失敗了它照跑，結論照樣算綠。``timeout-minutes`` 缺席一樣紅，
   沒有上限等於可以無聲卡死（GitHub 的預設 360 分鐘不是宣告，是沒人想過這件事）；
   有寫但超過卡上登記的 ``max_timeout_minutes``、或不是整數字面值，也紅。
4. **不准有空 job**——每個 job 至少要有一步真的在跑檢查或測試（``run:`` 的內容、或它呼叫
   的腳本，命中卡上登記的 ``work_markers``）。只有 checkout／setup 的 job 永遠綠，
   掛成 required check 就是一個只會回綠的閘；登記收尾另要求恰好一步包抄寫員、帶完整 needs 環境值、命令整串等於登記的 verdict_run。
5. **受保護集合 P，每一步都要留得下離開碼**——P 是必要名單、登記收尾與登記簿的分片；必要名單讀掃描根自己的
   ``governance/required-status-checks.txt``（版控裡那份「必須擋合併的 job 名」名單，
   ``rule-card-required-fields`` 讀的是同一份），**不是卡上另外登記一個名字**：同一個名字
   兩個家的話，job 一改名這一條就靜默停用。那份檔不在、讀不開、裡面一個名字都沒有，
   或列了名字而掃到的 workflow 裡沒有任何 job 叫那個名字，一律回 2——沒有對象就不出結論
   （同一跑有第⑧條的重複鍵時例外：剖析出來的結構不可信，先報重複鍵、另外印一句缺席的 job 名）。
   名單裡那些 job 底下每一個 ``run:``，``run: |`` 區塊裡的每一行命令也各算一個，而**一行裡
   用 ``&&``／``;``／``||``／``|`` 串起來的每一段又各自算一步**——要嘛那一段**開頭**就是卡上
   登記的 ``wrapper_command``（抄寫員：把那一步真實的離開碼記成一片收據、原封不動回那個
   離開碼的那一層），要嘛它的前幾個字命中卡上登記的 ``plumbing_first_words``
   （水管步驟：裝依賴、抓分支、搬檔案那種，本來就沒有判決可記）。
   命中水管的段落還要逐個 shell token（命令字）排除卡上登記的
   ``plumbing_forbidden_option_prefixes``（水管禁用選項前綴）：token 等於前綴、長選項用
   ``=`` 帶值、或短選項黏值都算；這能擋 apt 的執行前 hook 與外部設定檔。另依
   ``plumbing_forbidden_argument_markers`` 的具名比法排除本地套件參數：``token_contains``
   比 token 是否含有標記，``token_ends_with`` 比 token 是否以標記結尾。本地套件檔的安裝
   腳本（postinst）可以跑任意命令，所以不能當成沒有判決可記的水管。兩份禁單都只套用在
   命中水管的段，抄寫員段不受影響。
   為什麼要切段（2026-09-10 補的縫）：只判整行開頭的話，``uv sync --locked && 跑一支檢查``
   會因為開頭那一段是水管就整行放行，後面那支檢查的離開碼不會進收據，而卡面看起來守著。
   切段共用 :func:`ci_command_segments._split_outside_quotes`，引號裡的分隔符不切（``grep 'a|b'`` 只有一段），
   ``&&``／``||`` 排在單一個 ``|`` 前面所以邏輯或不會被當成管線切開。抄寫員那一段 ``--``
   之後是它要跑的子程序、離開碼由它記，不另外判；但 ``--`` 之後被分隔符切出來的下一段是
   shell 層的另一個命令，抄寫員管不到，照判。
   兩邊比的都是**命令開頭**，不是子字串：比子字串的話，一行 ``# governance.status.record_step``
   的行內註解、或把抄寫員塞在 ``--`` 後面，都能假裝包了；水管那半比子字串的話
   ``uv run python -m governance.checks.x`` 會因為開頭是 ``uv`` 就被放行。比對前先照 shell 的讀法
   斷行、接續行、剝註解（只認字首的 ``#``；引號裡、字中的不是註解），但**引號留著**
   （見 :func:`ci_command_segments._quoted_chunks`）；沒在同一行收尾的引號、字面外的 ``<<``（heredoc）、
   展開（``$(``、``${``、``$[``、反引號，雙引號裡的也算；``<(``、``>(`` 只認引號外）這一支不追，一律紅。
   **認不出的行一律紅（fail-closed，2026-09-10 補的縫）**：「殘骸」只認原始那一行去頭尾空白
   後是空的、或以 ``#`` 開頭那兩種；其他每一行都要分類得出來，分類不出來就紅。舊版拿剝掉
   引號的版本去切段，於是整行被一對引號包住的命令（``"blueprint/remap_cards.py"`` 這種腳本
   路徑、``"uv run python -m governance.checks.x --scan-root ."``、單引號版本）剝完是空字串，被當成
   殘骸跳過、段數 0、不紅——而 shell 真的會執行那一行，那是一條可用的繞道（issue #104）。
   同一條順手補上：一段命令在字面外有單一個 ``&``（丟到背景跑，不只結尾）一律紅，shell 不等它，
   離開碼一定不會被記，包了抄寫員也一樣。
   沒包又不是水管的那一步，在收據裡只剩雲端記的紅綠，0／1／2 三種結局分不開；2026-09-10 之前
   ruff 那一步就是這樣，理由只寫在 workflow 的註解裡，沒有機器在守。
   **只看受保護集合 P**：集合外的工作不套用⑤；收尾與分片的登記、唯一性、上傳覆蓋另由輔助模組守。
6. **重試次數要跟卡上登記的一樣**——任何一份 workflow 裡，只要某一步的 ``env:`` 有
   ``PUSH_MAX_ATTEMPTS`` 這個鍵（推機器分支撞到非快進時重疊上去再推幾次才放棄），它的值
   （yaml 寫成字串，先轉整數，轉不成回 2）必須**等於**卡上登記的 ``push_max_attempts``。
   為什麼這張卡管重試次數：重試用完那一步就回非零、那個 job 紅，所以它是「不准無聲死掉」的
   門檻不是裝飾——調小了收據會在還沒推上去之前就放棄，那一跑的證據跟著消失。
   刻意用**相等**不是 ``timeout-minutes`` 那種「不准超過上限」：太小會提早放棄、太大會讓
   撞車那一跑一直重推佔著 runner，兩邊都不對，所以只有一個值算數，改它就改卡、走 PR。

7. **受保護工作不能跳過或被過濾**——P 中的 job 不准有 job 層 ``if:`` 或 needs；登記收尾以整串條件和依賴集合驗證。所在 workflow
   必須有 ``on.pull_request``，且不能有卡上禁用的路徑或分支過濾鍵；事件類型必須包含卡上
   為該 job 登記的全部事件。名單中的 job 沒在卡上登記就回 2。PyYAML 把 ``on`` 讀成布林
   ``True`` 時也照樣讀到；這個語法差異不能讓整條規矩失效。

8. **同一層不准有重複的鍵**——YAML 剖析器遇到同一張表裡兩個同名的鍵，靜靜只留最後一個。
   #326 實見：新步驟插進別一步的 ``name:`` 與 ``run:`` 之間，那一步有兩個 ``run:``、前一步
   沒了 ``run:``，十三支檢查全綠。剖析成表之前，用 ``yaml.compose`` 的節點樹自己走一遍抓。

9. **每一步要嘛有 ``run:`` 要嘛有 ``uses:``**——兩個都沒有的步驟什麼都沒做，多半就是上面那種
   被擠走的結果。

**為什麼用 pyyaml 而不是自己剖析。** 這幾條要分得清 job 層與 step 層的同名鍵
（``continue-on-error`` 兩層都能寫，意思不同）、要把 ``timeout-minutes`` 讀成數字比大小、
要看得懂流式寫法與引號、還要拿到 ``run: |`` 區塊真正的內容（區塊摺疊符號由剖析器吃掉，
管線那一條才不會把 YAML 的 ``|`` 當成 shell 的管線）。縮排／正則的子集剖析每一種都繞得
過去，等於一個只會回綠的檢查。pyyaml 不在的時候這支檢查回 2，不會自己退回猜。

**「卡宣告的 job 必須存在且被呼叫」不在這裡**——那一關歸 ``rule-card-required-fields``
（見它的第一關三條交叉驗證）。兩張卡掃同一個對象會互相遮蔽，所以這裡不重複。

**血債那一半**：``guard-of-guards-silently-dies`` 只記關聯不算血債，理由寫在卡的
``related_lessons_why``——那筆事故的五個載體全是本機 hook，這張卡在事故當下咬不到；
留下來的是同一個失敗形狀換了載體（守衛無聲死掉、離開碼被吞掉），今天的載體是 workflow。

**已知的縫**（照抄不遮，見卡面「刻意沒管的事」）：cron 心跳那一段沒做（要上網）；
順序條款沒有機器判準；step 層的 ``if:`` 不管；``jobs.<id>.uses:`` 那種 job 會被第 4 條
誤判成空 job（今天零對象）；管線那一條把管線塞進 ``bash -c "..."`` 的字串就繞得過去
（那是第 2 條的洞，第 5 條會把整段判成認不出而紅）；第 5 條把 ``run: |`` 區塊當成一行一個
命令看，四種誤紅——抄寫員後面用 Tab 隔開參數、反斜線續行的第二行、``sh -c``／``bash -c``
包起來的那一整段、``FOO=1 前綴`` 那種環境變數前綴——四個都是零對象、而且都是紅得安全的
方向（2026-09-10 起第 5 條沒有已知的漏放）；第 6 條只看 step 層的 ``env:``。

**副作用，寫出來不遮**：整支檢查回 2（讀不到門檻、讀不到那份名單、workflow 剖不開）會蓋掉
同一跑其他五條的判決——那一跑就只有「這一跑不算數」一句話。這裡可以接受：2 一樣擋合併，
不會無聲，跟「回綠但其實什麼都沒掃」不是同一件事。
"""
from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

from governance.exit_codes import ToolBroken, note, run
from governance.checks.ci_command_segments import (
    ARGUMENT_MARKER_COMPARISONS,
    ARGUMENT_MARKERS_KEY,
    _command_lines,
    _split_outside_quotes,
    _receipt_step_problems,
)
from governance.checks.ci_required_gate import (
    FINALIZER_FIELDS, FINALIZER_KEY, Finalizer, finalizer_registry_paths,
    finalizer_problems, job_identity_problems, protected_ids_unique, read_finalizers,
    required_trigger_problems, step_if_problems, upload_overwrite_problems,
)
from governance.loader import RULES_DIR, setting_int, setting_strings

try:
    import yaml
except ImportError as exc:  # pyyaml 不在就回 2，不准退回猜
    # 刻意不寫 `yaml = None`：那會把「模組」這個名字指成 None，型別上是一個謊
    # （而且只能靠抑制註解壓下去）。缺不缺席由下面這個字串說，用的地方問它。
    YAML_IMPORT_ERROR = str(exc)
else:
    YAML_IMPORT_ERROR = ""

# 這支檢查在卡裡的名字。門檻只從「宣告了這支檢查」的那張卡讀。
CHECK_REL = "governance/checks/ci_jobs_cannot_die_quietly.py"

WORKFLOW_DIR = ".github/workflows"
WORKFLOW_SUFFIXES = (".yml", ".yaml")

# 「哪幾個 job 擋得住合併」的名單。這一份是版控裡的期望，rule-card-required-fields 讀的是
# 同一份——第 5 條適用哪幾個 job 就從這裡讀，不在卡上另外登記一次（同一個名字兩個家的話，
# job 一改名這一條就靜默停用，而卡面看起來還在守）。
REQUIRED_CHECKS_FILE = "governance/required-status-checks.txt"

# 第 6 條的對象：step 的 env 裡這個鍵。值是門檻，門檻住在卡上。
PUSH_ATTEMPTS_ENV = "PUSH_MAX_ATTEMPTS"

# 「這一步／這個 job 失敗了也不擋」的鍵。兩層都能寫，兩層都咬。
CONTINUE_KEY = "continue-on-error"
TIMEOUT_KEY = "timeout-minutes"

# ``run:`` 裡叫得動的腳本長什麼樣（只認進得了版控、掃描根底下的檔）。跟
# green-must-be-real-green 用同一個形狀。
SCRIPT_RE = re.compile(r"[\w./-]+\.(?:sh|bash|py)")

# 第 2 條認的管線：前後都不是 ``|`` 的那一個 ``|`` 才是管線，``||`` 是邏輯或不是管線。
PIPE_RE = re.compile(r"(?<!\|)\|(?!\|)")

# 門檻的形狀。打錯字的門檻等於沒有門檻，所以多一個鍵、少一個鍵、型別不對，一律回 2。
LIST_KEYS = (
    "swallow_snippets",
    "forbidden_job_ifs",
    "work_markers",
    "pipefail_markers",
    "pipefail_shells",
    "plumbing_first_words",
    "plumbing_forbidden_option_prefixes",
)
INT_KEYS = ("max_timeout_minutes", "push_max_attempts")
# 第 5 條的一格：抄寫員那一串命令（比的是命令開頭，所以登記的是整串命令不是一個字樣）。
# 「哪幾個 job 適用」刻意不在這裡——那一格的家是 governance/required-status-checks.txt。
TEXT_KEYS = ("wrapper_command", "required_child_prefix")
REQUIRED_EVENTS_KEY = "required_pull_request_events"
FORBIDDEN_FILTERS_KEY = "forbidden_pull_request_filters"
STEP_IF_KEY = "allowed_required_step_ifs"
SETTINGS_KEYS = (*LIST_KEYS, ARGUMENT_MARKERS_KEY, *INT_KEYS, *TEXT_KEYS, REQUIRED_EVENTS_KEY, FORBIDDEN_FILTERS_KEY, STEP_IF_KEY, FINALIZER_KEY)


def _card_files(scan_root: Path, files: list[Path]) -> list[Path]:
    """掃描面的一組：所有規矩卡。門檻從卡上讀，所以每一張都要打開找自己那一張。"""
    return sorted(f for f in files if f.parent == scan_root / RULES_DIR and f.suffix == ".toml")


def _workflow_files(scan_root: Path, files: list[Path]) -> list[Path]:
    """掃描面的一組：``.github/workflows/`` 底下進得了版控的 workflow。"""
    workflow_dir = scan_root / WORKFLOW_DIR
    return sorted(
        f for f in files if f.parent == workflow_dir and f.suffix.casefold() in WORKFLOW_SUFFIXES
    )


def _required_jobs(scan_root: Path, files: list[Path]) -> frozenset[str]:
    """第 5 條適用哪幾個 job：讀掃描根自己的 ``governance/required-status-checks.txt``。

    一行一個 job 名，``#`` 開頭的註解行與空行不算。那份檔不在版控裡、讀不開、或裡面一個
    名字都沒有，一律 raise :class:`ToolBroken`——沒有名單就沒有對象，不出結論。
    """
    path = scan_root / REQUIRED_CHECKS_FILE
    if path not in files:
        raise ToolBroken(
            f"版控裡沒有 {REQUIRED_CHECKS_FILE}——第 5 條適用哪幾個 job 就是從那份名單讀的，"
            "讀不到名單等於沒有對象，這一跑不算數"
        )
    names = [
        line.strip()
        for line in _read(path, REQUIRED_CHECKS_FILE).splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not names:
        raise ToolBroken(
            f"{REQUIRED_CHECKS_FILE} 裡一個 job 名都沒有（只有註解與空行）"
            "——空名單等於第 5 條沒在管，那比紅更糟，所以回 2"
        )
    return frozenset(names)


def targets(scan_root: Path, files: list[Path]) -> list[Path]:
    """這支檢查真的會讀的檔：workflow ＋ 所有規矩卡 ＋ 必要檢查名單 ＋ workflow 裡叫到的腳本。

    腳本那一組刻意用整份 workflow 的原文抓候選（不是只抓 ``run:`` 裡的），寧可把掃描面
    報大一點，也不要漏報——漏報的那個檔就是別人看不到的洞。
    名單那一份只在它進得了版控時列進來：不在的時候 :func:`_required_jobs` 會回 2，
    而卡上宣告的 scope 在那棵樹上同樣展開不出這個檔，兩邊還是相等的。
    """
    picked = [*_workflow_files(scan_root, files), *_card_files(scan_root, files)]
    required = scan_root / REQUIRED_CHECKS_FILE
    if required in files:
        picked.append(required)
    for rel in finalizer_registry_paths(_card_settings(scan_root, files)):
        if scan_root / rel in files:
            picked.append(scan_root / rel)
    for wf in _workflow_files(scan_root, files):
        rel = str(wf.relative_to(scan_root))
        for script_rel, _ in _called_scripts(_read(wf, rel), scan_root, files):
            picked.append(scan_root / script_rel)
    return sorted(set(picked))


def _card_settings(scan_root: Path, files: list[Path]) -> dict[str, object]:
    """從掃描根自己的 ``governance/rules/`` 讀這支檢查的門檻。

    找不到、找到多張、或門檻的形狀不對，一律 raise ToolBroken——沒有門檻就不出結論。
    """
    mine: list[tuple[Path, dict[str, object]]] = []
    for path in _card_files(scan_root, files):
        try:
            data = tomllib.loads(path.read_bytes().decode("utf-8"))
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            raise ToolBroken(f"讀不開 {path.relative_to(scan_root)}：{exc}") from exc
        if data.get("check") == CHECK_REL:
            mine.append((path, data))
    if len(mine) != 1:
        raise ToolBroken(
            f"{scan_root}/{RULES_DIR} 底下宣告 check={CHECK_REL} 的卡有 {len(mine)} 張，要剛好 1 張"
            "——門檻只寫在卡上，讀不到門檻這一跑就不算數"
        )
    path, data = mine[0]
    rel = str(path.relative_to(scan_root))
    settings = data.get("settings")
    if not isinstance(settings, dict):
        raise ToolBroken(f"{rel} 沒有 [settings] 表（上限與名單都寫在那裡）")
    _settings_problems(settings, rel)
    _finalizer_settings_problems(settings, rel)
    return settings


def _finalizer_settings_problems(settings: dict[str, object], rel: str) -> None:
    """獨立驗設定，既有設定驗證已到行數上限；判準字串只從卡讀。"""
    entries = settings.get(FINALIZER_KEY)
    if not isinstance(entries, dict):
        raise ToolBroken(f"{rel} 的 {FINALIZER_KEY} 必須是表（沒有例外也要明寫空表）")
    for job_id, entry in entries.items():
        if not isinstance(job_id, str) or not job_id.strip() or not isinstance(entry, dict):
            raise ToolBroken(f"{rel} 的 {FINALIZER_KEY} 必須是非空 job id 到設定表")
        if set(entry) != set(FINALIZER_FIELDS) or not all(
            isinstance(entry.get(key), str) and entry[key].strip() for key in FINALIZER_FIELDS
        ):
            raise ToolBroken(f"{rel} 的 {FINALIZER_KEY}.{job_id} 必須剛好有非空整串設定 {list(FINALIZER_FIELDS)}")
        registry = Path(entry["registry"])
        if registry.is_absolute() or ".." in registry.parts:
            raise ToolBroken(f"{rel} 的 {FINALIZER_KEY}.{job_id}.registry 必須指向掃描根裡的相對路徑")


def _settings_problems(settings: dict[str, object], rel: str) -> None:
    bad: list[str] = []
    extra = [k for k in settings if k not in SETTINGS_KEYS]
    if extra:
        bad.append(f"多了不認識的鍵 {sorted(extra)}，只認 {list(SETTINGS_KEYS)}（打錯字的門檻等於沒門檻）")
    for key in LIST_KEYS:
        value = settings.get(key)
        if not isinstance(value, list) or not all(isinstance(x, str) and x.strip() for x in value):
            bad.append(f"{key} 必須是字串 list，實際是 {value!r}")
        elif not value:
            bad.append(f"{key} 不准是空 list——空的名單等於這一條沒在管")
    events = settings.get(REQUIRED_EVENTS_KEY)
    if not isinstance(events, dict) or not events or not all(
        isinstance(job, str) and job and isinstance(names, list) and names
        and all(isinstance(name, str) and name for name in names)
        for job, names in events.items()
    ):
        bad.append(f"{REQUIRED_EVENTS_KEY} 必須是 job 到非空事件名單的表，實際是 {events!r}")
    filters = settings.get(FORBIDDEN_FILTERS_KEY)
    if not isinstance(filters, list) or not filters or not all(
        isinstance(name, str) and name for name in filters
    ):
        bad.append(f"{FORBIDDEN_FILTERS_KEY} 必須是非空鍵名單，實際是 {filters!r}")
    step_ifs = settings.get(STEP_IF_KEY)
    if not isinstance(step_ifs, dict) or not all(
        isinstance(job, str) and isinstance(steps, dict) and all(
            isinstance(label, str) and isinstance(expr, str) and label and expr
            for label, expr in steps.items()
        ) for job, steps in step_ifs.items()
    ):
        bad.append(f"{STEP_IF_KEY} 必須是 job 到步驟名與 if 值的表，實際是 {step_ifs!r}")
    marker_table = settings.get(ARGUMENT_MARKERS_KEY)
    if not isinstance(marker_table, dict):
        bad.append(
            f"{ARGUMENT_MARKERS_KEY} 必須是表，底下只准有"
            f" {list(ARGUMENT_MARKER_COMPARISONS)}，實際是 {marker_table!r}"
        )
    else:
        marker_keys = set(marker_table)
        expected_marker_keys = set(ARGUMENT_MARKER_COMPARISONS)
        if marker_keys != expected_marker_keys:
            bad.append(
                f"{ARGUMENT_MARKERS_KEY} 必須剛好有 {list(ARGUMENT_MARKER_COMPARISONS)}，"
                f"實際是 {sorted(marker_keys)}"
            )
        for comparison in ARGUMENT_MARKER_COMPARISONS:
            markers = marker_table.get(comparison)
            if (
                not isinstance(markers, list)
                or not markers
                or not all(isinstance(marker, str) and marker for marker in markers)
            ):
                bad.append(
                    f"{ARGUMENT_MARKERS_KEY}.{comparison} 必須是非空字串 list，實際是 {markers!r}"
                )
    for key in TEXT_KEYS:
        text = settings.get(key)
        if not isinstance(text, str) or not text.strip():
            bad.append(f"{key} 必須是非空字串，實際是 {text!r}——讀不到判準這一條就等於沒在管")
    for key in INT_KEYS:
        number = settings.get(key)
        if not isinstance(number, int) or isinstance(number, bool) or number < 1:
            bad.append(
                f"{key} 必須是 1 以上的整數（門檻的家在卡上，這裡讀不到就沒有尺），實際是 {number!r}"
                "——上限寫成 0 或不寫等於沒有上限，重試次數寫成 0 等於不准重試"
            )
    if bad:
        raise ToolBroken(f"{rel} 的 [settings] 形狀不對：" + "；".join(bad))


def _read(path: Path, rel: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ToolBroken(f"讀不到 {rel}：{exc}") from exc


def _workflow(path: Path, rel: str) -> dict[str, object]:
    """剖析一份 workflow。看不懂的一律回 2，不准假裝乾淨也不准當成違規。"""
    text = _read(path, rel)
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ToolBroken(f"{rel} 不是解得開的 YAML（{exc}）——我沒看懂就不出結論") from exc
    if not isinstance(data, dict):
        raise ToolBroken(f"{rel} 剖析出來不是一張表（實際是 {type(data).__name__}），我看不懂")
    jobs = data.get("jobs")
    if not isinstance(jobs, dict) or not jobs:
        raise ToolBroken(f"{rel} 的 jobs: 解不出任何 job（實際是 {jobs!r}），這份 workflow 我看不懂")
    return data


def _duplicate_key_problems(rel: str, text: str) -> list[str]:
    """第⑧條：同一層重複的鍵。剖析器會靜靜只留最後一個——新步驟插進別一步的 name: 與 run: 之間，那一步就有兩個
    run:、前一步沒了 run:（#326 實見）。剖析成 dict 之前，自己走一遍節點樹抓。"""
    try:
        root = yaml.compose(text, Loader=yaml.SafeLoader)
    except yaml.YAMLError as exc:
        raise ToolBroken(f"{rel} 不是解得開的 YAML（{exc}）——我沒看懂就不出結論") from exc
    pending: list[yaml.Node] = [] if root is None else [root]
    found: list[tuple[int, str]] = []
    # 別名（*名字）在節點樹裡是同一個物件：不記走過的，指回自己的別名會永遠走不完、層層套的別名會按指數長（複查）。
    visited: set[int] = set()
    while pending:
        node = pending.pop()
        if id(node) in visited:
            continue
        visited.add(id(node))
        if isinstance(node, yaml.SequenceNode):
            pending.extend(node.value)
        if not isinstance(node, yaml.MappingNode):
            continue
        seen: dict[str, int] = {}
        for key, value in node.value:
            pending.append(value)
            if not isinstance(key, yaml.ScalarNode):
                continue
            line = key.start_mark.line + 1
            if key.value in seen:
                found.append((line, f"{rel} 第 {line} 行：同一層重複的鍵 {key.value!r}（第 {seen[key.value]} 行已經有了）"
                                    "——剖析只留最後一個，前一個靜靜不見"))
            seen.setdefault(key.value, line)
    return [message for _, message in sorted(found)]


def _step_shape_problems(step_where: str, step: dict[str, object]) -> list[str]:
    """第⑨條：每一步要嘛有 run: 要嘛有 uses:。"""
    if "run" in step or "uses" in step:
        return []
    return [f"{step_where} 既沒有 run: 也沒有 uses:——這一步什麼都沒做；多半是別一步插進來時把它的 run: "
            "擠成重複鍵、剖析只留了別人的（#326）"]


def _default_shell(holder: object) -> str:
    """``defaults.run.shell``。沒寫就是空字串（GitHub 在 Linux 上的預設是 bash -e，沒有 pipefail）。"""
    if not isinstance(holder, dict):
        return ""
    defaults = holder.get("defaults")
    if not isinstance(defaults, dict):
        return ""
    block = defaults.get("run")
    if not isinstance(block, dict):
        return ""
    shell = block.get("shell")
    return shell if isinstance(shell, str) else ""


def _continue_problems(where: str, holder: dict[str, object]) -> list[str]:
    """第 1 條：``continue-on-error`` 只准缺席或明寫 false。"""
    if CONTINUE_KEY not in holder:
        return []
    value = holder[CONTINUE_KEY]
    if value is False:
        return []
    return [
        f"{where} 寫了 {CONTINUE_KEY}: {value!r}"
        "——那一步／那個 job 紅了也不擋，job 照樣算成功、required check 照樣綠。"
        "雲端那一跑是唯一權威（docs/decisions/cloud-run-authority-and-receipts.md），"
        "權威自己把紅漂成綠，等於沒有閘"
    ]


def _pipelines(body: str) -> list[str]:
    """多段管線的那幾行。行由 :func:`ci_command_segments._command_lines` 斷（跨行引號、接續行、註解跟第 5 條
    同一把尺），只在非字面的片段依單一個 ``|`` 切段，切得出兩段以上才算。不先剝掉字面再重掃：
    ``uv sync ''#x | 命令`` 剝完是 ``uv sync #x | 命令``，字中的 ``#`` 跑到字首、管線被當註解丟掉。"""
    return [line for line in _command_lines(body) if len(_split_outside_quotes(line, PIPE_RE)) > 1]


def _swallow_problems(
    where: str,
    body: str,
    settings: dict[str, object],
    shell: str,
) -> list[str]:
    """第 2 條：離開碼不准被吞掉（``run:`` 的內容，以及它呼叫的腳本，用同一把尺）。"""
    bad: list[str] = []
    for snippet in setting_strings(settings, "swallow_snippets"):
        if snippet in body:
            bad.append(
                f"{where} 把離開碼吞掉了（命中卡上登記的 {snippet!r}）"
                "——檢查回 1 也會變成 shell 回 0，紅的被漂成綠的"
            )
    for line in body.splitlines():
        if line.strip() == "exit " + "0":
            bad.append(
                f"{where} 有單獨一行 `exit 0`"
                "——前面那幾行的離開碼被它整個蓋掉，跟接在分號後面是同一回事"
            )
            break
    markers = [m for m in setting_strings(settings, "pipefail_markers") if m in body]
    shells = setting_strings(settings, "pipefail_shells")
    # 刻意比整串、不比第一個詞：GitHub 只有在 shell 寫成關鍵字（就是 `bash` 這個字）時才幫你
    # 加 -eo pipefail；寫成自訂命令（`bash -e {0}`）它照字面跑，沒有 pipefail。比第一個詞會把
    # 後者誤判成合規。
    if not markers and shell.strip() not in shells:
        pipes = _pipelines(body)
        if pipes:
            bad.append(
                f"{where} 用了沒有 pipefail 的多段管線（{pipes[0]!r}）"
                "——GitHub 在 Linux 上預設的 run: 是 `bash -e {0}`，沒有 pipefail，"
                "管線只回最後一段的離開碼，前面那段紅了看不見。"
                f"要嘛自己開 pipefail，要嘛把 shell 指成卡上登記的 {list(shells)}"
            )
    return bad


def _called_scripts(body: str, scan_root: Path, files: list[Path]) -> list[tuple[str, str]]:
    """``run:`` 裡叫到、進得了版控的腳本：[(相對路徑, 內容)]。"""
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for token in SCRIPT_RE.findall(body):
        rel = token.lstrip("./")
        if rel in seen:
            continue
        path = scan_root / rel
        if path not in files or not path.is_file():
            continue
        seen.add(rel)
        out.append((rel, _read(path, rel)))
    return out


def _push_attempts_problems(
    step_where: str, step: dict[str, object], settings: dict[str, object]
) -> list[str]:
    """第 6 條：這一步的 ``env:`` 若登記了重試次數，值必須等於卡上那個數。"""
    env = step.get("env")
    if not isinstance(env, dict) or PUSH_ATTEMPTS_ENV not in env:
        return []
    raw = env[PUSH_ATTEMPTS_ENV]
    if isinstance(raw, bool) or not isinstance(raw, str | int):
        raise ToolBroken(
            f"{step_where} 的 env.{PUSH_ATTEMPTS_ENV} 是 {raw!r}，我讀不成一個整數"
            "——讀不到那一步實際用的次數就不出結論"
        )
    try:
        actual = int(str(raw).strip())
    except ValueError as exc:
        raise ToolBroken(
            f"{step_where} 的 env.{PUSH_ATTEMPTS_ENV} 是 {raw!r}，轉不成整數（{exc}）"
            "——算出來才知道的次數等於沒有登記，這一跑不算數"
        ) from exc
    expected = setting_int(settings, "push_max_attempts")
    if actual == expected:
        return []
    return [
        f"{step_where} 的 env.{PUSH_ATTEMPTS_ENV} 是 {actual}，卡上登記的是 {expected}"
        "——重試用完那一步就回非零、那個 job 紅，所以這是「不准無聲死掉」的門檻："
        "調小了收據會在推上去之前先放棄，那一跑的證據跟著消失。"
        "門檻只有一個家（卡的 [settings]），要改就改卡、走 PR"
    ]


def _timeout_problems(where: str, job: dict[str, object], cap: int) -> list[str]:
    """第 3 條後半：``timeout-minutes`` 必須在場，而且是不超過上限的整數字面值。"""
    if TIMEOUT_KEY not in job:
        return [
            f"{where} 沒有 {TIMEOUT_KEY}"
            "——沒有上限等於可以無聲卡死。GitHub 的預設 360 分鐘不是宣告，是沒人想過這件事；"
            f"上限只寫在卡的 [settings]（今天是 {cap} 分鐘）"
        ]
    value = job[TIMEOUT_KEY]
    if isinstance(value, bool) or not isinstance(value, int):
        return [
            f"{where} 的 {TIMEOUT_KEY} 是 {value!r}，必須是整數字面值"
            "——算出來才知道的上限等於沒有上限（算錯就回到 GitHub 的預設）"
        ]
    if value < 1:
        return [f"{where} 的 {TIMEOUT_KEY} 是 {value}，必須是 1 以上的分鐘數"]
    if value > cap:
        return [
            f"{where} 的 {TIMEOUT_KEY} 是 {value}，超過卡上登記的上限 {cap} 分鐘"
            "——開得比實跑時間大太多等於沒有上限，要改寬就改卡、走 PR"
        ]
    return []


def _if_problems(where: str, job: dict[str, object], markers: list[str]) -> list[str]:
    """第 3 條前半：job 的 ``if:`` 不准把紅漂成綠。"""
    raw = job.get("if")
    if not isinstance(raw, str):
        return []
    bad: list[str] = []
    for marker in markers:
        if marker in raw:
            bad.append(
                f"{where} 的 if: 寫了 {raw!r}（命中卡上登記的 {marker!r}）"
                "——前置失敗了它照跑，結論還是算綠，紅被漂成綠"
            )
    return bad


def _job_problems(
    rel: str,
    name: str,
    job: object,
    wf_shell: str,
    settings: dict[str, object],
    scan_root: Path,
    files: list[Path],
    required_jobs: frozenset[str], finalizers: dict[str, Finalizer],
) -> list[str]:
    where = f"{rel} 的 job {name!r}"
    if not isinstance(job, dict):
        raise ToolBroken(f"{where} 剖析出來不是一張表（實際是 {type(job).__name__}），我看不懂")

    bad: list[str] = []
    bad += _continue_problems(where, job)
    bad += [] if name in finalizers else _if_problems(where, job, setting_strings(settings, "forbidden_job_ifs"))
    if name in required_jobs and "if" in job and name not in finalizers:
        bad.append(f"{where} 是必要檢查卻有 job 層 if:——條件跳過會回報 Success，沒驗就亮綠")
    bad += job_identity_problems(where, name, job, required_jobs, finalizers.get(name))
    bad += _timeout_problems(where, job, setting_int(settings, "max_timeout_minutes"))

    # 第 5 條對受保護集合 P 成立：必要名單、卡上登記的收尾與分片，
    # 全部由 check 傳進來，沒有登記的工作不多開例外。
    is_receipt_job = name in required_jobs
    job_shell = _default_shell(job) or wf_shell
    steps = job.get("steps")
    if not isinstance(steps, list) or not steps:
        return bad + [
            f"{where} 底下沒有任何步驟（steps={steps!r}）"
            "——空 job 永遠綠，掛成 required check 就是一個只會回綠的閘"
        ]

    does_work = name in finalizers  # 收尾的判決步驟改由完整形狀驗證，缺席只報該條。
    markers = setting_strings(settings, "work_markers")
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            raise ToolBroken(f"{where} 第 {index + 1} 步剖析出來不是一張表（{step!r}），我看不懂")
        raw_label = step.get("name")
        label = raw_label if isinstance(raw_label, str) else f"第 {index + 1} 步"
        step_where = f"{where} 的{label}"
        bad += _continue_problems(step_where, step)
        if is_receipt_job:
            bad += step_if_problems(step_where, name, label, step, settings[STEP_IF_KEY], finalizers.get(name))
        bad += _push_attempts_problems(step_where, step, settings)
        bad += _step_shape_problems(step_where, step)

        body = step.get("run")
        if not isinstance(body, str):
            continue
        shell = step.get("shell") if isinstance(step.get("shell"), str) else job_shell
        bad += _swallow_problems(f"{step_where} 的 run:", body, settings, shell or "")
        if is_receipt_job:
            bad += _receipt_step_problems(f"{step_where} 的 run:", body, settings)
        if any(marker in body for marker in markers):
            does_work = True
        for script_rel, text in _called_scripts(body, scan_root, files):
            bad += _swallow_problems(f"{step_where} 呼叫的腳本 {script_rel}", text, settings, "")
            if any(marker in text for marker in markers):
                does_work = True

    if not does_work:
        bad.append(
            f"{where} 沒有任何一步在跑檢查或測試（run: 的內容與它呼叫的腳本都沒命中"
            f"卡上登記的 {markers}）——只有 checkout／setup 的 job 永遠綠，等於沒有檢查"
        )
    return bad


def check(scan_root: Path, files: list[Path]) -> list[str]:
    if YAML_IMPORT_ERROR:
        raise ToolBroken(
            f"pyyaml 不在這個環境裡（{YAML_IMPORT_ERROR}）——這張卡要對 workflow 做真的 YAML 剖析，"
            "沒有剖析器就不出結論。它登記在 pyproject.toml 的正式依賴裡，跑 `uv sync --locked`"
        )
    settings = _card_settings(scan_root, files)
    required_jobs = _required_jobs(scan_root, files)

    workflows = _workflow_files(scan_root, files)
    if not workflows:
        raise ToolBroken(
            f"{scan_root}/{WORKFLOW_DIR} 底下一份版控裡的 workflow 都沒有"
            "——沒掃到 workflow 不等於沒有違規，這一跑不算數"
        )

    # 先把每一份剖開、把 job 名收齊，再下判斷：名單上的 job 一個都掃不到的時候，正確答案是
    # 「這一跑不算數」（2），不是「掃過了、很乾淨」（0）——沒有對象就不出結論。
    parsed: list[tuple[str, dict[str, object], dict[str, object]]] = []
    seen_jobs: set[str] = set()
    duplicates: list[str] = []
    for path in workflows:
        rel = str(path.relative_to(scan_root))
        data = _workflow(path, rel)
        # 第⑧條緊接剖析就判：重複的是頂層 jobs: 這種，晚一點會先撞上「名單上的 job 掃不到」，把人帶錯方向（複查）。
        duplicates += _duplicate_key_problems(rel, _read(path, rel))
        jobs = data["jobs"]
        if not isinstance(jobs, dict):
            # _workflow 已經驗過這一格是一張非空的表；寫出來是為了讓型別看得見，
            # 而且形狀真的壞掉的時候是回 2（這一跑不算數），不是炸出追蹤訊息。
            raise ToolBroken(f"{rel} 的 jobs: 不是一張表（實際是 {jobs!r}），我看不懂")
        seen_jobs |= {str(name) for name in jobs}
        parsed.append((rel, data, jobs))

    missing = sorted(required_jobs - seen_jobs)
    # 有重複鍵時剖析出來的結構不可信（頂層 jobs: 重複，第一塊裡的 job 就不見了）：先報重複鍵，不丟「掃不到」。
    if missing and duplicates:
        note(f"{REQUIRED_CHECKS_FILE} 列的 {missing} 在掃到的 workflow 裡找不到；這一跑有重複鍵、剖析出來的結構不可信，"
             "先修重複鍵，下一跑再看這幾個是不是真的缺席")
    if missing and not duplicates:
        raise ToolBroken(
            f"{REQUIRED_CHECKS_FILE} 列了 {missing}，可是掃到的 workflow 裡沒有任何 job 叫這些名字"
            f"（掃到的是 {sorted(seen_jobs)}）——第 5 條在這棵樹上一個對象都沒有，"
            "那不是乾淨，是量錯了對象：名單改了 workflow 沒跟上，或反過來"
        )

    if duplicates:
        return duplicates  # 結構已不可信，先報⑧；登記簿與其他判準下一跑再判。
    finalizers = read_finalizers(scan_root, files, settings)
    protected = required_jobs | frozenset(finalizers) | frozenset(s for f in finalizers.values() for s in f.shards)
    bad = protected_ids_unique(parsed, protected)
    for rel, data, jobs in parsed:
        bad += required_trigger_problems(rel, data, jobs, protected, settings)
        bad += finalizer_problems(rel, jobs, finalizers, settings)
        bad += upload_overwrite_problems(rel, jobs, protected)
        wf_shell = _default_shell(data)
        for name, job in jobs.items():
            bad += _job_problems(
                rel, str(name), job, wf_shell, settings, scan_root, files, protected, finalizers
            )
    return bad


if __name__ == "__main__":
    sys.exit(
        run(
            check,
            description=(
                "雲端工作不准無聲死掉：吞離開碼、漂綠、沒有上限、空 job、"
                "擋合併工作每步留收據、重試次數一致、必要檢查不能跳過或過濾、不准重複鍵、"
                "每步要有 run 或 uses，一律守住"
            ),
            targets=targets,
        )
    )
