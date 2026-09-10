#!/usr/bin/env python3
"""雲端那一跑不准無聲死掉。

掃描面是掃描根自己那一層的 ``.github/workflows/*.yml``／``*.yaml``（不含樣本樹裡的道具），
加上 ``run:`` 呼叫、進得了版控的腳本，再加上版控裡的 ``governance/required-status-checks.txt``
（那份名單說哪幾個 job 擋得住合併）。門檻與名單全部只寫在卡的 ``[settings]`` 裡，
讀不到就回 2（工具自壞），不回 0。六條：

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
3. **job 不准漂綠、不准沒有上限**——job 的 ``if:`` 出現卡上登記的 ``forbidden_job_ifs``
   （``always()``）就紅：前置失敗了它照跑，結論照樣算綠。``timeout-minutes`` 缺席一樣紅，
   沒有上限等於可以無聲卡死（GitHub 的預設 360 分鐘不是宣告，是沒人想過這件事）；
   有寫但超過卡上登記的 ``max_timeout_minutes``、或不是整數字面值，也紅。
4. **不准有空 job**——每個 job 至少要有一步真的在跑檢查或測試（``run:`` 的內容、或它呼叫
   的腳本，命中卡上登記的 ``work_markers``）。只有 checkout／setup 的 job 永遠綠，
   掛成 required check 就是一個只會回綠的閘。
5. **擋得住合併的那幾個 job，每一步都要留得下離開碼**——哪幾個 job 適用，讀的是掃描根自己的
   ``governance/required-status-checks.txt``（版控裡那份「必須擋合併的 job 名」名單，
   ``rule-card-required-fields`` 讀的是同一份），**不是卡上另外登記一個名字**：同一個名字
   兩個家的話，job 一改名這一條就靜默停用。那份檔不在、讀不開、裡面一個名字都沒有，
   或列了名字而掃到的 workflow 裡沒有任何 job 叫那個名字，一律回 2——沒有對象就不出結論。
   名單裡那些 job 底下每一個 ``run:``，``run: |`` 區塊裡的每一行命令也各算一個，而**一行裡
   用 ``&&``／``;``／``||``／``|`` 串起來的每一段又各自算一步**——要嘛那一段**開頭**就是卡上
   登記的 ``wrapper_command``（抄寫員：把那一步真實的離開碼記成一片收據、原封不動回那個
   離開碼的那一層），要嘛它的前幾個字命中卡上登記的 ``plumbing_first_words``
   （水管步驟：裝依賴、抓分支、搬檔案那種，本來就沒有判決可記）。
   為什麼要切段（2026-09-10 補的縫）：只判整行開頭的話，``uv sync --locked && 跑一支檢查``
   會因為開頭那一段是水管就整行放行，後面那支檢查的離開碼不會進收據，而卡面看起來守著。
   切段共用 :func:`_split_outside_quotes`，引號裡的分隔符不切（``grep 'a|b'`` 只有一段），
   ``&&``／``||`` 排在單一個 ``|`` 前面所以邏輯或不會被當成管線切開。抄寫員那一段 ``--``
   之後是它要跑的子程序、離開碼由它記，不另外判；但 ``--`` 之後被分隔符切出來的下一段是
   shell 層的另一個命令，抄寫員管不到，照判。
   兩邊比的都是**命令開頭**，不是子字串：比子字串的話，一行 ``# governance.status.record_step``
   的行內註解、或把抄寫員塞在 ``--`` 後面，都能假裝包了；水管那半比子字串的話
   ``uv run python -m governance.checks.x`` 會因為開頭是 ``uv`` 就被放行。比對前先剝掉
   **引號外**的 ``#`` 之後的內容（引號裡的 ``#``、``echo "a#b"`` 那種，不是註解），
   但**引號留著**（見 :func:`_command_text`）。
   **認不出的行一律紅（fail-closed，2026-09-10 補的縫）**：「殘骸」只認原始那一行去頭尾空白
   後是空的、或以 ``#`` 開頭那兩種；其他每一行都要分類得出來，分類不出來就紅。舊版拿剝掉
   引號的版本去切段，於是整行被一對引號包住的命令（``"blueprint/remap_cards.py"`` 這種腳本
   路徑、``"uv run python -m governance.checks.x --scan-root ."``、單引號版本）剝完是空字串，被當成
   殘骸跳過、段數 0、不紅——而 shell 真的會執行那一行，那是一條可用的繞道（issue #104）。
   同一條順手補上：一段命令以 ``&`` 結尾（丟到背景跑）一律紅，shell 不等它，離開碼一定不會
   被記，包了抄寫員也一樣。
   沒包又不是水管的那一步，在收據裡只剩雲端記的紅綠，0／1／2 三種結局分不開；2026-09-10 之前
   ruff 那一步就是這樣，理由只寫在 workflow 的註解裡，沒有機器在守。
   **只看那幾個 job**：同一份 workflow 裡別的 job、別的 workflow（狀態頁那一份、票務守衛
   那一份）一律不管——那些 job 本來就在做別的事，包抄寫員沒有意義。
6. **重試次數要跟卡上登記的一樣**——任何一份 workflow 裡，只要某一步的 ``env:`` 有
   ``PUSH_MAX_ATTEMPTS`` 這個鍵（推機器分支撞到非快進時重疊上去再推幾次才放棄），它的值
   （yaml 寫成字串，先轉整數，轉不成回 2）必須**等於**卡上登記的 ``push_max_attempts``。
   為什麼這張卡管重試次數：重試用完那一步就回非零、那個 job 紅，所以它是「不准無聲死掉」的
   門檻不是裝飾——調小了收據會在還沒推上去之前就放棄，那一跑的證據跟著消失。
   刻意用**相等**不是 ``timeout-minutes`` 那種「不准超過上限」：太小會提早放棄、太大會讓
   撞車那一跑一直重推佔著 runner，兩邊都不對，所以只有一個值算數，改它就改卡、走 PR。

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

from governance.exit_codes import ToolBroken, run
from governance.loader import RULES_DIR, setting_int, setting_strings, setting_text

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

# 認引號用的：**只有這一把尺**，第 2 條（管線）與第 5 條（切段、分類）認引號認的都是它，
# 兩份會各自漂。差別在拿它做什麼：第 2 條把引號裡的內容整段剝掉（``grep 'a|b'`` 不是管線），
# 第 5 條只拿它標出「哪一段是引號裡面」，內容原封不動留著（見 :func:`_command_text`）。
QUOTED_RE = re.compile(r"'[^']*'|\"[^\"]*\"")

# 一段命令丟到背景跑的尾巴。``&`` 結尾的那一段 shell 不等它，離開碼一定不會被記，
# 所以第 5 條一律紅——包了抄寫員也一樣（抄寫員本人被丟到背景，收據跟著沒人等）。
BACKGROUND_SUFFIX = "&"

# 第 5 條把一行 shell 切成一段一段用的分隔符。正則的交替是**有序**的，所以 ``&&`` 與 ``||``
# 排在單一個 ``|`` 前面——邏輯或先被吃掉，不會被當成管線切開（舊版靠一個佔位字元做同一件事，
# 現在切段與管線共用 :func:`_split_outside_quotes`，佔位字元跟著退休）。
SEGMENT_SEP_RE = re.compile(r"&&|\|\||;|\|")

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
)
INT_KEYS = ("max_timeout_minutes", "push_max_attempts")
# 第 5 條的一格：抄寫員那一串命令（比的是命令開頭，所以登記的是整串命令不是一個字樣）。
# 「哪幾個 job 適用」刻意不在這裡——那一格的家是 governance/required-status-checks.txt。
TEXT_KEYS = ("wrapper_command",)
SETTINGS_KEYS = (*LIST_KEYS, *INT_KEYS, *TEXT_KEYS)


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
    return settings


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
        "雲端那一跑是唯一權威（docs/decisions/cloud-run-is-the-only-authority.md），"
        "權威自己把紅漂成綠，等於沒有閘"
    ]


def _quoted_chunks(text: str) -> list[tuple[str, bool]]:
    """把一行 shell 切成 ``(片段, 這一段在引號裡嗎)``，順序原封不動。

    引號認的是 :data:`QUOTED_RE`。剝註解（:func:`_command_text`）與切段
    （:func:`_split_outside_quotes`）都從這裡拿同一份切法，兩邊才不會對「哪裡是引號裡面」
    有兩種看法。
    """
    chunks: list[tuple[str, bool]] = []
    cursor = 0
    for quoted in QUOTED_RE.finditer(text):
        chunks.append((text[cursor : quoted.start()], False))
        chunks.append((quoted.group(0), True))
        cursor = quoted.end()
    chunks.append((text[cursor:], False))
    return chunks


def _strip_comment(raw: str) -> str:
    """一行 shell 剝掉引號裡的內容、再剝掉行內註解（引號外第一個 ``#`` 之後全丟）。

    **第 2 條（管線）專用**。它要判的是「這一行有沒有多段管線」，引號裡的 ``|`` 不算管線，
    整段剝掉最省事。第 5 條**不能**用這一把尺：剝掉引號之後
    ``"uv run python -m governance.checks.x --scan-root ."``（整行被一對引號包住，shell 照樣
    會執行它）會變成空字串，於是「沒有命令就沒有離開碼可記」把它當殘骸跳過——那是 2026-09-10
    修掉的漏放。第 5 條改用 :func:`_command_text`（保留引號）。
    引號裡的 ``#``（``echo "a#b"``）不是註解，所以先剝引號再切。
    """
    return QUOTED_RE.sub("", raw).split("#", 1)[0]


def _command_text(raw: str) -> str:
    """一行 shell 剝掉行內註解，**引號原封不動留著**（第 5 條分類用的那一把尺）。

    跟 :func:`_strip_comment` 的差別只有一件事：引號裡的內容留著。留著才分得出「這一段是不是
    整行被引號包住」——被包住的那一段不是空的、也不是抄寫員、也不是水管，所以第 5 條判它紅
    （fail-closed：認不出就紅）。``#`` 在引號裡面（``echo "a#b"``）不是註解，所以只在引號
    外面的片段裡找第一個 ``#``。
    """
    kept: list[str] = []
    for chunk, is_quoted in _quoted_chunks(raw):
        if is_quoted:
            kept.append(chunk)
            continue
        if "#" in chunk:
            kept.append(chunk.split("#", 1)[0])
            break
        kept.append(chunk)
    return "".join(kept)


def _split_outside_quotes(text: str, separator: re.Pattern[str]) -> list[str]:
    """依 ``separator`` 把一行 shell 切段，引號裡的分隔符不算分隔符。

    引號認的是 :data:`QUOTED_RE`（經 :func:`_quoted_chunks`）：引號那一段原封不動接回去，
    只有引號外面命中的才切，所以 ``grep 'a|b'``、``sh -c "a && b"`` 都只有一段。
    回來的每一段去掉頭尾空白、空的一律丟掉——``a |  | b`` 中間那一段沒有命令。
    這裡丟掉空段**不是**放行：第 5 條在外面守著「一行切完一段都不剩就紅」（見
    :func:`_receipt_step_problems`），空段從此不會變成一個安靜的放行。

    **引號追蹤是有對象的，不是多餘的防禦**：第 5 條餵進來的是 :func:`_command_text` 的產物
    （引號留著），``sh -c "a && b"`` 的 ``&&`` 在引號裡、不切，整段當一個命令去分類（認不出，
    紅）。第 2 條餵進來的是 :func:`_strip_comment` 的產物（引號已經剝掉），對它來說這幾行
    只是不做事而已——兩條路共用一個函式，是為了「哪裡是引號裡面」只有一種看法。
    """
    parts: list[str] = []
    buffer = ""
    for chunk, is_quoted in _quoted_chunks(text):
        if is_quoted:
            buffer += chunk
            continue
        pieces = separator.split(chunk)
        buffer += pieces[0]
        for piece in pieces[1:]:
            parts.append(buffer)
            buffer = piece
    parts.append(buffer)
    return [part.strip() for part in parts if part.strip()]


def _pipelines(body: str) -> list[str]:
    """多段管線的那幾行。先剝掉引號與行內註解，再依單一個 ``|`` 切段，切得出兩段以上才算。"""
    hits: list[str] = []
    for raw in body.splitlines():
        if len(_split_outside_quotes(_strip_comment(raw), PIPE_RE)) > 1:
            hits.append(raw.strip())
    return hits


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


def _command_lines(body: str) -> list[str]:
    """一段 ``run:`` 裡真的是命令的那幾行。空行與 ``#`` 開頭的註解行不算。"""
    return [line.strip() for line in body.splitlines() if line.strip() and not line.strip().startswith("#")]


def _is_plumbing(line: str, prefixes: list[str]) -> bool:
    """這一行是不是水管步驟：比命令的**前幾個字**，不是比子字串。

    登記一個字就比第一個字（``git``），登記兩個字就比前兩個字（``uv sync``）——後者是刻意的：
    ``uv`` 底下什麼都跑得起來，整個 ``uv`` 放行等於這一條沒在管。
    """
    words = line.split()
    return any(words[: len(head)] == head for head in (prefix.split() for prefix in prefixes) if head)


def _is_wrapped(command: str, wrapper: str) -> bool:
    """這一行是不是**開頭**就在叫抄寫員。比開頭不比子字串，理由見模組說明第 5 條。

    後面接什麼都可以（``-- 真正要跑的命令``），但抄寫員出現在別的位置一律不算：塞在
    ``--`` 後面的那一串是被抄寫員跑的對象，不是抄寫員本人。
    """
    return command == wrapper or command.startswith(wrapper + " ")


def _segment_verdict(command: str, wrapper: str, plumbing: list[str]) -> str:
    """一段命令認不認得出來：認得出（留得下離開碼）回空字串，認不出回一句為什麼。

    **fail-closed**：只認兩種正面形狀——開頭等於卡上登記的 ``wrapper_command``（抄寫員：
    把那一步真實離開碼記成一片收據的那一層），或前幾個字命中卡上登記的
    ``plumbing_first_words``（水管步驟：本來就沒有判決可記的那幾種）。其他一律紅，
    包括看不懂的：``$(...)``、反引號、``eval "..."``、``sh -c "..."``、``FOO=1 ...``、
    ``exec``／``time``／``nohup`` 開頭那幾種，這一支都認不出來，所以都紅。

    判定用的字串**保留引號**，所以引號包住的 token 不算命中任何一種：``"uv sync" --locked``
    的第一個字是 ``"uv``，命不中水管；整行被包住的
    ``"uv run python -m governance.status.record_step ..."`` 也命不中抄寫員。

    背景執行先判：``&`` 結尾那一段 shell 不等它，離開碼一定不會被記——連抄寫員被丟到背景
    都一樣（收據沒人等），所以這一格排在兩種正面形狀前面。
    """
    if command.endswith(BACKGROUND_SUFFIX):
        return (
            "以 `&` 結尾，被丟到背景跑——shell 不等它，那一段的離開碼一定不會被記，"
            "包了抄寫員也一樣（抄寫員本人在背景，收據沒人等）"
        )
    if _is_wrapped(command, wrapper) or _is_plumbing(command, plumbing):
        return ""
    return (
        f"開頭不是抄寫員（卡上登記的 {wrapper!r}），"
        f"前幾個字也不在卡上登記的水管名單 {plumbing} 裡——這一段我認不出是什麼。"
        "認不出就紅（fail-closed）：離開碼進不了收據時，"
        "0（乾淨）／1（抓到違規）／2（這一跑不算數）在雲端記的紅綠裡分不開，"
        "等於那一步只留下一個顏色。"
        "引號包住的 token 不算命中任何一種（整行包一對引號，shell 照樣執行它）；"
        "抄寫員寫在行內註解裡、或塞在 `--` 後面當被跑的對象，都不算包"
    )


def _receipt_step_problems(
    step_where: str, body: str, settings: dict[str, object]
) -> list[str]:
    """第 5 條：擋合併那幾個 job 底下這一步的每一**段**命令，要嘛包抄寫員、要嘛是水管。

    一行裡用 ``&&``／``;``／``||``／``|`` 串起來的每一段各自算一步（2026-09-10 補的縫）：
    只看整行開頭的話，``uv sync --locked && uv run python -m governance.checks.x --scan-root .``
    會因為開頭那一段是水管就整行放行，後面那支檢查的離開碼進不了收據，而卡面看起來守著。
    切段用 :func:`_split_outside_quotes`，引號裡的分隔符不切。

    **殘骸只認一種形狀**（2026-09-10 改成 fail-closed）：原始那一行去頭尾空白後是空的、
    或以 ``#`` 開頭——那兩種由 :func:`_command_lines` 濾掉。其他每一行都必須被分類，
    分類不出來就紅。舊版拿剝掉引號的版本去切段，於是整行被一對引號包住的命令剝完變成空字串，
    被「沒有命令就沒有離開碼可記」當殘骸跳過、段數 0、不紅——而 shell 真的會執行它，
    那是一條可用的繞道（issue #104）。現在切段吃的是 :func:`_command_text`（引號留著）。

    抄寫員那一段 ``--`` 之後不用另外處理：``--`` 之後是抄寫員要跑的子程序，離開碼由抄寫員
    原封不動記下來、原封不動回傳，而那一整段的開頭就是抄寫員，所以它整段合格。``--`` 之後
    出現的分隔符會切出**下一段**，那是 shell 層的另一個命令（抄寫員管不到它），照判。
    """
    wrapper = setting_text(settings, "wrapper_command")
    plumbing = setting_strings(settings, "plumbing_first_words")
    bad: list[str] = []
    for line in _command_lines(body):
        commands = _split_outside_quotes(_command_text(line), SEGMENT_SEP_RE)
        if not commands:
            bad.append(
                f"{step_where} 的 `{line}` 切完一段命令都不剩——這一行我認不出是什麼。"
                "殘骸只認「原始那一行去頭尾空白後是空的、或以 `#` 開頭」那兩種，"
                "其他一律要分類得出來，分類不出來就紅（fail-closed）"
            )
            continue
        for index, command in enumerate(commands, start=1):
            verdict = _segment_verdict(command, wrapper, plumbing)
            if not verdict:
                continue
            bad.append(f"{step_where} 的 `{line}` 第 {index} 段 `{command}` {verdict}")
    return bad


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
    required_jobs: frozenset[str],
) -> list[str]:
    where = f"{rel} 的 job {name!r}"
    if not isinstance(job, dict):
        raise ToolBroken(f"{where} 剖析出來不是一張表（實際是 {type(job).__name__}），我看不懂")

    bad: list[str] = []
    bad += _continue_problems(where, job)
    bad += _if_problems(where, job, setting_strings(settings, "forbidden_job_ifs"))
    bad += _timeout_problems(where, job, setting_int(settings, "max_timeout_minutes"))

    # 第 5 條只對「擋得住合併」的那幾個 job 成立（名單由版控裡的
    # governance/required-status-checks.txt 決定），別的 job 不管。
    is_receipt_job = name in required_jobs
    job_shell = _default_shell(job) or wf_shell
    steps = job.get("steps")
    if not isinstance(steps, list) or not steps:
        return bad + [
            f"{where} 底下沒有任何步驟（steps={steps!r}）"
            "——空 job 永遠綠，掛成 required check 就是一個只會回綠的閘"
        ]

    does_work = False
    markers = setting_strings(settings, "work_markers")
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            raise ToolBroken(f"{where} 第 {index + 1} 步剖析出來不是一張表（{step!r}），我看不懂")
        label = step.get("name") if isinstance(step.get("name"), str) else f"第 {index + 1} 步"
        step_where = f"{where} 的{label}"
        bad += _continue_problems(step_where, step)
        bad += _push_attempts_problems(step_where, step, settings)

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
    for path in workflows:
        rel = str(path.relative_to(scan_root))
        data = _workflow(path, rel)
        jobs = data["jobs"]
        if not isinstance(jobs, dict):
            # _workflow 已經驗過這一格是一張非空的表；寫出來是為了讓型別看得見，
            # 而且形狀真的壞掉的時候是回 2（這一跑不算數），不是炸出追蹤訊息。
            raise ToolBroken(f"{rel} 的 jobs: 不是一張表（實際是 {jobs!r}），我看不懂")
        seen_jobs |= {str(name) for name in jobs}
        parsed.append((rel, data, jobs))

    missing = sorted(required_jobs - seen_jobs)
    if missing:
        raise ToolBroken(
            f"{REQUIRED_CHECKS_FILE} 列了 {missing}，可是掃到的 workflow 裡沒有任何 job 叫這些名字"
            f"（掃到的是 {sorted(seen_jobs)}）——第 5 條在這棵樹上一個對象都沒有，"
            "那不是乾淨，是量錯了對象：名單改了 workflow 沒跟上，或反過來"
        )

    bad: list[str] = []
    for rel, data, jobs in parsed:
        wf_shell = _default_shell(data)
        for name, job in jobs.items():
            bad += _job_problems(
                rel, str(name), job, wf_shell, settings, scan_root, files, required_jobs
            )
    return bad


if __name__ == "__main__":
    sys.exit(
        run(
            check,
            description=(
                "雲端工作不准無聲死掉：吞離開碼、漂綠、沒有上限、空 job、"
                "擋合併那幾個 job 有沒包的步驟、重試次數跟卡上登記的不一樣，一律紅"
            ),
            targets=targets,
        )
    )
