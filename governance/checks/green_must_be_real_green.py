#!/usr/bin/env python3
"""測試那盞綠燈要是真的綠：只讀 pytest 產的 junit 收據，不讀人抄的數字。

收據路徑、地板過期的倍數與**籃子表**寫在卡自己的 ``[junit]`` 表裡（``path`` ＋
``floor_stale_ratio`` ＋ ``[[junit.bucket]]``），**這支檢查沒有預設值**：掃描根底下沒有
任何一張卡宣告 ``[junit]``，就回 2 說「沒東西可判」，不會自己挑一個路徑去看。

**一份成績單、兩個籃子（決策紙 docs/decisions/engine-first-block-config-shape.md 的第三題）**

一次 pytest 只交一份 junit 收據（``tests/conftest.py`` 的規矩），但收據裡按 **``classname``
的前綴**分成幾個籃子：治理層的考卷住 ``tests/``、引擎的住 ``tests/engine/``，於是
``classname`` 長成 ``tests.test_build_status`` 與 ``tests.engine.test_aosr_runtime``。
每個籃子各自一個「最少題數」與一條「地板過期」線，引擎長大的時候不會把治理層那條網稀釋掉。

**逐個 ``<testcase>`` 分籃，不把 ``<testsuite>`` 屬性加總。** 一個 ``testcase`` 的
``classname`` 命中卡上登記的哪一個前綴就落進那一籃；**取最長符合的前綴**（``tests.`` 與
``tests.engine.`` 同時符合時算 ``tests.engine.``）。**沒命中任何前綴的 ``testcase`` 一律紅**
——沒被分進任何籃子的考卷就是沒有人守的題，加了新目錄卻忘了登記籃子會在這裡亮燈。

**第一組 收據本身（v2 事故 skips-disguise-red-as-green 的形狀）**

1. 收據不存在 → 紅。沒有收據就是沒有綠。這一條刻意不是 2 也不是 0：「檔不存在就當乾淨」
   正是 v2 的病根，而「檔不存在所以我沒法判」在這張卡的語境下也是假話——收據該由卡宣告的
   那個 job 產出來，它不在就是那一跑沒跑測試。
2. ``classname`` 是空的 → 紅，而且訊息要說得出**這是收集期爆掉，不是分籃分錯**。pytest 收到
   collection error 時交出來的 ``testcase`` 就長這樣（``classname=""``、通常裡面一個
   ``<error>``），那一題根本沒跑到；把它當成「不屬於任何籃子」會誤導人去改籃子表。
   這一條刻意不看有沒有 ``<error>``：空 ``classname`` 自己就是「這個 testcase 講不出它屬於
   哪一籃」，有沒有那個元素都要紅。
3. 沒落進任何籃子的 ``classname`` → 紅。分籃分錯跟收集期爆掉是兩件事，訊息分開寫。
4. 收據自己的 ``<testsuite tests="…">`` 屬性加起來不等於實際數到的 ``<testcase>`` 個數 → 紅。
   收集數改成逐個 testcase 數之後，屬性那一格就沒有人讀了，而雲端收據
   （``governance/status/build_receipt.py``）抄的正是那個屬性：一份 ``tests`` 寫 9999、實際只有
   幾個 ``testcase`` 的收據，門口判綠、帳本卻記下一個沒人驗過的數字。這一條把兩邊對起來，
   那張收據自相矛盾就紅。
5. ``skipped == 0``。屬性與 ``<skipped>`` 元素數取大的那個，所以把屬性改成 0 也躲不掉。
   v2 那一跑是 3,311 collected／12 skipped／failures 0／離開碼 0，當年這就算「綠」；
   D5 併入後那 12 支真跑起來，露出 5 條本來就存在的紅。
6. 收集數 0 → 紅（pytest 離開碼 5「沒收到測試」就是這個形狀；v2 的 REPO-5 還撞過 cwd 錯、
   離開碼 4、SKIPPED 0 次而測試一次都沒跑）。收集數為 0 的時候只報這一條，不再重複報
   「低於地板」，省得看不出真正的病。
7. **每一個籃子**的收集數 ``>=`` 那一籃的 ``collected_floor``。地板是登記在卡上的數字，
   某籃掉下去就是那一籃有測試不見了。
8. **每一個籃子**的收集數 ``<=`` 那一籃的 ``collected_floor × floor_stale_ratio``。地板離
   實跑數太遠等於沒有地板：地板 16、實跑 74 的時候掉掉一半測試也還在地板上面，離開碼照樣
   是 0。所以「地板過期」本身就是一條紅，訊息直接寫該怎麼修（加卡的 PR 順手把那一籃的
   ``collected_floor`` 調到實跑數）。倍數寫在卡上不寫死在程式裡；跟上一條互斥，同一籃不會
   兩邊都報。刻意不改成「地板 = 卡數 × 回合數」由機器算：回合數今天是 6，明天多一回合就要
   改程式，而且 ``tests/`` 底下不只後設測試那幾支，算出來的數字會跟實跑數對不上（issue #35）。
9. ``failures``／``errors`` 都必須是 0。真綠的定義裡沒有「有紅但我當它綠」這一種。

``skipped``、收集數 0、``failures``／``errors`` 這三條刻意留在**整份收據**這一層（不逐籃報），
因為它們判的是 v2 那幾筆事故的形狀——整份收據的 skip／collect error／紅；逐籃報會讓同一筆病依
籃子數量重複印好幾次。逐籃算出來的 ``failures``／``errors``／``skipped`` 三格只是順手數出來，
**判定不在那裡**。

**第二組 收據的來源與跑法（不然收據可以是任何一跑留下的）**

10. 卡宣告的那個 job 裡（含它 ``run:`` 呼叫的、進得了版控的腳本）必須真的有一步在跑 pytest。
    測試步驟整個被拿掉、收據卻還在，綠燈就跟這一跑的程式碼沒有關係了。
11. 那些跑 pytest 的地方必須把 junit 寫到卡宣告的那個路徑（``--junitxml=<path>``），
    路徑不一樣也算沒綁上。
12. 正式跑法只有一種：不准 ``--deselect``／``--ignore``／``-k`` 排除清單，不准 ``--collect-only``
    冒充跑過，不准把離開碼吞掉（``continue-on-error: true`` 與吞掉失敗的 shell 字樣）。
    v2 的 deselect 邏輯藏在 ``nightly_hermetic.sh`` 裡，所以第 10～12 條都要遞迴進 workflow
    ``run:`` 呼叫的腳本一起掃，光看 yaml 看不出來。
13. pytest 設定只准寫在 ``pyproject.toml``。另一份 ``pytest.ini``／帶 pytest 段的 ``tox.ini``
   ／``setup.cfg`` 就是第二套跑法，裸 pytest 跟 CI 跑的不再是同一套。

**第三組 考卷住哪裡（放錯籃子不算放外面）**

14. **掃描面上的一支考卷如果載入了 ``aosr.``（＝它是引擎的考卷），它就必須住那一籃的目錄。**
    那一籃的目錄**從卡上的籃子表推出來**：一個 ``classname`` 前綴就是一條點分開的模組路徑
    （``tests.engine.`` → ``tests/engine``），檢查程式自己不再抄一次路徑。只有一支考卷的
    **classname 前綴**（去掉最後一段的那個模組路徑）落在那一籃的目錄裡才算住對。
    這一條要擋的不是「沒進籃子」——那有第 3 條。沒進籃子會被擋下；**放錯籃子不會**：一支引擎
    的考卷寫進治理層那個資料夾，它照樣進治理層那一籃、照樣不紅，而下一張卡又會被要求把治理層
    的地板往上調，那個數字就被引擎的題灌大了。之後治理層真的掉了考卷，還在那個被灌大的地板
    之上，離開碼照樣是 0。

    判準是 ``ast`` 解析出來的，不是正則猜的。**抓得到與抓不到逐條列出來**，不讓這一條聽起來
    比實際強：

    * 抓得到——``import aosr``／``import aosr.x``／``from aosr import …``／``from aosr.x import …``；
      以及把**字面字串**交給 ``import_module``／``__import__`` 這兩個名字（位置參數或
      ``name=`` 關鍵字都算，``from importlib import import_module`` 之後的 ``import_module(…)``
      一樣算）。
    * 抓不到——相對 import（``from . import aosr``，那指的是同一棵套件樹裡的名字，不是引擎套件，
      刻意跳過）；用變數／``+``／f-string 拼出來的模組名字；把 ``import_module``／``__import__``
      **改名**之後呼叫的（``from importlib import import_module as im`` 之後的 ``im(…)``）；
      ``getattr(importlib, "import_module")(…)``；``exec``／``eval``。

    後面這幾種的成本都是零，靜態解析拿不到，這一條今天就是抓不到，不是漏寫。規則改寬之前
    不許把抓不到的說成抓得到。

**沒做的那一條，以及為什麼**

卡的第 2 版規格還要求「收據的 mtime 必須晚於 job 開始時間，否則回 2」，這支檢查沒做：
沒有可靠、可重現的「job 開始時間」來源——``git checkout`` 會把整棵樹的 mtime 設成同一時刻，
所以進版控的樣本表達不出「這是上一跑留下的舊檔」，寫了也證明不了它會咬。改用同效而且驗得到
的綁定：收據路徑進 ``.gitignore``（雲端每一跑都是新鮮檔案，舊檔搭不了便車），而且第 10～11 條
要求同一個 job 裡真的有一步在產它。「拿舊收據冒充」這一條今天靠的是這個組合，不是時間戳。

**乾淨樹那一回合的收據哪裡來**

收據不進版控，剛 clone 的樹裡沒有它。後設測試在跑第一回合之前先跑一次真的全套把它產出來
（見 ``tests/conftest.py``）。**這支檢查自己絕不補一份**——尺自己造證據，就是 v2 的病。
"""
from __future__ import annotations

import ast
import re
import sys
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Iterator
from pathlib import Path

from governance.checks.rule_card_required_fields import job_blocks
from governance.exit_codes import ToolBroken, run
from governance.loader import RULES_DIR, Card, card_problems, load_card

WORKFLOW_DIR = ".github/workflows"
PYTEST_CONFIG_HOME = "pyproject.toml"

# 第二套 pytest 設定會住在哪。marker 是空字串代表「這個檔存在就算」。
RIVAL_CONFIGS = (
    ("pytest.ini", ""),
    ("tox.ini", "[pytest]"),
    ("setup.cfg", "[tool:pytest]"),
)

# 排除／假跑的旗標。跑 pytest 的地方出現任何一個，正式跑法就不只一種。
FORBIDDEN_FLAGS = (
    ("--deselect", "從清單裡逐一挑掉測試——v2 的 nightly 就是這樣把 54 條假紅藏起來的"),
    ("--ignore", "整個目錄不跑，收據上看不出少了什麼"),
    ("-k ", "用表達式挑測試，裸 pytest 跟 CI 跑的就不是同一套"),
    ("--collect-only", "只收集不跑，收據會是 0 個測試卻長得像跑過"),
)

# 「把失敗吞掉」的字樣。跟上一張卡一樣刻意用片段拼出來：這支檢查自己也在
# governance/checks/ 的掃描面裡，寫成完整字面值會被那張卡咬到。
_NULL = "/dev" + "/null"
_OR = "|" + "|"
SWALLOW_SNIPPETS = (
    _OR + " true",
    _OR + "true",
    _OR + " :",
    ";" + " true",
    "set" + " +e",
    "2>" + _NULL,
    "2> " + _NULL,
    ">" + _NULL + " 2>&1",
)
SWALLOW_YAML_KEYS = ("continue-on-error: true", "continue-on-error: True")

# workflow 的 run: 裡叫得動的腳本長什麼樣（只認進得了版控、掃描根底下的檔）。
SCRIPT_RE = re.compile(r"[\w./-]+\.(?:sh|bash|py)")
RUN_RE = re.compile(r"^(?P<pre>\s*(?:-\s+)?)run\s*:\s*(?P<rest>.*)$")

# 整份收據這一層的三個計數：屬性名與對應的元素標籤。**收集數（tests）不在這裡**——
# 收集數改成逐個 <testcase> 數，屬性可以被改，所以那三個還是屬性與元素取大的那個。
COUNT_PAIRS = (("skipped", "skipped"), ("failures", "failure"), ("errors", "error"))
# 一個籃子的四個數字。收集數是逐個 <testcase> 數出來的。
#
# ``failures``／``errors``／``skipped`` 這三格**刻意不逐籃判**：那三條規矩判的是整份收據有
# 沒有把紅當綠（v2 事故的形狀），逐籃報只會讓同一筆病依籃子數量重複印；判定留在第一組那一段
# （整份收據），這裡逐籃算只是順手把它們一起數出來。**不要以為逐籃的 skip 有在守**——真的
# 在守的是第一組那三條。逐籃真正拿來判的只有 ``tests``（地板與地板過期）。
BUCKET_COUNTS = ("tests", "failures", "errors", "skipped")

# 第三組（考卷住哪裡）用的零件。
#
# 引擎的套件名：一支考卷的 import 命中它（或它的子模組）就是「引擎的考卷」。這不是門檻、
# 也不是可以調的路徑，是這個 repo 新家的名字，跟 ``pyproject.toml`` 的 ``pythonpath`` 同一
# 件事。
ENGINE_PACKAGE = "aosr"
# 那一種動態載入要算：``importlib.import_module("aosr.…")`` 與 ``__import__("aosr.…")``
# （兩個都是**字面字串**，靜態 ast 拿得到）。用變數拼出來的不算——靜態解析拿不到執行期的值，
# 這條限制照實寫在模組說明與卡的人話裡。``__import__`` 收在裡面是因為它跟 ``import_module``
# 一樣直覺、而且一樣靜態可見：不放進來就等於留一條零成本的繞法。
DYNAMIC_IMPORT_FUNCTIONS = ("import_module", "__import__")
# 那兩個函式的正式參數名（``import_module(name, package=None)``／``__import__(name, ...)``）。
# 關鍵字參數只看這一個名字，理由寫在 :func:`_literal_module_name`。
MODULE_NAME_KEYWORD = "name"
# 一個 ``classname`` 前綴的點分開來就是一條模組路徑（``tests.engine.`` → ``tests/engine``）。
MODULE_SEPARATOR = "."
PYTHON_SUFFIX = ".py"
# 考卷樹在掃描根底下的哪一層。第三組要讀的就是這裡底下的 ``.py``（由 pytest 的 ``testpaths``
# 決定、寫在 ``pyproject.toml``）。這不是門檻，是這棵樹的座標。
TEST_DIR = "tests"


def _module_dir(prefix: str) -> str:
    """把一個 ``classname`` 前綴換成它對應的目錄（``tests.engine.`` → ``tests/engine``）。

    這是「引擎考卷住哪」這件事唯一的來源：卡上的籃子表。檢查程式刻意不在這裡寫死任何
    路徑——卡上改了前綴（例如多一籃），這裡自己會跟著改，不會出現第二份宣告。
    """
    return "/".join(part for part in prefix.split(MODULE_SEPARATOR) if part)


def _engine_dirs(card: Card) -> list[str]:
    """這一跑要拿來判「引擎考卷住哪」的目錄：卡上**最深**的幾個籃子前綴換成的目錄。

    取最深的幾個，不是每一個：``tests.`` 換出來是 ``tests``（整個考卷樹），它上面還掛著
    ``tests.engine.`` 那一籃。算進 ``tests`` 的話，每一支考卷都「住對地方」——守門就死了
    （第一版就是這樣，兩個必紅樣本當場變綠）。真正說得出「引擎的考卷住哪裡」的是最細的
    那幾層：今天只有 ``tests.engine.`` 一個，所以那個答案是 ``tests/engine``。

    卡上只剩一個最上層的籃子（``tests.``）時，換出來就是 ``tests``：那不是「引擎考卷的
    目的地」，是整個考卷樹，這時候這一條沒有更細的去處可指、不回報任何目錄——判準只說
    「引擎的考卷要住卡上籃子表推出來的那一籃」，卡上沒有比較細的籃子時就沒有對象。
    """
    dirs = [directory for prefix, _floor in card.junit_buckets if (directory := _module_dir(prefix))]
    deepest = [
        directory
        for directory in dirs
        if not any(other != directory and other.startswith(directory + "/") for other in dirs)
    ]
    # 只有一層、而且就是考卷樹本身（``tests``）＝卡上沒有更細的籃子可指。
    if deepest == [TEST_DIR]:
        return []
    return sorted(set(deepest))


def _bucket_of(card: Card, rel: str) -> str:
    """一支考卷落在哪一籃：拿它的 ``classname``（去掉最後一段的模組路徑）比前綴，取最長符合。

    檔案路徑與 ``classname`` 的關係是 pytest 自己定的：``tests/engine/test_aosr_runtime.py``
    的 ``classname`` 是 ``tests.engine.test_aosr_runtime``，所以去掉最後一段就是
    ``tests.engine``。

    比對逐段來（不是用 :func:`_bucket_prefix` 那種字串前綴）：一個前綴 ``tests.`` 的段是
    ``("tests",)``，命中 ``tests`` 這一層與它底下每一段；``tests.engine.`` 的段是
    ``("tests", "engine")``，只命中那一層與底下，取最長的那個。拿 ``tests.engine`` 去比
    ``tests.`` 字串前綴是不合的（`.` 之後沒有東西），那會讓每一支引擎考卷都被判成「住錯」。
    """
    classname = rel.removesuffix(PYTHON_SUFFIX).replace("/", MODULE_SEPARATOR)
    module = classname.rsplit(MODULE_SEPARATOR, 1)[0] if MODULE_SEPARATOR in classname else ""
    segments = tuple(part for part in module.split(MODULE_SEPARATOR) if part)
    hit = ""
    depth = 0
    for prefix, _floor in card.junit_buckets:
        wanted = tuple(part for part in prefix.split(MODULE_SEPARATOR) if part)
        if wanted and segments[: len(wanted)] == wanted and len(wanted) > depth:
            hit = prefix
            depth = len(wanted)
    return hit


def _called_name(func: ast.expr) -> str:
    """一個呼叫對象攤平後的名字（``import_module``／``importlib.import_module`` → 最後一段）。"""
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _literal_module_name(call: ast.Call) -> str:
    """這個呼叫交給動態載入的模組名字是不是字面字串；是就回那個字串，不是就回空字串。

    位置參數與關鍵字參數都看：``import_module("aosr.runtime")`` 與
    ``import_module(name="aosr.runtime")`` 一樣是靜態看得見的字面值，只讀位置參數會漏掉後者
    （找碴實測：關鍵字那一種放 `tests/` 根層回 0）。關鍵字只認名叫 ``name`` 的那一個
    （``import_module`` 與 ``__import__`` 的正式參數名都是它）；別的關鍵字不看——那不是模組名。
    """
    candidates: list[ast.expr] = list(call.args[:1])
    candidates += [kw.value for kw in call.keywords if kw.arg == MODULE_NAME_KEYWORD]
    for candidate in candidates:
        if isinstance(candidate, ast.Constant) and isinstance(candidate.value, str):
            return candidate.value
    return ""


def _import_targets(tree: ast.AST) -> Iterator[str]:
    """這支考卷載入了哪些模組（``ast`` 解析出來的，不是正則猜的）。

    抓得到：``import aosr``、``import aosr.x``、``from aosr import …``、``from aosr.x import …``
    （相對 import 不算，見下），以及把**字面字串**交給 ``import_module``／``__import__``
    這兩個名字（位置參數或 ``name=`` 關鍵字都算）。
    抓不到：用變數／``+``／f-string 拼出來的、把 ``import_module``／``__import__`` 改名之後
    呼叫的、``getattr`` 現抓函式的、``exec``／``eval``。這幾種成本都是零，模組說明與卡面
    逐條照實寫出來，不讓這支檢查聽起來比實際強。
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            # 相對 import（``from . import aosr``、``from .. import aosr``）指的是同一棵套件樹裡
            # 的那個名字，不是引擎套件——``node.level`` 大於零一律跳過。這不是「剛好漏掉」：
            # 跳過是刻意的，理由是它與 ``aosr`` 這個套件無關。
            if node.level == 0 and node.module:
                yield node.module
        elif isinstance(node, ast.Call) and _called_name(node.func) in DYNAMIC_IMPORT_FUNCTIONS:
            literal = _literal_module_name(node)
            if literal:
                yield literal


def _loads_engine_module(name: str) -> bool:
    """這個模組名字是不是 ``aosr`` 或它的子模組（``aosr``、``aosr.runtime``、``aosr.x.y``）。"""
    return name == ENGINE_PACKAGE or name.startswith(ENGINE_PACKAGE + MODULE_SEPARATOR)


def _placement_group_problems(
    card: Card, scan_root: Path, sources: Iterable[Path]
) -> list[str]:
    """第三組：載入 ``aosr.`` 的考卷必須住在那一籃的目錄裡。

    ``sources`` 是掃描面上的考卷（``.py``）。落點由 :func:`_bucket_of` 算出來，不是拿字串
    比對拼出來的——卡上換一個籃子前綴，這裡跟著換。
    """
    buckets = card.junit_buckets
    if not buckets:
        return []
    owned = _engine_dirs(card)
    if not owned:
        # 卡上沒有比考卷樹本身更細的籃子時，這一條判不出「引擎考卷該住哪裡」——沒有對象
        # 就不開槍。少了這一行，``owned`` 空集合會讓**每一支**載入引擎套件的考卷都被判紅
        # （訊息還會說「卡說那一籃的目錄是 []」），那是無差別假紅，不是守門。
        return []
    bad: list[str] = []
    for path in sorted(sources):
        rel = path.relative_to(scan_root).as_posix()
        text = _read(path, rel)
        try:
            tree = ast.parse(text, filename=rel)
        except SyntaxError as exc:
            raise ToolBroken(
                f"{rel} 剖不開（{exc}）——這一條是 ast 判的，看不懂的檔我不出結論"
            ) from exc
        if not any(_loads_engine_module(name) for name in _import_targets(tree)):
            continue
        own = _bucket_of(card, rel)
        if not own:
            bad.append(
                f"{rel} 載入了 {ENGINE_PACKAGE}.（它是引擎的考卷），可是它的 classname 沒命中"
                f"卡 {card.id} 上任何一個籃子前綴（{list(buckets)}）——沒進籃子由第 3 條紅，"
                "這裡不重複報"
            )
            continue
        own_dir = _module_dir(own)
        if own_dir in owned:
            continue
        bad.append(
            f"{rel} 載入了 {ENGINE_PACKAGE}.（它是引擎的考卷），可是它住在籃子 {own!r}"
            f"（目錄 {own_dir}）底下——卡 {card.id} 說那一籃的目錄是 {owned}"
            "（從籃子表的 classname 前綴推出來的）。"
            "放錯籃子不算放外面：它會進錯籃子、不紅，還會把那一籃的地板灌大，"
            "以後那一籃真的掉了考卷也還在那個被灌大的地板之上、離開碼照樣是 0"
        )
    return bad


def _card_files(scan_root: Path, files: list[Path]) -> list[Path]:
    """掃描面的一組：所有規矩卡（收據路徑與地板只寫在卡上，所以每一張都要打開）。"""
    return sorted(f for f in files if f.parent == scan_root / RULES_DIR and f.suffix == ".toml")


def _workflow_files(scan_root: Path, files: list[Path]) -> list[Path]:
    """掃描面的一組：``.github/workflows/`` 底下進得了版控的 workflow。"""
    return sorted(
        f for f in files if f.parent == scan_root / WORKFLOW_DIR and f.suffix in (".yml", ".yaml")
    )


def _test_files(scan_root: Path, files: list[Path]) -> list[Path]:
    """掃描面的一組：考卷樹底下的 ``.py``。

    第三組（考卷住哪裡）讀的是考卷自己的原始碼，所以這一組要進列舉面。
    """
    base = scan_root / TEST_DIR
    return sorted(f for f in files if f.suffix == PYTHON_SUFFIX and f.is_relative_to(base))


def targets(scan_root: Path, files: list[Path]) -> list[Path]:
    """這支檢查真的會讀的檔：所有規矩卡 ＋ workflow ＋ 那個 job 叫到的腳本 ＋ 對手設定檔
    ＋ 考卷樹底下的 ``.py``。

    收據自己（卡的 ``[junit]`` path）不在裡面：它刻意不進版控（被 .gitignore 蓋住），
    所以列舉集合裡本來就沒有它，宣告那一邊也不會有。這一條是照實記，不是漏掉。
    """
    picked = [*_card_files(scan_root, files), *_workflow_files(scan_root, files)]
    picked += _test_files(scan_root, files)
    for wf in _workflow_files(scan_root, files):
        rel = str(wf.relative_to(scan_root))
        for script_rel, _ in _called_scripts([_read(wf, rel)], scan_root, files):
            picked.append(scan_root / script_rel)
    for name, _ in RIVAL_CONFIGS:
        rival = scan_root / name
        if rival in files:
            picked.append(rival)
    home = scan_root / PYTEST_CONFIG_HOME
    if home in files:
        picked.append(home)
    return sorted(set(picked))


def _junit_cards(scan_root: Path, files: list[Path]) -> list[Card]:
    """掃描根底下宣告了 ``[junit]`` 的卡。卡自己壞掉是第一張卡的事，這裡跳過。"""
    cards: list[Card] = []
    for path in _card_files(scan_root, files):
        if card_problems(path, scan_root):
            continue
        card = load_card(path, scan_root)
        if card.junit:
            cards.append(card)
    return cards


def _read(path: Path, rel: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ToolBroken(f"讀不到 {rel}：{exc}") from exc


def _receipt_root(target: Path, rel: str) -> ET.Element:
    """把收據剖成 XML 樹。解不開、或找不到任何 ``<testsuite>``，一律回 2，不猜。"""
    text = _read(target, rel)
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise ToolBroken(f"{rel} 不是解得開的 XML（{exc}）——我沒看懂就不出結論") from exc
    if not list(root.iter("testsuite")):
        raise ToolBroken(f"{rel} 裡找不到任何 <testsuite>，這份收據我看不懂")
    return root


def _bucket_prefix(classname: str, buckets: tuple[tuple[str, int], ...]) -> str:
    """一個 classname 落在哪一籃：取**最長符合**的前綴。沒命中回空字串。"""
    hit = ""
    for prefix, _floor in buckets:
        if classname.startswith(prefix) and len(prefix) > len(hit):
            hit = prefix
    return hit


def _empty_counts() -> dict[str, int]:
    return {name: 0 for name in BUCKET_COUNTS}


def _suite_attr_totals(root: ET.Element, rel: str) -> dict[str, int]:
    """整份收據的 testsuite 屬性加總（只有 skipped／failures／errors 三個）。"""
    totals = {attr: 0 for attr, _tag in COUNT_PAIRS}
    for suite in root.iter("testsuite"):
        for attr, _tag in COUNT_PAIRS:
            raw = suite.get(attr, "0")
            try:
                totals[attr] += int(raw)
            except ValueError as exc:
                raise ToolBroken(f"{rel} 的 <testsuite {attr}={raw!r}> 不是整數") from exc
    return totals


def _suite_tests_total(root: ET.Element, rel: str) -> int:
    """整份收據的 ``<testsuite tests="…">`` 屬性加總。

    這一格逐籃分籃之後就沒人讀了，而雲端收據（``governance/status/build_receipt.py``）抄的
    正是它——所以要拿它跟實際數到的 ``<testcase>`` 個數對帳，不然收據可以自己說一個沒人驗過
    的收集數。
    """
    total = 0
    for suite in root.iter("testsuite"):
        raw = suite.get("tests", "0")
        try:
            total += int(raw)
        except ValueError as exc:
            raise ToolBroken(f"{rel} 的 <testsuite tests={raw!r}> 不是整數") from exc
    return total


def _placement_problems(
    card: Card,
    rel: str,
    unassigned: list[str],
    unlabelled: list[str],
) -> list[str]:
    """每個 ``<testcase>`` 都必須落進某一籃，而且 classname 不能是空的。"""
    bad: list[str] = []
    prefixes = [prefix for prefix, _floor in card.junit_buckets]
    if unassigned:
        bad.append(
            f"{rel} 裡有 {len(unassigned)} 個 <testcase> 的 classname 沒命中任何一個籃子前綴"
            f"（卡 {card.id} 登記的籃子：{prefixes}）——沒被分進任何籃子的考卷一律紅，"
            "不然它會變成沒有人守的題。像是：" + "、".join(unassigned[:3])
        )
    if unlabelled:
        bad.append(
            f"{rel} 裡有 {len(unlabelled)} 個 <testcase> 沒有 classname（例如 "
            + "、".join(unlabelled[:3])
            + "）——這是**收集期就爆掉**的檔，不是分籃分錯：pytest 收到 collection error 時"
            "交出來的 testcase 就是這個形狀（classname 空；通常還帶一個 <error>，但沒有那個"
            "元素一樣是空的），那一題根本沒跑到。先修那個收集錯誤，再看它修好之後的 classname"
            " 該落進哪一籃"
        )
    return bad


def _floor_problems(card: Card, rel: str, counts: dict[str, dict[str, int]]) -> list[str]:
    """每一個籃子各自比地板與地板過期倍數（第一組的第 7、8 條）。"""
    bad: list[str] = []
    for prefix, floor in card.junit_buckets:
        got = counts[prefix]["tests"]
        if got < floor:
            bad.append(
                f"{rel} 的籃子 {prefix!r} 只收集到 {got} 個測試，卡 {card.id} 上登記的地板是 "
                f"{floor}——這一籃有測試不見了，離開碼卻還是 0"
            )
        elif got > floor * card.junit_stale_ratio:
            bad.append(
                "地板過期，加卡的 PR 要順手把那一籃的 collected_floor 調到實跑數："
                f"{rel} 的籃子 {prefix!r} 實際收集到 {got} 個測試，卡 {card.id} 上登記的地板還是 "
                f"{floor}（超過 {floor} × {card.junit_stale_ratio} = "
                f"{floor * card.junit_stale_ratio:g}）"
                "——地板離實跑數太遠等於沒有地板：掉掉一大半測試也還在地板上面，離開碼照樣是 0"
            )
    return bad


def _split_testcases(
    root: ET.Element, card: Card
) -> tuple[dict[str, dict[str, int]], list[str], list[str], list[ET.Element]]:
    """逐個 ``<testcase>`` 分籃。

    回傳 ``(每一籃的計數, 沒命中任何前綴的 classname, 沒有 classname 的名字, 全部 testcase)``。

    沒有 classname 的 testcase 先在這裡挑出來，跟「有 classname 但沒命中前綴」分開。這一支
    就是「收集期爆掉」那一條規矩：它一被拿掉，那些 testcase 只會被下面的 guard 跳過、沒有
    任何一條咬得到，對應的必紅樣本（``case-empty-classname-without-error``）當場變綠
    ——規則死了後設測試看得到。
    """
    cases = list(root.iter("testcase"))
    counts = {prefix: _empty_counts() for prefix, _floor in card.junit_buckets}
    unassigned: list[str] = []
    unlabelled = [
        case.get("name") or "<沒有 name>" for case in cases if not (case.get("classname") or "")
    ]
    for case in cases:
        classname = case.get("classname") or ""
        if not classname:
            # 空 classname 由上面那一支負責歸類與回報；這裡只是不讓它掉進「沒命中任何前綴」
            # 那一條——收集期爆掉跟分籃分錯是兩件事，訊息要分得開。
            continue
        prefix = _bucket_prefix(classname, card.junit_buckets)
        if not prefix:
            unassigned.append(classname)
            continue
        bucket = counts[prefix]
        bucket["tests"] += 1
        for attr, tag in COUNT_PAIRS:
            bucket[attr] += len(case.findall(tag))
    return counts, unassigned, unlabelled, cases


def _receipt_problems(card: Card, scan_root: Path) -> list[str]:
    """第一組：收據本身講的是真綠嗎。逐個 ``<testcase>`` 分籃，再各籃各自判。"""
    rel = card.junit_path
    target = scan_root / rel
    if not target.is_file():
        return [
            f"卡 {card.id} 宣告的收據 {rel} 不存在——沒有收據就是沒有綠。"
            "「檔不存在就當乾淨」是 v2 假綠的病根，這支檢查也不准自己補一份"
        ]

    root = _receipt_root(target, rel)
    counts, unassigned, unlabelled, cases = _split_testcases(root, card)
    bad = _placement_problems(card, rel, unassigned, unlabelled)
    collected = sum(bucket["tests"] for bucket in counts.values()) + len(unassigned) + len(unlabelled)
    # 對帳刻意用「實際有幾個 <testcase> 元素」而不是上面那個 collected：collected 是分籃之後
    # 算出來的，空 classname 那一支有沒有在跑會改變它；拿它對帳會讓兩條規矩互相遮蔽
    # （空 classname 那一支一關掉，collected 就少一個，對帳那條反而亮起來）。
    declared = _suite_tests_total(root, rel)
    if declared != len(cases):
        bad.append(
            f"{rel} 自己跟自己打架：<testsuite tests=…> 那幾格加起來是 {declared}，實際數到 "
            f"{len(cases)} 個 <testcase>——收集數改成逐個 testcase 數之後，屬性那一格就沒有人讀了，"
            "而雲端收據（governance/status/build_receipt.py）抄的正是那個屬性：不把兩邊對起來，"
            "一份屬性寫很大、實際題數很少的收據會門口判綠、帳本卻記下一個沒人驗過的數字"
        )

    attrs = _suite_attr_totals(root, rel)
    for attr, tag in COUNT_PAIRS:
        # 屬性可以被改，元素不會憑空消失：兩邊取大的那個。
        attrs[attr] = max(attrs[attr], sum(1 for _ in root.iter(tag)))
    if attrs["skipped"] > 0:
        bad.append(
            f"{rel} 的 skipped={attrs['skipped']}，必須是 0"
            f"（收集 {collected} 個）——資源缺席要 fail，不准 skip 成非紅。"
            "v2 事故 skips-disguise-red-as-green 當下就是 3,311 collected／12 skipped／離開碼 0，"
            "那 12 支後來真跑起來露出 5 條紅"
        )
    if collected == 0:
        bad.append(
            f"{rel} 收集到 0 個測試卻被當成綠"
            "——pytest 離開碼 5「沒收到測試」就是這個形狀，一個測試都沒跑就沒有綠可言"
        )
    else:
        bad += _floor_problems(card, rel, counts)
    if attrs["failures"] > 0 or attrs["errors"] > 0:
        bad.append(
            f"{rel} 裡 failures={attrs['failures']}、errors={attrs['errors']}，"
            "卻被當成綠——真綠的定義裡沒有「有紅但我當它綠」這一種"
        )
    return bad


def _run_bodies(block: str) -> list[str]:
    """把一個 job 原文裡每個 ``run:`` 的內容挖出來（含 ``run: |`` 那種區塊）。

    刻意只挖 run 的內容、不掃整個 job 原文：step 的名字是給人看的中文，掃進去會誤咬。
    """
    lines = block.splitlines()
    bodies: list[str] = []
    i = 0
    while i < len(lines):
        match = RUN_RE.match(lines[i])
        if match is None:
            i += 1
            continue
        indent = len(match.group("pre"))
        rest = match.group("rest").strip()
        i += 1
        if rest and rest[0] not in "|>":
            bodies.append(rest)
            continue
        body: list[str] = []
        while i < len(lines):
            line = lines[i]
            if line.strip() and (len(line) - len(line.lstrip())) <= indent:
                break
            body.append(line)
            i += 1
        bodies.append("\n".join(body))
    return bodies


def _called_scripts(bodies: list[str], scan_root: Path, files: list[Path]) -> list[tuple[str, str]]:
    """run: 裡叫到的、進得了版控的腳本：[(相對路徑, 內容)]。"""
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for body in bodies:
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


def _source_problems(card: Card, scan_root: Path, files: list[Path]) -> list[str]:
    """第二組：收據的來源綁死了嗎，跑法只有一種嗎。"""
    workflows = _workflow_files(scan_root, files)
    if not workflows:
        return [
            f"卡 {card.id} 宣告 job={card.job!r} 要產收據，但 {WORKFLOW_DIR} 底下一份 workflow 都沒有"
            "——沒有人在跑測試，樹裡那份收據就是遺物"
        ]

    blocks: dict[str, str] = {}
    for wf in workflows:
        rel = str(wf.relative_to(scan_root))
        for name, block in job_blocks(_read(wf, rel), rel).items():
            blocks.setdefault(name, block)
    if card.job not in blocks:
        return [
            f"卡 {card.id} 宣告 job={card.job!r}，但 {WORKFLOW_DIR} 裡沒有這個 job"
            f"（實際有 {sorted(blocks)}）——沒有那個 job 就沒有人產收據"
        ]

    block = blocks[card.job]
    bodies = _run_bodies(block)
    places = [(f"{card.job} 的 run:", body) for body in bodies]
    places += _called_scripts(bodies, scan_root, files)

    testing = [(where, text) for where, text in places if "pytest" in text]
    if not testing:
        return [
            f"卡 {card.id} 宣告 job={card.job!r}，那個 job（含它呼叫的腳本）裡沒有任何一步在跑 pytest"
            "——測試步驟整個被拿掉了，收據還在也只是上一跑的遺物"
        ]

    bad: list[str] = []
    flag = f"--junitxml={card.junit_path}"
    if not any(flag in text for _, text in testing):
        bad.append(
            f"卡 {card.id} 宣告收據在 {card.junit_path}，但 job={card.job!r} 跑 pytest 的地方"
            f"沒有一處帶 {flag}——收據跟這一跑沒有綁死，樹裡那份可以是任何一跑留下的、甚至是手寫的"
        )
    for where, text in testing:
        for snippet, why in FORBIDDEN_FLAGS:
            if snippet in text:
                bad.append(f"{where} 跑 pytest 帶了 {snippet.strip()}：{why}——正式跑法只准一種")
        for snippet in SWALLOW_SNIPPETS:
            if snippet in text:
                bad.append(f"{where} 把離開碼吞掉了（{snippet!r}）——pytest 回 4／5 也會變成綠")
    for key in SWALLOW_YAML_KEYS:
        if key in block:
            bad.append(f"job={card.job!r} 寫了 {key}——測試紅了也不擋，綠燈就沒有意義")

    for name, marker in RIVAL_CONFIGS:
        path = scan_root / name
        if path not in files or not path.is_file():
            continue
        if marker and marker not in _read(path, name):
            continue
        bad.append(
            f"多了一份 pytest 設定 {name}——pytest 設定只准寫在 {PYTEST_CONFIG_HOME}，"
            "第二套設定等於第二種跑法，裸 pytest 跟 CI 跑的就不是同一套了"
        )
    return bad


def check(scan_root: Path, files: list[Path]) -> list[str]:
    cards = _junit_cards(scan_root, files)
    if not cards:
        raise ToolBroken(
            f"{scan_root}/{RULES_DIR} 底下沒有任何一張讀得出來的卡宣告 [junit]"
            "——收據路徑與地板只寫在卡上，這支檢查沒有預設值，所以這一跑沒東西可判"
        )
    bad: list[str] = []
    for card in cards:
        bad += _receipt_problems(card, scan_root)
        bad += _source_problems(card, scan_root, files)
        bad += _placement_group_problems(card, scan_root, _test_files(scan_root, files))
    return bad


if __name__ == "__main__":
    sys.exit(
        run(
            check,
            description="測試的綠燈要是真的綠：junit 收據不准有 skip，每一籃的收集數不准低於卡上的地板，"
            "載入了引擎套件的考卷必須住在那一籃的目錄裡",
            targets=targets,
        )
    )
