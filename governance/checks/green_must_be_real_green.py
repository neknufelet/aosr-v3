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
   collection error 時交出來的 ``testcase`` 就長這樣（``classname=""``、裡面一個 ``<error>``），
   那一題根本沒跑到；把它當成「不屬於任何籃子」會誤導人去改籃子表。
3. 沒落進任何籃子的 ``classname`` → 紅。分籃分錯跟收集期爆掉是兩件事，訊息分開寫。
4. ``skipped == 0``。屬性與 ``<skipped>`` 元素數取大的那個，所以把屬性改成 0 也躲不掉。
   v2 那一跑是 3,311 collected／12 skipped／failures 0／離開碼 0，當年這就算「綠」；
   D5 併入後那 12 支真跑起來，露出 5 條本來就存在的紅。
5. 收集數 0 → 紅（pytest 離開碼 5「沒收到測試」就是這個形狀；v2 的 REPO-5 還撞過 cwd 錯、
   離開碼 4、SKIPPED 0 次而測試一次都沒跑）。收集數為 0 的時候只報這一條，不再重複報
   「低於地板」，省得看不出真正的病。
6. **每一個籃子**的收集數 ``>=`` 那一籃的 ``collected_floor``。地板是登記在卡上的數字，
   某籃掉下去就是那一籃有測試不見了。
7. **每一個籃子**的收集數 ``<=`` 那一籃的 ``collected_floor × floor_stale_ratio``。地板離
   實跑數太遠等於沒有地板：地板 16、實跑 74 的時候掉掉一半測試也還在地板上面，離開碼照樣
   是 0。所以「地板過期」本身就是一條紅，訊息直接寫該怎麼修（加卡的 PR 順手把那一籃的
   ``collected_floor`` 調到實跑數）。倍數寫在卡上不寫死在程式裡；跟第 6 條互斥，同一籃不會
   兩邊都報。刻意不改成「地板 = 卡數 × 回合數」由機器算：回合數今天是 6，明天多一回合就要
   改程式，而且 ``tests/`` 底下不只後設測試那幾支，算出來的數字會跟實跑數對不上（issue #35）。
8. ``failures``／``errors`` 都必須是 0。真綠的定義裡沒有「有紅但我當它綠」這一種。
   第 4、5、8 條刻意留在**整份收據**這一層（不逐籃報），因為它們判的是 v2 那幾筆事故的形狀
   ——整份收據的 skip／collect error／紅；逐籃報會讓同一筆病依籃子數量重複印好幾次。

**第二組 收據的來源與跑法（不然收據可以是任何一跑留下的）**

9. 卡宣告的那個 job 裡（含它 ``run:`` 呼叫的、進得了版控的腳本）必須真的有一步在跑 pytest。
   測試步驟整個被拿掉、收據卻還在，綠燈就跟這一跑的程式碼沒有關係了。
10. 那些跑 pytest 的地方必須把 junit 寫到卡宣告的那個路徑（``--junitxml=<path>``），
    路徑不一樣也算沒綁上。
11. 正式跑法只有一種：不准 ``--deselect``／``--ignore``／``-k`` 排除清單，不准 ``--collect-only``
    冒充跑過，不准把離開碼吞掉（``continue-on-error: true`` 與吞掉失敗的 shell 字樣）。
    v2 的 deselect 邏輯藏在 ``nightly_hermetic.sh`` 裡，所以第 9～11 條都要遞迴進 workflow
    ``run:`` 呼叫的腳本一起掃，光看 yaml 看不出來。
12. pytest 設定只准寫在 ``pyproject.toml``。另一份 ``pytest.ini``／帶 pytest 段的 ``tox.ini``
   ／``setup.cfg`` 就是第二套跑法，裸 pytest 跟 CI 跑的不再是同一套。

**沒做的那一條，以及為什麼**

卡的第 2 版規格還要求「收據的 mtime 必須晚於 job 開始時間，否則回 2」，這支檢查沒做：
沒有可靠、可重現的「job 開始時間」來源——``git checkout`` 會把整棵樹的 mtime 設成同一時刻，
所以進版控的樣本表達不出「這是上一跑留下的舊檔」，寫了也證明不了它會咬。改用同效而且驗得到
的綁定：收據路徑進 ``.gitignore``（雲端每一跑都是新鮮檔案，舊檔搭不了便車），而且第 9～10 條
要求同一個 job 裡真的有一步在產它。「拿舊收據冒充」這一條今天靠的是這個組合，不是時間戳。

**乾淨樹那一回合的收據哪裡來**

收據不進版控，剛 clone 的樹裡沒有它。後設測試在跑第一回合之前先跑一次真的全套把它產出來
（見 ``tests/conftest.py``）。**這支檢查自己絕不補一份**——尺自己造證據，就是 v2 的病。
"""
from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
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
BUCKET_COUNTS = ("tests", "failures", "errors", "skipped")


def _card_files(scan_root: Path, files: list[Path]) -> list[Path]:
    """掃描面的一組：所有規矩卡（收據路徑與地板只寫在卡上，所以每一張都要打開）。"""
    return sorted(f for f in files if f.parent == scan_root / RULES_DIR and f.suffix == ".toml")


def _workflow_files(scan_root: Path, files: list[Path]) -> list[Path]:
    """掃描面的一組：``.github/workflows/`` 底下進得了版控的 workflow。"""
    return sorted(
        f for f in files if f.parent == scan_root / WORKFLOW_DIR and f.suffix in (".yml", ".yaml")
    )


def targets(scan_root: Path, files: list[Path]) -> list[Path]:
    """這支檢查真的會讀的檔：所有規矩卡 ＋ workflow ＋ 那個 job 叫到的腳本 ＋ 對手設定檔。

    收據自己（卡的 ``[junit]`` path）不在裡面：它刻意不進版控（被 .gitignore 蓋住），
    所以列舉集合裡本來就沒有它，宣告那一邊也不會有。這一條是照實記，不是漏掉。
    """
    picked = [*_card_files(scan_root, files), *_workflow_files(scan_root, files)]
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
            "交出來的 testcase 就是這個形狀（classname 空、裡面一個 <error>），那一題根本沒跑到。"
            "先修那個收集錯誤，再看它修好之後的 classname 該落進哪一籃"
        )
    return bad


def _floor_problems(card: Card, rel: str, counts: dict[str, dict[str, int]]) -> list[str]:
    """第 6～7 條：每一個籃子各自比地板與地板過期倍數。"""
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
    counts = {prefix: _empty_counts() for prefix, _floor in card.junit_buckets}
    unassigned: list[str] = []
    unlabelled: list[str] = []
    for case in root.iter("testcase"):
        classname = case.get("classname") or ""
        if not classname:
            unlabelled.append(case.get("name") or "<沒有 name>")
            continue
        prefix = _bucket_prefix(classname, card.junit_buckets)
        if not prefix:
            unassigned.append(classname)
            continue
        bucket = counts[prefix]
        bucket["tests"] += 1
        for attr, tag in COUNT_PAIRS:
            bucket[attr] += len(case.findall(tag))

    bad = _placement_problems(card, rel, unassigned, unlabelled)
    collected = sum(bucket["tests"] for bucket in counts.values()) + len(unassigned) + len(unlabelled)

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
    return bad


if __name__ == "__main__":
    sys.exit(
        run(
            check,
            description="測試的綠燈要是真的綠：junit 收據不准有 skip，每一籃的收集數不准低於卡上的地板",
            targets=targets,
        )
    )
