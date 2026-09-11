#!/usr/bin/env python3
"""規矩卡必填欄位 ＋ 附的樣本真的會咬 ＋ 血債編號與新卡票號。

四關，缺一不可：

**第一關（欄位）**
每張卡的必填欄位、列舉值、掛載點四段都由 :mod:`governance.loader` 驗。
這一關再多一道交叉驗證，不然「執行者」只是換個地方申報假的：

1. 卡宣告的 ``job`` 必須真的出現在 ``.github/workflows/*.yml`` 的某個 job；
2. 那個 job 的步驟裡必須真的呼叫這張卡宣告的 check 模組；
3. 那個 job 必須列在版控裡的 ``governance/required-status-checks.txt``。

三者任一對不上就紅。v2 的 L01 就是「CLAUDE.md 申報執行牙是 pre-commit doc-topology，
而 .git/hooks 底下只有 .sample」——申報了執行者，它其實沒在跑。

**第二關（會咬）**
欄位填滿 ≠ 守衛會咬。對每張卡，把它宣告的每一份必紅樣本（含控制樣本）餵給它自己
宣告的檢查模組，退出碼必須是 1。回 0 就代表這張卡附的是不會咬的樣本，等於沒有守衛
（v2 的五支守衛四支假綠就是這樣過關的）。

**第三關（血債編號要解析得到）**
卡面 ``blood_debt`` 與 ``related_lessons`` 是自報欄位：填一串編出來的編號，載入器收下、
所有檢查照樣綠，那是事故 ``verifier-trusts-self-reported-fields`` 的形狀。這一關把那些
編號拿回 ``<掃描根>/v2-audit/lessons.json`` 的 ``incidents[*].id`` 對，對不到就紅。
讀不到、讀不懂那份檔（不是 JSON／沒有 ``incidents``／``incidents`` 不是 list／某一筆缺 id／
是空的）一律 raise :class:`ToolBroken`（回 2）——沒有原始檔可對，這一跑不算數。
出處：決策紙 ``docs/decisions/card-admission-threshold.md`` 的第一條判準。

**第四關（新卡要有票號）**
卡面 ``admission_issue`` 是候選票的號碼。卡名在舊卡名單裡的不問（那 27 張在開票制度
之前就立了，一個字都不用改）；不在名單裡的就是新卡，沒寫或寫了不是正整數都紅。
名單見下面 :data:`LEGACY_CARD_NAMES`。出處同上一張紙的程序那一節。

掃描面只限 ``governance/`` 底下的結構化宣告、``.github/workflows/``、以及上面那兩關
真的打開來對的 ``v2-audit/lessons.json`` 與 ``blueprint/cards-38.json``，
**不掃全樹散文**：找碴席實測今天的乾淨樹裡 ``v2-audit/lessons.json`` 自己就含
「執行牙」「會擋下」字樣，掃全樹自然語言的寫法會在還沒有任何規矩卡之前就把乾淨樹判紅。
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

from governance.exit_codes import ToolBroken, VIOLATION, note, run
from governance.loader import RULES_DIR, Card, card_problems, load_card

WORKFLOW_DIR = ".github/workflows"
REQUIRED_CHECKS_FILE = "governance/required-status-checks.txt"

# 血債編號對回原始檔的那份檔（相對掃描根）。掃描面宣告的就是這一個檔，不是整個 v2-audit/：
# 旁邊那份 lessons.md 這支檢查一個字都沒讀，宣告了它就會讓宣告面大於實際列舉集合。
LESSONS_REL = "v2-audit/lessons.json"
# 新卡票號那關要讀的舊卡名單來源（相對掃描根）。
BLUEPRINT_REL = "blueprint/cards-38.json"
LEGACY_NOTE = "舊卡名單：卡名在這裡面的不問票號（那些卡在開票制度之前就立了，一個字都不用改）"
ADMISSION_FIELD = "admission_issue"
# 一個正整數的形狀。TOML 把 ``admission_issue = 7`` 讀成 int、``= "7"`` 讀成 str，
# 兩種都是「正的整數」；``1.0``／``abc`` 不是。
POSITIVE_INT_SHAPE = re.compile(r"-?\d+")

# 開票制度之前就立好的 27 張卡。內容取自 ``blueprint/cards-38.json`` 的
# ``meta.establishment``：``established``（29 個名字，其中 4 個已經併進別的卡、
# 今天不在樹上）加上 ``extra_rules_not_in_38``（那兩張不在 38 張裡、但也是在名單之前立的）。
# 為什麼名單寫在這裡而不是卡的 ``[settings]``：第四關要跑得進必紅樣本樹，而樣本樹裡
# 宣告檢查程式的是樣本卡、不是這張卡——名單住在卡的 settings 的話，每一棵樣本樹都會
# 因為「找不到宣告這支檢查的卡」回 2。名單不是門檻（調它不會讓紅變綠，只會把既有卡
# 從「不用票號」改成「要票號」），所以也不必登記進 thresholds-live-only-in-registry 的
# 例外清單。這份名單只增不減：減一個名字，那張既有卡當場被要求補票號。
LEGACY_CARD_NAMES = frozenset(
    {
        "assertions-not-pinned-to-counts",
        "check-exit-code-honest",
        "ci-jobs-cannot-die-quietly",
        "commit-author-allowlisted",
        "decision-paper-structure",
        "derived-content-rendered-not-handwritten",
        "doc-frontmatter-and-dates",
        "doc-size-cap",
        "enforcer-must-be-machine-in-vcs",
        "entry-files-rendered-from-registry",
        "exemptions-need-expiry",
        "file-placement-allowlist",
        "four-roles-different-actors",
        "green-must-be-real-green",
        "identity-strings-generated",
        "issues-closed-only-by-merged-pr",
        "merge-gate-read-back",
        "no-model-names-in-entry-files",
        "prove-the-bite",
        "receipt-authority-is-the-cloud-run",
        "receipt-schema-complete",
        "refs-and-links-resolve",
        "rule-card-required-fields",
        "scan-scope-has-no-holes",
        "secrets-never-committed",
        "status-page-computed-not-typed",
        "style-guard",
        "tests-isolated-from-real-env",
        "thresholds-live-only-in-registry",
        "type-guard",
        "uv-single-entrypoint",
    }
)

# 第二關會遞迴（卡的樣本裡也有卡）。深度到 2 就不再往下，不然會沒完沒了。
MAX_BITE_DEPTH = 2
DEPTH_ENV = "AOSR_BITE_DEPTH"
BITE_TIMEOUT = 300


def _read_data(scan_root: Path, rel: str, what: str) -> object:
    """讀一份對判準有權威的 JSON。讀不到、讀不懂一律 raise :class:`ToolBroken`。

    絕不吞成「沒有違規」：這兩份檔（``lessons.json`` 與 ``cards-38.json``）是那兩關
    唯一的原始檔，沒有它們可對，這一跑就不算數。
    """
    path = scan_root / rel
    if not path.is_file():
        raise ToolBroken(f"讀不到 {rel}（{what}）——原始檔不在，這一跑不出結論")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ToolBroken(f"{rel} 讀不開（{exc}）——我沒看懂就不出結論") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ToolBroken(f"{rel} 不是合法 JSON（{exc}）——我沒看懂就不出結論") from exc


def _incident_ids(scan_root: Path) -> set[str]:
    """``v2-audit/lessons.json`` 裡每一筆事故的 id。形狀不對就回 2，不回 0。"""
    data = _read_data(scan_root, LESSONS_REL, "血債編號要對回原始檔")
    if not isinstance(data, dict):
        raise ToolBroken(f"{LESSONS_REL} 的最外層不是一張表——我沒看懂就不出結論")
    incidents = data.get("incidents")
    if not isinstance(incidents, list):
        raise ToolBroken(f"{LESSONS_REL} 沒有 incidents 這份清單——血債編號沒有東西可以對")
    if not incidents:
        raise ToolBroken(f"{LESSONS_REL} 的 incidents 是空的——血債編號沒有東西可以對")
    ids: set[str] = set()
    for index, incident in enumerate(incidents):
        if not isinstance(incident, dict):
            raise ToolBroken(f"{LESSONS_REL} 的第 {index + 1} 筆事故不是一張表——我沒看懂就不出結論")
        ident = incident.get("id")
        if not isinstance(ident, str) or not ident.strip():
            raise ToolBroken(f"{LESSONS_REL} 的第 {index + 1} 筆事故沒有 id——我沒看懂就不出結論")
        ids.add(ident.strip())
    return ids


def _blueprint_names(scan_root: Path, key: str) -> list[str]:
    """舊卡名單的一格（``meta.establishment`` 底下的字串 list）。形狀不對就回 2。"""
    data = _read_data(scan_root, BLUEPRINT_REL, "舊卡名單的來源")
    if not isinstance(data, dict):
        raise ToolBroken(f"{BLUEPRINT_REL} 的最外層不是一張表——我沒看懂就不出結論")
    meta = data.get("meta")
    if not isinstance(meta, dict):
        raise ToolBroken(f"{BLUEPRINT_REL} 沒有 meta 這張表——舊卡名單住在那裡")
    establishment = meta.get("establishment")
    if not isinstance(establishment, dict):
        raise ToolBroken(f"{BLUEPRINT_REL} 沒有 meta.establishment——舊卡名單住在那裡")
    value = establishment.get(key)
    if not isinstance(value, list) or not all(isinstance(name, str) and name.strip() for name in value):
        raise ToolBroken(
            f"{BLUEPRINT_REL} 的 meta.establishment.{key} 必須是字串 list，實際是 {value!r}"
            "——讀不到舊卡名單就不出結論"
        )
    return [name.strip() for name in value]


def _legacy_card_names(scan_root: Path) -> set[str]:
    """舊卡名單＝登記簿的兩格聯集：``established`` ＋ ``extra_rules_not_in_38``。

    只讀 ``established`` 不夠：今天樹上 27 張卡裡有兩張（``issues-closed-only-by-merged-pr``
    、``uv-single-entrypoint``）被 blueprint 歸在 ``extra_rules_not_in_38``，只認前一格
    會把這兩張既有卡當成新卡、當場要求補票號，整棵樹紅。
    """
    names = set(_blueprint_names(scan_root, "established"))
    names |= set(_blueprint_names(scan_root, "extra_rules_not_in_38"))
    if not names:
        raise ToolBroken(f"{BLUEPRINT_REL} 的舊卡名單是空的——沒有名單就分不出新卡與舊卡")
    return names


def _blood_debt_problems(card: Card, data: dict[str, object], known: set[str]) -> list[str]:
    """第三關：``blood_debt`` 與 ``related_lessons`` 裡每一個編號都要解析得到。"""
    bad: list[str] = []
    for field in ("blood_debt", "related_lessons"):
        value = data.get(field, [])
        if not isinstance(value, list):
            continue
        for ident in value:
            if isinstance(ident, str) and ident.strip() and ident.strip() not in known:
                bad.append(
                    f"卡 {card.id} 的 {field} 寫了事故編號 {ident.strip()!r}，"
                    f"但 {LESSONS_REL} 的 incidents[*].id 裡找不到它"
                    "——血債是自報欄位，填一串編出來的編號看不出來"
                )
    return bad


def _admission_problems(card: Card, data: dict[str, object], legacy: set[str]) -> list[str]:
    """第四關：新卡（卡名不在舊卡名單裡）一定要有正的整數票號。"""
    if card.id in legacy:
        return []
    if ADMISSION_FIELD not in data:
        return [
            f"卡 {card.id} 不在舊卡名單裡（新卡），但沒有 {ADMISSION_FIELD}"
            f"——新卡要寫它從哪一號 GitHub 票討論出來（{LEGACY_NOTE}）"
        ]
    raw = data[ADMISSION_FIELD]
    if isinstance(raw, bool) or not isinstance(raw, (int, str)):
        return [
            f"卡 {card.id} 的 {ADMISSION_FIELD}={raw!r} 既不是整數也不是整數字串"
            "——票號要是一個正整數"
        ]
    if not POSITIVE_INT_SHAPE.fullmatch(str(raw).strip()) or int(str(raw).strip()) < 1:
        return [
            f"卡 {card.id} 的 {ADMISSION_FIELD}={raw!r} 不是正整數"
            "——0、負數、字串與小數都不是票號"
        ]
    return []


def job_blocks(text: str, source: str) -> dict[str, str]:
    """從一份 workflow yml 裡挖出 {job 名: 那個 job 的原文}。green-must-be-real-green 也用它。

    刻意只認 YAML 的一個子集（``jobs:`` 底下同一層縮排的鍵），不引入 yaml 依賴。
    看不懂的檔一律 raise ToolBroken——寧可回 2 說「我沒看懂」，也不要假裝乾淨回 0。
    """
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if re.match(r"^jobs\s*:\s*$", line):
            start = i + 1
            break
    if start is None:
        raise ToolBroken(f"{source} 裡找不到頂層的 jobs:，這份 workflow 我看不懂")

    body = lines[start:]
    indent = None
    for line in body:
        if line.strip() and not line.lstrip().startswith("#"):
            indent = len(line) - len(line.lstrip())
            break
    if not indent:
        raise ToolBroken(f"{source} 的 jobs: 底下沒有任何 job")

    jobs: dict[str, list[str]] = {}
    current: str | None = None
    for line in body:
        if not line.strip() or line.lstrip().startswith("#"):
            if current:
                jobs[current].append(line)
            continue
        here = len(line) - len(line.lstrip())
        if here < indent:
            break
        match = re.match(rf"^ {{{indent}}}([A-Za-z0-9_.\-]+)\s*:", line)
        if here == indent and match:
            current = match.group(1)
            jobs[current] = [line]
        elif current:
            jobs[current].append(line)
    if not jobs:
        raise ToolBroken(f"{source} 的 jobs: 底下解不出任何 job 名")
    return {name: "\n".join(rows) for name, rows in jobs.items()}


def _mount_problems(card: Card, scan_root: Path, files: list[Path]) -> list[str]:
    """交叉驗證：宣告的 job 有沒有真的在崗。"""
    bad: list[str] = []
    workflows = _workflow_files(scan_root, files)
    if not workflows:
        return [f"卡 {card.id} 宣告 job={card.job!r}，但 {WORKFLOW_DIR} 底下一份 workflow 都沒有"]

    found: dict[str, str] = {}
    for wf in workflows:
        for name, block in job_blocks(wf.read_text(encoding="utf-8"), str(wf.relative_to(scan_root))).items():
            found.setdefault(name, block)

    if card.job not in found:
        bad.append(
            f"卡 {card.id} 宣告 job={card.job!r}，但 {WORKFLOW_DIR} 裡沒有這個 job"
            f"（實際有 {sorted(found)}）——申報了執行者，它其實不在崗"
        )
    elif card.check_module not in found[card.job] and card.check not in found[card.job]:
        bad.append(
            f"卡 {card.id} 宣告 job={card.job!r}，那個 job 存在，"
            f"但它的步驟裡沒有呼叫這張卡宣告的檢查模組 {card.check_module}"
        )

    expect = scan_root / REQUIRED_CHECKS_FILE
    if expect not in files:
        bad.append(f"卡 {card.id} 宣告 job={card.job!r}，但版控裡沒有 {REQUIRED_CHECKS_FILE}，無從確認它會卡住合併")
    else:
        listed = [
            ln.strip()
            for ln in expect.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")
        ]
        if card.job not in listed:
            bad.append(
                f"卡 {card.id} 宣告 job={card.job!r}，但它沒被列在 {REQUIRED_CHECKS_FILE}"
                f"（實際列了 {listed}）——不是 required check 就擋不住合併"
            )
    return bad


def _bite_problems(card: Card, scan_root: Path, depth: int) -> list[str]:
    """第二關：卡宣告的每一份必紅樣本，餵給它自己的檢查都必須回 1。"""
    if depth >= MAX_BITE_DEPTH:
        note(f"第二關（會咬）在深度 {depth} 停止遞迴，{card.id} 這一層只驗欄位")
        return []

    cases = card.negative_cases(scan_root)
    control = card.control_path(scan_root)
    if control.is_dir() and control not in cases:
        cases.append(control)
    if not cases:
        return [f"卡 {card.id} 宣告的必紅樣本目錄底下沒有任何樣本——第二關無從證明它會咬"]

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    # 不要在樣本樹裡留 __pycache__：那些是掃描面上的垃圾，管路徑的檢查會看到。
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env[DEPTH_ENV] = str(depth + 1)
    # 不寫 .pyc：上一張卡撞過子程序拿 __pycache__ 舊位元碼、改了程式卻沒生效的坑。
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    bad: list[str] = []
    for case in cases:
        try:
            proc = subprocess.run(
                [sys.executable, "-m", card.check_module, "--scan-root", str(case)],
                cwd=scan_root,
                env=env,
                capture_output=True,
                text=True,
                timeout=BITE_TIMEOUT,
            )
        except subprocess.TimeoutExpired as exc:
            raise ToolBroken(f"跑 {card.check_module} 餵 {case} 超過 {BITE_TIMEOUT} 秒") from exc
        if proc.returncode != VIOLATION:
            # 把子程序自己說的最後一句帶出來：只寫「回 2」看不出是哪一條路回的 2
            # （2026-09-10 雲端實測：merge-gate-read-back 的樣本在雲端回 2、本機回 1，沒有這一句查不下去）。
            said = (proc.stderr.strip().splitlines() or ["（子程序沒說話）"])[-1][:300]
            bad.append(
                f"卡 {card.id} 的樣本 {case.relative_to(scan_root)} 餵給 {card.check_module} 回 "
                f"{proc.returncode}，應為 1——填滿欄位不算有牙，附的樣本必須真的讓檢查回 1；它說：{said}"
            )
    return bad


def _card_files(scan_root: Path, files: list[Path]) -> list[Path]:
    """掃描面的一組：所有規矩卡。"""
    return sorted(f for f in files if f.parent == scan_root / RULES_DIR and f.suffix == ".toml")


def _workflow_files(scan_root: Path, files: list[Path]) -> list[Path]:
    """掃描面的一組：``.github/workflows/`` 底下進得了版控的 workflow。"""
    return sorted(
        f for f in files if f.parent == scan_root / WORKFLOW_DIR and f.suffix in (".yml", ".yaml")
    )


def targets(scan_root: Path, files: list[Path]) -> list[Path]:
    """這支檢查真的會讀／會判的檔：所有規矩卡 ＋ workflow ＋ required 名單 ＋ 卡指到的檢查模組
    ＋ 血債編號與舊卡名單那兩份原始檔。

    檢查模組算在裡面是因為第一關真的對它下判斷（「check 指向的模組不存在」是一筆違規），
    而 ``governance/checks/`` 那層的套件標記不算——沒有卡指到它，它不是檢查程式。
    必紅樣本樹不算：第二關是把**另一支程式**餵給那些樹，這支檢查自己不讀它們的內容。
    那兩份原始檔算在裡面是因為第三、第四關真的打開來讀（掃描面宣告了就要真的掃到）。
    """
    picked = [*_card_files(scan_root, files), *_workflow_files(scan_root, files)]
    expect = scan_root / REQUIRED_CHECKS_FILE
    if expect in files:
        picked.append(expect)
    for rel in (LESSONS_REL, BLUEPRINT_REL):
        if scan_root / rel in files:
            picked.append(scan_root / rel)
    for path in _card_files(scan_root, files):
        try:
            data = tomllib.loads(path.read_bytes().decode("utf-8"))
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            raise ToolBroken(f"讀不開 {path.relative_to(scan_root)}：{exc}") from exc
        declared = data.get("check")
        if isinstance(declared, str) and scan_root / declared in files:
            picked.append(scan_root / declared)
    return sorted(set(picked))


def check(scan_root: Path, files: list[Path]) -> list[str]:
    depth = int(os.environ.get(DEPTH_ENV, "0"))
    rules = _card_files(scan_root, files)
    if not rules:
        raise ToolBroken(f"{scan_root}/{RULES_DIR} 底下一張版控裡的規矩卡都沒有——這一跑沒掃到東西")
    # 兩份原始檔先讀：讀不到就回 2，不進到逐張卡的判斷（第三、第四關唯一的證據就是它們）。
    known = _incident_ids(scan_root)
    legacy = _legacy_card_names(scan_root)

    bad: list[str] = []
    for path in rules:
        problems = card_problems(path, scan_root)
        if problems:
            bad += [f"{path.relative_to(scan_root)}：{p}" for p in problems]
            continue
        card = load_card(path, scan_root)
        data = tomllib.loads(path.read_bytes().decode("utf-8"))
        bad += _blood_debt_problems(card, data, known)
        bad += _admission_problems(card, data, legacy)
        bad += _mount_problems(card, scan_root, files)
        bad += _bite_problems(card, scan_root, depth)
    return bad


if __name__ == "__main__":
    sys.exit(
        run(
            check,
            description="規矩卡必填欄位、附的樣本真的會讓檢查回 1，且血債編號與新卡票號都對得回原始檔",
            targets=targets,
        )
    )
