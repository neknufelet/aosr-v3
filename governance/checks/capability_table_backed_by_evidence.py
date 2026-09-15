#!/usr/bin/env python3
"""能力與驗證範圍表要有證據撐：validated 的每一條指名的考卷與答案檔都真的在，每個入口都真的有模組。

票 #315 的機器版。表住 ``table_path``（一個入口一節、一個條件組合一條，每條標 validated／experimental／
unsupported），前端與 agent 之後只讀這張表決定開不開選項；所以「標了 validated 卻沒有證據」不能出現。

三顆牙（判準全從卡的 ``[settings]`` 讀，程式裡沒有預設值）：

1. **validated 要有證據**——每一條 ``status`` 等於 ``status_validated`` 的組合，``evidence`` 至少指名一個
   考卷節點「考卷檔::測試函式」（考卷住 ``tests_dir`` 底下的 .py、函式名以 ``test_prefix`` 開頭、檔在版控裡、
   用程式結構找得到那個函式）；答案檔（住 ``answers_dir`` 底下、副檔名登記在 ``answer_suffixes``）可以加、
   但單獨不算證據——資料沒有考卷去比它就不算驗過。
2. **入口要真的有模組**——每一節 ``module`` 那個點記法對應的 .py 在 ``product_dir`` 底下版控裡。
3. **狀態只認三個值**——不在 ``status_values`` 裡的字樣紅（自創的狀態等於沒有狀態）。

「指名的考卷跑過且過」不在這裡判：收集考卷證明每條指名的節點收集得到、綠卡證明整份收據沒紅沒跳，
跟精度契約卡同一條鏈（決策紙 precision-contracts-third-tooth-by-collection-and-green.md 的做法）。

**回 2**：讀不到卡的 settings、表不存在或剖不開、表的形狀不對（沒有 entry、entry 沒有 name／module／capability）、
指名的考卷檔剖不開。
"""
from __future__ import annotations

import ast
import sys
import tomllib
from pathlib import Path
from typing import NamedTuple

from governance.exit_codes import ToolBroken, note, run
from governance.loader import setting_strings, setting_text

CARD_ID = "capability-table-backed-by-evidence"
RULES_DIR = "governance/rules"
PYTHON_SUFFIX = ".py"
TEXT_KEYS = ("table_path", "product_dir", "answers_dir", "tests_dir", "test_prefix", "status_validated")
LIST_KEYS = ("status_values", "answer_suffixes")
SETTINGS_KEYS = (*TEXT_KEYS, *LIST_KEYS)
ENTRY_KEY = "entry"
CAPABILITY_KEY = "capability"
NAME_KEY = "name"
MODULE_KEY = "module"
STATUS_KEY = "status"
EVIDENCE_KEY = "evidence"
NODE_SEPARATOR = "::"


class Settings(NamedTuple):
    table_path: str
    product_dir: str
    answers_dir: str
    tests_dir: str
    test_prefix: str
    status_validated: str
    status_values: tuple[str, ...]
    answer_suffixes: tuple[str, ...]


class Row(NamedTuple):
    entry: str
    module: str
    position: int
    status: str
    evidence: tuple[str, ...]


def _card_data(scan_root: Path, files: list[Path]) -> dict[str, object]:
    mine: list[dict[str, object]] = []
    for path in sorted(f for f in files if f.parent == scan_root / RULES_DIR and f.suffix == ".toml"):
        try:
            data = tomllib.loads(path.read_bytes().decode("utf-8"))
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            raise ToolBroken(f"讀不開 {path.relative_to(scan_root)}：{exc}") from exc
        if data.get("id") == CARD_ID:
            mine.append(data)
    if len(mine) != 1:
        raise ToolBroken(f"{scan_root}/{RULES_DIR} 底下 id={CARD_ID} 的卡有 {len(mine)} 張，要剛好一張——判準只寫在卡上")
    return mine[0]


def read_settings(scan_root: Path, files: list[Path]) -> Settings:
    settings = _card_data(scan_root, files).get("settings")
    if not isinstance(settings, dict):
        raise ToolBroken(f"id={CARD_ID} 的卡沒有 [settings] 表")
    extra = [k for k in settings if k not in SETTINGS_KEYS]
    missing = [k for k in SETTINGS_KEYS if k not in settings]
    if extra or missing:
        raise ToolBroken(f"id={CARD_ID} 的卡 [settings] 鍵不對：多了 {sorted(extra)}、少了 {missing}")
    values = tuple(setting_strings(settings, "status_values"))
    validated = setting_text(settings, "status_validated")
    if not values or validated not in values:
        raise ToolBroken(f"status_validated={validated!r} 不在 status_values {list(values)} 裡")
    suffixes = tuple(setting_strings(settings, "answer_suffixes"))
    if not suffixes:
        raise ToolBroken("answer_suffixes 不准是空 list")
    return Settings(
        table_path=setting_text(settings, "table_path"),
        product_dir=setting_text(settings, "product_dir").rstrip("/") + "/",
        answers_dir=setting_text(settings, "answers_dir").rstrip("/") + "/",
        tests_dir=setting_text(settings, "tests_dir").rstrip("/") + "/",
        test_prefix=setting_text(settings, "test_prefix"),
        status_validated=validated,
        status_values=values,
        answer_suffixes=suffixes,
    )


def read_rows(scan_root: Path, files: list[Path], settings: Settings) -> list[Row]:
    """表的形狀：``[[entry]]`` 一節一個入口，底下 ``[[entry.capability]]`` 一條一個組合。壞掉就 ToolBroken。"""
    path = scan_root / settings.table_path
    if path not in files:
        raise ToolBroken(f"能力表 {settings.table_path} 不在這棵樹的版控裡——沒有表就沒有對象")
    try:
        data = tomllib.loads(path.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ToolBroken(f"{settings.table_path} 剖不開：{exc}") from exc
    entries = data.get(ENTRY_KEY)
    if not isinstance(entries, list) or not entries:
        raise ToolBroken(f"{settings.table_path} 裡沒有任何 [[{ENTRY_KEY}]]——表是空的")
    rows: list[Row] = []
    for order, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise ToolBroken(f"{settings.table_path} 第 {order} 節不是一張表")
        name = entry.get(NAME_KEY)
        module = entry.get(MODULE_KEY)
        if not (isinstance(name, str) and name.strip()) or not (isinstance(module, str) and module.strip()):
            raise ToolBroken(f"{settings.table_path} 第 {order} 節缺 {NAME_KEY} 或 {MODULE_KEY}")
        caps = entry.get(CAPABILITY_KEY)
        if not isinstance(caps, list) or not caps:
            raise ToolBroken(f"{settings.table_path} 入口 {name} 底下沒有任何 [[{ENTRY_KEY}.{CAPABILITY_KEY}]]——入口沒有任何組合就沒有東西可宣告")
        for index, cap in enumerate(caps, start=1):
            if not isinstance(cap, dict):
                raise ToolBroken(f"{settings.table_path} 入口 {name} 第 {index} 條不是一張表")
            status = cap.get(STATUS_KEY)
            evidence = cap.get(EVIDENCE_KEY, [])
            if not isinstance(status, str):
                raise ToolBroken(f"{settings.table_path} 入口 {name} 第 {index} 條沒有 {STATUS_KEY}")
            if not isinstance(evidence, list) or not all(isinstance(e, str) for e in evidence):
                raise ToolBroken(f"{settings.table_path} 入口 {name} 第 {index} 條的 {EVIDENCE_KEY} 不是字串 list")
            rows.append(Row(name.strip(), module.strip(), index, status.strip(), tuple(e.strip() for e in evidence)))
    return rows


def _defines_function(tree: ast.Module, name: str) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return True
    return False


def _evidence_problem(scan_root: Path, known: set[Path], row: Row, item: str, settings: Settings) -> str:
    """一項證據：考卷節點或答案檔。回空字串代表沒問題。"""
    where = f"能力表 {row.entry} 第 {row.position} 條"
    if NODE_SEPARATOR in item:
        file_part, _sep, func = item.partition(NODE_SEPARATOR)
        if not file_part.startswith(settings.tests_dir) or not file_part.endswith(".py"):
            return f"{where} 的證據 {item}：考卷要住 {settings.tests_dir} 底下的 .py（寫 src 裡的函式不算考卷）"
        if not func.split("[", 1)[0].startswith(settings.test_prefix):
            return f"{where} 的證據 {item}：測試函式名要以 {settings.test_prefix} 開頭，pytest 才會收它"
        target = scan_root / file_part
        if target not in known:
            return f"{where} 的證據 {item}：考卷檔 {file_part} 不在版控裡"
        try:
            tree = ast.parse(target.read_bytes().decode("utf-8"), filename=file_part)
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            raise ToolBroken(f"{file_part} 剖不開：{exc}") from exc
        func_name = func.split("[", 1)[0]
        if not func_name or not _defines_function(tree, func_name):
            return f"{where} 的證據 {item}：{file_part} 裡沒有定義 {func_name}（用程式結構找，註解裡寫一行不算）"
        return ""
    target = scan_root / item
    if not item.startswith(settings.answers_dir) or not any(item.endswith(s) for s in settings.answer_suffixes):
        return f"{where} 的證據 {item} 既不是「考卷檔::測試函式」也不是 {settings.answers_dir} 底下的答案檔"
    if target not in known:
        return f"{where} 的證據 {item}：答案檔不在版控裡"
    return ""


def _module_paths(module: str, settings: Settings) -> tuple[str, str]:
    """模組點記法對應的兩種檔：單檔模組，或套件的 __init__.py。"""
    base = settings.product_dir + module.replace(".", "/")
    package_marker = "__init__" + PYTHON_SUFFIX
    return base + PYTHON_SUFFIX, f"{base}/{package_marker}"


def check(scan_root: Path, files: list[Path]) -> list[str]:
    settings = read_settings(scan_root, files)
    rows = read_rows(scan_root, files, settings)
    known = set(files)
    bad: list[str] = []
    seen_modules: dict[str, str] = {}
    for row in rows:
        if row.status not in settings.status_values:
            bad.append(f"能力表 {row.entry} 第 {row.position} 條的 status={row.status!r} 不在 {list(settings.status_values)} 裡——自創的狀態等於沒有狀態")
        if row.status == settings.status_validated:
            if not any(NODE_SEPARATOR in item for item in row.evidence):
                bad.append(
                    f"能力表 {row.entry} 第 {row.position} 條標了 {settings.status_validated} 卻沒有指名任何考卷節點"
                    "——答案檔只是資料，沒有考卷去比它就不算驗過；沒證據的宣告前端會照樣開選項"
                )
            for item in row.evidence:
                problem = _evidence_problem(scan_root, known, row, item, settings)
                if problem:
                    bad.append(problem)
        if row.entry not in seen_modules:
            seen_modules[row.entry] = row.module
            if not any((scan_root / rel) in known for rel in _module_paths(row.module, settings)):
                bad.append(f"能力表入口 {row.entry} 指的模組 {row.module} 在 {settings.product_dir} 底下沒有對應的檔（單檔或套件）——表寫了程式沒有")
    note(f"能力表 {len(seen_modules)} 個入口、{len(rows)} 條組合，其中 validated {sum(1 for r in rows if r.status == settings.status_validated)} 條")
    return bad


def targets(scan_root: Path, files: list[Path]) -> list[Path]:
    """掃描面：判決取決於「檔在不在這幾個集合裡」，所以列的是那幾個集合本身，跟卡上宣告的一樣。"""
    settings = read_settings(scan_root, files)
    picked: set[Path] = {scan_root / settings.table_path}
    product = scan_root / settings.product_dir.rstrip("/")
    tests = scan_root / settings.tests_dir.rstrip("/")
    answers = scan_root / settings.answers_dir.rstrip("/")
    picked.update(f for f in files if f.suffix == ".py" and product in f.parents)
    picked.update(f for f in files if f.suffix == ".py" and f.parent == tests)
    picked.update(f for f in files if f.parent == answers and any(f.name.endswith(s) for s in settings.answer_suffixes))
    picked.update(f for f in files if f.parent == scan_root / RULES_DIR and f.suffix == ".toml")
    known = set(files)
    return sorted(p for p in picked if p in known)


if __name__ == "__main__":
    sys.exit(run(check, description="能力與驗證範圍表要有證據撐：validated 指名的考卷與答案檔都在、入口都有模組、狀態只認三個值", targets=targets))
