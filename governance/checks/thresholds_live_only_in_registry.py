#!/usr/bin/env python3
"""門檻數字只准住在卡的登記簿：檢查程式與卡的人話裡不准再寫一次。

**查形狀，不按數值比對。** 舊設計是「拿登記簿上的值去 grep 全樹」，那條路實測走不通：
門檻數量到兩個的當下就開始誤傷（pytest 的「沒收到測試」離開碼會被當成測試地板重寫），
而且同值同單位不同意思的數字分不出來。所以這裡改成問一個機器答得出來的問題——
**這個數字是不是寫在「模組載入時就算出來」的位置**。

兩條規則，名單與例外只寫在卡的 ``[settings]``，讀不到就回 2（工具自壞），不回 0：

1. **程式裡不准有載入時就算好的門檻**。掃描面是掃描根自己的 ``governance/*.py`` 與
   ``governance/checks/*.py``（那兩層，用 ``git ls-files`` 的集合）。算「載入時就算好」的
   位置有三種：模組層級的賦值（藏在模組層的 ``if``／``try`` 底下、或類別身體裡一樣算）、
   函式簽章的預設值、以及賦值運算式裡包起來的容器與 lambda。**函式身體裡的字面值不看**
   ——那是演算法（切片、索引、位移），不是門檻。放行只有兩種：值就是離開碼約定的那三個數
   （唯一一份住在 ``governance/exit_codes.py``，這裡 import 進來用，不自己再寫一次），
   或具名登記在卡的 ``[[settings.allow]]``（``file``＋``name``＋``reason``）。名字命中卡上
   登記的門檻樣式（``MAX_*``、``*_TIMEOUT`` 這些）時，值落在離開碼白名單裡也一樣要登記
   ——不然「深度上限」這種門檻只要把值寫成那三個數之一就整條繞過去。

2. **卡的人話不准重抄自己登記的門檻**。掃描根自己的 ``governance/rules/*.toml``，每張卡的
   ``human`` 欄裡出現的數字，不准等於同一張卡登記表（哪幾張表由卡上的 ``registry_tables``
   決定）裡登記的值。人話抄一次，之後改的是登記表那一份，人話那一份不會跟著改。
   值等於離開碼那三個數的門檻放掉——卡的人話到處在寫「回 2 不回 0」，不放掉會誤判。

血債與關聯事故寫在卡上（``governance/rules/thresholds-live-only-in-registry.toml``），
這裡不重複；那張卡刻意沒管的四件事也在卡面。

這支程式自己受這條規矩管（它就在掃描面裡）：從頭到尾沒有一個模組層級的數字字面值。
"""
from __future__ import annotations

import ast
import re
import sys
import tomllib
from collections.abc import Iterator
from fnmatch import fnmatchcase
from pathlib import Path

from governance.exit_codes import CLEAN, TOOL_BROKEN, VIOLATION, ToolBroken, run
from governance.loader import CHECKS_DIR, RULES_DIR

# 這張卡的 id。名單與例外只從「id 是這個」的那張卡讀。為什麼靠 id 認卡而不靠 check 欄：
# 必紅樣本是一棵棵獨立的迷你掃描根，每棵樹裡都得放一張卡（名單在卡上），而樣本樹裡沒有
# governance/checks/ 底下那支程式——寫了 check 欄，每份樣本都會多一筆無關的違規。
CARD_ID = "thresholds-live-only-in-registry"

# 掃描面的第一層：治理層自己那一層的 .py。第二層是 loader 的 CHECKS_DIR。
GOVERNANCE_DIR = "governance"
PYTHON_SUFFIX = ".py"
TOML_SUFFIX = ".toml"
HUMAN_FIELD = "human"

# 門檻的形狀。打錯字的名單等於沒有名單，所以多一個鍵、少一個鍵、型別不對，一律回 2。
LIST_KEYS = ("threshold_name_patterns", "registry_tables")
ALLOW_KEY = "allow"
ALLOW_ENTRY_KEYS = ("file", "name", "reason")
SETTINGS_KEYS = (*LIST_KEYS, ALLOW_KEY)

# 唯一的值白名單：離開碼約定那三個數。刻意 import 名字而不是寫值——這張卡自己的規矩就是
# 「同一個數字不准有第二個家」，它們的家是 governance/exit_codes.py。
EXIT_CODE_VALUES = (CLEAN, VIOLATION, TOOL_BROKEN)

# 人話裡的數字。兩頭的 lookbehind／lookahead 是為了不把識別字裡的數字讀成數字：
# `v2 事故`、`L07`、`T20`、`F401` 不算，`200 行` 算。
NUMBER_RE = re.compile(r"(?<![0-9A-Za-z_])(\d+(?:\.\d+)?)(?![0-9A-Za-z_])")

# 不往裡面看的節點：函式身體。簽章的預設值另外單獨看（那是載入時就算好的）。
FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)


def _read_text(path: Path, rel: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ToolBroken(f"讀不到 {rel}：{exc}") from exc


def _load_toml(path: Path, rel: str) -> dict[str, object]:
    try:
        return tomllib.loads(_read_text(path, rel))
    except tomllib.TOMLDecodeError as exc:
        raise ToolBroken(f"{rel} 解不開（{exc}）——我沒看懂就不出結論") from exc


def _card_files(scan_root: Path, files: list[Path]) -> list[Path]:
    return sorted(f for f in files if f.parent == scan_root / RULES_DIR and f.suffix == TOML_SUFFIX)


def _card_settings(scan_root: Path, files: list[Path]) -> dict[str, object]:
    """從掃描根自己的 ``governance/rules/`` 讀這張卡的名單與例外清單。

    找不到、找到多張、或形狀不對，一律 raise ToolBroken——讀不到登記簿就不出結論。
    """
    mine: list[tuple[str, dict[str, object]]] = []
    for path in _card_files(scan_root, files):
        rel = path.relative_to(scan_root).as_posix()
        data = _load_toml(path, rel)
        if data.get("id") == CARD_ID:
            mine.append((rel, data))
    if len(mine) != 1:
        raise ToolBroken(
            f"{scan_root}/{RULES_DIR} 底下 id={CARD_ID} 的卡有 {len(mine)} 張，要剛好一張"
            "——名單與例外清單只寫在卡上，讀不到這一跑就不算數"
        )
    rel, data = mine[0]
    settings = data.get("settings")
    if not isinstance(settings, dict):
        raise ToolBroken(f"{rel} 沒有 [settings] 表（門檻樣式、登記表清單與例外清單都寫在那裡）")
    _assert_settings(settings, rel)
    return settings


def _assert_settings(settings: dict[str, object], rel: str) -> None:
    bad: list[str] = []
    extra = [k for k in settings if k not in SETTINGS_KEYS]
    if extra:
        bad.append(f"多了不認識的鍵 {sorted(extra)}，只認 {list(SETTINGS_KEYS)}（打錯字的名單等於沒名單）")
    for key in LIST_KEYS:
        value = settings.get(key)
        if key not in settings:
            bad.append(f"缺 {key}")
        elif not isinstance(value, list) or not value or not all(isinstance(s, str) and s.strip() for s in value):
            bad.append(f"{key} 必須是非空的字串 list，實際是 {value!r}")
    allow = settings.get(ALLOW_KEY, [])
    if not isinstance(allow, list) or not all(isinstance(x, dict) for x in allow):
        bad.append(f"{ALLOW_KEY} 必須是 [[settings.allow]] 表陣列（沒有也要明寫 {ALLOW_KEY} = []），實際是 {allow!r}")
    else:
        for index, entry in enumerate(allow):
            where = f"[[settings.{ALLOW_KEY}]] 第 {index + 1} 條"
            missing = [k for k in ALLOW_ENTRY_KEYS if k not in entry]
            if missing:
                bad.append(f"{where} 缺 {missing}——例外要說清楚是哪個檔、哪個名字、為什麼不是門檻")
            odd = [k for k in entry if k not in ALLOW_ENTRY_KEYS]
            if odd:
                bad.append(f"{where} 多了不認識的鍵 {sorted(odd)}，只認 {list(ALLOW_ENTRY_KEYS)}")
            for key in ALLOW_ENTRY_KEYS:
                value = entry.get(key)
                if key in entry and (not isinstance(value, str) or not value.strip()):
                    bad.append(f"{where} 的 {key} 必須是非空字串，實際是 {value!r}")
    if bad:
        raise ToolBroken(f"{rel} 的 [settings] 形狀不對：" + "；".join(bad))


def _numbers(node: ast.AST) -> list[float]:
    """一段運算式裡所有的數字字面值（負號跟著算，true／false 不算）。"""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        inner = _numbers(node.operand)
        return [-value for value in inner] if isinstance(node.op, ast.USub) else inner
    if isinstance(node, ast.Constant):
        ok = isinstance(node.value, (int, float)) and not isinstance(node.value, bool)
        return [node.value] if ok else []
    found: list[float] = []
    for child in ast.iter_child_nodes(node):
        found += _numbers(child)
    return found


def _default_entries(node: ast.FunctionDef | ast.AsyncFunctionDef, prefix: str) -> Iterator[tuple[str, str, int, list[float]]]:
    """函式簽章的預設值。位置參數的預設值靠尾端對齊，關鍵字參數一對一。"""
    args = node.args
    positional = [*args.posonlyargs, *args.args]
    paired = list(zip(positional[len(positional) - len(args.defaults) :], args.defaults))
    paired += [(a, d) for a, d in zip(args.kwonlyargs, args.kw_defaults) if d is not None]
    for arg, default in paired:
        values = _numbers(default)
        if values:
            yield f"{prefix}{node.name}({arg.arg})", arg.arg, default.lineno, values


def _load_time_entries(body: list[ast.stmt], prefix: str) -> Iterator[tuple[str, str, int, list[float]]]:
    """走一份檔裡「模組載入時就算好」的位置，回 (顯示名, 比對名, 行號, 數字)。

    進得去的：模組層級、模組層的 ``if``／``try``／``with``／``for`` 底下、類別身體、
    以及函式簽章的預設值。進不去的：函式身體（那裡的數字是演算法，不是門檻）。
    """
    for node in body:
        if isinstance(node, FUNCTION_NODES):
            yield from _default_entries(node, prefix)
            continue
        if isinstance(node, ast.ClassDef):
            yield from _load_time_entries(node.body, f"{prefix}{node.name}.")
            continue
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)) and node.value is not None:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [ast.unparse(t) for t in targets]
            values = _numbers(node.value)
            if values:
                shown = "／".join(names) or "<賦值>"
                yield f"{prefix}{shown}", names[-1] if names else shown, node.lineno, values
            continue
        for field in ("body", "orelse", "finalbody"):
            inner = getattr(node, field, None)
            if isinstance(inner, list) and all(isinstance(s, ast.stmt) for s in inner):
                yield from _load_time_entries(inner, prefix)
        for handler in getattr(node, "handlers", []) or []:
            yield from _load_time_entries(handler.body, prefix)


def _scanned_python(scan_root: Path, files: list[Path]) -> list[Path]:
    homes = (scan_root / GOVERNANCE_DIR, scan_root / CHECKS_DIR)
    return sorted(f for f in files if f.suffix == PYTHON_SUFFIX and f.parent in homes)


def targets(scan_root: Path, files: list[Path]) -> list[Path]:
    """這支檢查真的會讀的檔：治理層那兩層的 ``.py`` ＋ 所有規矩卡。

    第①條讀程式（載入時就算好的門檻），第②條讀卡的人話與登記簿。別處的 .py 不掃
    （測試自己的數字不是門檻），樣本樹裡的道具也不掃。
    """
    return sorted({*_scanned_python(scan_root, files), *_card_files(scan_root, files)})


def _threshold_pattern(name: str, patterns: list[str]) -> str:
    """名字命中的第一個門檻樣式（大小寫都拉成大寫再比）。沒命中回空字串。"""
    bare = name.rsplit(".", maxsplit=1)[-1].upper()
    for pattern in patterns:
        if fnmatchcase(bare, pattern.upper()):
            return pattern
    return ""


def _programs_problems(
    scan_root: Path, files: list[Path], settings: dict[str, object], used: set[tuple[str, str]]
) -> list[str]:
    """第①條：程式裡不准有載入時就算好的門檻。"""
    patterns = [str(p) for p in settings["threshold_name_patterns"]]  # type: ignore[union-attr]
    registered = {
        (str(entry["file"]), str(entry["name"])) for entry in settings.get(ALLOW_KEY, [])  # type: ignore[union-attr,index]
    }
    programs = _scanned_python(scan_root, files)
    if not programs:
        raise ToolBroken(
            f"{scan_root} 底下 {GOVERNANCE_DIR}／{CHECKS_DIR} 那兩層一支版控裡的 .py 都沒有"
            "——第①條掃到的是空集合，「沒找到違規」不算數"
        )

    bad: list[str] = []
    for path in programs:
        rel = path.relative_to(scan_root).as_posix()
        text = _read_text(path, rel)
        try:
            tree = ast.parse(text, filename=rel)
        except SyntaxError as exc:
            raise ToolBroken(f"{rel} 剖不開（{exc}）——我沒看懂就不出結論") from exc
        for shown, bare, lineno, values in _load_time_entries(tree.body, ""):
            key = (rel, shown)
            if key in registered:
                used.add(key)
                continue
            hit = _threshold_pattern(bare, patterns)
            if not all(value in EXIT_CODE_VALUES for value in values):
                bad.append(
                    f"{rel}:{lineno} {shown} 在模組載入時就把數字寫死了"
                    f"（{_shown_values(values)}）——門檻只准住在卡的登記簿，檢查程式從卡讀進來；"
                    f"真的不是判決門檻（列舉、協定碼、看門狗秒數）就在卡 {CARD_ID} 的"
                    f" [[settings.{ALLOW_KEY}]] 具名登記並寫理由"
                )
            elif hit:
                bad.append(
                    f"{rel}:{lineno} {shown} 的名字命中卡上登記的門檻樣式 {hit!r} 又帶了數字"
                    f"（{_shown_values(values)}）——值落在離開碼白名單裡也不放行，不然任何門檻"
                    f"只要把值寫成離開碼那三個數之一就整條繞過去了。真的不是判決門檻就在卡"
                    f" {CARD_ID} 的 [[settings.{ALLOW_KEY}]] 具名登記並寫理由"
                )
    return bad


def _shown_values(values: list[float]) -> str:
    return "、".join(str(value) for value in values)


def _registered_numbers(data: dict[str, object], tables: list[str]) -> set[float]:
    """一張卡在登記表裡登記的數值（遞迴進子表與陣列，離開碼那三個數不算）。"""
    found: set[float] = set()

    def walk(node: object) -> None:
        if isinstance(node, bool):
            return
        if isinstance(node, (int, float)):
            found.add(node)
        elif isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    for table in tables:
        if table in data:
            walk(data[table])
    return {value for value in found if value not in EXIT_CODE_VALUES}


def _human_numbers(human: str) -> set[float]:
    found: set[float] = set()
    for match in NUMBER_RE.finditer(human):
        raw = match.group(1)
        found.add(float(raw) if "." in raw else int(raw))
    return found


def _cards_problems(scan_root: Path, files: list[Path], settings: dict[str, object]) -> list[str]:
    """第②條：卡的人話不准重抄自己登記的門檻。"""
    tables = [str(t) for t in settings["registry_tables"]]  # type: ignore[union-attr]
    bad: list[str] = []
    for path in _card_files(scan_root, files):
        rel = path.relative_to(scan_root).as_posix()
        data = _load_toml(path, rel)
        human = data.get(HUMAN_FIELD)
        if not isinstance(human, str):
            continue
        registered = _registered_numbers(data, tables)
        for number in sorted(_human_numbers(human) & registered):
            bad.append(
                f"{rel} 的 {HUMAN_FIELD} 欄寫了 {number}，那個數字已經登記在同一張卡的"
                f" {tables} 裡——人話重抄一次門檻，之後改的是登記表那一份，人話那一份不會跟著改，"
                "那個數字當場開始漂（v2 「規範檔重抄數值必漂」的形狀）。人話改成講形狀"
                "（「超過卡上登記的個數」），數字留在登記表"
            )
    return bad


def check(scan_root: Path, files: list[Path]) -> list[str]:
    settings = _card_settings(scan_root, files)
    used: set[tuple[str, str]] = set()
    bad = _programs_problems(scan_root, files, settings, used)
    bad += _cards_problems(scan_root, files, settings)

    allow = settings.get(ALLOW_KEY, [])
    stale = [
        f"{entry['file']}／{entry['name']}"  # type: ignore[index]
        for entry in allow  # type: ignore[union-attr]
        if (str(entry["file"]), str(entry["name"])) not in used  # type: ignore[index]
    ]
    if stale:
        print(
            f"NOTE: 卡上這幾條例外這一跑沒有被用到，可能已經過期，該回卡上刪掉：{stale}",
            file=sys.stderr,
        )
    return bad


if __name__ == "__main__":
    sys.exit(
        run(
            check,
            description="門檻數字只准住在卡的登記簿，檢查程式與卡的人話裡不准再寫一次",
            targets=targets,
        )
    )
