#!/usr/bin/env python3
"""基礎物理量單一來源：產品程式不准手寫聲速那幾個數、凍結答案要記自己的條件、設定檔不准存 ρc。

票 #296 的機器版（改窄後的三條；原草案「答案檔頭要等於設定檔」刪掉——凍結案例用案例自己的條件）。
上一代事故 physics-constants-duplicated-without-ssot：ρ₀ 三處三值、考卷拿實作值當期望自證。

三條（判準全從卡的 ``[settings]`` 讀，程式裡沒有預設值）：

1. **產品程式不准手寫基礎物理量**——``product_dir`` 底下每一支 .py：呼叫時關鍵字引數名命中
   ``physics_names``（聲速、密度、參考聲壓、黏度、ρc）而值是數字（字面值、純數字運算式、或欄位工廠
   ``Field(default=…)`` 包著數字）即紅；模組層級、類別身體、函式簽章預設值裡同名（不分大小寫）的數字也紅；
   除了 ``config_dir`` 那一層之外，匯入或呼叫 ``loader_name``（讀產品設定的那支函式，取別名一樣）也紅；
   設定層自己在模組載入時就呼叫它（模組層、類別身體、簽章預設值）也紅——物理條件由呼叫端傳進來，
   不是模組自己去拿預設，也不准設定層先讀好包成第二個家。
2. **凍結答案要記自己的物理條件**——``blueprint/`` 底下檔名命中 ``answer_patterns`` 的 json，
   ``parameters`` 裡要有 ``speed_keys`` 之一；除了 ``rho_c_optional_patterns`` 命中的（只有幾何、
   沒有材料的答案）之外還要有 ``rho_c_keys`` 之一。條件記在答案檔頭，讀答案的考卷才有東西可餵。
3. **產品設定不准存 ρc**——``physics_toml`` 裡任何鍵名含 ``forbidden_key_parts`` 之一即紅：
   ρc 由密度乘聲速算，存一份就是第二個家。

**刻意沒管**：考卷從答案檔取物理條件、不手寫——性質考卷開答案檔拿幾何再自己給聲速是合理的，機器分不出
「該用答案檔的」與「自己給的」，那一格靠驗收席看 diff。函式裡的賦值與運算字面值不管（呼叫時的關鍵字引數在哪都咬）；
取了別的名字的變數（``c = 343.0``）、位置引數、字典字面值裡的數字、自己用 tomllib 讀那份設定檔，都看不到。

**回 2**：讀不到卡的 settings、設定檔不存在或剖不開、某份受管答案 json 剖不開、產品 .py 剖不開。
"""
from __future__ import annotations

import ast
import fnmatch
import json
import sys
import tomllib
from pathlib import Path
from typing import NamedTuple

from governance.exit_codes import ToolBroken, note, run
from governance.loader import setting_strings, setting_text

CARD_ID = "physics-constants-single-source"
RULES_DIR = "governance/rules"
PYTHON_SUFFIX = ".py"
TEXT_KEYS = ("product_dir", "config_dir", "loader_name", "physics_toml", "answers_dir")
LIST_KEYS = (
    "physics_names",
    "answer_patterns",
    "speed_keys",
    "rho_c_keys",
    "rho_c_optional_patterns",
    "forbidden_key_parts",
)
SETTINGS_KEYS = (*TEXT_KEYS, *LIST_KEYS)
PARAMETERS_KEY = "parameters"


class Settings(NamedTuple):
    product_dir: str
    config_dir: str
    loader_name: str
    physics_toml: str
    answers_dir: str
    physics_names: tuple[str, ...]
    answer_patterns: tuple[str, ...]
    speed_keys: tuple[str, ...]
    rho_c_keys: tuple[str, ...]
    rho_c_optional_patterns: tuple[str, ...]
    forbidden_key_parts: tuple[str, ...]


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
    texts = {key: setting_text(settings, key) for key in TEXT_KEYS}
    lists = {key: tuple(setting_strings(settings, key)) for key in LIST_KEYS}
    for key in ("physics_names", "answer_patterns", "speed_keys", "rho_c_keys", "forbidden_key_parts"):
        if not lists[key]:
            raise ToolBroken(f"id={CARD_ID} 的卡 [settings] {key} 不准是空 list")
    return Settings(
        product_dir=texts["product_dir"].rstrip("/") + "/",
        config_dir=texts["config_dir"].rstrip("/") + "/",
        loader_name=texts["loader_name"],
        physics_toml=texts["physics_toml"],
        answers_dir=texts["answers_dir"].rstrip("/") + "/",
        physics_names=lists["physics_names"],
        answer_patterns=lists["answer_patterns"],
        speed_keys=lists["speed_keys"],
        rho_c_keys=lists["rho_c_keys"],
        rho_c_optional_patterns=lists["rho_c_optional_patterns"],
        forbidden_key_parts=lists["forbidden_key_parts"],
    )


# ── 第 1 條：產品程式 ────────────────────────────────────────────────────────


FIELD_FACTORIES = ("Field", "field")


def _callee_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _is_number(node: ast.expr) -> bool:
    """數字字面值、兩邊都是數字的四則與次方運算（``1.2 * 343.0`` 正是上一代事故的原形），
    或資料模型的欄位工廠 ``Field(343.0)``／``Field(default=343.0)``／``field(default=343.0)`` 包著數字。"""
    if isinstance(node, ast.Call) and _callee_name(node) in FIELD_FACTORIES:
        first = node.args[0] if node.args else None
        default = next((k.value for k in node.keywords if k.arg == "default"), None)
        return any(v is not None and _is_number(v) for v in (first, default))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return _is_number(node.operand)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
        return _is_number(node.left) and _is_number(node.right)
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool)


def _product_python(scan_root: Path, files: list[Path], settings: Settings) -> list[Path]:
    home = scan_root / settings.product_dir.rstrip("/")
    return sorted(f for f in files if f.suffix == PYTHON_SUFFIX and home in f.parents)


def _default_entries(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[tuple[str, int]]:
    """函式簽章裡預設值是數字的參數名與行號（``def f(sound_speed_m_s=343.0)`` 也是第二個家）。"""
    args = node.args
    positional = [*args.posonlyargs, *args.args]
    paired = list(zip(positional[len(positional) - len(args.defaults):], args.defaults))
    paired += [(a, d) for a, d in zip(args.kwonlyargs, args.kw_defaults) if d is not None]
    return [(arg.arg, default.lineno) for arg, default in paired if _is_number(default)]


def _load_time_assignments(body: list[ast.stmt]) -> list[tuple[str, int]]:
    """模組層級（含模組層的 if／try 底下）與類別身體裡的賦值目標名與行號，加上函式簽章的數字預設值。"""
    out: list[tuple[str, int]] = []
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out += _default_entries(node)
            continue
        if isinstance(node, ast.ClassDef):
            out += _load_time_assignments(node.body)
            continue
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None and _is_number(node.value):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    out.append((target.id, node.lineno))
            continue
        for field in ("body", "orelse", "finalbody"):
            inner = getattr(node, field, None)
            if isinstance(inner, list) and all(isinstance(s, ast.stmt) for s in inner):
                out += _load_time_assignments(inner)
        for handler in getattr(node, "handlers", []) or []:
            out += _load_time_assignments(handler.body)
    return out


def _load_time_calls(body: list[ast.stmt], loader_name: str) -> list[int]:
    """模組載入時就會執行的呼叫（模組層、類別身體、簽章預設值、模組層的 if／try 底下）裡叫到載入器的行號。"""
    out: list[int] = []
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defaults = [*node.args.defaults, *[d for d in node.args.kw_defaults if d is not None]]
            for default in defaults:
                out += [n.lineno for n in ast.walk(default) if isinstance(n, ast.Call) and _callee_name(n) == loader_name]
            continue
        if isinstance(node, ast.ClassDef):
            out += _load_time_calls(node.body, loader_name)
            continue
        nested: list[ast.stmt] = []
        for field in ("body", "orelse", "finalbody"):
            inner = getattr(node, field, None)
            if isinstance(inner, list) and all(isinstance(s, ast.stmt) for s in inner):
                nested += inner
        for handler in getattr(node, "handlers", []) or []:
            nested += handler.body
        if nested:
            out += _load_time_calls(nested, loader_name)
            continue
        out += [n.lineno for n in ast.walk(node) if isinstance(n, ast.Call) and _callee_name(n) == loader_name]
    return out


def _product_problems(scan_root: Path, files: list[Path], settings: Settings) -> list[str]:
    bad: list[str] = []
    names = {n.lower() for n in settings.physics_names}
    config_home = scan_root / settings.config_dir.rstrip("/")
    for path in _product_python(scan_root, files, settings):
        rel = path.relative_to(scan_root).as_posix()
        try:
            tree = ast.parse(path.read_bytes().decode("utf-8"), filename=rel)
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            raise ToolBroken(f"{rel} 剖不開：{exc}——產品程式有一支讀不了，這一跑不算數") from exc
        for name, lineno in _load_time_assignments(tree.body):
            if name.lower() in names:
                bad.append(f"{rel}:{lineno} 把 {name} 寫成載入時就算好的數字（模組常數、類別欄位或簽章預設值）——基礎物理量只准住 {settings.physics_toml}，這是第二個家")
        in_config = config_home in path.parents
        if in_config:
            for lineno in _load_time_calls(tree.body, settings.loader_name):
                bad.append(
                    f"{rel}:{lineno} 設定層在模組載入時就呼叫 {settings.loader_name}——載入時就把預設讀進來再給別層用，"
                    "等於把產品預設包成第二個家；設定層只准在函式裡（呼叫端要的時候）讀"
                )
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and not in_config:
                if any(alias.name == settings.loader_name for alias in node.names):
                    bad.append(f"{rel}:{node.lineno} 在設定層之外把 {settings.loader_name} 匯進來（取別名也一樣）——物理模組不准自己去拿產品預設")
                continue
            if not isinstance(node, ast.Call):
                continue
            for keyword in node.keywords:
                if keyword.arg and keyword.arg.lower() in names and _is_number(keyword.value):
                    bad.append(
                        f"{rel}:{node.lineno} 呼叫時把 {keyword.arg} 手寫成數字——物理條件從設定檔或案例檔傳進來，不准在產品程式裡寫死"
                    )
            if not in_config and _callee_name(node) == settings.loader_name:
                bad.append(
                    f"{rel}:{node.lineno} 在設定層之外呼叫 {settings.loader_name}——物理模組不准自己去拿產品預設，條件由呼叫端傳進來（凍結案例用案例自己的）"
                )
    return bad


# ── 第 2 條：凍結答案 ────────────────────────────────────────────────────────


def _recorded_number(value: object) -> bool:
    """答案檔記的值：數字，或上一代那種 ``{"dec": "343.0", "hex": "0x1.57p+8"}`` 兩格的寫法（hex 解得回浮點）。"""
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, dict):
        raw = value.get("hex")
        if isinstance(raw, str):
            try:
                float.fromhex(raw)
            except ValueError:
                return False
            return True
        raw = value.get("dec")
        if isinstance(raw, str):
            try:
                float(raw)
            except ValueError:
                return False
            return True
    return False


def _answer_files(scan_root: Path, files: list[Path], settings: Settings) -> list[Path]:
    home = scan_root / settings.answers_dir.rstrip("/")
    return sorted(
        f for f in files if f.parent == home and any(fnmatch.fnmatchcase(f.name, p) for p in settings.answer_patterns)
    )


def _answer_problems(scan_root: Path, files: list[Path], settings: Settings) -> list[str]:
    bad: list[str] = []
    for path in _answer_files(scan_root, files, settings):
        rel = path.relative_to(scan_root).as_posix()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ToolBroken(f"{rel} 剖不開：{exc}——受管答案讀不了，這一跑不算數") from exc
        params = data.get(PARAMETERS_KEY) if isinstance(data, dict) else None
        if not isinstance(params, dict):
            bad.append(f"{rel} 沒有 {PARAMETERS_KEY} 這一節——凍結答案要記自己的物理條件，沒有就沒東西可餵給考卷")
            continue
        if not any(_recorded_number(params.get(key)) for key in settings.speed_keys):
            bad.append(f"{rel} 的 {PARAMETERS_KEY} 沒有聲速（{list(settings.speed_keys)} 之一）——答案是在哪個聲速下算的沒寫，考卷只能猜或拿產品預設")
        optional = any(fnmatch.fnmatchcase(path.name, p) for p in settings.rho_c_optional_patterns)
        if not optional and not any(_recorded_number(params.get(key)) for key in settings.rho_c_keys):
            bad.append(f"{rel} 的 {PARAMETERS_KEY} 沒有 ρc（{list(settings.rho_c_keys)} 之一）——有材料的答案沒記 ρc，阻抗就對不回吸音率")
    return bad


# ── 第 3 條：設定檔 ─────────────────────────────────────────────────────────


def _toml_problems(scan_root: Path, files: list[Path], settings: Settings) -> list[str]:
    path = scan_root / settings.physics_toml
    if path not in files:
        raise ToolBroken(f"設定檔 {settings.physics_toml} 不在這棵樹的版控裡——沒有唯一來源就不出結論")
    try:
        data = tomllib.loads(path.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ToolBroken(f"{settings.physics_toml} 剖不開：{exc}") from exc
    bad: list[str] = []

    def walk(table: dict[str, object], prefix: str) -> None:
        for key, value in table.items():
            padded = f"_{key.lower()}_"
            if any(f"_{part.lower()}_" in padded for part in settings.forbidden_key_parts):
                bad.append(f"{settings.physics_toml} 存了 {prefix}{key}——ρc 由密度乘聲速算，存一份就是第二個家，之後改密度它不會跟著動")
            if isinstance(value, dict):
                walk(value, f"{prefix}{key}.")

    walk(data, "")
    return bad


def targets(scan_root: Path, files: list[Path]) -> list[Path]:
    settings = read_settings(scan_root, files)
    picked = set(_product_python(scan_root, files, settings))
    picked.update(_answer_files(scan_root, files, settings))
    picked.add(scan_root / settings.physics_toml)
    picked.update(f for f in files if f.parent == scan_root / RULES_DIR and f.suffix == ".toml")
    known = set(files)
    return sorted(p for p in picked if p in known)


def check(scan_root: Path, files: list[Path]) -> list[str]:
    settings = read_settings(scan_root, files)
    bad = _product_problems(scan_root, files, settings)
    bad += _answer_problems(scan_root, files, settings)
    bad += _toml_problems(scan_root, files, settings)
    note(
        f"產品 .py {len(_product_python(scan_root, files, settings))} 支，受管答案 {len(_answer_files(scan_root, files, settings))} 份，"
        f"設定檔 {settings.physics_toml}"
    )
    return bad


if __name__ == "__main__":
    sys.exit(run(check, description="基礎物理量單一來源：產品不准手寫聲速那幾個數、凍結答案要記自己的條件、設定檔不准存 ρc", targets=targets))
