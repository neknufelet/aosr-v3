#!/usr/bin/env python3
"""精度契約的尺只住一份登記簿：產品程式不准再定義契約常數，改值要帶新紙，每條要有變異考卷。

決策紙 docs/decisions/precision-contract-thresholds-live-in-one-registry.md 的機器版（票 #312）。
尺有三層：容差值、比對公式、受驗案例。這張卡守的是第一層的住處與改法，第二、三層由登記簿指名的
變異考卷守（考卷有沒有跑過且過由綠卡守，這裡只守「指名的考卷真的在」）。

三顆牙（判準全從卡的 ``[settings]`` 讀，程式裡沒有預設值）：

1. **產品程式不准有契約常數的形狀**——``product_dir`` 底下每一支 .py，模組載入時就算好的數字
   （模組層級常數、類別身體、函式簽章預設值；藏在模組層的 ``if``／``try`` 底下一樣算），名字命中
   ``constant_name_patterns`` 任一樣式即紅。判形狀不判數值：v2 的病是同一個數字住三個地方，
   抓的是「第二個家」這件事，不是某個值。
2. **登記簿改值要帶新紙**——本次提交範圍（``base...head``，跟 tests-land-with-code 同一種三點差異）
   裡登記簿有哪幾條的值或人看寫法變了（含新增），每一條指名的決策紙必須是同範圍**新增**的檔、
   住在 ``decision_dirs`` 之一、內文出現那條的名字與新的人看寫法、``status`` 是 ``status_accepted``。
   一份無關的紙不算：紙上要對得上「哪個契約、新值」。
3. **每條指得到紙與考卷**——登記簿每一條的 ``decision_paper`` 在 ``decision_dirs`` 之一找得到；
   ``mutant_test`` 的檔在版控裡、而且那支檔真的定義了那個測試函式（用程式結構找，不比字串）。
   「跑過且過」不在這裡判：考卷 tests/engine/test_precision_contracts.py 證明每條指名的節點收集得到，
   綠卡證明整份收據沒有 failures、沒有 skip；三件事合起來才是「跑過且過」（決策紙
   precision-contracts-third-tooth-by-collection-and-green.md）。

借的零件：範圍解析（含 AOSR_RANGE_BASE／AOSR_RANGE_HEAD／GITHUB_EVENT_NAME 三個環境變數的判法）借
tests-land-with-code 那支，模組載入時就算好的數字借 thresholds-live-only-in-registry 那支；那兩支改行為這張卡跟著變。

**範圍怎麼定**：真的 git 工作樹走 tests-land-with-code 那條路（環境變數給 base／head，不然從主線
分支點算），舊登記簿從**共同分支點**讀——主線在分支之後自己改過的值不算這支合併請求改的；
必紅樣本用 ``fixture_history_file`` 宣告快照（base 與 head 各一棵子樹，選填一棵 main 代表主線在分支點
之後自己動過），檢查在系統暫存區重建真的提交再比。真的 git 工作樹根出現那份宣告檔一律回 2。

**回 2**：讀不到卡的 settings、登記簿不存在或剖不開、拿不到範圍、快照宣告壞掉、git 叫不動。
"""
from __future__ import annotations

import ast
import fnmatch
import shutil
import sys
import tempfile
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

from governance.checks.tests_land_with_code import (
    _fixture_env,
    _merge_base,
    _rev_parse,
    _run_git,
    _toplevel,
    parse_diff,
    resolve_range,
)
from governance.checks.thresholds_live_only_in_registry import _load_time_entries
from governance.exit_codes import ToolBroken, note, run
from governance.loader import setting_strings, setting_text

CARD_ID = "precision-contracts-live-in-one-registry"
RULES_DIR = "governance/rules"
PYTHON_SUFFIX = ".py"
MARKDOWN_SUFFIX = ".md"
FRONTMATTER_FENCE = "---"
TEXT_KEYS = (
    "registry_path",
    "product_dir",
    "fixture_history_file",
    "status_field",
    "status_accepted",
    "name_field",
    "value_field",
    "display_field",
    "paper_field",
    "mutant_field",
)
LIST_KEYS = ("decision_dirs", "constant_name_patterns")
SETTINGS_KEYS = (*TEXT_KEYS, *LIST_KEYS)
HISTORY_KEYS = ("base", "head")
HISTORY_OPTIONAL = ("main",)


class Settings(NamedTuple):
    registry_path: str
    product_dir: str
    fixture_history_file: str
    status_field: str
    status_accepted: str
    name_field: str
    value_field: str
    display_field: str
    paper_field: str
    mutant_field: str
    decision_dirs: tuple[str, ...]
    constant_name_patterns: tuple[str, ...]


class Entry(NamedTuple):
    name: str
    value: float
    display: str
    paper: str
    mutant: str


class Source(NamedTuple):
    """要比的那一段：``fork`` 是共同分支點（舊登記簿從這裡讀），``spec`` 是餵給 git diff 的範圍寫法。"""

    work_tree: Path
    fork: str
    head: str
    spec: str
    label: str
    cleanup: Callable[[], None]
    env: dict[str, str] | None


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
        raise ToolBroken(
            f"{scan_root}/{RULES_DIR} 底下 id={CARD_ID} 的卡有 {len(mine)} 張，要剛好一張——判準只寫在卡上"
        )
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
    for key, value in lists.items():
        if not value:
            raise ToolBroken(f"id={CARD_ID} 的卡 [settings] {key} 不准是空 list")
    return Settings(
        registry_path=texts["registry_path"],
        product_dir=texts["product_dir"].rstrip("/") + "/",
        fixture_history_file=texts["fixture_history_file"],
        status_field=texts["status_field"],
        status_accepted=texts["status_accepted"],
        name_field=texts["name_field"],
        value_field=texts["value_field"],
        display_field=texts["display_field"],
        paper_field=texts["paper_field"],
        mutant_field=texts["mutant_field"],
        decision_dirs=tuple(d.rstrip("/") + "/" for d in lists["decision_dirs"]),
        constant_name_patterns=lists["constant_name_patterns"],
    )


def parse_registry(raw: bytes, where: str, settings: Settings) -> dict[str, Entry]:
    """登記簿的形狀：``[[contract]]`` 一條一節，五格都要有。壞掉就 ToolBroken。"""
    try:
        data = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ToolBroken(f"{where} 剖不開：{exc}——尺讀不到就不出結論") from exc
    rows = data.get("contract")
    if not isinstance(rows, list) or not rows:
        raise ToolBroken(f"{where} 裡沒有任何 [[contract]]——登記簿是空的，尺不存在")
    out: dict[str, Entry] = {}
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ToolBroken(f"{where} 第 {index} 條不是一張表")
        name = row.get(settings.name_field)
        value = row.get(settings.value_field)
        display = row.get(settings.display_field)
        paper = row.get(settings.paper_field)
        mutant = row.get(settings.mutant_field)
        if not (isinstance(name, str) and name.strip()):
            raise ToolBroken(f"{where} 第 {index} 條沒有 {settings.name_field}")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ToolBroken(f"{where} 第 {index} 條（{name}）的 {settings.value_field} 不是數字：{value!r}")
        for field, got in ((settings.display_field, display), (settings.paper_field, paper), (settings.mutant_field, mutant)):
            if not (isinstance(got, str) and got.strip()):
                raise ToolBroken(f"{where} 第 {index} 條（{name}）的 {field} 不是非空字串：{got!r}")
        if name in out:
            raise ToolBroken(f"{where} 有兩條同名 {name}")
        out[name] = Entry(name, float(value), str(display).strip(), str(paper).strip(), str(mutant).strip())
    return out


def _read_registry(scan_root: Path, files: list[Path], settings: Settings) -> dict[str, Entry]:
    path = scan_root / settings.registry_path
    if path not in files:
        raise ToolBroken(f"登記簿 {settings.registry_path} 不在這棵樹的版控裡——沒有尺就不出結論")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ToolBroken(f"登記簿 {settings.registry_path} 讀不開：{exc}") from exc
    return parse_registry(raw, settings.registry_path, settings)


# ── 牙 1：產品程式的契約常數 ─────────────────────────────────────────────────


def _product_python(scan_root: Path, files: list[Path], settings: Settings) -> list[Path]:
    home = scan_root / settings.product_dir.rstrip("/")
    return sorted(f for f in files if f.suffix == PYTHON_SUFFIX and home in f.parents)


def _constant_problems(scan_root: Path, files: list[Path], settings: Settings) -> list[str]:
    bad: list[str] = []
    for path in _product_python(scan_root, files, settings):
        rel = path.relative_to(scan_root).as_posix()
        try:
            tree = ast.parse(path.read_bytes().decode("utf-8"), filename=rel)
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            raise ToolBroken(f"{rel} 剖不開：{exc}——產品程式有一支讀不了，這一跑不算數") from exc
        for shown, name, lineno, values in _load_time_entries(tree.body, ""):
            # 比對不分大小寫：模組常數是大寫，函式簽章的參數名是小寫（tolerance_rel=2**-20 那種預設值也是第二個家）。
            if any(fnmatch.fnmatchcase(name.upper(), pattern.upper()) for pattern in settings.constant_name_patterns):
                bad.append(
                    f"{rel}:{lineno} 的 {shown} 在模組載入時就寫死了數字 {values}——名字長得像精度契約的界線"
                    f"（命中卡上登記的樣式），契約門檻只准住 {settings.registry_path}，產品程式要從呼叫端收參數"
                )
    return bad


# ── 牙 3：每條指得到紙與考卷 ─────────────────────────────────────────────────


def _frontmatter_status(text: str, settings: Settings) -> str:
    lines = text.splitlines()
    if not lines or lines[0].strip() != FRONTMATTER_FENCE:
        return ""
    for line in lines[1:]:
        if line.strip() == FRONTMATTER_FENCE:
            break
        if line.startswith(settings.status_field + ":"):
            return line.split(":", 1)[1].strip().strip('"')
    return ""


def _find_paper(scan_root: Path, files: list[Path], paper: str, settings: Settings) -> Path | None:
    for home in settings.decision_dirs:
        candidate = scan_root / home / paper
        if candidate in files:
            return candidate
    return None


def _reference_problems(scan_root: Path, files: list[Path], entries: dict[str, Entry], settings: Settings) -> list[str]:
    bad: list[str] = []
    for entry in entries.values():
        if "/" in entry.paper or not entry.paper.endswith(MARKDOWN_SUFFIX):
            bad.append(f"登記簿 {entry.name} 的 {settings.paper_field}={entry.paper!r} 要寫成決策紙的檔名（不帶目錄、.md 結尾）")
        elif _find_paper(scan_root, files, entry.paper, settings) is None:
            bad.append(
                f"登記簿 {entry.name} 指名的決策紙 {entry.paper} 在 {list(settings.decision_dirs)} 都找不到"
                "——尺的出處是紙，沒有紙的門檻是一個沒有證據的數字"
            )
        file_part, sep, func = entry.mutant.partition("::")
        target = scan_root / file_part
        if not sep or not func or target not in files:
            bad.append(f"登記簿 {entry.name} 的 {settings.mutant_field}={entry.mutant!r} 指到的考卷檔不在版控裡（要寫成 檔案::測試函式）")
            continue
        try:
            tree = ast.parse(target.read_bytes().decode("utf-8"), filename=file_part)
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            raise ToolBroken(f"{file_part} 剖不開：{exc}") from exc
        func_name = func.split("[", 1)[0]
        if not _defines_function(tree, func_name):
            bad.append(f"登記簿 {entry.name} 的變異考卷 {entry.mutant}：{file_part} 裡沒有定義 {func_name}（用程式結構找，不比字串，註解裡寫一行 def 不算）——指名的考卷不存在等於這條契約沒有變異考卷")
    return bad


def _defines_function(tree: ast.Module, name: str) -> bool:
    """那支考卷檔有沒有真的定義那個測試函式（模組層級或類別身體裡，非同步也算）。"""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return True
    return False


# ── 牙 2：改值要帶新紙 ───────────────────────────────────────────────────────


def _read_history(decl: Path) -> dict[str, str]:
    try:
        text = decl.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ToolBroken(f"讀不到樣本的快照宣告 {decl}：{exc}") from exc
    out: dict[str, str] = {}
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        tokens = line.split()
        if len(tokens) != 2 or tokens[0] not in (*HISTORY_KEYS, *HISTORY_OPTIONAL):
            raise ToolBroken(f"{decl}:{lineno} 只認 `base <子樹>`／`main <子樹>`／`head <子樹>`，實際是 {line!r}")
        rel = tokens[1]
        if rel.startswith("/") or ".." in rel.split("/") or ".git" in rel.split("/"):
            raise ToolBroken(f"{decl}:{lineno} 快照子樹 {rel!r} 不准是絕對路徑、不准夾 ..、不准碰 .git")
        out[tokens[0]] = rel
    missing = [k for k in HISTORY_KEYS if k not in out]
    if missing:
        raise ToolBroken(f"{decl} 缺 {missing} 那一行——兩棵快照都要宣告")
    return out


def _copy_snapshot(scan_root: Path, rel: str, work: Path, decl: Path) -> None:
    source = (scan_root / rel).resolve()
    if not source.is_dir() or scan_root.resolve() not in source.parents:
        raise ToolBroken(f"{decl} 宣告的快照子樹 {rel} 不在樣本樹裡（或不是目錄）")
    for child in work.iterdir():
        if child.name == ".git":
            continue
        shutil.rmtree(child) if child.is_dir() else child.unlink()
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ToolBroken(f"{decl} 的快照 {rel} 裡有連結 {path.name}，不准")
        if path.is_file():
            target = work / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())


def _materialize(scan_root: Path, decl: Path) -> Source:
    history = _read_history(decl)
    tmp = Path(tempfile.mkdtemp(prefix="aosr-contracts-"))

    def cleanup() -> None:
        try:
            shutil.rmtree(tmp)
        except OSError as exc:
            raise ToolBroken(f"清不掉自己開的暫存目錄 {tmp}：{exc}") from exc

    work = tmp / "repo"
    try:
        work.mkdir()
        env = _fixture_env(tmp)
        _run_git(["-c", "init.defaultBranch=main", "init", "--quiet"], work, what="開樣本用的暫存 repo", env=env)
        def snapshot(key: str) -> str:
            _copy_snapshot(scan_root, history[key], work, decl)
            _run_git(["add", "-A"], work, what=f"加入快照 {key}", env=env)
            _run_git(
                ["commit", "--allow-empty", "--no-verify", "--quiet", "-m", f"樣本快照（{key}）"],
                work,
                what=f"提交快照 {key}",
                env=env,
            )
            return _rev_parse(work, "HEAD")

        fork = snapshot("base")
        # 主線在分支點之後自己也動過（有 main 快照才有）：三點差異不會把它算到候選身上。
        main_tip = snapshot("main") if "main" in history else fork
        _run_git(["switch", "--quiet", "-c", "candidate", fork], work, what="切到候選分支", env=env)
        head = snapshot("head")
    except Exception:
        cleanup()
        raise
    return Source(work, fork, head, f"{main_tip}...{head}", f"{fork[:9]}...{head[:9]}（樣本 {scan_root.name}）", cleanup, env)


def _nothing() -> None:
    """真的工作樹不用清什麼。"""


def _resolve_source(scan_root: Path, settings: Settings) -> Source:
    decl = scan_root / settings.fixture_history_file
    top = _toplevel(scan_root)
    if top is not None and top == scan_root:
        if decl.exists():
            raise ToolBroken(f"這是真的 git 工作樹的根，卻放著樣本用的 {settings.fixture_history_file}——真歷史不准被一個檔案繞過")
        rng = resolve_range(scan_root)
        spec = rng.diff_args[0]
        left, sep, head = spec.partition("...")
        if not sep:
            left, _sep, head = spec.partition("..")
        # 舊登記簿從共同分支點讀，不從主線頂端讀：主線在分支之後自己改過的值不算這支合併請求改的。
        fork = _merge_base(scan_root, left, head) if sep else left
        if not fork:
            raise ToolBroken(f"{left[:9]} 與 {head[:9]} 算不出共同分支點——沒有分支點就分不出誰改了登記簿")
        return Source(scan_root, fork, head, spec, rng.label, _nothing, None)
    if decl.is_file():
        return _materialize(scan_root, decl)
    raise ToolBroken(f"{scan_root} 既不是 git 工作樹的根，也沒有 {settings.fixture_history_file}——拿不到範圍，這一跑不算數")


def _show(source: Source, rev: str, rel: str) -> bytes | None:
    """那顆提交裡那個檔的內容；檔不在那顆提交裡回 None（先問 cat-file，不靠錯誤字串），git 壞掉回 2。"""
    exists = _run_git(["cat-file", "-e", f"{rev}:{rel}"], source.work_tree, what=f"問 {rev[:9]} 有沒有 {rel}", env=source.env, allow=(0, 1, 128))
    if exists.returncode != 0:
        return None
    return _run_git(["show", f"{rev}:{rel}"], source.work_tree, what=f"讀 {rev[:9]}:{rel}", env=source.env).stdout.encode("utf-8")


def _range_problems(source: Source, files_in_head: dict[str, str], settings: Settings) -> list[str]:
    """牙 2。``files_in_head`` 是差異裡新增的檔：路徑 → 狀態。"""
    head_raw = _show(source, source.head, settings.registry_path)
    if head_raw is None:
        return []
    head_entries = parse_registry(head_raw, f"{source.head[:9]}:{settings.registry_path}", settings)
    base_raw = _show(source, source.fork, settings.registry_path)
    # 登記簿在共同分支點還不存在（開張、或搬了路徑）：每一條都算新增，每一條都要帶紙——
    # 不豁免，不然改 registry_path 就能整份繞過（第一輪找碴點的）。
    base_entries = parse_registry(base_raw, f"{source.fork[:9]}:{settings.registry_path}", settings) if base_raw is not None else {}
    bad: list[str] = []
    for name, entry in head_entries.items():
        before = base_entries.get(name)
        if before is not None and before.value == entry.value and before.display == entry.display:
            continue
        what = "新增" if before is None else f"從 {before.display} 改成 {entry.display}"
        homes = [f"{home}{entry.paper}" for home in settings.decision_dirs]
        added = [rel for rel in homes if files_in_head.get(rel) == "A"]
        if not added:
            bad.append(
                f"登記簿 {name} 這一段{what}，但它指名的決策紙 {entry.paper} 不是這一段新增的檔"
                "——改尺要開新紙取代舊紙，同一支合併請求帶進來；沒有新紙的改值就是心證改"
            )
            continue
        raw = _show(source, source.head, added[0])
        text = raw.decode("utf-8", "replace") if raw is not None else ""
        if name not in text or entry.display not in text:
            bad.append(
                f"登記簿 {name} 這一段{what}，同範圍新增的紙 {added[0]} 內文沒有同時出現契約名 {name} 與新寫法 {entry.display}"
                "——一份無關的紙不算帶紙，紙上要對得上哪個契約、改成多少"
            )
            continue
        status = _frontmatter_status(text, settings)
        if status != settings.status_accepted:
            bad.append(f"登記簿 {name} 指名的新紙 {added[0]} 的 {settings.status_field} 是 {status!r}，不是 {settings.status_accepted!r}——沒拍板的紙撐不起改尺")
    return bad


def _added_files(source: Source) -> dict[str, str]:
    """差異裡每個路徑的狀態字母。不做改名偵測（新紙被配對成改名會誤紅），剖析借 tests-land-with-code 的。"""
    out = _run_git(["diff", "--name-status", "-z", "--no-renames", source.spec], source.work_tree, what=f"量 {source.label} 的差異", env=source.env).stdout
    seen: dict[str, str] = {}
    for entry in parse_diff(out):
        for side in entry.sides:
            seen[side] = entry.status[0]
    return seen


def targets(scan_root: Path, files: list[Path]) -> list[Path]:
    settings = read_settings(scan_root, files)
    picked = set(_product_python(scan_root, files, settings))
    picked.add(scan_root / settings.registry_path)
    for home in settings.decision_dirs:
        picked.update(f for f in files if f.parent == scan_root / home.rstrip("/") and f.suffix == MARKDOWN_SUFFIX)
    # 變異考卷實際指到哪些檔就讀哪些（今天都在 tests/engine/，將來指到別處也跟著讀）。
    for entry in _read_registry(scan_root, files, settings).values():
        file_part = entry.mutant.partition("::")[0]
        if (scan_root / file_part) in set(files):
            picked.add(scan_root / file_part)
    picked.update(f for f in files if f.parent == scan_root / RULES_DIR and f.suffix == ".toml")
    return sorted(p for p in picked if p in set(files))


def check(scan_root: Path, files: list[Path]) -> list[str]:
    settings = read_settings(scan_root, files)
    entries = _read_registry(scan_root, files, settings)
    source = _resolve_source(scan_root, settings)
    try:
        bad = _constant_problems(scan_root, files, settings)
        bad += _reference_problems(scan_root, files, entries, settings)
        bad += _range_problems(source, _added_files(source), settings)
    finally:
        source.cleanup()
    note(f"登記簿 {len(entries)} 條，範圍 {source.label}，產品 .py {len(_product_python(scan_root, files, settings))} 支")
    return bad


if __name__ == "__main__":
    sys.exit(run(check, description="精度契約的尺只住一份登記簿：產品不准定義契約常數、改值要帶新紙、每條指得到紙與考卷", targets=targets))
