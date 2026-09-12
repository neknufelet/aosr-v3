"""票 #134 第二刀：讀標準答案檔、逐格比對的共用零件。

這一支不是考卷（檔名不以 ``test_`` 開頭），是考卷共用的四件事：

1. **答案檔在哪、誰決定它有哪些 case。** ``blueprint/materials_cut2_answers.json``，由
   ``blueprint/generate_materials_cut2_answers.py`` 在唯讀的 donor 工作樹上跑出來。
   **人不碰裡面的數字**：要改就重跑產生器。「有哪些 case」由
   ``blueprint/materials_cut2_cases.py`` 那張表決定。
2. **怎麼比。** 借第一刀那一支 :func:`~tests.engine._materials_answers.differences`——比較器
   只有一份，第一刀的控制組（``test_materials_judge_control.py``）驗的就是它。
3. **跑新家的入口。** :func:`engine_lookup` 把表上的名字換成新家那一支模組，
   :func:`run_here` 拿同一支探針在新家跑一筆。兩邊只差「去哪一個套件拿模組」。
4. **x64 那一格的探針**（:func:`x64_identity_here`）：另起一個開了 x64 的行程，在**新家**
   問同一條 tuple——同一個行程裡問不出這件事（常數在 import 時就算完了）。
"""
from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType

from blueprint import materials_cut2_cases as cases
from blueprint import materials_cut2_probe as probe
from blueprint import materials_cut2_steps as steps
from tests.engine._materials_answers import as_mapping, differences

__all__ = [
    "ANSWER_PATH",
    "answer_id_counts",
    "answer_ids",
    "as_mapping",
    "case_by_id",
    "differences",
    "duplicate_answer_ids",
    "duplicate_declared_ids",
    "engine_lookup",
    "engine_module_paths",
    "frozen_block",
    "load_answers",
    "modules_of",
    "probe_block",
    "result_of",
    "run_here",
    "x64_identity_here",
]

# 答案檔的位置：從這一支往上兩層是 repo 根（跟 cwd 無關）。
ANSWER_PATH: Path = Path(__file__).resolve().parents[2] / "blueprint" / "materials_cut2_answers.json"

# 檔頭一定要有的那幾格（少一格就不是證據）。
REQUIRED_DONOR_KEYS: tuple[str, ...] = ("tag", "commit", "clean")
REQUIRED_ENV_KEYS: tuple[str, ...] = ("versions", "backend", "x64")


def load_answers(path: Path | None = None) -> dict[str, object]:
    """讀答案檔。檔頭少一格就當場炸——沒有 donor 出身或環境的答案檔不算證據。"""
    source = ANSWER_PATH if path is None else path
    with source.open(encoding="utf-8") as handle:
        data: object = json.load(handle)
    root = as_mapping(data, source.name)
    if root.get("schema") != probe.ANSWER_SCHEMA:
        raise AssertionError(
            f"{source.name} 的 schema 是 {root.get('schema')!r}，這一支讀的是 "
            f"{probe.ANSWER_SCHEMA}——產生器改了形狀就要一起改讀的人"
        )
    donor = as_mapping(root.get("donor"), f"{source.name} 的 donor 檔頭")
    for key in REQUIRED_DONOR_KEYS:
        if not donor.get(key):
            raise AssertionError(f"{source.name} 的 donor 檔頭少了 {key}")
    if donor.get("clean") is not True:
        raise AssertionError(
            f"{source.name} 的 donor 檔頭沒有 clean=true"
            "——那代表它是在一棵被改過的樹上跑出來的，不是上一代的值"
        )
    env = as_mapping(root.get("env"), f"{source.name} 的 env 檔頭")
    for key in REQUIRED_ENV_KEYS:
        if key not in env:
            raise AssertionError(
                f"{source.name} 的 env 檔頭少了 {key}"
                "——dtype 跟著 x64 開關與函式庫版本走，沒記就比不出兩邊是不是同一組設定"
            )
    return root


def modules_of(root: dict[str, object] | None = None) -> dict[str, object]:
    """答案檔的 ``modules`` 那一塊。"""
    answers = load_answers() if root is None else root
    return as_mapping(answers.get("modules"), "答案檔的 modules")


def probe_block(name: str, root: dict[str, object] | None = None) -> dict[str, object]:
    """答案檔裡某一支探針那一格（另起行程才問得出來的那幾件事）。"""
    answers = load_answers() if root is None else root
    probes = as_mapping(answers.get("probes"), "答案檔的 probes")
    if name not in probes:
        raise AssertionError(f"答案檔裡沒有探針 {name!r}——這一格沒有裁判")
    return as_mapping(probes[name], f"答案檔的 probes.{name}")


def _case_records(module: dict[str, object], name: str) -> list[dict[str, object]]:
    """一支模組那一塊裡的每一筆 case（收窄成表）。"""
    records = module.get("cases")
    if not isinstance(records, list):
        raise AssertionError(f"答案檔的 {name} cases 不是一串東西")
    return [as_mapping(item, f"答案檔 {name} 的一筆 case") for item in records]


def answer_id_counts(root: dict[str, object] | None = None) -> Counter[str]:
    """答案檔裡每一個 id 出現幾次（一筆一格地數，集合看不到重複）。"""
    counts: Counter[str] = Counter()
    for name, raw in modules_of(root).items():
        module = as_mapping(raw, f"答案檔的 modules.{name}")
        for constant in as_mapping(module.get("constants"), f"{name} 的常數表"):
            counts[f"{name}.const.{constant}"] += 1
        for alias in as_mapping(module.get("aliases"), f"{name} 的別名表"):
            counts[f"{name}.alias.{alias}"] += 1
        for record in _case_records(module, name):
            case_id = record.get("id")
            if not isinstance(case_id, str) or not case_id:
                raise AssertionError(f"答案檔 {name} 有一筆 case 沒有 id：{record!r}")
            counts[case_id] += 1
    return counts


def answer_ids(root: dict[str, object] | None = None) -> set[str]:
    """答案檔裡每一個 id 的集合。"""
    return set(answer_id_counts(root))


def duplicate_answer_ids(root: dict[str, object] | None = None) -> list[str]:
    """答案檔裡出現不只一次的 id（空清單＝沒有重複）。"""
    return sorted(case_id for case_id, count in answer_id_counts(root).items() if count > 1)


def duplicate_declared_ids() -> list[str]:
    """case 表裡出現不只一次的 id（空清單＝沒有重複）。"""
    seen: Counter[str] = Counter(cases.declared_id_list())
    return sorted(name for name, count in seen.items() if count > 1)


def result_of(case_id: str, root: dict[str, object] | None = None) -> dict[str, object]:
    """答案檔裡那一筆 case 的結果格（找不到就當場炸——那一格沒有裁判）。"""
    module_name = case_id.split(".", 1)[0]
    modules = modules_of(root)
    if module_name not in modules:
        raise AssertionError(f"答案檔裡沒有 {module_name}——產生器沒跑到它，或表的名單變了")
    module = as_mapping(modules[module_name], f"答案檔的 modules.{module_name}")
    for record in _case_records(module, module_name):
        if record.get("id") == case_id:
            return as_mapping(record.get("result"), f"答案檔 {case_id} 的 result")
    raise AssertionError(f"答案檔裡沒有 case {case_id!r}——這一格沒有裁判")


def frozen_block(module: str, key: str, name: str, root: dict[str, object] | None = None) -> object:
    """答案檔裡某一支模組的常數／別名那一格（``key`` 是 ``constants`` 或 ``aliases``）。"""
    modules = modules_of(root)
    if module not in modules:
        raise AssertionError(f"答案檔裡沒有 {module}")
    table = as_mapping(as_mapping(modules[module], f"modules.{module}").get(key), f"{module}.{key}")
    if name not in table:
        raise AssertionError(f"答案檔裡沒有 {module}.{key}.{name}——這一格沒有裁判")
    return table[name]


def engine_lookup(name: str) -> ModuleType:
    """表上的模組名換成**新家**那一支真的模組。"""
    return importlib.import_module(cases.lookup_name(name, "engine"))


def engine_module_paths() -> dict[str, str]:
    """表上每一個名字在新家這一邊的 import 路徑（訊息裡那一段要換成同一個替身）。"""
    names = [*cases.MODULES, *cases.INGREDIENTS]
    return {name: cases.lookup_name(name, "engine") for name in names}


def run_here(case: steps.CaseEntry) -> dict[str, object]:
    """在**新家**跑一筆 case，回傳跟答案檔同一個形狀的結果格。"""
    return probe.run_case(engine_lookup, case, engine_module_paths())


def case_by_id(case_id: str) -> steps.CaseEntry:
    """case 表裡那一筆（找不到就當場炸）。"""
    module_name = case_id.split(".", 1)[0]
    for case in cases.cases_for(module_name):
        if case["id"] == case_id:
            return case
    raise AssertionError(f"case 表裡沒有 {case_id!r}")


# 另起那個行程要跑的東西：載入新家的 freq_axis，用**同一支 codec** 把那條 tuple 編成記號。
X64_SNIPPET: str = """
import importlib, json, sys

from blueprint import materials_cut1_probe as codec

import jax

module = importlib.import_module(sys.argv[1])
sys.stdout.write(json.dumps({
    "x64": bool(jax.config.jax_enable_x64),
    "identity": codec.encode(module.FREQS_HZ_IDENTITY),
}))
"""


def x64_identity_here(workdir: Path) -> dict[str, object]:
    """另起一個**開了 x64** 的行程，在新家問 ``FREQS_HZ_IDENTITY``。

    ``workdir`` 是 pytest 的 ``tmp_path``（不是真的 repo）。子行程的 ``PYTHONPATH`` 由
    **這個模組實際從哪裡被載入**推出來，不是寫死 repo 路徑——整個套件被換成突變版本跑的
    時候，子行程載到的也要是同一份。
    """
    repo_root = Path(__file__).resolve().parents[2]
    script = workdir / "probe_x64_identity.py"
    script.write_text(X64_SNIPPET, encoding="utf-8")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(repo_root), str(repo_root / "src")])
    env["JAX_ENABLE_X64"] = "true"
    env["JAX_PLATFORMS"] = "cpu"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    done = subprocess.run(
        [sys.executable, str(script), cases.lookup_name("freq_axis", "engine")],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if done.returncode != 0:
        raise AssertionError(f"子行程載入新家的 freq_axis 失敗：{done.stderr.strip()[:500]}")
    parsed: object = json.loads(done.stdout)
    return as_mapping(parsed, "子行程印出來的那一格")
