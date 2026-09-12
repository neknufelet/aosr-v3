"""票 #134 第一候選：讀標準答案檔、逐格比對的共用零件。

這一支不是考卷（檔名不以 ``test_`` 開頭），是考卷共用的三件事：

1. **答案檔在哪、誰決定它有哪些 case。** ``blueprint/materials_cut1_answers.json``，由
   ``blueprint/generate_materials_cut1_answers.py`` 在唯讀的 donor 工作樹上跑出來。
   **人不碰裡面的數字**：要改就重跑產生器。「有哪些 case」由
   ``blueprint/materials_cut1_cases.py`` 那張表決定，考卷斷言兩邊的 id 集合**相等**、
   而且各自**不重複**（集合相等看不出「同一個 id 兩筆」，票 #163 的洞）。

2. **怎麼比。** :func:`differences` 逐格走過兩邊的記號，回傳「哪一格不一樣」的清單。
   不寫成一個布林：一個布林在幾百題裡說不出是哪一格壞了，而且很容易被寫成「兩邊都用
   同一支函式算出來所以一定相等」的假比對。

3. **跑新家的入口。** :func:`engine_lookup` 把表上的名字換成新家那一支模組，
   :func:`run_here` 拿同一支探針（產生器用的也是它）在新家跑一筆。兩邊只差「去哪一個套件
   拿模組」——跑法是同一份程式，比出來的差異才會是「新家跟上一代不一樣」，而不是「兩支
   跑法不一樣」。
"""
from __future__ import annotations

import copy
import importlib
import json
from collections import Counter
from pathlib import Path
from types import ModuleType

from blueprint import materials_cut1_cases as cases
from blueprint import materials_cut1_probe as probe
from blueprint import materials_cut1_steps as steps

# 答案檔的位置：從這一支往上兩層是 repo 根（跟 cwd 無關）。
ANSWER_PATH: Path = Path(__file__).resolve().parents[2] / "blueprint" / "materials_cut1_answers.json"

# 檔頭一定要有的那幾格（少一格就不是證據）。
REQUIRED_DONOR_KEYS: tuple[str, ...] = ("tag", "commit", "clean")
REQUIRED_ENV_KEYS: tuple[str, ...] = ("versions", "backend", "x64")


def as_mapping(node: object, where: str) -> dict[str, object]:
    """把一個節點收窄成一層表。不是表就當場炸——我沒看懂就不出結論。"""
    if not isinstance(node, dict):
        raise AssertionError(f"{where} 不是一層表：{node!r}")
    return {str(key): value for key, value in node.items()}


def load_answers(path: Path | None = None) -> dict[str, object]:
    """讀答案檔。檔頭少一格就當場炸——沒有 donor 出身或環境的答案檔不算證據。

    ``path`` 只有控制組會用到（把一份刻意的壞答案檔餵進來，證明裁判真的咬得住）。
    """
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


def _case_records(module: dict[str, object], name: str) -> list[dict[str, object]]:
    """一支模組那一塊裡的每一筆 case（收窄成表）。"""
    records = module.get("cases")
    if not isinstance(records, list):
        raise AssertionError(f"答案檔的 {name} cases 不是一串東西")
    return [as_mapping(item, f"答案檔 {name} 的一筆 case") for item in records]


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
    """表上的模組名換成**新家**那一支真的模組。

    這一支跟產生器那邊的 ``donor_lookup`` 是同一個位置上的兩顆螺絲：探針收一個
    ``lookup``，donor 那邊交 donor 的模組、這邊交新家的。跑法（步驟怎麼跑、值怎麼編碼）
    是同一份程式。
    """
    return importlib.import_module(cases.MODULES[name]["engine"])


def run_here(case: steps.CaseEntry) -> dict[str, object]:
    """在**新家**跑一筆 case，回傳跟答案檔同一個形狀的結果格。"""
    return probe.run_case(engine_lookup, case)


def case_by_id(case_id: str) -> steps.CaseEntry:
    """case 表裡那一筆（找不到就當場炸）。"""
    module_name = case_id.split(".", 1)[0]
    for case in cases.cases_for(module_name):
        if case["id"] == case_id:
            return case
    raise AssertionError(f"case 表裡沒有 {case_id!r}")


def deep_copy(root: dict[str, object]) -> dict[str, object]:
    """整份答案檔的副本（控制組要動手腳，正本不准動）。"""
    return copy.deepcopy(root)


def differences(actual: object, expected: object, where: str = "") -> list[str]:
    """逐格比兩個記號，回傳「哪一格不一樣」。空清單＝完全相同。

    刻意回清單不回布林：布林在幾百題裡說不出是哪一格壞了。這一支**不做任何容差**——
    浮點在答案檔裡是精確的十六進位寫法，這張票的合約是「數值與行為跟上一代一致」，
    不是「差不多」。
    """
    if isinstance(expected, dict) and isinstance(actual, dict):
        return _map_differences(actual, expected, where)
    if isinstance(expected, list) and isinstance(actual, list):
        return _list_differences(actual, expected, where)
    if type(actual) is not type(expected) or actual != expected:
        return [f"{where or '<根>'}：新家是 {actual!r}，答案檔是 {expected!r}"]
    return []


def _map_differences(actual: dict[object, object], expected: dict[object, object], where: str) -> list[str]:
    """兩層表逐鍵比（少一鍵、多一鍵都算不一樣）。"""
    out: list[str] = []
    for key in sorted(set(expected) - set(actual), key=repr):
        out.append(f"{where}.{key}：新家少了這一格（答案檔是 {expected[key]!r}）")
    for key in sorted(set(actual) - set(expected), key=repr):
        out.append(f"{where}.{key}：新家多了這一格（{actual[key]!r}）")
    for key in expected:
        if key in actual:
            out.extend(differences(actual[key], expected[key], f"{where}.{key}"))
    return out


def _list_differences(actual: list[object], expected: list[object], where: str) -> list[str]:
    """兩串東西逐格比（長度不一樣直接算不一樣）。"""
    if len(actual) != len(expected):
        return [f"{where}：新家有 {len(actual)} 格，答案檔有 {len(expected)} 格"]
    out: list[str] = []
    for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
        out.extend(differences(left, right, f"{where}[{index}]"))
    return out
