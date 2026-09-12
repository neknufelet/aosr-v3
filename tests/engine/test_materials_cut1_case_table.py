"""票 #134 第一候選：case 表與覆蓋帳自己的守門。

決策紙 ``engine-not-imported-new-home-grows-block-by-block`` 寫著：一塊裡哪些公開函式
**沒有寫死期望值的測試**，要列出來、列著沒補完不准關票——那一條原本靠人看。這一支把它
變成機器看得見的東西：每一個公開符號都要指得出是哪幾筆在判它，而且指到的那幾筆必須**真的
存在**（指到一個不存在的 id 是最舒服的假帳）。

**今天只有三支**（``response``／``source``／``registry``）。另外三支（``freq_axis``／
``experiment_schema``／``material_loader``）連 case 都還沒宣告，這一支會把「它們不在表上」
當成事實記下來——不是當成已經驗過。
"""
from __future__ import annotations

import pytest

from blueprint import materials_cut1_cases as cases

# 這一段交的三支。另外三支還沒進表，由 test_the_other_three_modules_are_not_claimed_yet 記著。
DELIVERED: tuple[str, ...] = ("registry", "response", "source")
NOT_YET: tuple[str, ...] = ("experiment_schema", "freq_axis", "material_loader")


def test_the_table_declares_exactly_the_modules_this_cut_delivers() -> None:
    """表上就是這一段交的那三支（多一支少一支都要當場看見）。"""
    assert sorted(cases.MODULES) == sorted(DELIVERED)


@pytest.mark.parametrize("module", NOT_YET)
def test_the_other_three_modules_are_not_claimed_yet(module: str) -> None:
    """另外三支**不在表上**：沒宣告 case 就是沒有裁判，不是「已經驗過」。"""
    assert module not in cases.MODULES


def test_no_declared_id_is_repeated() -> None:
    """case 表自己的 id 不准重複（同一個 id 兩筆，另一筆就是沒有人看的孤兒）。"""
    listed = cases.declared_id_list()
    assert len(listed) == len(set(listed)), "case 表有重複的 id"


@pytest.mark.parametrize("module", DELIVERED)
def test_every_case_has_an_id_steps_and_a_report(module: str) -> None:
    """每一筆 case 都要有 id、至少一個步驟、至少一個要記的名字。"""
    for case in cases.cases_for(module):
        assert case["id"].startswith(f"{module}."), f"{case['id']} 的 id 沒有掛在它自己的模組底下"
        assert case["steps"], f"{case['id']} 一個步驟都沒有"
        assert case["report"], f"{case['id']} 沒有要記的名字——那它比不到任何東西"


@pytest.mark.parametrize("module", DELIVERED)
def test_every_public_symbol_names_at_least_one_judge(module: str) -> None:
    """**沒有裁判就不准關票**那一條：每一個公開符號都要指得出判它的那幾筆。"""
    coverage = cases.coverage_for(module)
    assert coverage, f"{module} 一個公開符號都沒有登記"
    empty = sorted(name for name, judges in coverage.items() if not judges)
    assert empty == [], f"{module} 這幾個公開符號指不出任何裁判：{empty}"


@pytest.mark.parametrize("module", DELIVERED)
def test_every_named_judge_actually_exists(module: str) -> None:
    """覆蓋帳指到的每一筆都必須真的在表上（指到不存在的 id 是最舒服的假帳）。"""
    known = {case["id"] for case in cases.cases_for(module)}
    constants = set(cases.constants_for(module))
    aliases = set(cases.aliases_for(module))
    missing: list[str] = []
    for symbol, judges in cases.coverage_for(module).items():
        for judge in judges:
            if judge.startswith("const:"):
                if judge.removeprefix("const:") not in constants:
                    missing.append(f"{symbol} -> {judge}")
            elif judge.startswith("alias:"):
                if judge.removeprefix("alias:") not in aliases:
                    missing.append(f"{symbol} -> {judge}")
            elif judge not in known and judge.split(".", 1)[0] != module:
                missing.append(f"{symbol} -> {judge}")
            elif judge not in known:
                missing.append(f"{symbol} -> {judge}")
    assert missing == [], f"{module} 的覆蓋帳指到不存在的東西：{missing}"


@pytest.mark.parametrize("module", DELIVERED)
def test_every_frozen_constant_and_alias_is_in_the_coverage_map(module: str) -> None:
    """凍結的常數與型別別名也要出現在覆蓋帳上（凍了值卻沒人認領一樣是沒有帳）。"""
    coverage = cases.coverage_for(module)
    for name in cases.constants_for(module):
        assert name in coverage, f"{module}.{name} 凍了值卻不在覆蓋帳上"
    for name in cases.aliases_for(module):
        assert name in coverage, f"{module}.{name} 凍了值卻不在覆蓋帳上"


def test_the_registry_dunder_interface_is_accounted_for() -> None:
    """``MaterialRegistry()``／``in``／``len()`` 也要有帳。

    「名字不以底線開頭」那一種盤點法看不到這三個，但呼叫端天天在用——這一條把它們釘在
    覆蓋帳上，免得下一段長新家的時候整組漏掉。
    """
    coverage = cases.coverage_for("registry")
    for name in ("MaterialRegistry.__init__", "MaterialRegistry.__contains__", "MaterialRegistry.__len__"):
        assert coverage.get(name), f"{name} 沒有裁判"
