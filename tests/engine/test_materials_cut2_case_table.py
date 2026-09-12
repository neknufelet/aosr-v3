"""票 #134 第二刀：case 表與覆蓋帳自己的守門。

決策紙 ``engine-not-imported-new-home-grows-block-by-block`` 寫著：一塊裡哪些公開函式
**沒有寫死期望值的測試**，要列出來、列著沒補完不准關票——那一條原本靠人看。這一支把它變成
機器看得見的東西：每一個公開符號都要指得出是哪幾筆在判它，而且指到的那幾筆必須**真的存在**
（指到一個不存在的 id 是最舒服的假帳）。

這一支**不載入新家的產品模組**，只讀 case 表與答案檔——所以三支產品還沒長出來之前，它就
應該是綠的。產品那一份逐筆比對在 ``test_materials_cut2_product.py``。
"""
from __future__ import annotations

import pytest

from blueprint import materials_cut2_cases as cases
from tests.engine import _materials_cut2_answers as answers

# 這一刀交的三支。第一刀那三支不在這裡重判（它們的答案已經在雲端綠過）。
DELIVERED: tuple[str, ...] = ("experiment_schema", "freq_axis", "material_loader")
ALREADY_JUDGED: tuple[str, ...] = ("registry", "response", "source")


def test_the_table_declares_exactly_the_modules_this_cut_delivers() -> None:
    """表上就是這一刀交的那三支（多一支少一支都要當場看見）。"""
    assert sorted(cases.MODULES) == sorted(DELIVERED)


@pytest.mark.parametrize("module", ALREADY_JUDGED)
def test_the_first_cut_modules_are_not_judged_again_here(module: str) -> None:
    """第一刀那三支不在這張表上——重判一次只會多一份會漂開的第二意見。"""
    assert module not in cases.MODULES


def test_response_is_an_ingredient_not_a_judged_module() -> None:
    """``response`` 只是材料：case 會叫它建頻率軸，但這一刀不判它。"""
    assert "response" in cases.INGREDIENTS
    assert "response" not in cases.MODULES
    assert cases.lookup_name("response", "engine").startswith("aosr.materials.")


def test_an_unknown_module_name_is_refused() -> None:
    """表上沒有的名字要當場炸——解析不出來卻回一個像樣的東西是最舒服的假帳。"""
    with pytest.raises(KeyError):
        cases.lookup_name("no_such_module", "engine")


def test_no_declared_id_is_repeated() -> None:
    """case 表自己的 id 不准重複（同一個 id 兩筆，另一筆就是沒有人看的孤兒）。"""
    assert answers.duplicate_declared_ids() == []


def test_no_answer_id_is_repeated() -> None:
    """答案檔裡的 id 也不准重複。"""
    assert answers.duplicate_answer_ids() == []


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
            elif judge.startswith("probe:"):
                if judge.removeprefix("probe:") not in cases.PROBES:
                    missing.append(f"{symbol} -> {judge}")
            elif judge not in known:
                missing.append(f"{symbol} -> {judge}")
    assert missing == [], f"{module} 的覆蓋帳指到不存在的東西：{missing}"


@pytest.mark.parametrize("module", DELIVERED)
def test_every_frozen_constant_is_in_the_coverage_map(module: str) -> None:
    """凍結的常數與型別別名也要出現在覆蓋帳上（凍了值卻沒人認領一樣是沒有帳）。"""
    coverage = cases.coverage_for(module)
    for name in [*cases.constants_for(module), *cases.aliases_for(module)]:
        assert name in coverage, f"{module}.{name} 凍了值卻不在覆蓋帳上"


@pytest.mark.parametrize("probe_name", cases.PROBES)
def test_every_declared_probe_has_an_answer(probe_name: str) -> None:
    """宣告的每一支探針在答案檔裡都要有一格（宣告了沒量到就是沒有裁判）。"""
    block = answers.probe_block(probe_name)
    assert {"on", "off"} <= set(block), f"探針 {probe_name} 的答案缺了 on／off"


@pytest.mark.parametrize("module", DELIVERED)
def test_every_answer_record_actually_carries_something_to_compare(module: str) -> None:
    """答案檔裡每一筆都要**真的帶著東西**：要嘛記了值，要嘛記了它炸什麼。

    這一條擋的是「一筆空答案」：``{"values": {}}`` 在比較器眼裡跟新家的空答案完全相等，
    等於那一筆沒有人看。
    """
    empty: list[str] = []
    for case in cases.cases_for(module):
        result = answers.result_of(case["id"])
        raised = result.get("raised")
        values = result.get("values")
        if isinstance(raised, dict) and raised.get("type"):
            continue
        if isinstance(values, dict) and values:
            continue
        empty.append(case["id"])
    assert empty == [], f"{module} 這幾筆答案是空的，比不到任何東西：{empty}"


@pytest.mark.parametrize("module", DELIVERED)
def test_every_reported_name_has_an_answer_of_its_own(module: str) -> None:
    """跑完那幾筆：``report`` 上的每一個名字在答案檔裡都要各自有一格。

    ``report`` 寫了三個名字、答案檔只有兩格，那第三個就是沒有人比的——集合相等那一條
    看不到這件事（它只數 case，不數 case 裡面的格子）。
    """
    gaps: list[str] = []
    for case in cases.cases_for(module):
        result = answers.result_of(case["id"])
        values = result.get("values")
        if not isinstance(values, dict):
            continue  # 這一筆記的是 raised（炸在第幾步），沒有 values 那一格
        for name in case["report"]:
            if name not in values:
                gaps.append(f"{case['id']}.{name}")
    assert gaps == [], f"{module} 這幾格 report 了卻沒有答案：{gaps}"
