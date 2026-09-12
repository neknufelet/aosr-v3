"""票 #134 第一候選：**裁判自己的控制組**。

新家 ``aosr.materials`` 已經實作了，產品的逐筆比對住
``tests/engine/test_materials_cut1_product.py``。**這一支不是那一份**：它驗的是裁判本身
（**這一支全綠不代表產品跟上一代對過了，那句話由產品那一份負責**）。為什麼非驗不可：
第一版的編碼器實測有三個碰撞（``/tmp`` 那份 ``encoding-collisions.json``）——

* ``encode([1]) == encode((1,))``——``MaterialRegistry.ids()`` 從 ``list`` 變 ``tuple`` 看不見；
* ``encode({1}) == encode(frozenset({1}))``——``SOURCE_KINDS`` 從 ``frozenset`` 變成可改的
  ``set`` 看不見；
* ``encode({1: "x"}) == encode({"1": "x"})``——字典的鍵從整數變字串看不見。

公開容器的種類與鍵的型別是呼叫端看得到的行為，不是「怎麼寫的」。這一支每一組都給**兩腳**：
先用第一版的寫法證明它真的擋不住（不然這一條就是在自我恭維），再證明現在的寫法擋得住。
"""
from __future__ import annotations

from typing import Final

import pytest

from blueprint import materials_cut1_cases as cases
from blueprint import materials_cut1_probe as probe
from tests.engine import _materials_answers as answers


def old_encode(value: object) -> dict[str, object]:
    """第一版的編碼器（**只保留那三格會碰撞的分支**），當成控制組的左腳。

    它是刻意留著的壞尺：下面每一組都先用它量一次，證明「舊寫法說兩邊一樣」，再用現在的
    :func:`blueprint.materials_cut1_probe.encode` 量一次，證明「現在說不一樣」。少了左腳，
    這些斷言就只是在說「我現在這樣寫」，證不出它補到了什麼。
    """
    if isinstance(value, (frozenset, set)):
        return {"kind": "set", "items": [old_encode(item) for item in sorted(value, key=repr)]}
    if isinstance(value, (list, tuple)):
        return {"kind": "seq", "items": [old_encode(item) for item in value]}
    if isinstance(value, dict):
        return {"kind": "map", "items": {str(key): old_encode(item) for key, item in value.items()}}
    return probe.encode(value)


# 三組「同值、不同型別」的東西。每一組的兩邊在舊寫法底下編出來一模一樣。
COLLIDING_PAIRS: Final[list[tuple[str, object, object]]] = [
    ("list 與 tuple（ids() 的回傳容器）", ["m1", "m2"], ("m1", "m2")),
    ("set 與 frozenset（SOURCE_KINDS 能不能被改）", {"analytic"}, frozenset({"analytic"})),
    ("字典鍵的型別（整數 1 與字串 '1'）", {1: "x"}, {"1": "x"}),
]


@pytest.mark.parametrize(("name", "left", "right"), COLLIDING_PAIRS)
def test_the_old_codec_was_blind_to_this_pair(name: str, left: object, right: object) -> None:
    """控制組的左腳：這一組在**第一版**的編碼器底下編出來一模一樣（`name` 是它）。

    這一條不是在驗產品，是在驗「下一條真的有對象」。它如果哪天紅了，代表那個碰撞已經
    不存在，下一條就該一起重寫——不是把這一條刪掉。
    """
    assert old_encode(left) == old_encode(right), f"{name}：這一組在舊寫法底下沒有碰撞，控制組失去對象"


@pytest.mark.parametrize(("name", "left", "right"), COLLIDING_PAIRS)
def test_the_codec_now_separates_the_pair(name: str, left: object, right: object) -> None:
    """控制組的右腳：同一組在**現在**的編碼器底下必須不一樣（`name` 是它）。"""
    assert probe.encode(left) != probe.encode(right), f"{name}：現在的編碼器還是分不出這兩種"


def test_the_comparison_says_where_it_differs() -> None:
    """比對回的是「哪一格不一樣」，而且同一個值比自己要回空清單（裁判不亂咬）。"""
    encoded = probe.encode({"ids": ["m1"], "kind": "analytic"})
    assert answers.differences(encoded, encoded) == []
    changed = probe.encode({"ids": ("m1",), "kind": "analytic"})
    assert answers.differences(changed, encoded), "容器從 list 變 tuple，比對竟然說一樣"


def test_a_changed_float_is_caught_bit_for_bit() -> None:
    """浮點差一個最小單位也要紅（答案檔存的是十六進位的精確寫法，不是四捨五入）。"""
    expected = probe.encode(0.1)
    nudged = probe.encode(0.1 + 2.0**-56)
    assert nudged != expected
    assert answers.differences(nudged, expected)


def test_an_int_is_not_a_float_and_a_bool_is_not_an_int() -> None:
    """``1``／``1.0``／``True`` 是三種東西：凍結值換了型別就是行為變了。"""
    assert probe.encode(1) != probe.encode(1.0)
    assert probe.encode(1) != probe.encode(True)


def test_array_dtype_and_shape_are_part_of_the_answer() -> None:
    """陣列的 ``dtype`` 與 ``shape`` 改掉要紅（``complex64`` 掉成 ``float32`` 是相位不見了）。"""
    block = answers.result_of("response.MaterialResponse.from_impedance.complex_with_default_scattering")
    values = answers.as_mapping(block.get("values"), "那一筆的 values")
    array = answers.as_mapping(values.get("m_Z_surface"), "Z_surface 那一格")
    assert array.get("kind") == "array"
    assert array.get("dtype") == "complex64"
    assert array.get("shape") == [3]
    mutated = dict(array)
    mutated["dtype"] = "float32"
    assert answers.differences(mutated, array)
    mutated_shape = dict(array)
    mutated_shape["shape"] = [1]
    assert answers.differences(mutated_shape, array)


def test_the_real_answer_file_has_unique_ids() -> None:
    """控制組的右邊：版控裡那一份答案檔與 case 表的 id 都不重複。"""
    assert answers.duplicate_answer_ids() == []
    assert answers.duplicate_declared_ids() == []


def test_the_answer_file_matches_the_case_table_exactly() -> None:
    """答案檔的 id 集合必須**等於** case 表宣告的集合（少一筆紅、多一筆也紅）。"""
    declared = cases.declared_ids()
    present = answers.answer_ids()
    assert sorted(declared - present) == [], "答案檔少了 case 表宣告的這幾筆"
    assert sorted(present - declared) == [], "答案檔有 case 表沒宣告的這幾筆"
    assert present, "答案檔一筆 id 都沒有——那就沒有任何裁判"


def _first_case_id() -> str:
    """答案檔裡第一筆 case 的 id（挑一筆真的、不是自己造一個玩具）。"""
    present = sorted(answers.answer_ids())
    for case_id in present:
        if ".const." not in case_id and ".alias." not in case_id:
            return case_id
    raise AssertionError("答案檔裡一筆 case 都沒有")


def _drop_case(case_id: str) -> dict[str, object]:
    """**真的那一份**答案檔 ＋ 拔掉指名的那一筆（正本不動）。"""
    root = answers.deep_copy(answers.load_answers())
    modules = answers.as_mapping(root.get("modules"), "答案檔的 modules")
    for name, raw in modules.items():
        module = answers.as_mapping(raw, f"modules.{name}")
        records = module.get("cases")
        if isinstance(records, list):
            kept = [item for item in records if not (isinstance(item, dict) and item.get("id") == case_id)]
            if len(kept) != len(records):
                module["cases"] = kept
                modules[name] = module
                root["modules"] = modules
                return root
    raise AssertionError(f"答案檔裡找不到 {case_id!r}——這一條就沒有對象")


def _duplicate_case(case_id: str) -> dict[str, object]:
    """**真的那一份**答案檔 ＋ 把指名的那一筆複製一份（同一個 id 兩筆）。"""
    root = answers.deep_copy(answers.load_answers())
    modules = answers.as_mapping(root.get("modules"), "答案檔的 modules")
    for name, raw in modules.items():
        module = answers.as_mapping(raw, f"modules.{name}")
        records = module.get("cases")
        if not isinstance(records, list):
            continue
        for item in list(records):
            if isinstance(item, dict) and item.get("id") == case_id:
                records.append(answers.deep_copy(item))
                module["cases"] = records
                modules[name] = module
                root["modules"] = modules
                return root
    raise AssertionError(f"答案檔裡找不到 {case_id!r}——這一條就沒有對象")


@pytest.mark.parametrize("case_id", sorted(cases.case_id_list()))
def test_dropping_this_one_case_is_caught(case_id: str) -> None:
    """**逐筆刪的突變，一筆一題**：答案檔少了這一筆，集合比對要當場紅。

    「有幾筆」跟「每一筆都被拿去比」是兩件事。只挑第一筆刪，證得出來的也只有第一筆——
    所以這一條對 case 表宣告的**每一筆**各出一題（餵的都是真的那一份答案檔現拔一筆）。
    """
    broken = _drop_case(case_id)
    assert case_id not in answers.answer_ids(broken), "拔掉之後那個 id 還在——突變沒生效"
    assert cases.declared_ids() - answers.answer_ids(broken) == {case_id}
    assert case_id in answers.answer_ids(), "正本被動到了"


@pytest.mark.parametrize("frozen_id", sorted(cases.constant_id_list()))
def test_dropping_this_one_frozen_value_is_caught(frozen_id: str) -> None:
    """凍結的常數與型別別名也一樣：少一格就少一個裁判，集合比對要紅。"""
    module_name, kind, name = frozen_id.split(".", 2)
    table_key = "constants" if kind == "const" else "aliases"
    root = answers.deep_copy(answers.load_answers())
    modules = answers.as_mapping(root.get("modules"), "答案檔的 modules")
    module = answers.as_mapping(modules[module_name], f"modules.{module_name}")
    table = answers.as_mapping(module.get(table_key), f"{module_name}.{table_key}")
    del table[name]
    module[table_key] = table
    modules[module_name] = module
    root["modules"] = modules
    assert cases.declared_ids() - answers.answer_ids(root) == {frozen_id}
    assert frozen_id in answers.answer_ids(), "正本被動到了"


@pytest.mark.parametrize("case_id", sorted(cases.case_id_list()))
def test_duplicating_this_one_case_is_caught_by_uniqueness(case_id: str) -> None:
    """同一個 id 兩筆：集合看不出來（所以唯一性要另一條），唯一性那一條要咬得住。"""
    broken = _duplicate_case(case_id)
    assert answers.answer_ids(broken) == answers.answer_ids(), "複製一筆竟然改變了 id 集合"
    assert case_id in answers.duplicate_answer_ids(broken), "複製一筆之後裁判說沒有重複"
    assert answers.duplicate_answer_ids() == [], "正本被動到了"


@pytest.mark.parametrize("case_id", sorted(cases.case_id_list()))
def test_changing_this_one_answer_makes_the_product_test_red(case_id: str) -> None:
    """**這一筆的期望值改掉，比對一定要紅**——證明每一筆真的走到比較器。

    上面那幾條證的是「少一筆看得見」，這一條證的是「在的那一筆真的被比」：拿新家跑出來
    的結果去跟**動過手腳的**答案比，必須有差異；跟原本那一筆比，必須沒有差異。
    """
    actual = answers.run_here(answers.case_by_id(case_id))
    expected = answers.result_of(case_id)
    assert answers.differences(actual, expected) == [], "這一筆本來就對不上，控制組沒有對象"
    tampered = answers.deep_copy(expected)
    tampered["__tampered__"] = {"kind": "str", "v": "x"}
    assert answers.differences(actual, tampered), "期望值多一格，比對竟然說一樣"


def test_a_changed_expected_value_is_caught() -> None:
    """期望值那一格改掉要紅；同一筆拿回自己比要綠（裁判在好的輸入上不亂咬）。"""
    case_id = _first_case_id()
    block = answers.result_of(case_id)
    assert answers.differences(block, block) == []
    broken = answers.deep_copy(block)
    broken["values"] = {"BOGUS": {"kind": "str", "v": "x"}}
    assert answers.differences(broken, block)


def test_a_dropped_reported_value_is_caught() -> None:
    """一筆 case 裡**少報一格**也要紅（整筆還在，但那一格沒有人比了）。"""
    case_id = "response.MaterialResponse.from_impedance.complex_with_default_scattering"
    block = answers.result_of(case_id)
    values = answers.as_mapping(block.get("values"), "那一筆的 values")
    assert len(values) > 1, "這一筆只有一格，換一筆當對象"
    trimmed = {name: item for name, item in list(values.items())[1:]}
    assert answers.differences(trimmed, values)


def test_a_raised_case_cannot_quietly_become_a_value() -> None:
    """「本來會炸」的那一筆如果哪天不炸了要紅（守門被拿掉是行為變了）。"""
    block = answers.result_of("response.MaterialResponse.from_impedance.scattering_wrong_shape_raises")
    assert "raised" in block, "這一筆在答案檔裡不是例外——控制組挑錯對象"
    became_value = {"values": {"mat": {"kind": "none"}}}
    assert answers.differences(became_value, block)


def test_the_header_must_carry_donor_provenance_and_environment() -> None:
    """檔頭的 donor 出身與環境三格少一格就不算證據（讀的時候當場炸）。"""
    root = answers.load_answers()
    donor = answers.as_mapping(root.get("donor"), "donor 檔頭")
    assert donor.get("tag") == "v3-donor"
    assert donor.get("clean") is True
    commit = donor.get("commit")
    # 只驗「它是一串十六進位、而且不是空的」，**不鎖長度**：規矩卡
    # assertions-not-pinned-to-counts 咬把數量寫死的斷言，而「幾個字」正是那一種。
    assert isinstance(commit, str) and commit != ""
    assert set(commit) <= set("0123456789abcdef"), f"donor 的 commit 不是十六進位：{commit!r}"
    env = answers.as_mapping(root.get("env"), "env 檔頭")
    assert env.get("x64") is False, "答案是在 x64 關著的時候量的，這一格是答案的一部分"
    assert env.get("backend") == "cpu"
