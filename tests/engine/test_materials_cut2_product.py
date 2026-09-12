"""票 #134 第二刀：**新家對上一代的逐筆比對**（產品考卷）。

case 表宣告的**每一個 id** 在這裡都有一題：每一筆 case 真的在 ``aosr.materials`` 上跑一遍，
每一個凍結常數真的從新家的模組上讀一次，再跟唯讀 donor 跑出來的標準答案**逐格**比（值、
dtype、shape、複數的實部虛部、pydantic 的 json schema、模型 dump 出來的每一格、以及每一種
錯誤的型別、訊息與「炸在第幾步」）。

**不是只比 id 集合。** 集合相等只證明兩邊「有同樣幾格」，證不出任何一格的值被比過。這裡
每一個 id 各自是一題；而「參數化出來的題目就是宣告的那些」由
:func:`test_every_declared_id_has_a_test_of_its_own` 當場對帳——少一題會被看見。
"""
from __future__ import annotations

import pytest

from blueprint import materials_cut2_cases as cases
from blueprint import materials_cut2_probe as probe
from tests.engine import _materials_cut2_answers as answers

CASE_IDS: list[str] = sorted(cases.case_id_list())
CONSTANT_IDS: list[str] = sorted(cases.constant_id_list())


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_a_case_behaves_like_the_donor(case_id: str) -> None:
    """一筆 case 在新家跑出來的結果，要跟 donor 的標準答案逐格相同。"""
    actual = answers.run_here(answers.case_by_id(case_id))
    expected = answers.result_of(case_id)
    gaps = answers.differences(actual, expected, case_id)
    assert gaps == [], "新家跟上一代對不上：\n" + "\n".join(gaps)


@pytest.mark.parametrize("frozen_id", CONSTANT_IDS)
def test_a_frozen_value_matches_the_donor(frozen_id: str) -> None:
    """一個凍結常數（或型別別名）在新家讀出來的值，要跟 donor 的凍結值逐格相同。"""
    module_name, kind, name = frozen_id.split(".", 2)
    module = answers.engine_lookup(module_name)
    if kind == "const":
        actual = probe.encode(getattr(module, name))
        expected = answers.frozen_block(module_name, "constants", name)
    else:
        actual = probe.encode(list(probe.alias_values(module, name)))
        expected = answers.frozen_block(module_name, "aliases", name)
    gaps = answers.differences(actual, expected, frozen_id)
    assert gaps == [], "凍結值對不上：\n" + "\n".join(gaps)


def test_every_declared_id_has_a_test_of_its_own() -> None:
    """參數化出來的題目**就是** case 表宣告的那些（少一題、多一題都要當場看見）。"""
    parametrised = {*CASE_IDS, *CONSTANT_IDS}
    assert parametrised == cases.declared_ids()
    assert parametrised == answers.answer_ids()


def test_no_id_is_counted_twice_on_either_side() -> None:
    """表上與答案檔裡的 id 都不准重複（集合相等看不出「同一個 id 兩筆」）。"""
    assert answers.duplicate_declared_ids() == []
    assert answers.duplicate_answer_ids() == []


def test_the_environment_matches_the_one_the_answers_were_measured_on() -> None:
    """這一跑的 JAX 設定要跟量標準答案那一跑同一組。

    ``x64`` 一開，同一段程式的 dtype 從 ``float32`` 變 ``float64``、值也跟著變——那時候
    上面每一題的紅是「跑在另一組設定上」，不是「新家寫錯了」。這一條把那件事講在前面。
    """
    import jax
    import jax.numpy as jnp

    env = answers.as_mapping(answers.load_answers().get("env"), "答案檔的 env 檔頭")
    # 問的是那個開關**看得見的後果**（預設浮點是不是 64 位元），不是去讀設定物件的某個
    # 屬性：後果才是答案檔裡那些 dtype 的來源，而屬性名會跟著函式庫漂。
    x64_here = jnp.asarray(1.0).dtype == jnp.float64
    assert bool(x64_here) is env.get("x64"), (
        "這一跑的 x64 設定跟量答案那一跑不一樣——先對齊設定再看上面的紅"
    )
    assert str(jax.default_backend()) == env.get("backend")


def test_the_libraries_that_write_the_error_messages_match_too() -> None:
    """寫那些錯誤訊息的函式庫要是同一個系列。

    這一刀的答案裡有幾十格是**它們寫的字**（``Input should be greater than 0``、yaml 的
    行號與位置）。版本一換，那些字就可能換——那時候上面的紅是「換了函式庫」，不是「新家
    寫錯了」。這一條把那件事講在前面。

    **比到第二格（``2.13``）為止，不比第三格**，兩個理由：pydantic 自己寫進訊息尾巴的那個
    網址就是 ``errors.pydantic.dev/<第一格>.<第二格>/``，第三格根本不出現；而且這一刀交件
    時實測過——標準答案那一跑（donor 的環境，pydantic 2.13.4）與新家這一跑的環境
    （2.13.5）各跑一次同一份產生器，104 筆 case 逐格**零差異**。第三格也比的話，那是一個
    會為了證明不出差異的東西而紅的裁判。第一、二格一換就紅：那時候訊息真的會變。
    """
    from importlib import metadata

    versions = answers.as_mapping(
        answers.as_mapping(answers.load_answers().get("env"), "env 檔頭").get("versions"),
        "env 檔頭的 versions",
    )
    for name in ("pydantic", "PyYAML"):
        recorded = str(versions.get(name))
        here = metadata.version(name)
        assert here.split(".")[:2] == recorded.split(".")[:2], (
            f"{name} 的版本跟量答案那一跑不是同一個系列（這一跑 {here}、"
            f"答案檔 {recorded!r}）——錯誤訊息那幾格的紅先看這裡"
        )
