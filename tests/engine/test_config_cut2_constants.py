"""這一刀那 9 支模組的**凍結常數**考卷：逐個拿 ``getattr(模組, NAME)`` 跟 donor 比。

**這一支在補什麼洞（獨立驗證的修 B）。** 原本的形狀只有
``test_config_cut2_loader_probes.py`` 那一支在比答案檔，而它的參數化清單是
``cases.cut2_case_ids()``——那是**探針**。常數雖然在 case 表宣告了、答案檔也有、
``test_every_declared_case_has_exactly_one_answer`` 的集合相等也過，**但沒有一條考卷
拿 ``module.NAME`` 去跟答案檔那一筆比**。後果（獨立驗證量到的）：`fem_lane` 的
11 個可改常數改壞之後，整套 `tests/engine` 照樣綠——那一支自己的 docstring 卻寫著
「`cut2_fem_lane` 的裁判就是那 21 筆凍結值」，那句話當時是假的。

**這一支的形狀。** 參數化清單就是 case 表宣告的常數（``constant_ids()`` 裡這一刀那
一組），一條都不另外抄；每一筆比的是「新家模組上那一格的值」與「donor 在上一代跑出來
的凍結值」，逐位元（浮點走 ``is_approx``：``nan`` 兩邊都是 ``nan``、``inf`` 相等、
tuple／dict／純量都蓋到）。

覆蓋不到的：**載入期算出來的常數**（`fem_lane` 的 `SPLAY_CAP`／`MIN_OCTAGON_EDGE_M`／
六個 `POSITION_*`）——它們的值由這一支蓋到，但「它們是從 TOML 算出來的」那條線由
`tests/engine/test_config_cut2_loader_probes.py` 的突變考卷蓋到；三支私有載入器自己的
錯誤訊息仍然沒有逐筆比。
"""
from __future__ import annotations

import importlib
from typing import cast

import pytest

from blueprint import config_cut1_cases as cases
from tests.engine._config_answers import constant, is_approx

# 這一刀那 9 支模組宣告的每一筆**常數**（`<表名>.const.<NAME>`），排序好當參數化清單。
# 清單就是 case 表（`cut2_case_ids()` 過濾出 `const.` 那些），不另外抄一份；
# 刪一筆常數就少一題，而 `test_every_declared_case_has_exactly_one_answer` 會紅。
CUT2_CONSTANT_IDS: tuple[str, ...] = tuple(
    sorted(case_id for case_id in cases.constant_ids() if ".const." in case_id and case_id.startswith("cut2_"))
)


def test_the_constant_list_is_not_empty() -> None:
    """清單不是空的（空的參數化清單會讓這一支變成「什麼都沒比」的假綠）。"""
    assert CUT2_CONSTANT_IDS


@pytest.mark.parametrize("constant_id", CUT2_CONSTANT_IDS)
def test_constant_matches_the_donor_answer(constant_id: str) -> None:
    """這一筆常數：新家模組上那一格的值，逐位元等於 donor 的凍結值。

    ``case_id`` 是 ``<表名>.const.<NAME>``：表名換成新家的模組名（``cut2_fem_lane`` →
    ``aosr.config.fem_lane``），``NAME`` 就是模組上那一格。
    """
    table_name, _, name = constant_id.split(".", 2)
    module = importlib.import_module(f"aosr.config.{cases.engine_module_name(table_name)}")
    assert hasattr(module, name), f"{constant_id}：新家模組上沒有這一格"
    actual = cast(object, getattr(module, name))
    assert is_approx(actual, constant(table_name, name)), (
        f"{constant_id}：新家那一格跟 donor 的凍結值不一樣\n"
        f"  donor: {constant(table_name, name)!r}\n"
        f"  新家 : {actual!r}"
    )
