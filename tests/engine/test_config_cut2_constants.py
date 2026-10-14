"""這一刀那 9 支模組的**凍結常數**考卷：逐個拿 ``getattr(模組, NAME)`` 跟 donor 比。

**這一支在補什麼洞（獨立驗證的修 B）。** 原本的形狀只有
``test_config_cut2_loader_probes.py`` 那一支在比答案檔，而它的參數化清單是
``cases.cut2_case_ids()``——那是**探針**。常數雖然在 case 表宣告了、答案檔也有、
``test_every_declared_case_has_at_least_one_answer`` 的集合相等也過，**但沒有一條考卷
拿 ``module.NAME`` 去跟答案檔那一筆比**。後果（獨立驗證量到的）：`fem_lane` 的
11 個可改常數改壞之後，整套 `tests/engine` 照樣綠——那一支自己的 docstring 卻寫著
「`cut2_fem_lane` 的裁判就是那 21 筆凍結值」，那句話當時是假的。

**這一支的形狀。** 參數化清單就是 case 表宣告的常數（``constant_ids()`` 裡這一刀那
一組），一條都不另外抄；每一筆比的是「新家模組上那一格的值」與「donor 在上一代跑出來
的凍結值」，逐位元（浮點走 ``is_approx``：``nan`` 兩邊都是 ``nan``、``inf`` 相等、
tuple／dict／純量都蓋到）。

覆蓋不到的：**載入期算出來的常數**（`fem_lane` 的 `SPLAY_CAP`／`MIN_OCTAGON_EDGE_M`／
六個 `POSITION_*`）——它們的值由這一支蓋到，但「它們是從 TOML 算出來的」那條線由
`tests/engine/test_config_cut2_fem_lane_wiring.py` 蓋到（票 #162 補的：餵一份改過值的
`fem_lane.toml`、reload，常數要跟著變）。三支私有載入器的真檔與所有報錯點則由
`tests/engine/test_config_cut2_loader_probes.py` 逐筆比上一代的回傳值、例外型別與訊息（票 #156）。

**類別常數比身分（票 #166，補票 #155 留下的那一格）。** 類別那一種常數（`CalibrationConfig`
那一類 13 筆）在答案檔裡是 ``{"kind": "type", "name": "…"}``，上面那一條只比**那個名字**。
欄位對不對由 ``cut2_*.load.real`` 那一筆探針逐欄比（載入器回傳的物件收成普通容器）；
但「模組匯出的這個名字，指的是不是載入器實際建出來的那一個類別物件」原本沒有人守——
匯出一個同名、欄位不同的類別，而載入器照舊用原本那一個，兩條都綠。
``test_class_constant_is_the_class_the_loader_builds`` 補這一格：跑這一支每一筆成功的
探針，把回傳物件連同巢狀欄位一路走下去收集碰到的實例類別，匯出的那一個要**以身分**
（``is``）出現在裡面。答案檔不用改：身分在新家這一邊比，donor 那一邊只負責說「這一格是類別」。
清點（票 #166 第 1 步）：13 筆都由同一支模組的 ``load.real`` 探針建得出來，沒有「從來沒被
載入過、只剩名字」的類別。控制組是 ``test_a_same_named_impostor_turns_a_judge_red``。
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, fields, is_dataclass, make_dataclass
from typing import cast

import pytest
from pydantic import BaseModel

from blueprint import config_cut1_cases as cases
from tests.engine._config_answers import (
    as_plain,
    constant,
    decode,
    expected_block,
    is_approx,
    loader_case_run,
    module_answers,
    probe_by_id,
    probe_raised,
)

# 這一刀那 9 支模組宣告的每一筆**常數**（`<表名>.const.<NAME>`），排序好當參數化清單。
# 清單就是 case 表（`cut2_case_ids()` 過濾出 `const.` 那些），不另外抄一份；
# 刪一筆常數就少一題，而 `test_every_declared_case_has_at_least_one_answer` 會紅。
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


def _answer_kind(constant_id: str) -> object:
    """答案檔裡那一筆常數的記號種類（donor 那一邊說它是什麼）。"""
    table_name, _, name = constant_id.split(".", 2)
    constants = cast(dict[str, object], module_answers(table_name)["constants"])
    return cast(dict[str, object], constants[name]).get("kind")


# donor 說「這一格是類別」的那幾筆（答案檔記成 `{"kind": "type"}`），從上面那一串過濾，不另外抄。
CUT2_CLASS_CONSTANT_IDS: tuple[str, ...] = tuple(
    case_id for case_id in CUT2_CONSTANT_IDS if _answer_kind(case_id) == "type"
)


def _instance_types(value: object) -> list[type]:
    """從載入器回傳的物件一路往下走，收集碰到的每一個實例的類別。

    dataclass 走 ``fields``、pydantic 模型走類別上的 ``model_fields``，容器逐項進去；
    純量與 ``None`` 不算（它們不是這一刀匯出的類別）。
    """
    found: list[type] = []
    pending: list[object] = [value]
    while pending:
        item = pending.pop()
        if item is None or isinstance(item, (bool, int, float, str, type)):
            continue
        if isinstance(item, dict):
            pending.extend(cast(dict[object, object], item).values())
            continue
        if isinstance(item, (list, tuple)):
            pending.extend(cast(list[object], item))
            continue
        found.append(type(item))
        if is_dataclass(item):
            pending.extend(getattr(item, field.name) for field in fields(item))
            continue
        model_fields = getattr(type(item), "model_fields", None)
        if isinstance(model_fields, dict):
            pending.extend(getattr(item, str(name)) for name in cast(dict[object, object], model_fields))
    return found


def _successful_case_ids(table_name: str) -> list[str]:
    """這一支模組宣告的探針裡，donor 說會成功的那幾筆。"""
    return [
        str(case["id"])
        for case in cases.cases_for(table_name)
        if probe_raised(expected_block(probe_by_id(str(case["id"])))) is None
    ]


def _cases_that_build(cls: type, table_name: str) -> list[str]:
    """哪幾筆成功探針的回傳物件（含巢狀欄位）用到了**這一個**類別物件（比身分）。"""
    hits: list[str] = []
    for case_id in _successful_case_ids(table_name):
        with loader_case_run(case_id) as actual:
            if any(found is cls for found in _instance_types(actual)):
                hits.append(case_id)
    return hits


def _case_matches_donor(case_id: str) -> bool:
    """這一筆成功探針，新家跑出來的值是否跟 donor 一樣（同載入器探針那一支的成功分支）。"""
    expected = expected_block(probe_by_id(case_id))
    with loader_case_run(case_id) as actual:
        if isinstance(actual, dict) and "raised" in actual:
            return False
        return is_approx(as_plain(actual), as_plain(decode(expected)))


class _WalkLeafModel(BaseModel):
    """走訪零件控制組的最底層：pydantic 模型，只有純量。"""

    value: int


class _WalkMiddleModel(BaseModel):
    """pydantic 模型的欄位裡掛另一個模型。"""

    leaf: _WalkLeafModel


class _WalkDictModel(BaseModel):
    """只出現在 dict 的值裡的那一種（dict 那一條路沒走就收不到它）。"""

    value: int


@dataclass(frozen=True)
class _WalkChild:
    """dataclass 的欄位裡掛 pydantic 模型。"""

    middle: _WalkMiddleModel


@dataclass(frozen=True)
class _WalkRoot:
    """最外層：dataclass，欄位是 tuple 與 dict 包著的子層。"""

    children: tuple[_WalkChild, ...]
    by_name: dict[str, _WalkDictModel]


def test_instance_types_walks_dataclass_model_and_container_fields() -> None:
    """走訪零件自己的控制組：dataclass 欄位、pydantic 欄位、tuple 與 dict 裡面的都要收到。"""
    leaf = _WalkLeafModel(value=1)
    root = _WalkRoot(children=(_WalkChild(middle=_WalkMiddleModel(leaf=leaf)),), by_name={"a": _WalkDictModel(value=2)})
    found = _instance_types(root)
    for cls in (_WalkRoot, _WalkChild, _WalkMiddleModel, _WalkLeafModel, _WalkDictModel):
        assert any(item is cls for item in found), f"走訪沒收到 {cls.__name__}：{found}"


def test_the_class_constant_list_is_not_empty() -> None:
    """donor 說是類別的常數至少有一筆（空清單會讓下面兩條變成什麼都沒比）。"""
    assert CUT2_CLASS_CONSTANT_IDS


@pytest.mark.parametrize("constant_id", CUT2_CLASS_CONSTANT_IDS)
def test_class_constant_is_the_class_the_loader_builds(constant_id: str) -> None:
    """匯出的類別就是載入器實際建出來的那一個（``is``，不是同名）。"""
    table_name, _, name = constant_id.split(".", 2)
    module = importlib.import_module(f"aosr.config.{cases.engine_module_name(table_name)}")
    exported = cast(object, getattr(module, name))
    assert isinstance(exported, type), f"{constant_id}：donor 那一格是類別，新家那一格是 {exported!r}"
    assert _cases_that_build(exported, table_name), (
        f"{constant_id}：這一支沒有一筆成功探針的回傳物件用到匯出的這個類別物件——"
        "名字指的不是載入器實際建出來的那一個"
    )


@pytest.mark.parametrize("constant_id", CUT2_CLASS_CONSTANT_IDS)
def test_a_same_named_impostor_turns_a_judge_red(constant_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """控制組：模組上那一格換成同名、欄位不同的類別，至少一位裁判要紅。

    名字那一條本來就照樣綠（只比名字）。載入器在呼叫當下用模組上的名字去建物件的，會建出
    冒牌貨或炸掉——載入器探針紅；載入器用的是別處綁好的原本那一個（巢狀欄位的型別在類別
    定義時就綁定了）——身分那一條紅。兩條都綠就是這一格又沒人守了。
    """
    table_name, _, name = constant_id.split(".", 2)
    module = importlib.import_module(f"aosr.config.{cases.engine_module_name(table_name)}")
    impostor = make_dataclass(name, [("impostor_only_field", int)])
    monkeypatch.setattr(module, name, impostor)
    assert is_approx(cast(object, getattr(module, name)), constant(table_name, name)), "名字那一條應該照樣綠"
    identity_green = bool(_cases_that_build(impostor, table_name))
    probes_green = all(_case_matches_donor(case_id) for case_id in _successful_case_ids(table_name))
    assert not (identity_green and probes_green), f"{constant_id}：換成同名冒牌貨，身分與探針兩條都綠"
