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

**類別常數比欄位表與身分（票 #166，補票 #155 留下的那一格）。** 類別那一種常數
（`CalibrationConfig` 那一類 13 筆）上面那一條只比**名字**；載入器探針比的是回傳物件的
dump，而 pydantic 照**上一層宣告的類別**輸出，同名子類別多出來的欄位、序列化時排除的欄位
都看不到。所以另外兩條：

* ``test_class_constant_has_the_donor_field_table``——匯出的類別宣告的欄位名（照宣告順序、
  含排除的欄位）等於 donor 那一邊的欄位表（答案檔類別記號的 ``fields``，產生器在上一代量的）。
  v3 刻意多加的欄位具名列在 ``ADDED_IN_V3``，每一格都必須是序列化時排除的。
* ``test_class_constant_is_the_class_at_its_home``——每一筆在 ``CLASS_HOMES`` 登記一個家：
  哪一筆成功探針、回傳物件底下哪一格（空路徑是回傳物件本身）。那一格的物件類別**就是**
  匯出的那一個（``is``，子類別不算）；巢狀的那一格，上一層模型宣告的欄位型別也要是它。
  只在登記的那一格比，冒牌貨藏在別的欄位裡不算數。

控制組是 ``test_a_same_named_impostor_turns_a_judge_red``（六種冒牌貨）與
``test_home_judge_controls``（冒牌貨藏在別的欄位、子類別坐在家裡、上一層宣告的不是它）。
**這兩條不比的**：欄位的型別、預設值與驗證器（只比欄位名）；以及同名、欄位名完全相同的
子類別在**最外層**由載入器真的建出來——那時匯出的就是載入器用的那一個、欄位表也跟上一代
一樣，沒有東西可以分辨，算綠。
"""
from __future__ import annotations

import importlib
import types
from dataclasses import field, fields, is_dataclass, make_dataclass
from typing import cast

import pytest
from pydantic import BaseModel, ConfigDict, Field, create_model
from pydantic.fields import FieldInfo

from blueprint import config_cut1_cases as cases
from tests.engine._config_answers import (
    constant,
    constant_kind,
    constant_type_fields,
    decode,
    expected_block,
    is_approx,
    loader_case_run,
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


# donor 說「這一格是類別」的那幾筆（答案檔記成 `{"kind": "type"}`），從上面那一串過濾，不另外抄。
CUT2_CLASS_CONSTANT_IDS: tuple[str, ...] = tuple(
    case_id for case_id in CUT2_CONSTANT_IDS if constant_kind(*case_id.split(".const.")) == "type"
)

# v3 在上一代的類別上**刻意多加**的欄位（上一代的欄位表裡沒有）。每一格都必須是序列化時排除的
# 欄位，載入器回傳物件的 dump 才會跟上一代同一份；要加一格就寫出是哪一張票、為什麼。
ADDED_IN_V3: dict[str, tuple[str, ...]] = {
    # 票 #560：TMM（傳遞矩陣法）要的空氣條件；舊四欄案例照樣載入，dump 不變。
    "cut2_physics_constants.const.PhysicsConstants": ("specific_heat_ratio", "prandtl_number", "atmospheric_pressure_pa"),
}

# 每一筆類別常數的家：哪一筆成功探針、回傳物件底下哪一格（空路徑＝回傳物件本身）。
# 路徑另由 `test_class_homes_cover_every_class_constant` 核對 donor 那一筆答案在同一格真的有東西。
CLASS_HOMES: dict[str, tuple[str, tuple[str, ...]]] = {
    "cut2_calibration.const.CalibrationConfig": ("cut2_calibration.load.real", ()),
    "cut2_calibration.const.CalibrationBandConfig": ("cut2_calibration.load.real", ("band",)),
    "cut2_calibration.const.CalibrationRtConfig": ("cut2_calibration.load.real", ("rt",)),
    "cut2_calibration.const.CalibrationSmoothingConfig": ("cut2_calibration.load.real", ("smoothing",)),
    "cut2_mat_continuation.const.MatContinuationConfig": ("cut2_mat_continuation.load.real", ()),
    "cut2_membrane_opt.const.MembraneOptConfig": ("cut2_membrane_opt.load.real", ()),
    "cut2_physics_constants.const.PhysicsConstants": ("cut2_physics_constants.load.real", ()),
    "cut2_scoring.const.ScoringConfig": ("cut2_scoring.load.real", ()),
    "cut2_scoring.const.SpatialScoringConfig": ("cut2_scoring.load.real", ("spatial",)),
    "cut2_scoring.const.AsymmetryScoringConfig": ("cut2_scoring.load.real", ("asymmetry",)),
    "cut2_scoring.const.DecayScoringConfig": ("cut2_scoring.load.real", ("decay",)),
    "cut2_scoring.const.SmoothingScoringConfig": ("cut2_scoring.load.real", ("smoothing",)),
    "cut2_stereo_layout.const.StereoLayoutConfig": ("cut2_stereo_layout.load.real", ()),
}


def _module_of(constant_id: str) -> types.ModuleType:
    """這一筆常數住的新家模組。"""
    table_name = constant_id.split(".", 1)[0]
    return importlib.import_module(f"aosr.config.{cases.engine_module_name(table_name)}")


def _declared_fields(cls: type) -> list[str]:
    """一個類別宣告的欄位名，照宣告順序（dataclass 走 ``fields``、pydantic 走類別上的 ``model_fields``）。"""
    if is_dataclass(cls):
        return [field.name for field in fields(cls)]
    model_fields = getattr(cls, "model_fields", None)
    if isinstance(model_fields, dict):
        return [str(name) for name in cast(dict[object, object], model_fields)]
    raise AssertionError(f"{cls!r} 既不是 dataclass 也不是 pydantic 模型，欄位表比不了")


def _field_table_holds(constant_id: str, exported: object) -> bool:
    """匯出的類別宣告的欄位表＝donor 的欄位表＋``ADDED_IN_V3`` 那幾格（而且那幾格都是排除的）。"""
    if not isinstance(exported, type):
        return False
    table_name, _, name = constant_id.split(".", 2)
    added = ADDED_IN_V3.get(constant_id, ())
    if _declared_fields(exported) != [*constant_type_fields(table_name, name), *added]:
        return False
    model_fields = cast(dict[str, FieldInfo], getattr(exported, "model_fields", {}))
    return all(model_fields[field].exclude is True for field in added)


def _home_holds(constant_id: str, exported: object, actual: object) -> bool:
    """在登記的那一格，物件的類別就是 ``exported``；巢狀的那一格，上一層宣告的欄位型別也是它。"""
    _, path = CLASS_HOMES[constant_id]
    parent: object = None
    item = actual
    for step in path:
        parent, item = item, getattr(item, step, None)
    if type(item) is not exported:
        return False
    if parent is None:
        return True
    model_fields = getattr(type(parent), "model_fields", None)
    if not isinstance(model_fields, dict):
        raise AssertionError(f"{constant_id}：這一刀的巢狀類別只掛在 pydantic 模型底下，這一格不是")
    return cast(dict[str, FieldInfo], model_fields)[path[-1]].annotation is exported


def _home_judge(constant_id: str, exported: object) -> bool:
    """跑這一筆的家那一筆探針，看登記的那一格是不是 ``exported``。"""
    case_id, _ = CLASS_HOMES[constant_id]
    with loader_case_run(case_id) as actual:
        return _home_holds(constant_id, exported, actual)


def test_the_class_constant_list_is_not_empty() -> None:
    """donor 說是類別的常數至少有一筆（空清單會讓下面幾條變成什麼都沒比）。"""
    assert CUT2_CLASS_CONSTANT_IDS


def test_class_homes_cover_every_class_constant() -> None:
    """家的登記＝類別常數的集合；每個家是同一支模組 donor 說會成功的探針，那一格在 donor 答案裡真的有物件。"""
    assert set(CLASS_HOMES) == set(CUT2_CLASS_CONSTANT_IDS)
    assert set(ADDED_IN_V3) <= set(CUT2_CLASS_CONSTANT_IDS)
    for constant_id, (case_id, path) in CLASS_HOMES.items():
        assert case_id.split(".", 1)[0] == constant_id.split(".", 1)[0], f"{constant_id} 的家不在同一支模組"
        expected = expected_block(probe_by_id(case_id))
        assert probe_raised(expected) is None, f"{constant_id} 的家 {case_id} 在 donor 是炸掉的那一筆"
        node = decode(expected)
        for step in path:
            assert isinstance(node, dict) and step in node, f"{constant_id}：donor 的 {case_id} 底下沒有 {step!r} 那一格"
            node = cast(dict[str, object], node)[step]
        assert isinstance(node, dict), f"{constant_id}：donor 的 {case_id} 在 {path!r} 那一格不是物件"


@pytest.mark.parametrize("constant_id", CUT2_CLASS_CONSTANT_IDS)
def test_class_constant_has_the_donor_field_table(constant_id: str) -> None:
    """匯出的類別宣告的欄位名（照順序、含排除的）＝donor 的欄位表＋v3 具名多加的那幾格。"""
    name = constant_id.split(".", 2)[2]
    exported = cast(object, getattr(_module_of(constant_id), name))
    assert isinstance(exported, type), f"{constant_id}：donor 那一格是類別，新家那一格是 {exported!r}"
    assert _field_table_holds(constant_id, exported), (
        f"{constant_id}：欄位表跟 donor 不一樣\n"
        f"  donor＋v3 多加：{[*constant_type_fields(*constant_id.split('.const.')), *ADDED_IN_V3.get(constant_id, ())]}\n"
        f"  新家          ：{_declared_fields(exported)}"
    )


@pytest.mark.parametrize("constant_id", CUT2_CLASS_CONSTANT_IDS)
def test_class_constant_is_the_class_at_its_home(constant_id: str) -> None:
    """匯出的類別就是載入器在登記那一格實際建出來的那一個（``is``，不是同名、不是子類別）。"""
    name = constant_id.split(".", 2)[2]
    exported = cast(object, getattr(_module_of(constant_id), name))
    assert _home_judge(constant_id, exported), (
        f"{constant_id}：{CLASS_HOMES[constant_id]} 那一格的物件不是匯出的這個類別——"
        "名字指的不是載入器實際建出來的那一個"
    )


def _impostor(original: type, flavor: str) -> type:
    """同名的冒牌類別，六種。

    ``rejects-fields``：只有自己一格，收不下設定檔的欄位。``accepts-anything``：什麼都收、另外多一格。
    ``reordered``：欄位名一樣、宣告順序反過來（只有一格的那幾個等於原樣，靠家那一條擋）。
    ``subclass-extra``：原本那一個的子類別，多一格欄位。``subclass-excluded``：同上，但那一格序列化時
    排除（dump 看不到）。``same-fields-subclass``：子類別、欄位一格不多——只有身分分得出來。
    """
    name = original.__name__
    if flavor == "rejects-fields":
        return make_dataclass(name, [("impostor_only_field", int)])
    if flavor == "accepts-anything":
        return create_model(name, __config__=ConfigDict(extra="allow"), impostor_only_field=(int, 0))
    if flavor == "reordered":
        if is_dataclass(original):
            frozen = bool(getattr(original, "__dataclass_params__").frozen)
            return make_dataclass(name, [(item.name, item.type) for item in reversed(fields(original))], frozen=frozen)
        model_fields = cast(dict[str, FieldInfo], getattr(original, "model_fields"))
        reordered = dict(reversed(model_fields.items()))
        namespace: dict[str, object] = {"__annotations__": {key: info.annotation for key, info in reordered.items()}, **reordered}
        return type(name, (BaseModel,), namespace)
    if is_dataclass(original):
        extra = [] if flavor == "same-fields-subclass" else [("impostor_only_field", int, field(default=0))]
        frozen = bool(getattr(original, "__dataclass_params__").frozen)
        return make_dataclass(name, extra, bases=(original,), frozen=frozen)
    model = cast(type[BaseModel], original)
    if flavor == "subclass-extra":
        return create_model(name, __base__=model, impostor_only_field=(int, 0))
    if flavor == "subclass-excluded":
        return create_model(name, __base__=model, impostor_only_field=(int, Field(default=0, exclude=True)))
    return create_model(name, __base__=model)


_FLAVORS = ("rejects-fields", "accepts-anything", "reordered", "subclass-extra", "subclass-excluded", "same-fields-subclass")
# 同名、欄位名一樣的子類別在最外層由載入器真的建出來時沒有東西可以分辨（見檔頭），那幾格不列。
_IMPOSTOR_CASES = [
    pytest.param(constant_id, flavor, id=f"{constant_id.split('.const.')[1]}-{flavor}")
    for constant_id in CUT2_CLASS_CONSTANT_IDS
    for flavor in _FLAVORS
    if not (flavor == "same-fields-subclass" and CLASS_HOMES[constant_id][1] == ())
]


@pytest.mark.parametrize(("constant_id", "flavor"), _IMPOSTOR_CASES)
def test_a_same_named_impostor_turns_a_judge_red(constant_id: str, flavor: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """控制組：模組上那一格換成同名冒牌貨，欄位表與家那兩位裁判至少一位紅（名字那一條照樣綠）。

    在呼叫當下用模組上的名字建物件的載入器（最外層），會建出冒牌貨或炸掉；巢狀那一格的型別
    在上一層類別定義時就綁定了，載入器照舊建原本那一個。哪一種都不准兩位裁判一起綠。
    """
    module = _module_of(constant_id)
    table_name, _, name = constant_id.split(".", 2)
    impostor = _impostor(cast(type, getattr(module, name)), flavor)
    monkeypatch.setattr(module, name, impostor)
    assert is_approx(cast(object, getattr(module, name)), constant(table_name, name)), "名字那一條應該照樣綠"
    assert not (_field_table_holds(constant_id, impostor) and _home_judge(constant_id, impostor)), (
        f"{constant_id}：換成 {flavor} 冒牌貨，欄位表與家兩位裁判都綠"
    )


@pytest.mark.parametrize(("constant_id", "other_field"), [
    ("cut2_physics_constants.const.PhysicsConstants", "specific_heat_ratio"),
    ("cut2_calibration.const.CalibrationBandConfig", "rt"),
])
def test_home_judge_controls(constant_id: str, other_field: str) -> None:
    """家那一條自己的控制組，三種形狀各要紅一次（原本那一個坐在家裡時要綠）。

    * 冒牌貨的實例塞在別的欄位、家裡仍是原本的類別——不在任何位置都算；
    * 家裡坐的是原本那一個的子類別——``is``，子類別不算；
    * 巢狀那一格：匯出的是欄位一樣的子類別、也真的坐在家裡，但上一層宣告的仍是原本那一個。
    """
    original = cast(type[BaseModel], getattr(_module_of(constant_id), constant_id.split(".", 2)[2]))
    impostor = _impostor(original, "rejects-fields")
    subclass = _impostor(original, "same-fields-subclass")
    case_id, path = CLASS_HOMES[constant_id]
    with loader_case_run(case_id) as actual:
        root = cast(BaseModel, actual)
        assert _home_holds(constant_id, original, root), "原本那一個在登記的那一格，這一條應該是綠的"
        planted = root.model_copy(update={other_field: impostor(impostor_only_field=0)})
        assert not _home_holds(constant_id, impostor, planted), f"{constant_id}：冒牌貨藏在 {other_field}，家那一條卻綠了"
        if path:
            seated = root.model_copy(update={path[-1]: subclass(**cast(BaseModel, getattr(root, path[-1])).model_dump())})
        else:
            seated = subclass(**root.model_dump())
        assert not _home_holds(constant_id, original, seated), f"{constant_id}：家裡坐的是子類別，家那一條卻綠了"
        if path:
            assert not _home_holds(constant_id, subclass, seated), f"{constant_id}：上一層宣告的不是匯出的那一個，家那一條卻綠了"
