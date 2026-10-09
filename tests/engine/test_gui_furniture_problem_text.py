"""第八支第三步之一：實跑問題的中文答案手寫；驗證原文仍由原驗證器產生。"""
from pathlib import Path
from typing import cast

import pytest

from aosr.gui.problem_text import field_name, plain_problems
from aosr.reporting.validation import SchemeProblem, validate_scheme
from tests.engine import _furniture_cases as furniture
from tests.engine import _speaker_setup_cases as speakers
from tests.engine._directivity import DIRECTIVITY
from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine.test_gui_browser import LONG_DECIMAL
from tests.engine.test_gui_problem_text import ENGLISH, _client


# 主案例用中文代號，讓兩支格式檢查直接看整行；英文使用者代號另題守原樣保留。
FIELD_CASES: dict[str, tuple[bool, str, object]] = {
    "width_blank": (False, "width_m", None), "width_negative": (False, "width_m", -1),
    "forward_blank": (False, "placement.forward_m", None),
    "yaw_45": (False, "placement.yaw_deg", 45),
    "glass_sofa": (False, "material", "glass"), "material_typo": (False, "material", "typo"),
    "cloud_relative": (True, "placement", {"forward_m": 1, "left_m": 0,
                                             "bottom_height_m": 0, "yaw_deg": 0}),
    "cloud_z_blank": (True, "placement.bottom_center_m", [2, 2, None]),
}


def _layout_document(case: str) -> dict[str, object]:
    item = furniture.relative_item(furniture_id="座席")
    placement = cast(dict[str, object], item["placement"])
    if case == "duplicate":
        return furniture.document(item, furniture.relative_item(furniture_id="座席"))
    if case == "floating":
        placement["bottom_height_m"] = 0.3
    elif case == "outside":
        placement["forward_m"] = 10
    elif case == "touching":
        return furniture.document(item, furniture.relative_item(furniture_id="另一席", placement={
            "forward_m": 1, "left_m": -0.5, "bottom_height_m": 0, "yaw_deg": 0}))
    elif case == "ear_inside":
        item["height_m"] = 1.5
        placement.update(forward_m=0, left_m=0)
    elif case == "speaker_inside":
        item["height_m"] = 1.5
        placement.update(forward_m=2, left_m=1)
    return furniture.document(item)


def _speaker_document(case: str) -> dict[str, object]:
    document = speakers.document("desk", left_z=1.0) if case == "height" else speakers.document()
    setup = cast(dict[str, object], document["speaker_setup"])
    cabinet = cast(dict[str, object], setup["cabinet"])
    if case == "desk_missing":
        setup["mount"] = "desk"
    elif case == "bookshelf_floor":
        setup["mount"] = "floor"
    elif case == "center_above":
        cabinet["acoustic_center_above_bottom_m"] = 0.4
    elif case == "cabinet_blank":
        cabinet["width_m"] = None
    return document


def case_document(case: str) -> dict[str, object]:
    """只組題目，不從產品的中文化函式反推答案。"""
    if case in FIELD_CASES:
        cloud, path, value = FIELD_CASES[case]
        item = furniture.cloud_item(furniture_id="天板") if cloud else furniture.relative_item(furniture_id="座席")
        target = item
        *parents, key = path.split(".")
        for parent in parents:
            target = cast(dict[str, object], target[parent])
        target[key] = value
        return furniture.document(item)
    if case in {"duplicate", "floating", "outside", "touching", "ear_inside", "speaker_inside"}:
        return _layout_document(case)
    return _speaker_document(case)


ANSWERS = [
    ("width_blank", ["第 1 件家具（沙發，座席）・寬：空著沒填"]),
    ("width_negative", ["第 1 件家具（沙發，座席）・寬：家具尺寸必須是有限正數"]),
    ("forward_blank", ["第 1 件家具（沙發，座席）・前方：空著沒填"]),
    ("yaw_45", ["第 1 件家具（沙發，座席）・相對角：家具角度只收數字 0／90／180／270 度"]),
    ("glass_sofa", ["第 1 件家具（沙發，座席）：材質「玻璃」不適用「沙發」"]),
    ("material_typo", ["第 1 件家具（沙發，座席）・材質：家具材質必須是五種材質之一：布面／皮面／木質／玻璃／吸音天雲"]),
    ("cloud_relative", ["第 1 件家具（天雲，天板）：天雲必須用房間座標"]),
    ("cloud_z_blank", ["第 1 件家具（天雲，天板）・底面中心 z：空著沒填"]),
    ("duplicate", ["第 1 件家具（沙發，座席）、第 2 件家具（沙發，座席）：家具代號不可重複"]),
    ("floating", ["第 1 件家具（沙發，座席）：貼地家具的底面必須在地板接觸界線內"]),
    ("outside", ["第 1 件家具（沙發，座席）：超出房間接觸界線"]),
    ("touching", ["第 1 件家具（沙發，座席）、第 2 件家具（沙發，另一席）：兩件家具的間隙必須超過接觸界線"]),
    ("ear_inside", ["主位 → 第 1 件家具（沙發，座席）：在家具內部超過接觸界線"]),
    ("speaker_inside", ["左聲道喇叭 → 第 1 件家具（沙發，座席）：在家具內部超過接觸界線"]),
    ("desk_missing", ["喇叭類型與擺法：喇叭放桌面必須剛好一件茶几或書桌，現在有 0 件"]),
    ("bookshelf_floor", ["喇叭類型與擺法：書架喇叭只能放腳架或桌面"]),
    ("center_above", ["喇叭箱體：聲學中心不可高過箱體頂面"]),
    ("cabinet_blank", ["箱寬：空著沒填"]),
    ("height", ["左聲道喇叭 z 座標：高度 1 m 跟擺法推出值不同：桌面頂 0.73 m＋聲學中心離箱底 0.205 m＝0.935 m"]),
]


def assert_chinese_lines(lines: list[str]) -> None:
    assert lines
    assert not [line for line in lines if ENGLISH.search(line)]
    assert not [line for line in lines if LONG_DECIMAL.search(line)]


@pytest.mark.parametrize("case,expected", ANSWERS)
def test_furniture_input_problem_lines_are_chinese(tmp_path: Path, case: str, expected: list[str]) -> None:
    document = case_document(case)
    written = validate_scheme(document, capabilities=furniture.CAPABILITIES, directivity=DIRECTIVITY)
    with _client(tmp_path) as client:
        response = client.post("/api/validate", json=document)
        assert response.status_code == 200
        problems = response.json()["problems"]
    lines = [problem["text"] for problem in problems]
    assert_chinese_lines(lines)
    assert lines == expected
    # 顯示翻譯與分支過濾不能改掉保留那支的驗證原句。
    assert all(detail in {str(problem) for problem in written} for row in problems for detail in row["details"])


@pytest.mark.parametrize("path,expected", [
    ("depth_m", "深"), ("height_m", "高"), ("furniture_id", "代號"), ("kind", "種類"),
    ("placement.left_m", "左方"), ("placement.bottom_height_m", "底面離地"),
    ("placement", "擺法"), ("placement.ListenerPlacement.yaw_deg", "相對角"),
    ("placement.RoomPlacement.yaw_deg", "房間角度"),
    ("placement.RoomPlacement.bottom_center_m.0", "底面中心 x"),
    ("placement.RoomPlacement.bottom_center_m.1", "底面中心 y"),
    ("placement.RoomPlacement.bottom_center_m.2", "底面中心 z"),
])
def test_furniture_field_names_follow_scheme_model(path: str, expected: str) -> None:
    # 第零格故意不是物件，定位仍要用文件的真正索引，不能過濾後重排。
    document = {"furniture": [None, furniture.relative_item(furniture_id="座席")]}
    label = field_name(f"furniture.1.{path}", document)
    assert label == f"第 2 件家具（沙發，座席）・{expected}"
    assert_chinese_lines([label])


@pytest.mark.parametrize("kind", [None, "unknown"])
def test_unreadable_kind_keeps_both_placement_branches(kind: object) -> None:
    document = {"furniture": [{"furniture_id": "未定", "kind": kind}]}
    written = (SchemeProblem("furniture.0.placement.ListenerPlacement.forward_m", "必填"),
               SchemeProblem("furniture.0.placement.RoomPlacement.bottom_center_m", "Field required"))
    problems = plain_problems(written, document)
    assert [row["text"] for row in problems] == [
        "第 1 件家具（種類未明，未定）・前方：空著沒填",
        "第 1 件家具（種類未明，未定）・底面中心：缺這一格"]
    assert {path for row in problems for path in cast(list[str], row["paths"])} == {p.path for p in written}
    assert_chinese_lines([cast(str, row["text"]) for row in problems])


def test_furniture_ids_are_preserved_in_parentheses_even_with_dots() -> None:
    document = furniture.document(furniture.relative_item(furniture_id="seat.with.dots", width_m=None))
    written = validate_scheme(document, capabilities=furniture.CAPABILITIES, directivity=DIRECTIVITY)
    problems = plain_problems(written, document)
    assert [row["text"] for row in problems] == ["第 1 件家具（沙發，seat.with.dots）・寬：空著沒填"]
    # 使用者代號是明定保留的名字；只扣這個完整括號後查英文，不能扣其他訊息。
    assert_chinese_lines([cast(str, row["text"]).replace("（沙發，seat.with.dots）", "") for row in problems])


def test_height_display_rounds_numbers_but_validation_details_stay_exact() -> None:
    document = case_document("height")
    written = validate_scheme(document, capabilities=furniture.CAPABILITIES, directivity=DIRECTIVITY)
    assert [p.message for p in written] == [
        "喇叭 left 的高度 1.0 m 跟擺法推出值不同：桌面頂 0.73 m＋聲學中心離箱底 0.205 m＝0.9349999999999999 m"]
    problems = plain_problems(written, document)
    assert problems[0]["details"] == [str(written[0])]
    assert_chinese_lines([cast(str, p["text"]) for p in problems])


def test_height_display_uses_at_most_four_decimal_places() -> None:
    written = SchemeProblem("speakers.left.z", "喇叭 left 的高度 1.23456789 m 跟擺法推出值不同："
                            "桌面頂 0.7300000000000001 m＋聲學中心離箱底 0.205 m＝0.9349999999999999 m")
    problems = plain_problems((written,), {})
    assert [row["text"] for row in problems] == [
        "左聲道喇叭 z 座標：高度 1.2346 m 跟擺法推出值不同：桌面頂 0.73 m＋聲學中心離箱底 0.205 m＝0.935 m"]
    assert problems[0]["details"] == [str(written)]
    assert_chinese_lines([cast(str, row["text"]) for row in problems])


@pytest.mark.parametrize("kind,branch,title", [
    ("sofa", "ListenerPlacement", "沙發"), ("chair", "ListenerPlacement", "座椅"),
    ("coffee_table", "ListenerPlacement", "茶几"), ("desk", "ListenerPlacement", "書桌"),
    ("ceiling_cloud", "RoomPlacement", "天雲"),
])
def test_placement_filter_uses_kind_and_preserves_other_errors(kind: str, branch: str, title: str) -> None:
    document = {"furniture": [{"furniture_id": "物件", "kind": kind}]}
    written = (SchemeProblem("furniture.0.placement.ListenerPlacement.forward_m", "必填"),
               SchemeProblem("furniture.0.placement.RoomPlacement.bottom_center_m.2", "必填"),
               SchemeProblem("furniture.0.width_m", "必填"))
    problems = plain_problems(written, document)
    labels = [f"第 1 件家具（{title}，物件）・{field}" for field in
              (("底面中心 z", "寬") if kind == "ceiling_cloud" else ("前方", "寬"))]
    assert [row["fields"] for row in problems] == [labels]
    assert set(cast(list[str], problems[0]["paths"])) == {
        f"furniture.0.placement.{branch}." + ("bottom_center_m.2" if kind == "ceiling_cloud" else "forward_m"),
        "furniture.0.width_m"}
    assert_chinese_lines([cast(str, row["text"]) for row in problems])


@pytest.mark.parametrize("case,message", [
    ("kind_typo", "只能選「沙發」或「座椅」或「茶几」或「書桌」或「天雲」"),
    ("blank_id", "家具代號不可空白"),
    ("table_grounded", "懸空家具的底面必須高於地板並超過接觸界線"),
])
def test_furniture_codes_and_suspended_bottom_are_chinese(tmp_path: Path, case: str, message: str) -> None:
    item = furniture.relative_item(furniture_id="座席")
    if case == "kind_typo":
        item["kind"] = "typo"
    elif case == "blank_id":
        item["furniture_id"] = " "
    else:
        item.update(kind="desk", material="wood", height_m=0.03)
    with _client(tmp_path) as client:
        problems = client.post("/api/validate", json=furniture.document(item)).json()["problems"]
    assert [row["message"] for row in problems] == [message]
    assert_chinese_lines([row["text"] for row in problems])
