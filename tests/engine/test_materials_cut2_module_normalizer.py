"""票 #134 第二刀：**模組路徑只在一個地方換得掉**（正反控制）。

這一支守的是獨立審查實測出來的碰撞：正規化原本對整棵結果無差別做字串替換，於是
``material_id`` 這種**資料**如果剛好寫成 ``lib.config.material_loader`` 或
``aosr.materials.material_loader``，兩個**不同**的值會被換成同一個記號
（``normalized_equal=true``），裁判就再也看不見那一格變了。

現在只換一個地方：``raised`` 是 ``TypeError``、訊息又正好是 CPython 那一句
``<類別的完整名字>() argument after ** must be a mapping, not …`` 的開頭那一段。

* **反控制**：同樣那兩個字串當成 ``values`` 裡的資料，換完之後**仍然不同**；
  ``ValidationError`` 的訊息裡帶著同一段路徑（使用者輸入原字照印），一個字都不准動；
  長得不一樣的 ``TypeError`` 也不准動。
* **正控制**：真答案檔裡那三筆 ``TypeError``，donor 那一邊的原字與新家那一邊的原字，
  各自用自己的 mapping 換完之後**對得齊**，而且等於答案檔裡凍著的那一格。

這一支只讀答案檔與正規化那一支，**不載入產品模組**。
"""
from __future__ import annotations

import pytest

from blueprint import materials_cut2_cases as cases
from blueprint import materials_cut2_probe as probe
from tests.engine import _materials_cut2_answers as answers

# 兩個**資料**值，剛好長得像模組路徑（審查那一份 JSON 裡的原始兩筆）。
DONOR_LOOKALIKE: str = "lib.config.material_loader"
ENGINE_LOOKALIKE: str = "aosr.materials.material_loader"

# 這一筆 case 沒有暫存檔，所以暫存根給一個不會出現在任何字串裡的值。
NO_TMP: str = "/tmp/aosr-cut2-does-not-appear"

DONOR_PATHS: dict[str, str] = {
    name: cases.lookup_name(name, "donor") for name in [*cases.MODULES, *cases.INGREDIENTS]
}
ENGINE_PATHS: dict[str, str] = {
    name: cases.lookup_name(name, "engine") for name in [*cases.MODULES, *cases.INGREDIENTS]
}

# 那三筆 TypeError 各自是哪一支模組的類別（案表 id → 模組名、類別名）。
STAR_ARGS_CASES: tuple[tuple[str, str, str, str], ...] = (
    ("experiment_schema.load_experiment.empty_file_raises", "experiment_schema", "ExperimentConfig", "NoneType"),
    ("experiment_schema.load_experiment.a_yaml_list_raises", "experiment_schema", "ExperimentConfig", "list"),
    ("material_loader.load_materials.an_empty_yaml_file_raises", "material_loader", "MaterialSpec", "NoneType"),
)


def _star_args_message(module_path: str, class_name: str, got: str) -> str:
    """CPython 那一句原字（``F(**data)`` 但 data 不是一層表）。"""
    return f"{module_path}.{class_name}() argument after ** must be a mapping, not {got}"


def _values_result(text: str) -> dict[str, object]:
    """一筆「跑完了，記了一個字串值」的結果格。"""
    return {"values": {"material_id": {"kind": "str", "v": text}}}


def test_two_different_data_values_stay_different() -> None:
    """反控制①：長得像模組路徑的**資料**，兩邊換完之後仍然不同（這就是那個碰撞）。"""
    donor_side = probe.normalise_result(_values_result(DONOR_LOOKALIKE), NO_TMP, DONOR_PATHS)
    engine_side = probe.normalise_result(_values_result(ENGINE_LOOKALIKE), NO_TMP, ENGINE_PATHS)
    gaps = answers.differences(engine_side, donor_side, "lookalike")
    assert gaps != [], (
        "兩個不同的資料值被換成同一個記號了——正是審查重現的那個碰撞"
        f"（donor 側 {donor_side!r}、新家側 {engine_side!r}）"
    )


@pytest.mark.parametrize("text", [DONOR_LOOKALIKE, ENGINE_LOOKALIKE])
def test_a_data_value_is_returned_word_for_word(text: str) -> None:
    """反控制②：``values`` 裡的字串一個字都不動（連自己那一邊的路徑也不換）。"""
    for paths in (DONOR_PATHS, ENGINE_PATHS):
        assert probe.normalise_result(_values_result(text), NO_TMP, paths) == _values_result(text)


def test_a_validation_error_message_is_not_touched() -> None:
    """反控制③：``ValidationError`` 裡帶著同一段路徑（那是使用者輸入），不准動。"""
    message = (
        "1 validation error for MaterialSpec\nmaterial_id\n  Input should be a valid string "
        f"[type=string_type, input_value='{DONOR_LOOKALIKE}', input_type=str]"
    )
    result: dict[str, object] = {"raised": {"step": 0, "type": "ValidationError", "message": message}}
    assert probe.normalise_result(result, NO_TMP, DONOR_PATHS) == result


def test_a_type_error_with_another_shape_is_not_touched() -> None:
    """反控制④：同樣是 ``TypeError``，句子長得不一樣就不准動。

    這一條擋的是「只看例外種類就換」：那樣的話任何一個 ``TypeError`` 裡出現的路徑字串
    （可能是使用者餵進去的值）都會被悄悄換掉。
    """
    message = f"{DONOR_LOOKALIKE}.MaterialSpec() takes no arguments"
    result: dict[str, object] = {"raised": {"step": 1, "type": "TypeError", "message": message}}
    assert probe.normalise_result(result, NO_TMP, DONOR_PATHS) == result


def test_a_message_that_only_starts_like_that_sentence_is_not_touched() -> None:
    """反控制⑤：**整句**要一字不差才換；後面多接了東西就整條原字奉還。

    比對是整句對齊（正則兩頭都釘死），不是「開頭像就換」。多接的那一段可能是任何東西
    ——包含使用者餵進去的值——所以寧可把那一段會漂的路徑留著，也不要在一條沒見過形狀的
    訊息上動手。真答案檔裡那三筆是**正好**那一句（由下面的正控制逐筆對回凍結值）。
    """
    head = _star_args_message(DONOR_PATHS["material_loader"], "MaterialSpec", "NoneType")
    message = f"{head} ({DONOR_LOOKALIKE})"
    result: dict[str, object] = {"raised": {"step": 0, "type": "TypeError", "message": message}}
    assert probe.normalise_result(result, NO_TMP, DONOR_PATHS) == result


@pytest.mark.parametrize(("case_id", "module", "class_name", "got"), STAR_ARGS_CASES)
def test_the_three_star_args_messages_line_up_across_the_move(
    case_id: str, module: str, class_name: str, got: str
) -> None:
    """正控制：那三筆在 donor 與新家各自的原字，換完之後對得齊，也等於答案檔凍著的那一格。"""
    frozen = answers.as_mapping(answers.result_of(case_id).get("raised"), f"{case_id} 的 raised")
    donor_raw = _star_args_message(DONOR_PATHS[module], class_name, got)
    engine_raw = _star_args_message(ENGINE_PATHS[module], class_name, got)
    donor_side = probe.normalise_result(
        {"raised": {"step": frozen["step"], "type": "TypeError", "message": donor_raw}},
        NO_TMP,
        DONOR_PATHS,
    )
    engine_side = probe.normalise_result(
        {"raised": {"step": frozen["step"], "type": "TypeError", "message": engine_raw}},
        NO_TMP,
        ENGINE_PATHS,
    )
    assert answers.differences(engine_side, donor_side, case_id) == []
    assert answers.differences(engine_side, {"raised": frozen}, f"{case_id}.frozen") == []


def test_the_frozen_answers_only_carry_that_token_in_those_three_places() -> None:
    """答案檔裡 ``<MODULE:…>`` 只出現在那三筆的 ``raised`` 訊息裡，``values`` 一格都沒有。"""
    seen: list[str] = []
    for name in sorted(cases.MODULES):
        for case in cases.cases_for(name):
            result = answers.result_of(case["id"])
            raised = result.get("raised")
            values = result.get("values")
            if probe.TMP_TOKEN and _carries_token(values):
                seen.append(f"values:{case['id']}")
            if _carries_token(raised):
                seen.append(case["id"])
    assert sorted(seen) == sorted(item[0] for item in STAR_ARGS_CASES)


def _carries_token(node: object) -> bool:
    """這一格（或它底下任何一格）有沒有出現模組替身。"""
    if isinstance(node, str):
        return "<MODULE:" in node
    if isinstance(node, dict):
        return any(_carries_token(item) for item in node.values())
    if isinstance(node, list):
        return any(_carries_token(item) for item in node)
    return False
