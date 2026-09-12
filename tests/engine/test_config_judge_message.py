"""裁判自己的控制組（第二支）：``validate_message`` 的比對、以及答案檔／case 表的 id 唯一性。

**這一支在補什麼洞（票 #161 與 #163；兩張票同一個病：負責判斷「新家跟上一代一不一樣」的
工具自己會吞錯）。**

1. **#161：訊息裁判比錯欄位。** ``_config_answers._problems()`` 原本
   ``for chunk in message.split("[type=")[1:]`` 再從碎片裡找欄位——碎片裝的是切點**後面**
   那一截，而欄位在切點**之前**；於是每一筆的「欄位」都變成句子最後那行說明網址。實測：
   「左邊那個欄位壞掉」與「右邊那個欄位壞掉」被判成**一樣**。這一支把左右兩邊都釘住
   （只差欄位要紅、只差種類要紅、只差說明網址版本或輸入值容器要綠），並且用「把欄位比對
   拿掉」的舊寫法證明這些斷言真的咬得住。

2. **#163：id 沒有唯一性。** 答案檔同一個 id 塞兩筆、其中一筆期望值改成亂碼，集合相等
   照樣過（實測 244 題全過）；集合看不出「一對一」。這一支拿**真的那一份答案檔**現改
   （複製一筆、拔掉一筆、改掉一筆）餵給裁判，每一組的兩邊都給，並且用「現在那一份」
   當另一邊——控制組要餵得進真的那一份表，不是自己造一個玩具形狀自己說自己對。
"""
from __future__ import annotations

import copy
from collections.abc import Callable

import pytest

from tests.engine import _config_answers as answers

# ── 一則真的 pydantic 訊息（拿掉縮排之後的樣子）──────────────────────────────
# 形狀是 `sensitivity_db.0` 那一行、問題那一句（`[type=…]` 在這裡）、再一行說明網址。
_MESSAGE = """1 validation error for SplOutputConfig
sensitivity_db.0
  Input should be a finite number [type=finite_number, input_value=nan, input_type=float]
    For further information visit https://errors.pydantic.dev/2.13/v/finite_number"""


# `_MESSAGE` 裡那一格欄位路徑、以及控制組要換成的另一格。**寫死在這裡是刻意的**：這是夾具
# 訊息，不是被測物。控制組要能咬住「裁判報的欄位對不對」，被換掉的那一行就必須來自訊息
# 本身，不能來自裁判自己——裁判報錯時（#161：它報成說明網址那一行），拿它報的字串去換會
# 換到**另一行**，兩則訊息的 `_problems()` 反而不一樣，控制組就變成自己說自己對（實測：
# 舊裁判下那種寫法照樣綠，見 `_config_answers._problems` 的說明）。欄位由夾具釘住，裁判
# 報錯由 `test_the_judge_reports_the_field_not_the_doc_url` 當場紅。
_MESSAGE_FIELD = "sensitivity_db.0"
_MESSAGE_OTHER_FIELD = "playback_level_db"


def _other_field(message: str) -> str:
    """把訊息裡的欄位路徑換成另一個（其餘逐字不動）——「只差欄位」那一組的右腳。"""
    return message.replace(_MESSAGE_FIELD, _MESSAGE_OTHER_FIELD, 1)


def _other_kind(message: str) -> str:
    """把 ``[type=…]`` 那一格換成另一種（其餘逐字不動）——「只差問題種類」那一組。"""
    return message.replace("type=finite_number", "type=too_short", 1)


def _other_url_version(message: str) -> str:
    """只換說明網址的版本號（`/2.13/` → `/2.14/`）——刻意放掉的那一件，要判**相同**。"""
    return message.replace("/2.13/", "/2.14/", 1)


def _other_container(message: str) -> str:
    """只換輸入值的容器寫法（`nan` → `[nan]`）——刻意放掉的那一件，要判**相同**。"""
    return message.replace("input_value=nan", "input_value=[nan]", 1)


def test_the_judge_reports_the_field_not_the_doc_url() -> None:
    """**這是 #161 的洞本身。** 裁判抓到的「欄位」必須是欄位路徑，不是說明網址那一行。

    兩個方向都有：好的訊息抓到真欄位（＝夾具釘住的那一格）；把 ``[type=`` **前面**那一行
    拿掉（訊息壞了、形狀不是 pydantic 的）時回空字串——不是拿說明網址來墊，也不是猜一個
    看起來像欄位的東西。這一條刻意寫死 ``_MESSAGE_FIELD``：裁判報錯時要當場紅。
    """
    assert answers._problems(_MESSAGE) == [(_MESSAGE_FIELD, "finite_number")]
    truncated = "\n".join(line for line in _MESSAGE.splitlines() if _MESSAGE_FIELD not in line)
    assert answers._problems(truncated) == [("", "finite_number")]


def test_the_judge_catches_a_message_that_differs_only_in_the_field() -> None:
    """只差**欄位**的兩則訊息要判不同（#161 實測「判成一樣」的反面）。

    ``other`` 是拿夾具那一格欄位換掉的（不是拿裁判報的字串換）：換成說明網址那一行的舊
    裁判，兩則的 ``_problems()`` 一模一樣，這一條當場紅——反向控制跑得出來。
    """
    other = _other_field(_MESSAGE)
    assert other != _MESSAGE, "換欄位那一行沒有生效——這一組就沒有對象"
    assert answers.validate_message(_MESSAGE, other) is False


def test_the_judge_catches_a_message_that_differs_only_in_the_problem_kind() -> None:
    """只差**問題種類**的兩則訊息要判不同（控制組：壞掉的裁判在那個方向本來就紅）。"""
    assert answers.validate_message(_MESSAGE, _other_kind(_MESSAGE)) is False


@pytest.mark.parametrize(
    ("name", "shape"),
    [
        ("只差說明網址版本號", _other_url_version),
        ("只差輸入值的容器寫法", _other_container),
    ],
)
def test_the_judge_ignores_the_two_things_that_are_not_the_contract(
    name: str, shape: Callable[[str], str]
) -> None:
    """只差**說明網址版本號**或**輸入值容器寫法**的要判**相同**（刻意放掉的兩件，`name` 是它）。

    這兩件是「庫怎麼印」不是「哪裡壞了」：版本號跟著 pydantic 漂，容器寫法跟著呼叫端漂。
    少了這一條，收緊欄位比對的時候很容易順手把這兩件也變成契約（那是誤咬，比漏抓更糟）。
    """
    changed = shape(_MESSAGE)
    assert changed != _MESSAGE, f"{name} 那一格沒有生效——這一組就沒有對象"
    assert answers.validate_message(_MESSAGE, changed) is True


def test_the_whole_control_group_bites_the_old_broken_judge() -> None:
    """這一支自己也要紅得起來：把裁判換回 #161 那版（先切再從碎片找欄位），上面那條當場掛。

    這一段不是重複上面的斷言——它是**這一支考卷自己的牙**。舊寫法在「只差欄位」那一組
    回 ``True``（那就是票上實測到的假綠），而現在的裁判回 ``False``。
    """
    def broken_validate(actual: str, expected: str) -> bool:
        """#161 之前的判斷：欄位從 ``[type=`` 之後的碎片裡找（於是抓到說明網址那一行）。"""
        def problems(message: str) -> list[tuple[str, str]]:
            out: list[tuple[str, str]] = []
            for chunk in message.split("[type=")[1:]:
                kind = chunk.split(",")[0].split("]")[0].strip()
                head = chunk.strip().splitlines()[-1].strip()
                out.append((head, kind))
            return out

        if actual == expected:
            return True
        return bool(problems(actual)) and problems(actual) == problems(expected)

    other = _other_field(_MESSAGE)
    assert broken_validate(_MESSAGE, other) is True  # 壞裁判說「一樣」——這就是那個洞
    assert answers.validate_message(_MESSAGE, other) is False  # 真的裁判說「不一樣」


# ── #163：答案檔與 case 表的 id 唯一性（每一組都餵真的那一份答案檔）──────────
def _real_modules() -> dict[str, object]:
    """版控裡那一份答案檔的 ``modules``（**不是**玩具形狀）。"""
    return answers._as_mapping(answers.load_answers().get("modules"), "答案檔的 modules")


def _probes(modules: dict[str, object]) -> list[dict[str, object]]:
    """這一份答案檔裡每一筆探針（照檔上的順序），收成「一筆一格」的清單。"""
    out: list[dict[str, object]] = []
    for raw in modules.values():
        module = answers._as_mapping(raw, "答案檔的 modules.<一支>")
        entries = module.get("probes")
        assert isinstance(entries, list), "答案檔的探針不是一串東西"
        for item in entries:
            out.append(answers._as_mapping(item, "答案檔的一筆探針"))
    return out


def _duplicate_first_probe() -> tuple[dict[str, object], str]:
    """**真的那一份**答案檔 ＋ 多複製一筆（同一個 id 兩筆）——#163 的現場。

    回（那份 modules，被複製的 id）。只動回傳的那一份副本，正本不動。
    """
    modules = copy.deepcopy(_real_modules())
    probes = _probes(modules)
    assert probes, "那一份答案檔一筆探針都沒有——這一支就沒有對象"
    first = probes[0]
    case_id = str(first["id"])
    # 同一個 id 那筆住的那一支模組格：照 id 找，不照位置找（換一支模組也算得對）。
    for raw in modules.values():
        module = answers._as_mapping(raw, "答案檔的 modules.<一支>")
        entries = module.get("probes")
        assert isinstance(entries, list), "答案檔的探針不是一串東西"
        if any(isinstance(item, dict) and item.get("id") == case_id for item in entries):
            entries.append(copy.deepcopy(first))
            return modules, case_id
    raise AssertionError(f"那一份答案檔裡找不到 {case_id!r}——這一支就沒有對象")


def test_the_real_answer_file_has_unique_ids() -> None:
    """控制組的右邊：**版控裡那一份**答案檔與 case 表的 id 都唯一（裁判在好的輸入上要綠）。"""
    assert answers.duplicate_answer_case_ids() == []
    assert answers.duplicate_declared_case_ids() == []


def test_a_duplicate_id_in_the_answer_file_is_caught() -> None:
    """控制組的左邊：同一個 id 兩筆要被咬出來（票上實測「244 題全過」的那個現場）。

    第二筆的期望值是不是矛盾的**不影響這一條**：只要是同一個 id 出現兩次就該紅，因為
    比對只會挑到其中一筆，另一筆無聲消失。
    """
    modules, case_id = _duplicate_first_probe()
    found = answers.duplicate_answer_case_ids({"modules": modules})
    assert case_id in found, f"複製一筆同 id 之後裁判說沒有重複：{found}"
    # 對照：原本那一份沒有重複（同一支裁判、同一個入口，只差複製那一筆）。
    assert answers.duplicate_answer_case_ids({"modules": _real_modules()}) == []


def test_a_missing_id_is_caught_by_the_set_comparison() -> None:
    """少一筆要紅：集合相等那一條抓得到（左邊是「少一個具名的 id」，右邊是「沒少」）。"""
    declared = answers.declared_case_ids()
    present = answers.answer_case_ids()
    assert present, "答案檔一筆 id 都沒有——這一支就沒有對象"
    sample = sorted(present)[0]
    assert declared - {sample} != present, "少一個答案檔裡真的有的 id 之後集合還一樣"
    assert declared == present, "現在那一份答案檔就跟 case 表對不上"


def test_an_extra_id_is_caught() -> None:
    """多一筆也要紅：集合多一個**具名的 id** 就紅（右邊是「沒有多」）。"""
    present = answers.answer_case_ids()
    assert present == answers.declared_case_ids()
    assert present | {"spl_output.const.NOT_DECLARED"} != answers.declared_case_ids()


def test_a_duplicate_id_is_invisible_to_the_set_comparison() -> None:
    """同一支裁判的第二個方向：**複製同一個 id 不會讓集合多一筆**——所以唯一性要另一條。

    這一條是 #163 那個洞的形狀本身：集合相等在複製那一筆上完全看不出來（兩個集合一模
    一樣），而 ``duplicate_answer_case_ids`` 看得出來。兩條裁判比的是兩件事。
    """
    modules, _case_id = _duplicate_first_probe()
    before = answers.answer_case_ids({"modules": _real_modules()})
    after = answers.answer_case_ids({"modules": modules})
    assert after == before, "複製一筆同 id 竟然改變了 id 集合——那這一條的前提就錯了"
    assert answers.duplicate_answer_case_ids({"modules": modules}) != []


def _first_valued_probe(modules: dict[str, object]) -> dict[str, object]:
    """答案檔第一筆**成功**的探針（``expected`` 裡有 ``value`` 那一格）。

    ``raised`` 那一種沒有值可以比（它比的是例外訊息），所以這一支照形狀挑一筆有值的。
    """
    for record in _probes(modules):
        expected = record.get("expected")
        if isinstance(expected, dict) and "value" in expected:
            return record
    raise AssertionError("那一份答案檔裡沒有成功的探針——這一支就沒有對象")


def test_a_changed_expected_value_is_caught() -> None:
    """控制組的第三組：期望值那一格改掉時比對要紅（餵真的答案檔那一筆的期望值）。

    兩邊都給：拿那一筆自己的值回來比要綠（裁判在好的輸入上不亂咬），換成一個亂碼值要比
    不綠。比的是同一組輸入，差別只在期望值那一格。
    """
    record = _first_valued_probe(_real_modules())
    expected = answers._as_mapping(record.get("expected"), "答案檔一筆探針的 expected")
    value = answers.probe_value(expected)
    assert answers.is_approx(value, answers.probe_value(expected)) is True
    assert answers.is_approx(value, {"BOGUS": True}) is False
