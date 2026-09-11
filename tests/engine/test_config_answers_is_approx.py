"""裁判自己的控制組：``_config_answers.is_approx``（怎麼比「新家的值」與「donor 的凍結值」）。

**這一支在守什麼（總驗收的修 2）。** `is_approx` 是**每一條凍結值比對**必經的那一關
（cut1 11 支 ＋ cut2 9 支，20 支模組、上百個符號全靠它）。但它自己原本一條考卷都沒有：
把它改成 `return True`、或把 int/float 混在一起、或加一個容差，**整套考卷照樣全綠，
而那 20 支模組的比對全部變成假綠**——「只會回綠的檢查」長在裁判自己身上，是這個 repo
的頭號死因。

**這一支把它的行為逐條寫出來並釘住**（左右兩邊各一組，正反都有）：

1. 真的不同的值 → **不**近似（含 `0.1+0.2` vs `0.3`：這一位就是「沒有容差」那一條）；
2. 真的相同的浮點（含 `float.hex()` 還原出來的那種）→ 近似；
3. `nan`／`inf` 的處理（`nan` 只跟自己近似）；
4. 容器逐**元素**比、不是整體 `==`（tuple 與 list 算同一種，dict 的鍵集合要一樣）；
5. `bool` 與數字**不是**同一種（嚴的那一邊，理由寫在 `is_approx` 的 docstring）；
   `1` 與 `1.0` **是**同一種（同一個數字的兩種寫法）。
"""
from __future__ import annotations

import pytest

from tests.engine._config_answers import is_approx


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (1.0, 2.0),
        (1.0, 1.0000001),  # 這一條就是「沒有容差」：`pytest.approx` 會說 True，這裡必須 False
        (0.1 + 0.2, 0.3),  # 同一個道理，逐位元比
        (float("nan"), 1.0),
        (float("inf"), float("-inf")),
        ("a", "b"),
        (None, 0.0),
    ],
)
def test_different_values_are_not_approx(actual: object, expected: object) -> None:
    """真的不同的值 → **不**近似（裁判紅得起來；控制組的左邊）。"""
    assert is_approx(actual, expected) is False


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (1.0, 1.0),
        (0.30000000000000004, 0.1 + 0.2),
        (float.fromhex("0x1.0e00000000000p+7"), 135.0),  # 答案檔存 hex，還原回來的同一個浮點
        (float("nan"), float("nan")),
        (float("inf"), float("inf")),
        ("a", "a"),
        (None, None),
        (7, 7),
    ],
)
def test_same_values_are_approx(actual: object, expected: object) -> None:
    """真的相同的值 → 近似（控制組的右邊；少了這一邊，`return False` 也會過）。"""
    assert is_approx(actual, expected) is True


@pytest.mark.parametrize(
    ("actual", "expected", "same"),
    [
        (True, True, True),
        (False, False, True),
        (True, 1, False),  # 布林與整數不是同一種東西（`is_approx` docstring 選了嚴的那邊）
        (False, 0, False),
        (True, 1.0, False),
        (1, 1.0, True),  # 同一個數字的兩種寫法算同一種
        (6, 6.0, True),
        (6.5, 6, False),
    ],
)
def test_bool_int_and_float_boundaries_are_pinned(
    actual: object, expected: object, same: bool
) -> None:
    """`bool`／`int`／`float` 的界線：選定了就釘住（不是放著讓它漂）。"""
    assert is_approx(actual, expected) is same


def test_containers_are_compared_element_by_element() -> None:
    """容器逐元素比——不是比整體相等（整體 `==` 會讓「同一個物件」與「同樣內容」混在一起）。

    左右都給：內容一樣的（跨 tuple／list）要近似；**只有一個元素不同**的要紅。
    """
    assert is_approx([1.0, 2.0], [1.0, 2.0]) is True
    assert is_approx([1.0, 2.0], (1.0, 2.0)) is True  # tuple 與 list 算同一種
    assert is_approx((1.0, (2.0, 3.0)), [1.0, [2.0, 3.0]]) is True
    assert is_approx([1.0, 2.0], [1.0, 3.0]) is False
    assert is_approx([1.0, 2.0], [1.0, 2.0, 3.0]) is False
    assert is_approx({"a": [1.0, {"b": 2.0}]}, {"a": [1.0, {"b": 2.0}]}) is True
    assert is_approx({"a": [1.0, {"b": 2.0}]}, {"a": [1.0, {"b": 2.5}]}) is False
    assert is_approx({"a": 1.0}, {"a": 1.0, "b": 2.0}) is False  # 鍵集合要一樣
    assert is_approx({"a": 1.0, "b": 2.0}, {"a": 1.0, "c": 2.0}) is False


def test_this_control_group_bites_a_broken_judge() -> None:
    """這一支自己也要紅得起來：把裁判換成「永遠 True」，上面那幾條的左邊當場掛。

    這一段不是重複上面的斷言——它是**這一份考卷自己的牙**：沒有它，一支「全部用
    `is_approx` 但 `is_approx` 被改壞」的考卷會整批假綠，而這裡會先紅。
    """
    def always_true(actual: object, expected: object) -> bool:
        """刻意的壞裁判：不管餵什麼都說「一樣」（用來證明上面那些斷言真的咬得住）。"""
        return True

    assert always_true(1.0, 2.0) is True  # 壞裁判說「一樣」
    assert is_approx(1.0, 2.0) is False  # 真的裁判說「不一樣」——這就是這一支在守的那條線
