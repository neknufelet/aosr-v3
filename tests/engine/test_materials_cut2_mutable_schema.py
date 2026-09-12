"""``FrequencyAxisSpec`` 建構之後還改得動，改壞了由下游那一步自己炸。

模型沒有 frozen、也沒有 ``validate_assignment``：驗證器只在建構時跑一次。把型別收窄給
mypy 看**不准**順手多一道執行期守門——那會把「下游炸的 ``TypeError``」換成「這一層炸的
``ValueError``」，也會把上一代收不下的輸入（字串）悄悄轉成收得下。

清成 ``None`` 那三筆的訊息是穩定的，所以住在版本化答案檔裡逐字比。**字串那一筆不進答案檔**：
JAX 的訊息帶著函式的記憶體位址，每一跑都不一樣；這裡改成釘「例外的種類 ＋ 穩定片段」。
"""
from __future__ import annotations

import pytest

from aosr.materials.experiment_schema import FrequencyAxisSpec


def _linear_spec() -> FrequencyAxisSpec:
    """一份合法的 linear_hz 規格（下面各自改壞其中一格）。"""
    return FrequencyAxisSpec(mode="linear_hz", f_min_hz=100.0, f_max_hz=500.0, n_points=3)


def test_a_string_bound_still_reaches_jax_and_raises_type_error() -> None:
    """``f_min_hz`` 被改成字串：由 JAX 炸 ``TypeError``，**不是**這一層先轉成數字。

    上一代在這裡炸（JAX 不收字串）。若這一層先 ``float("100")``，這一筆會安靜地算出
    ``[100, 300, 500]``——一個上一代拒收的輸入變成收得下，那是放寬契約。
    """
    spec = _linear_spec()
    spec.f_min_hz = "100"  # type: ignore[assignment]  # expires=2026-12-08 reason=這一題要餵的就是型別不合的值，模型沒有 validate_assignment 所以塞得進去
    with pytest.raises(TypeError) as caught:
        spec.build()
    message = str(caught.value)
    assert "as an abstract array" in message
    assert "<class 'str'>" in message


@pytest.mark.parametrize("field", ["f_min_hz", "f_max_hz", "n_points"])
def test_clearing_a_required_field_raises_type_error_not_value_error(field: str) -> None:
    """清成 ``None`` 之後 build 炸的是 ``TypeError``（下游那一步炸的），不是 ``ValueError``。

    ``ValueError`` 是驗證器的語彙：在 build 裡補一道 ``None`` 檢查會讓呼叫端分不出「規格
    當初就不合法」與「規格被改壞了」。逐字比對那三筆住在答案檔裡，這一條只釘種類。
    """
    spec = _linear_spec()
    setattr(spec, field, None)
    with pytest.raises(TypeError):
        spec.build()
