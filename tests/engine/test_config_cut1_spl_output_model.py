"""``aosr.config.spl_output`` 的 ``SplOutputConfig``。

這一支是**帶走 v2 考卷**的那一筆：``test_m15_p5_2_inc4_absolute_spl`` 整支帶不走
（它 import 了 jax／numpy／physics_constants 與週邊腳本那一包），但那一支裡
**只靠 spl_output 就成立**的斷言照抄過來、一個字沒改邏輯：

* ``test_config_defaults_are_relative``：預設是相對值、``l_ref_db == 0.0``、``is_relative``；
* ``test_l_ref_db_is_additive_sum``：``(88.0, -6.0) -> 82.0``；
* ``test_config_rejects_nonfinite``：``nan``／``inf``／``-inf`` 建構時就丟
  ``ValidationError``（pydantic 預設會收下來，然後毒死每一個 SPL）；
* ``test_flat_equal_guard_rejects_unequal_per_source_sensitivity``：相等（在容差內）收下、
  不相等丟 ``ValueError``。

另外，case 表宣告的每一筆建構輸入（含兩個屬性）也逐筆跟 donor 比；判定與殘餘風險見
``tests/engine/test_config_cut1_table``（case 表住 ``blueprint/config_cut1_cases.py``）。
"""
from __future__ import annotations

from typing import TypedDict, cast

import pytest
from pydantic import ValidationError

import aosr.config.spl_output as spl_output

from tests.engine._config_answers import (
    case_args,
    expected_block,
    is_approx,
    probe_block,
    probe_by_id,
    probe_ids,
    probe_raised,
    probe_value,
    validate_message,
)


class ConfigArgs(TypedDict, total=False):
    """``SplOutputConfig`` 的建構參數（兩個欄位都有預設）。"""

    sensitivity_db: tuple[float, ...]
    playback_level_db: float


def test_defaults_are_relative() -> None:
    """預設值就是「沒有絕對參考」（donor 的預設是 0.0，不是編出來的 88 dB）。"""
    config = spl_output.SplOutputConfig()
    assert config.sensitivity_db == (spl_output.DEFAULT_SENSITIVITY_DB,)
    assert config.playback_level_db == spl_output.DEFAULT_PLAYBACK_LEVEL_DB
    assert config.l_ref_db == 0.0
    assert config.is_relative is True


def test_l_ref_db_is_additive_sum() -> None:
    """``l_ref`` 是靈敏度加播放位準（兩個都在 dB，所以是相加）。"""
    config = spl_output.SplOutputConfig(sensitivity_db=(88.0,), playback_level_db=-6.0)
    assert config.l_ref_db == 82.0
    assert config.is_relative is False


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_config_rejects_nonfinite(bad: float) -> None:
    """非有限的靈敏度／播放位準在建構時就要擋下來（``allow_inf_nan=False``）。"""
    with pytest.raises(ValidationError):
        spl_output.SplOutputConfig(sensitivity_db=(bad,))
    with pytest.raises(ValidationError):
        spl_output.SplOutputConfig(playback_level_db=bad)


def test_flat_equal_guard_rejects_unequal_per_source_sensitivity() -> None:
    """容差內的多來源收下、差太多的擋下來（輸出層的加法只對共同純量成立）。"""
    near = spl_output.SplOutputConfig(
        sensitivity_db=(85.0, 85.0 + spl_output.FLAT_EQUAL_TOL_DB / 2.0)
    )
    assert near.l_ref_db == pytest.approx(85.0, abs=spl_output.FLAT_EQUAL_TOL_DB)
    with pytest.raises(ValueError, match="physics-layer"):
        _ = spl_output.SplOutputConfig(sensitivity_db=(85.0, 88.0)).l_ref_db


def _check_attribute(config: spl_output.SplOutputConfig, expected: dict[str, object], key: str) -> None:
    """``l_ref_db``／``is_relative`` 這兩格：donor 說有值就比、說炸掉就要真的炸。"""
    own = probe_block(expected, key).get("raised")
    if own is not None:
        message = own["message"] if isinstance(own, dict) else None
        with pytest.raises(ValueError) as caught:
            getattr(config, key)
        assert str(caught.value) == message, f"{key} 的守門訊息跟 donor 不一樣"
        return
    assert is_approx(getattr(config, key), probe_value(expected, key)), f"{key} 跟 donor 不一樣"


def test_config_matches_donor_on_every_declared_case() -> None:
    """case 表宣告的每一筆建構輸入（含兩個屬性）都跟 donor 一樣。"""
    ids = sorted(probe_ids("spl_output"))
    assert ids, "答案檔裡沒有 SplOutputConfig 的 case——這一支就沒有對象"
    for case_id in ids:
        case = probe_by_id(case_id)
        args = cast(ConfigArgs, case_args(case))
        expected = expected_block(case)
        raised = probe_raised(expected)
        if raised is not None:
            message = raised.get("message")
            with pytest.raises(ValidationError) as caught:
                spl_output.SplOutputConfig(**args)
            assert validate_message(str(caught.value), str(message)), (
                f"pydantic 擋下來的原因跟 donor 不一樣：{case_id}"
            )
            continue
        config = spl_output.SplOutputConfig(**args)
        fields = probe_value(expected)
        actual = {
            "sensitivity_db": list(config.sensitivity_db),
            "playback_level_db": config.playback_level_db,
        }
        assert is_approx(actual, fields), f"欄位跟 donor 不一樣：{case_id}"
        _check_attribute(config, expected, "l_ref_db")
        _check_attribute(config, expected, "is_relative")
