"""``aosr.config.early_reflection.EarlyReflectionConfig``（這一刀**新蓋**的一支）。

這一個型別是從上一代 zoning 那一層**沉下來**的（上一代第 2 層的 perceptual 那一支反過來去拿它，
那條線分層卡一立就紅）。它的值沒有變——三個欄位名與種類跟上一代一樣——但它在這一刀的
新家是**新蓋的檔**，所以裁判不是「跟 donor 的型別比」，是把它自己的契約釘住：

1. 三個欄位叫什麼、是什麼（改了名字或種類，餵資料的那一天才會炸）；
2. **零預設**（上一代的註解寫「有預設就等於 15／10 有兩個家」）；
3. frozen（改不動）與 extra 不准多（多一個鍵就是有人在塞第二份真相）。

它蓋不到的是「這三個數字的值」——那在 ``perceptual.toml`` 裡，是第 2 刀的事。
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from aosr.config.early_reflection import EarlyReflectionConfig


def test_field_names_and_kinds_are_the_ones_the_toml_uses() -> None:
    """三個欄位名與種類（TOML 那個區塊餵得進來的形狀）。"""
    config = EarlyReflectionConfig(
        window_ms=15.0,
        required_attenuation_db=10.0,
        check_band_hz=(200.0, 8000.0),
    )
    assert config.window_ms == 15.0
    assert config.required_attenuation_db == 10.0
    assert config.check_band_hz == (200.0, 8000.0)


def test_every_field_is_required_with_no_default() -> None:
    """三個欄位都是必填、沒有預設（有預設就等於那兩個數字有兩個家）。"""
    for name in ("window_ms", "required_attenuation_db", "check_band_hz"):
        field = EarlyReflectionConfig.model_fields[name]
        assert field.is_required(), f"{name} 竟然有預設——那等於那個數字有兩個家"
    with pytest.raises(ValidationError):
        EarlyReflectionConfig(window_ms=15.0)  # type: ignore[call-arg]  # expires=2026-12-08 reason=這一條要驗的就是「少給必填欄位會炸」，所以刻意少給
    with pytest.raises(ValidationError):
        EarlyReflectionConfig(required_attenuation_db=10.0)  # type: ignore[call-arg]  # expires=2026-12-08 reason=同上
    with pytest.raises(ValidationError):
        EarlyReflectionConfig(check_band_hz=(200.0, 8000.0))  # type: ignore[call-arg]  # expires=2026-12-08 reason=同上


def test_field_annotations_are_the_declared_types() -> None:
    """三個欄位的**型別**（不是只驗名字）——`float` 改成 `int` 要在這裡紅（總驗收的修 4）。

    上面那一條驗的是「名字在、有值」，餵得進去；這一條驗的是**欄位宣告的型別**：
    ``window_ms`` 與 ``required_attenuation_db`` 是 ``float``，``check_band_hz`` 是
    ``tuple[float, float]``。把它們改成 ``int`` 或 ``list[float]`` 照樣餵得進去
    （pydantic 會轉），但那樣契約就換了一種——所以用 ``model_fields[...].annotation``
    把現在的事實釘住。

    （這兩個計量一個是毫秒、一個是 dB，都是連續量；上一代與 TOML 都寫浮點。）
    """
    assert EarlyReflectionConfig.model_fields["window_ms"].annotation is float
    assert EarlyReflectionConfig.model_fields["required_attenuation_db"].annotation is float
    assert EarlyReflectionConfig.model_fields["check_band_hz"].annotation == tuple[float, float]


def test_config_is_frozen() -> None:
    """建好之後改不動（frozen）。"""
    config = EarlyReflectionConfig(
        window_ms=15.0, required_attenuation_db=10.0, check_band_hz=(200.0, 8000.0)
    )
    with pytest.raises(ValidationError):
        setattr(config, "window_ms", 20.0)  # noqa: B010  # expires=2026-12-08 reason=這一條要驗的就是「賦值會炸」，而靜態賦值在嚴格模式下本來就不合法；setattr 才是「執行期真的去改」那個動作


def _with_extra_key() -> EarlyReflectionConfig:
    """建一個多帶一個鍵的（那個鍵逐字寫出來，靜態檢查看得見）。"""
    return EarlyReflectionConfig(
        window_ms=15.0,
        required_attenuation_db=10.0,
        check_band_hz=(200.0, 8000.0),
        extra_key=1,  # type: ignore[call-arg]  # expires=2026-12-08 reason=這一條要驗的就是「多一個鍵會炸」，所以那個鍵是刻意的
    )


def test_extra_keys_are_refused() -> None:
    """多一個鍵就當場炸（extra 不准多）。"""
    with pytest.raises(ValidationError):
        _with_extra_key()
