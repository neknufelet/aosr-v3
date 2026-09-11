"""新家最底層 :mod:`aosr.runtime` 的測試：它寫下去的到底是哪幾格、寫成什麼樣子。

刻意只用一個普通的 dict 當環境：測試不准改真的環境（規矩卡
``tests-isolated-from-real-env``），而這支函式本來就收得下「要寫進哪裡」。
斷言一律逐項具名比對，不比數量（規矩卡 ``assertions-not-pinned-to-counts``）。
"""
from __future__ import annotations

import pytest

from aosr import runtime


def test_writes_exactly_the_two_registered_names() -> None:
    """寫下去的就是這個模組登記的那兩格，一格不多一格不少。"""
    env: dict[str, str] = {}
    written = runtime.configure_jax(platforms="cpu", enable_x64=True, env=env)
    assert set(written) == {runtime.PLATFORMS_VAR, runtime.ENABLE_X64_VAR}
    assert set(env) == set(written)
    assert env == written


def test_platform_text_goes_in_verbatim_after_trimming() -> None:
    """平台那一格照交進來的樣子寫下去，只去掉前後空白。"""
    env: dict[str, str] = {}
    runtime.configure_jax(platforms="  cpu,cuda  ", enable_x64=False, env=env)
    assert env[runtime.PLATFORMS_VAR] == "cpu,cuda"


def test_precision_flag_is_written_as_text_both_ways() -> None:
    """精度那一格是字串不是布林：環境變數只吃字串，兩個方向都要看得懂。"""
    on: dict[str, str] = {}
    off: dict[str, str] = {}
    runtime.configure_jax(platforms="cpu", enable_x64=True, env=on)
    runtime.configure_jax(platforms="cpu", enable_x64=False, env=off)
    assert on[runtime.ENABLE_X64_VAR] == runtime.TRUE_TEXT
    assert off[runtime.ENABLE_X64_VAR] == runtime.FALSE_TEXT


def test_existing_keys_in_that_environment_are_left_alone() -> None:
    """交進來的環境裡原本有的別的格子不准被動到。"""
    env = {"UNRELATED": "keep me"}
    runtime.configure_jax(platforms="cpu", enable_x64=True, env=env)
    assert env["UNRELATED"] == "keep me"


def test_empty_platform_is_refused_and_nothing_is_written() -> None:
    """空的平台當場炸，而且一格都不准先寫下去——半套的設定比沒設定更難查。"""
    env: dict[str, str] = {}
    with pytest.raises(ValueError):
        runtime.configure_jax(platforms="   ", enable_x64=True, env=env)
    assert env == {}
