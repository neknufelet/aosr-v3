"""第六步動刀前由乾淨主線 2a2c060c8386680b2799a75b9af53be213c15756 實錄。

原指令：UV_CACHE_DIR=<本次空白暫存目錄> uv run --no-sync python -B - <<'PY'
由 test_reflection_window._inputs 的參考房、小房各建窗 0.015、0.02、1.0 秒，
再由 test_report_source_model._document(_analytic_input()) 建解析聲源窗 0.05 秒；
各場景另建篩查，對 model_dump_json().encode() 算 hashlib.sha256().hexdigest()。
以下 control_values 保留原 stdin 程式的同一組場景、頻率與散射，可重播：
uv run --no-sync python -B -c 'from tests.engine.test_furniture_window_control import control_values; print(control_values())'
"""
from __future__ import annotations

import hashlib

import pytest

from aosr.physics import report_io
from aosr.physics.reflection_screen import build_reflection_screen
from aosr.physics.reflection_window import build_reflection_window
from tests.engine import _directivity, test_reflection_window as window_cases
from tests.engine import test_report_source_model as source_cases


ANSWERS = {
    "reference_screen": "76c0aba076d8bc37d795cb74272a021a5ea91b72da9621efe92012132cad3ddf",
    "reference_0.015": "8bd1a9f9d437c976ad3ef78c22f61c219331d17bcb9e63ce2519d31b96bcc2d5",
    "reference_0.02": "cedae797c0e6a064c72bc2f13204ddbf73ed19fbc9ab1f151e3a83ccd5f0249b",
    "reference_1.0": "141f3f002df8b202027184ce3e2c6a3612426c9b96614b9e872fa3b78aa229be",
    "small_screen": "7f38c43e30c72cc8578faad52622007cfaaf7dea70bf84a8e5bfe5e4f71ece01",
    "small_0.015": "eb41189b53833a625331e4eaf540f253c13de0012201e461ea991d8259a18023",
    "small_0.02": "444c8ceceb1c80186d1429ad208cc170d42e44b768dc35a9c25dc957d998cebb",
    "small_1.0": "3942e6736bc8547ff9ef43113cd8b081ff37c5ccd82eade85501cb2bb9d74efb",
    "analytic_window": "910fe562e33a6bc6b2c54690851eb1ce88b0ad30d2d78ffb533364a7451b3bbd",
    "analytic_screen": "ff07c4064a4fa696d7bc63fa8333b276cd02f3d675901863a907aa0e76df6f95",
}


def control_values() -> dict[str, str]:
    values = {}
    for name, room in (("reference", window_cases._REFERENCE), ("small", window_cases._SMALL)):
        inputs = window_cases._inputs(room)
        screen = build_reflection_screen(inputs, window_cases._FREQUENCIES)
        values[f"{name}_screen"] = hashlib.sha256(screen.model_dump_json().encode()).hexdigest()
        for seconds in (0.015, 0.02, 1.0):
            window = window_cases._window(inputs, seconds)
            values[f"{name}_{seconds}"] = hashlib.sha256(window.model_dump_json().encode()).hexdigest()
    inputs = report_io.load_input_document(
        source_cases._document(source_cases._analytic_input()), source_cases._table(), _directivity.DIRECTIVITY)
    window = build_reflection_window(inputs, frequencies_hz=(1000.0,), scattering_coefficient=(0.0,), window_s=0.05)
    screen = build_reflection_screen(inputs, (1000.0,))
    values["analytic_window"] = hashlib.sha256(window.model_dump_json().encode()).hexdigest()
    values["analytic_screen"] = hashlib.sha256(screen.model_dump_json().encode()).hexdigest()
    return values


@pytest.fixture(scope="module")
def controls() -> dict[str, str]:
    return control_values()


@pytest.mark.parametrize("name", sorted(ANSWERS))
def test_unfurnished_window_and_screen_are_bitwise_main(name: str, controls: dict[str, str]) -> None:
    assert controls[name] == ANSWERS[name]
