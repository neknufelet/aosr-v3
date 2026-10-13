"""第六步動刀前由乾淨主線 2a2c060c8386680b2799a75b9af53be213c15756 實錄。

原指令：UV_CACHE_DIR=<本次空白暫存目錄> uv run --no-sync python -B - <<'PY'
由 test_reflection_window._inputs 的參考房、小房各建窗 0.015、0.02、1.0 秒，
再由 test_report_source_model._document(_analytic_input()) 建解析聲源窗 0.05 秒；
各場景另建篩查，對 model_dump_json().encode() 算 hashlib.sha256().hexdigest()。
以下 control_values 保留原 stdin 程式的同一組場景、頻率與散射，可重播：
uv run --no-sync python -B -c 'from tests.engine.test_furniture_window_control import control_values; print(control_values())'
第五步在 HEAD 5339182ca3fd1831a684a19ab5c03462eb5b7e1f 用工作區 785-tolerance-step5/evidence（repo 外）的錄製腳本重錄，整份舊雜湊同跑核過後拿掉。
現在 control_values 交產品原文；_furniture_window_answers.py 存原文骨架與逐格 hex。
所有窗的 relative_direct_energy 經 amplitude.path_pressure 的 cmath.exp 及 report_path_table._path_row 次方；
analytic_window 另經 source_directivity._values／two_parameter_pressure_factor 的 numpy.power／expm1。
direction_angles 經 report_path_table._direction 的 atan2／hypot；departure_off_axis_deg 經 off_axis_degrees 的 asin。
以上逐格①級、絕對 0。篩檢 face_retained_energy／round_trip_retained_energy 經 reflection_screen._retained_row
複數取模與次方，沒有正確捨入保證，保守當有關，用①級、絕對 0；不宣稱複數運算都保證正確捨入。
source_m／receiver_m、frequencies_hz、scattering_coefficient、window_s 是輸入；
direct_delay_s、next_uncomputed_earliest_relative_s、next_order_earliest_delay_s、round_trip_delay_s、
delay_s／distance_m／direction_vector 只經幾何四則／sqrt，全部留骨架逐字。
窗內外、覆蓋／數值驗證狀態與所有字串照舊逐字；本輸出沒有另列篩檢原因欄，不捏造欄位。
window_margin 核所有已枚舉路徑（含窗外及主報表階數），不是只看窗內 rows。
重錄已枚舉路徑的最小窗界距離 1.5548563424411938e-5 秒（small_0.015）；各窗斷言大於假設①級時間尺一千倍。
補到支援上限仍 not_provable 的 1 秒與解析窗，不能證明未枚舉更高階的窗界距離；已枚舉控制輸入沒有壓界。
這題證明骨架逐字與機器相關浮點在①級內，不再證明跨處理器逐位。
"""
from __future__ import annotations

import json
from typing import cast

import pytest

from aosr.physics import report_io
from aosr.physics.reflection_screen import build_reflection_screen
from aosr.physics.reflection_window import build_reflection_window
from aosr.physics.report_io import solver_inputs as _flow_inputs
from aosr.physics.room_paths import SUPPORTED_MAX_ORDER, image_source_paths
from tests.engine._cross_cpu_tolerance import GEOMETRIC_REL
from tests.engine._furniture_window_answers import ANSWERS
from tests.engine._json_control import assert_json_control
from tests.engine import _directivity, test_reflection_window as window_cases
from tests.engine import test_report_source_model as source_cases


def control_values() -> dict[str, str]:
    values = {}
    for name, room in (("reference", window_cases._REFERENCE), ("small", window_cases._SMALL)):
        inputs = window_cases._inputs(room)
        screen = build_reflection_screen(inputs, window_cases._FREQUENCIES, materials_by_wall=_flow_inputs(inputs).materials_by_wall)
        values[f"{name}_screen"] = screen.model_dump_json()
        for seconds in (0.015, 0.02, 1.0):
            window = window_cases._window(inputs, seconds)
            values[f"{name}_{seconds}"] = window.model_dump_json()
    inputs = report_io.load_input_document(
        source_cases._document(source_cases._analytic_input()), source_cases._table(), _directivity.DIRECTIVITY)
    window = build_reflection_window(inputs, frequencies_hz=(1000.0,), scattering_coefficient=(0.0,), window_s=0.05, materials_by_wall=_flow_inputs(inputs).materials_by_wall)
    screen = build_reflection_screen(inputs, (1000.0,), materials_by_wall=_flow_inputs(inputs).materials_by_wall)
    values["analytic_window"] = window.model_dump_json()
    values["analytic_screen"] = screen.model_dump_json()
    return values


def window_margin(name: str, text: str) -> tuple[float, float, tuple[int, ...]]:
    """最近已枚舉路徑距窗界及假設①級的時間尺；純量時間本身仍在骨架逐字比。"""
    document = json.loads(text)
    if name == "analytic_window":
        inputs = report_io.load_input_document(source_cases._document(source_cases._analytic_input()),
                                              source_cases._table(), _directivity.DIRECTIVITY)
    else:
        inputs = window_cases._inputs(window_cases._REFERENCE if name.startswith("reference") else window_cases._SMALL)
    paths = image_source_paths(inputs.room_m, inputs.source_m, inputs.receiver_m, inputs.sound_speed_m_s,
                               max_order=min(document["computed_order_k"] + 1, SUPPORTED_MAX_ORDER), materials=None)
    direct = document["direct_delay_s"]
    seconds = document["window_s"]
    nearest = min(paths, key=lambda path: abs(path.delay_s - direct - seconds))
    margin = abs(nearest.delay_s - direct - seconds)
    scale = GEOMETRIC_REL * max(nearest.delay_s, direct, seconds)
    return margin, scale, cast(tuple[int, ...], nearest.identity)


@pytest.fixture(scope="module")
def controls() -> dict[str, str]:
    return control_values()


@pytest.mark.parametrize("name", sorted(ANSWERS))
def test_unfurnished_window_and_screen_are_bitwise_main(name: str, controls: dict[str, str]) -> None:
    assert_json_control(controls[name], "window:" + name, ANSWERS[name])
    if not name.endswith("screen"):
        margin, tolerance, _ = window_margin(name, controls[name])
        assert margin > 1000 * tolerance, f"{name}：已枚舉路徑太靠近窗界"
