"""骨架必須保存原字面、順序與字串內的數字；只放行指定欄的浮點末位。"""
import json

import pytest

from tests.engine._cross_cpu_tolerance import GEOMETRIC_REL
from tests.engine._json_control import answer_cells, assert_json_control, float_leaves, snapshot


def test_only_machine_literals_are_masked() -> None:
    left = '{"rows":[{"relative_direct_energy":[1.0],"distance_m":2.0}],"text":"1.0"}'
    same = left.replace('[1.0]', '[1.0000000000000002]')
    assert snapshot(left, "path")[0] == snapshot(same, "path")[0]
    assert snapshot(left, "path")[1] != snapshot(same, "path")[1]
    for changed in (left.replace('2.0', '2.0000000000000004'), left.replace('"1.0"', '"1.1"'),
                    left.replace('1.0', '1', 1), left.replace(':', ': ', 1)):
        assert snapshot(left, "path")[0] != snapshot(changed, "path")[0]


def test_order_and_escaped_strings_stay_in_skeleton() -> None:
    left = '{"rows":[{"relative_direct_energy":[1.0]}],"text":"a\\\"1.0"}'
    swapped = '{"text":"a\\\"1.0","rows":[{"relative_direct_energy":[1.0]}]}'
    assert snapshot(left, "path")[0] != snapshot(swapped, "path")[0]


@pytest.mark.parametrize("family,text", [
    ("path", '{"rows":[{"direction_angles":{"azimuth_deg":1.0},"departure_off_axis_deg":2.0}]}'),
    ("window:reference_0.02", '{"rows":[{"relative_direct_energy":[1.0]}]}'),
    ("window:small_screen", '{"pairs":[{"face_retained_energy":[[1.0]],"round_trip_retained_energy":[2.0]}]}'),
    ("cli", '{"points":[{"direct_energy":1.0,"reflected_energy":2.0,"late_energy":3.0,"fem_energy":4.0,"scattering":5.0}]}'),
    ("cli", '{"bands":[{"fem_energy":1.0,"late_energy":2.0,"interference_energy":3.0}]}'),
    ("cli", '{"path_table":{"scattering_coefficient":[1.0],"rows":[{"relative_direct_energy":[2.0],"direction_angles":{"elevation_deg":3.0}}]}}'),
])
def test_transcendental_results_leave_skeleton(family: str, text: str) -> None:
    """漏分類 math／cmath、複數取模或次方衍生欄時，末位漂移會錯誤改動骨架。"""
    changed = text.replace('1.0', '1.0000000000000002')
    assert snapshot(text, family)[0] == snapshot(changed, family)[0]
    assert snapshot(text, family)[1] != snapshot(changed, family)[1]
    assert snapshot(text, family)[1] == list(float_leaves(json.loads(text)))


def test_cli_decay_from_basic_arithmetic_stays_bitwise() -> None:
    """快速衰減替身只用四則；不能連同 sin／cos 替身一起放寬。"""
    text = '{"bands":[{"t20_s":1.0,"t30_s":2.0}]}'
    assert snapshot(text, "cli")[0] != snapshot(text.replace('1.0', '1.0000000000000002'), "cli")[0]


def test_interference_uses_positive_sibling_answers() -> None:
    """把會相消的干涉只比相對值、或把絕對尺改由算出欄求，都不能通過此控制組。"""
    text = '{"points":[{"direct_energy":4.0,"reflected_energy":5.0,"interference_energy":1e-8}]}'
    skeleton, _ = snapshot(text, "cli")
    answer = (skeleton, (4.0.hex(), 5.0.hex(), (1e-8).hex()))
    _, values = snapshot(text.replace('4.0', '4.00000000000004'), "cli")
    assert answer_cells(values, "cli", answer)[-1].abs_ == 9.0 * GEOMETRIC_REL
    assert_json_control(text.replace('1e-8', '1.00001e-8'), "cli", answer)
    with pytest.raises(AssertionError, match="超出跨處理器容差"):
        assert_json_control(text.replace('1e-8', '1.001e-8'), "cli", answer)


def test_skeleton_change_is_red_even_when_every_cell_is_equal() -> None:
    """骨架只在 snapshot 算出來不夠：裁判要真的比它，字串、整數或鍵順序變了，逐格全等也紅。"""
    text = '{"rows":[{"relative_direct_energy":[1.0],"order":1}],"text":"a"}'
    skeleton, values = snapshot(text, "path")
    answer = (skeleton, tuple(value.hex() for _, value in values))
    assert_json_control(text, "path", answer)
    for changed in (text.replace('"a"', '"b"'), text.replace('"order":1', '"order":2'),
                    '{"text":"a","rows":[{"relative_direct_energy":[1.0],"order":1}]}'):
        with pytest.raises(AssertionError, match="原文骨架不同"):
            assert_json_control(changed, "path", answer)
