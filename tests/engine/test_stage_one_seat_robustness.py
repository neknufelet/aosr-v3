"""第一階段座位穩健性考卷（票 #558，老闆第 4 題）：相同曲線差為零；起伏總分相同但峰谷不同時差不為零。

穩健性＝每個周圍點各自對主位逐頻 dB 差的加權均方根，再取周圍點等權平均；共同聲源與音量基準，
不替各位置各自扣斜率或調平；最差方向只提醒、不淘汰。
"""
from __future__ import annotations

import numpy as np
import pytest

from aosr.scoring.stage_one_low_frequency_contract import StageOneSpeakerPayload
from tests.engine._stage_one_low_frequency_curves import (
    AXIS_HZ,
    DIRECTIONS,
    SCORING_RANGE_HZ,
    candidate,
    demo_curve_db,
    evaluate,
    gaussian_db,
    independent_own_line_answer,
    independent_weights,
    measured_payload,
    speaker,
    tilted_db,
)


def _by_direction(payload: StageOneSpeakerPayload) -> dict[str, float]:
    return {item.direction_relative_to_primary: item.difference_rms_db for item in payload.directions}


def test_identical_curves_differ_by_exactly_zero() -> None:
    """六點跟主位逐位相同：每個方向的曲線差、均方根與平均都剛好是 0（0 減 0，不用容差）。"""
    payload = measured_payload(candidate(speaker(demo_curve_db())))

    assert _by_direction(payload) == dict.fromkeys(DIRECTIONS, 0.0)
    assert payload.seat_robustness_rms_db == 0.0
    assert {value for item in payload.directions for _, value in item.difference_curve_db} == {0.0}


def test_same_ripple_score_with_peaks_turned_into_dips_is_not_zero() -> None:
    """主位 A＝自己的直線 ℓ＋殘差，前方 B＝ℓ−殘差（2ℓ−A）：兩點主項相同，峰在 B 變成谷。
    兩個起伏總分相減會是 0（老闆要排除的量法）；逐頻曲線差是 −2·殘差，前方那一點應剛好是 2 倍主項。
    ℓ 用 numpy.polyfit 獨立算（考卷自己的權重），不從評估器讀。"""
    a = demo_curve_db()
    _, slope, intercept = independent_own_line_answer(AXIS_HZ, a, SCORING_RANGE_HZ)
    line = slope * np.log2(AXIS_HZ) + intercept
    b = 2.0 * line - a
    payload = measured_payload(candidate(speaker(a, {"front": b})))
    b_as_primary = measured_payload(candidate(speaker(b)))

    ripple = payload.own_line_ripple_rms_db
    assert ripple > 1.0
    assert b_as_primary.own_line_ripple_rms_db == pytest.approx(ripple, rel=1e-12)
    assert abs(b_as_primary.own_line_ripple_rms_db - ripple) == pytest.approx(0.0, abs=1e-12)
    assert _by_direction(payload)["front"] == pytest.approx(2.0 * ripple, rel=1e-12)
    assert payload.seat_robustness_rms_db == pytest.approx(2.0 * ripple / len(DIRECTIONS), rel=1e-12)
    assert payload.strongest_peak_index is not None and b_as_primary.deepest_dip_index is not None
    assert (
        b_as_primary.features[b_as_primary.deepest_dip_index].center_frequency_hz
        == payload.features[payload.strongest_peak_index].center_frequency_hz
    )


def test_one_seat_three_db_louder_is_three_db_not_levelled_away() -> None:
    """只把前方整條加 3 dB：共同音量基準下前方差 3 dB；各自調平的話會變成 0。"""
    level = demo_curve_db()
    payload = measured_payload(candidate(speaker(level, {"front": level + 3.0})))

    assert _by_direction(payload) == pytest.approx(
        {"front": 3.0, "back": 0.0, "left": 0.0, "right": 0.0, "up": 0.0, "down": 0.0}, abs=1e-12
    )
    assert payload.seat_robustness_rms_db == pytest.approx(3.0 / len(DIRECTIONS), abs=1e-12)


def test_six_seats_are_averaged_with_equal_weight() -> None:
    """六點各差 1、2、3、−1、−2、−3 dB：每點均方根是差的絕對值，穩健性是平均 2.0（取最大會是 3.0）。"""
    level = demo_curve_db()
    offsets = dict(zip(DIRECTIONS, (1.0, 2.0, 3.0, -1.0, -2.0, -3.0), strict=True))
    payload = measured_payload(candidate(speaker(level, {key: level + value for key, value in offsets.items()})))

    assert _by_direction(payload) == pytest.approx({key: abs(value) for key, value in offsets.items()}, abs=1e-12)
    assert payload.seat_robustness_rms_db == pytest.approx(2.0, abs=1e-12)


def test_a_tilt_difference_is_kept_not_detrended_per_seat() -> None:
    """上方比主位多一條 1 dB／八度的斜線：逐點不扣自己的直線，所以差的就是那條斜線的加權均方根。"""
    level = demo_curve_db()
    extra = tilted_db(AXIS_HZ, 1.0, -6.0)
    payload = measured_payload(candidate(speaker(level, {"up": level + extra})))
    weights = independent_weights(AXIS_HZ, SCORING_RANGE_HZ)
    expected = float(np.sqrt(np.sum(weights * extra**2) / np.sum(weights)))

    assert expected > 0.5
    assert _by_direction(payload)["up"] == pytest.approx(expected, rel=1e-12)


def test_worst_direction_is_listed_and_does_not_eliminate() -> None:
    """下方差最大：列成最差方向（只提醒），這支喇叭照樣可估，穩健性仍是六點平均、不是最差那一點。"""
    level = demo_curve_db()
    surrounding = {
        "down": level + gaussian_db(AXIS_HZ, 45.0, -15.0, 0.1),
        "left": level + 0.5,
    }
    result = evaluate(candidate(speaker(level, surrounding))).speakers[0]
    assert result.state == "measured" and result.payload is not None
    payload = result.payload
    per_direction = _by_direction(payload)

    assert set(per_direction) == set(DIRECTIONS)
    assert (payload.worst_direction.direction_relative_to_primary, payload.worst_direction.receiver_id) == (
        "down", "down"
    )
    assert payload.worst_direction.difference_rms_db == max(per_direction.values())
    assert payload.seat_robustness_rms_db == pytest.approx(
        sum(per_direction.values()) / len(DIRECTIONS), rel=1e-12
    )
    assert payload.seat_robustness_rms_db < payload.worst_direction.difference_rms_db
