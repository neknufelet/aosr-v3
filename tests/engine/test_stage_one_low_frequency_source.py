"""第一階段低音量測的聲源考卷（票 #558，老闆第 3 題之二）：沒有額外聲源修正、資料未被改寫、假設標示存在。

dB 轉換、音量處理與扣線計分照常做，不要求處理後還等於原始曲線（老闆原話）。
「評估器收到的能量逐位等於報表的有限元素能量」要接報表，留到接結果檔那一支。
"""
from __future__ import annotations

import numpy as np
import pytest

from aosr.scoring.stage_one_low_frequency_contract import (
    IDEAL_SOURCE_ASSUMPTION,
    IDEAL_SOURCE_ASSUMPTION_TEXT,
    StageOneReasonCode,
)
from tests.engine._stage_one_low_frequency_curves import (
    AXIS_HZ,
    SCORING_RANGE_HZ,
    candidate,
    demo_curve_db,
    evaluate,
    measured_payload,
    speaker,
    tilted_db,
)

# 老闆 2026-10-19 原話逐字（#558 第 ② 塊第 3 題之二）；考卷自己抄一份，不從評估器讀。
_BOSS_SENTENCE = "未加入實際喇叭的低頻衰減，排名只適用於本次理想聲源基準；換播放系統後可能不同"


def test_flat_energy_stays_flat_without_any_source_correction() -> None:
    """平坦能量：斜率與主項都是零。偷套任何高通或低頻衰減，斜率或殘差就不會是零。"""
    payload = measured_payload(candidate(speaker(np.full(len(AXIS_HZ), 72.0))))

    assert payload.own_line_slope_db_per_octave == pytest.approx(0.0, abs=1e-12)
    assert payload.own_line_ripple_rms_db == pytest.approx(0.0, abs=1e-12)
    assert payload.own_line_intercept_db == pytest.approx(72.0, abs=1e-9)


def test_common_level_only_moves_the_intercept() -> None:
    """整組曲線乘同一個倍數（音量），只有截距平移 10·log10(倍數)；起伏、穩健性、斜率都不變。"""
    level = demo_curve_db()
    surrounding = {"front": level + tilted_db(AXIS_HZ, 0.5, -1.0), "up": level - 1.5}
    base = measured_payload(candidate(speaker(level, surrounding)))
    louder = measured_payload(candidate(speaker(
        level + 20.0, {key: value + 20.0 for key, value in surrounding.items()}
    )))

    assert louder.own_line_intercept_db == pytest.approx(base.own_line_intercept_db + 20.0, abs=1e-9)
    assert louder.own_line_slope_db_per_octave == pytest.approx(base.own_line_slope_db_per_octave, rel=1e-9)
    assert louder.own_line_ripple_rms_db == pytest.approx(base.own_line_ripple_rms_db, rel=1e-9)
    assert louder.seat_robustness_rms_db == pytest.approx(base.seat_robustness_rms_db, rel=1e-9)


def test_input_curves_are_not_rewritten() -> None:
    """評估前後輸入物件逐欄相等；周圍點的逐頻差就落在輸入的計分範圍軸點上，沒有重取樣。"""
    data = candidate(speaker(demo_curve_db(), {"left": demo_curve_db() + 2.0}))
    before = data.model_dump()
    payload = measured_payload(data)

    assert data.model_dump() == before
    inside = tuple(float(value) for value in AXIS_HZ if SCORING_RANGE_HZ[0] <= value <= SCORING_RANGE_HZ[1])
    for direction in payload.directions:
        assert tuple(frequency for frequency, _ in direction.difference_curve_db) == inside
    assert {item.direction_relative_to_primary: item.report_id for item in payload.directions} == {
        name: f"report-{name}" for name in ("front", "back", "left", "right", "up", "down")
    }
    assert payload.primary_report_id == "report-main"


def test_assumption_label_is_carried_on_every_result() -> None:
    """假設標示必帶：機器讀的固定字面值，加上老闆那句逐字；不可估的結果也帶，不寫「與喇叭無關」。"""
    keep = AXIS_HZ <= 150.0
    for data in (
        candidate(speaker(demo_curve_db())),
        candidate(speaker(demo_curve_db()[keep], frequencies=AXIS_HZ[keep])),
    ):
        result = evaluate(data)
        assert result.assumption == IDEAL_SOURCE_ASSUMPTION == "ideal_source_no_loudspeaker_low_frequency_rolloff"
        assert result.assumption_text == IDEAL_SOURCE_ASSUMPTION_TEXT == _BOSS_SENTENCE
        assert "與喇叭無關" not in result.assumption_text
        assert result.energy_column == "fem_energy"
    assert result.speakers[0].reason_codes == (StageOneReasonCode.SCORING_RANGE_GAP,)
