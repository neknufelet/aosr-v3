"""``aosr.config.crossover_axis`` 契約物件與建構函式的裁判。

v2 的考卷（``test_engine2_crossover_axis``）把 ``t_c_center``／``t_c_fade`` 與一個接縫
頻率寫死，再用它去驗別的層——那一支整個帶不走（它 import 了 dsp／materials／physics／
scoring 與腳本）。所以裁判改由 donor 標準答案接手：預設值與兩種覆寫各餵一組。

判定與殘餘風險見 ``tests/engine/test_config_cut1_table``（case 表住 ``blueprint/config_cut1_cases.py``）。
"""
from __future__ import annotations

from typing import TypedDict, cast

import aosr.config.crossover_axis as crossover_axis

from tests.engine._config_answers import (
    case_args,
    expected_block,
    is_approx,
    probe_by_id,
    probe_ids,
    probe_value,
)


class BuildArgs(TypedDict):
    """``build_canonical_crossover`` 的參數（``seam_f_s`` 是位置參數，其餘是關鍵字）。"""

    seam_f_s: float
    t_c_center: float
    t_c_fade: float


def test_build_canonical_crossover_matches_donor_on_every_declared_case() -> None:
    """case 表宣告的每一組 (接縫頻率, 中心, 淡出) 都跟 donor 一樣（逐欄比對）。"""
    ids = sorted(probe_ids("crossover_axis"))
    assert ids, "答案檔裡沒有 build_canonical_crossover 的 case——這一支就沒有對象"
    for case_id in ids:
        case = probe_by_id(case_id)
        args = cast(BuildArgs, case_args(case))
        built = crossover_axis.build_canonical_crossover(**args)
        actual = {
            "seam_f_s": built.seam_f_s,
            "t_c_center": built.t_c_center,
            "t_c_fade": built.t_c_fade,
        }
        assert is_approx(actual, probe_value(expected_block(case))), f"契約欄位跟 donor 不一樣：{case_id}"


def test_defaults_are_used_when_only_the_seam_frequency_is_given() -> None:
    """只給接縫頻率時，兩個時間常數走模組的預設常數（接線，不只是各自凍結）。"""
    built = crossover_axis.build_canonical_crossover(250.0)
    assert built.t_c_center == crossover_axis.DEFAULT_T_C_CENTER_S
    assert built.t_c_fade == crossover_axis.DEFAULT_T_C_FADE_S
