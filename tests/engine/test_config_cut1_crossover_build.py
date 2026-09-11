"""``aosr.config.crossover_axis`` 契約物件與建構函式的裁判。

v2 的考卷（``test_engine2_crossover_axis``）把 ``t_c_center``／``t_c_fade`` 與一個接縫
頻率寫死，再用它去驗別的層——那一支整個帶不走（它 import 了 dsp／materials／physics／
scoring 與腳本）。所以裁判改由 donor 標準答案接手：預設值與兩種覆寫各餵一組。

判定與殘餘風險見 ``tests/engine/test_config_cut1_table``。
"""
from __future__ import annotations

from typing import TypedDict, cast

import aosr.config.crossover_axis as crossover_axis

from tests.engine._config_answers import is_approx, module_answers, probe_value


class BuildArgs(TypedDict):
    """``build_canonical_crossover`` 的參數（``seam_f_s`` 是位置參數，其餘是關鍵字）。"""

    seam_f_s: float
    t_c_center: float
    t_c_fade: float


def _field(record: object, key: str) -> object:
    if not isinstance(record, dict) or key not in record:
        raise AssertionError(f"答案檔這一筆沒有 {key}：{record!r}")
    return record[key]


def _table(node: object, where: str) -> dict[str, object]:
    if not isinstance(node, dict):
        raise AssertionError(f"{where} 不是表：{node!r}")
    return {str(name): item for name, item in node.items()}


def test_build_canonical_crossover_matches_donor_on_every_probed_input() -> None:
    """每一組 (接縫頻率, 中心, 淡出) 的契約物件都跟 donor 一樣（逐欄比對）。"""
    probes = module_answers("crossover_axis")["probes"]
    assert isinstance(probes, list) and probes, "答案檔裡沒有 build_canonical_crossover 的探針"
    for record in probes:
        args = _table(_field(record, "args"), "答案檔這一筆的 args")
        expected = _table(_field(record, "expected"), "答案檔這一筆的 expected")
        built = crossover_axis.build_canonical_crossover(**cast(BuildArgs, args))
        actual = {
            "seam_f_s": built.seam_f_s,
            "t_c_center": built.t_c_center,
            "t_c_fade": built.t_c_fade,
        }
        assert is_approx(actual, probe_value(expected)), f"契約欄位跟 donor 不一樣：{args}"


def test_defaults_are_used_when_only_the_seam_frequency_is_given() -> None:
    """只給接縫頻率時，兩個時間常數走模組的預設常數（接線，不只是各自凍結）。"""
    built = crossover_axis.build_canonical_crossover(250.0)
    assert built.t_c_center == crossover_axis.DEFAULT_T_C_CENTER_S
    assert built.t_c_fade == crossover_axis.DEFAULT_T_C_FADE_S
