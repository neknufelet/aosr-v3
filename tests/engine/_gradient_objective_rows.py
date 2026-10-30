"""梯度帳本列的「舊合分與新合分並列」核對（#561 壬一）：兩本帳本（庚二、換樣板）的考卷共用。"""
from __future__ import annotations

import math
from collections.abc import Sequence
from functools import cache

import numpy as np

from aosr.config.paths import config_path
from aosr.config.quality_targets import load_quality_targets
from aosr.gradient.material_gradient_ledger import MaterialGradientRow
from aosr.gradient.template_ledger import TemplateRow
from aosr.scoring.gradient_objective_settings import GradientObjectiveSettings, gradient_objective_settings

PURPOSE = "dedicated_two_channel_listening_room"
# 新合分＝舊合分 ＋ w·P／ref 在編譯後算（兩次乘除與一次加，XLA 可以合成融合乘加），考卷用 Python 浮點重算：
# 差在幾個末位內。尺是新合分本身的四個末位（``np.spacing``），不是產品契約。
_ROUNDING_ULPS = 4


@cache
def registry_objective() -> GradientObjectiveSettings:
    return gradient_objective_settings(load_quality_targets(config_path("quality_targets.toml")), PURPOSE)


def assert_rows_split_scores(rows: Sequence[MaterialGradientRow | TemplateRow]) -> None:
    """每一列：是這一版（帶舊合分與三分項）、值＝舊合分 ＋ w·代理／ref（w、ref 從登記簿讀）、代理不小於 0、
    舊合分有限而且不大於值；舊合分可以單獨排序（每列都在、全部有限，照它排出的名次是一個全序）。"""
    settings = registry_objective()
    assert rows
    screening = []
    for row in rows:
        assert not row.is_legacy(), row.step
        value, old, (_, _, proxy) = row.scores()
        expected = old + settings.proxy_weight * proxy / settings.worse_reference
        tolerance = _ROUNDING_ULPS * float(np.spacing(max(abs(value), abs(expected))))
        assert abs(value - expected) <= tolerance, (row.step, value, expected)
        assert math.isfinite(old) and proxy >= 0.0 and old <= value, row.step
        screening.append(old)
    order = sorted(range(len(screening)), key=lambda index: (screening[index], index))
    assert sorted(screening) == [screening[index] for index in order]
