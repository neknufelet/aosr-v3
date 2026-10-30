"""梯度帳本列的「舊合分與新合分並列」核對（#561 壬一）：兩本帳本（庚二、換樣板）的考卷共用。"""
from __future__ import annotations

import math
from collections.abc import Sequence
from functools import cache

import numpy as np

from aosr.config.paths import config_path
from aosr.config.quality_targets import load_quality_targets
from aosr.gradient.material_gradient_ledger import SCORE_IDENTITY_ULPS, MaterialGradientRow
from aosr.gradient.template_ledger import TemplateRow
from aosr.scoring.gradient_objective_settings import GradientObjectiveSettings, gradient_objective_settings

PURPOSE = "dedicated_two_channel_listening_room"


@cache
def registry_objective() -> GradientObjectiveSettings:
    return gradient_objective_settings(load_quality_targets(config_path("quality_targets.toml")), PURPOSE)


def assert_rows_split_scores(rows: Sequence[MaterialGradientRow | TemplateRow]) -> None:
    """每一列：是這一版（帶舊合分與三分項）、值＝舊合分 ＋ w·代理／ref（w、ref 從登記簿讀；容差是帳本讀回驗證器
    同一把 ``SCORE_IDENTITY_ULPS`` 個末位）、舊合分與代理有限；舊合分可以單獨排序（每列都在、全部有限，照它排出的
    名次是一個全序）。代理可以是負的、新合分可以低於舊合分（審查 A），這裡不斷言正負。"""
    settings = registry_objective()
    assert rows
    screening = []
    for row in rows:
        assert not row.is_legacy(), row.step
        value, old, (_, _, proxy) = row.scores()
        expected = old + settings.proxy_weight * proxy / settings.worse_reference
        tolerance = SCORE_IDENTITY_ULPS * float(np.spacing(max(abs(value), abs(expected))))
        assert abs(value - expected) <= tolerance, (row.step, value, expected)
        assert math.isfinite(old) and math.isfinite(proxy), row.step
        screening.append(old)
    order = sorted(range(len(screening)), key=lambda index: (screening[index], index))
    assert sorted(screening) == [screening[index] for index in order]
