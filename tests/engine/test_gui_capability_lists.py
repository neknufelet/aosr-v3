"""兩份能力清單逐項照能力表原文；結果與比較用同一來源。"""
from __future__ import annotations

from pathlib import Path

import pytest

from aosr.config.capabilities import load_capabilities
from aosr.config.paths import config_path
from aosr.config.quality_targets import load_quality_targets
from aosr.gui.capability_view import capability_lists
from aosr.gui.compare_view import build_compare_view
from aosr.gui.result_view import build_result_view
from aosr.reporting.result import SchemeResult
from tests.engine.test_scheme_pipeline import shared_control_result


@pytest.fixture(scope="module")
def result(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> SchemeResult:
    return shared_control_result(tmp_path_factory, worker_id, "wall-1")


def test_capability_lists_preserve_nonempty_sections_verbatim() -> None:
    table = load_capabilities(config_path("capabilities.toml"))
    lists = capability_lists(table)
    for field in ("not_modeled", "manual_checks"):
        expected = tuple(item for entry in table.entry for item in getattr(entry, field))
        assert expected and lists[field] == expected


def test_result_and_compare_views_include_capability_lists(result: SchemeResult) -> None:
    targets = config_path("quality_targets.toml")
    view = build_result_view(result, quality_targets_path=targets)
    expected = capability_lists(load_capabilities(config_path("capabilities.toml")))
    assert view.not_modeled == expected["not_modeled"]
    assert view.manual_checks == expected["manual_checks"]
    comparison = build_compare_view(a_run_id="a" * 32, a=result, view_a=view,
                                    b_run_id="b" * 32, b=result, view_b=view,
                                    quality_targets=load_quality_targets(targets), run_date=result.run_date)
    assert comparison.not_modeled == tuple(dict.fromkeys(expected["not_modeled"]))
    assert comparison.manual_checks == tuple(dict.fromkeys(expected["manual_checks"]))
