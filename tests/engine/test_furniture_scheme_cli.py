"""命令列保留逐份代價與近似，集中原因只印一次；B1 取自第 13 條。"""
from datetime import date
from pathlib import Path

import pytest

from aosr.config.paths import config_path
from aosr.config.quality_targets import load_quality_targets
from aosr.reporting import scheme_cli
from aosr.reporting.compare import compare_results
from aosr.reporting.result import SchemeResult, save_result
from tests.engine._furniture_scheme_results import scheme_pair as scheme_pair
from tests.engine.test_gui_compare_view import _renamed


REASON = ("已含家具一次反射、遮擋與有限尺寸鏡面修正；未含家具與牆之間的多次反射、"
          "完整繞射，以及家具吸音對整房殘響的影響")


def test_cli_single_result_marks_approximate_category_line_ends(
    scheme_pair: tuple[SchemeResult, ...], capsys: pytest.CaptureFixture[str],
) -> None:
    furnished = scheme_pair[1]
    registry = load_quality_targets(config_path("quality_targets.toml"))
    ranking = compare_results((furnished,), quality_targets=registry, run_date=date(2026, 10, 8))
    scheme_cli._print_result(furnished, ranking)
    lines = capsys.readouterr().out.splitlines()
    approximate = {item.category.value for item in furnished.candidate.evaluations
                   if "furniture_model_approximate" in item.flags}
    for category in approximate:
        assert next(line for line in lines if line.strip().startswith(f"{category} | ")).endswith(" | 近似")
    assert REASON in lines


@pytest.mark.parametrize("both_furnished", [False, True])
def test_cli_comparison_prints_own_costs_conditions_and_one_reason(
    scheme_pair: tuple[SchemeResult, ...], tmp_path: Path, capsys: pytest.CaptureFixture[str], both_furnished: bool,
) -> None:
    furnished = scheme_pair[1]
    results = (furnished, _renamed(furnished, "second-furnished")) if both_furnished else scheme_pair
    paths = [tmp_path / f"{result.scheme.scheme_id}.json" for result in results]
    for path, result in zip(paths, results, strict=True):
        save_result(result, path)
    assert not scheme_cli.main(["compare", *(str(path) for path in paths), "--capabilities",
                               str(config_path("capabilities.toml")), "--run-date", "2026-10-08"])
    text = capsys.readouterr().out
    before, after = text.split(REASON)
    assert REASON not in before + after
    registry = load_quality_targets(config_path("quality_targets.toml"))
    blocks = {block.split("：", 1)[0]: block for block in text.split("方案 ")[1:]}
    for result in results:
        own = compare_results((result,), quality_targets=registry, run_date=date(2026, 10, 8))
        for row in own.rankable:
            for line in row.categories:
                assert (f"{line.identity.category.value} | measured | 代價 {line.category_cost} | "
                        in blocks[result.scheme.scheme_id])
    if not both_furnished:
        assert "評分條件不同；兩者計算涵蓋範圍不同" in text
    else:
        assert "兩者計算涵蓋範圍不同" not in text
