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
from tests.engine._furniture_scheme_results import shared_moved_furnished_result
from tests.engine._furniture_scheme_results import twodesks_k1_result as twodesks_k1_result
from tests.engine.test_gui_compare_view import _renamed, moved_primary_result
from tests.engine.test_scheme_pipeline import shared_control_result
from tests.engine._furniture_cli_control import MOVED_THEN_PLAIN, PLAIN_THEN_MOVED


REASON = ("已含家具一次反射、遮擋與有限尺寸鏡面修正；未含家具與牆之間的多次反射、"
          "完整繞射，以及家具吸音對整房殘響的影響")


def _compare_text(results: tuple[SchemeResult, ...], tmp_path: Path,
                  capsys: pytest.CaptureFixture[str], run_date: str = "2026-10-08") -> str:
    paths = [tmp_path / f"{result.scheme.scheme_id}.json" for result in results]
    for path, result in zip(paths, results, strict=True):
        save_result(result, path)
    assert not scheme_cli.main(["compare", *(str(path) for path in paths), "--capabilities",
                               str(config_path("capabilities.toml")), "--run-date", run_date])
    return capsys.readouterr().out


def _blocks(text: str) -> dict[str, list[str]]:
    return {block.split("：", 1)[0]: block.split("類別 | ", 1)[0].splitlines()
            for block in text.split("方案 ")[1:]}


def _table_cells(text: str, category: str) -> dict[str, str]:
    table = text.split("類別 | ", 1)[1].splitlines()
    names = table[0].split(" | ")
    cells = next(line for line in table[1:] if line.startswith(f"{category} | ")).split(" | ")[1:]
    return dict(zip(names, cells, strict=True))


def test_three_results_notes_belong_only_to_the_incompatible_candidate(
    scheme_pair: tuple[SchemeResult, ...], tmp_path: Path, capsys: pytest.CaptureFixture[str],
    twodesks_k1_result: SchemeResult,
) -> None:
    plain, furnished = scheme_pair
    second = twodesks_k1_result
    text = _compare_text((plain, furnished, second), tmp_path, capsys)
    blocks = _blocks(text)
    for result in (furnished, second):
        assert "評分條件不同" not in "\n".join(blocks[result.scheme.scheme_id])
        for category in ("reflections_and_echo", "listening_area_stability"):
            cell = _table_cells(text, category)[result.scheme.scheme_id]
            assert "評分條件不同" not in cell
            assert "兩者計算涵蓋範圍不同" not in cell
    reflection = next(line for line in blocks[plain.scheme.scheme_id]
                      if line.startswith("  reflections_and_echo | "))
    assert " | 評分條件不同；兩者計算涵蓋範圍不同" in reflection
    assert "評分條件不同；兩者計算涵蓋範圍不同" in _table_cells(text, "reflections_and_echo")[plain.scheme.scheme_id]


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
    blocks = _blocks(text)
    for result in results:
        # 答案走單份 compare 的主線既有主表輸出，不呼叫混比的自算代價 helper。
        own_lines = _compare_text((result,), tmp_path, capsys).splitlines()
        for category in ("timbre_balance", "listening_area_stability", "reflections_and_echo", "channel_matching", "reverberation"):
            own = next(line for line in own_lines if line.startswith(f"  {category} | "))
            cost = own.split(" | 代價 ", 1)[1].split(" | ", 1)[0]
            shown = next(line for line in blocks[result.scheme.scheme_id] if line.startswith(f"  {category} | "))
            assert f" | 代價 {cost} | " in shown
            assert _table_cells(text, category)[result.scheme.scheme_id].split("；", 1)[0] == cost
    if not both_furnished:
        plain = results[0]
        assert blocks[plain.scheme.scheme_id][0].endswith("：not_comparable")
        reflection = next(line for line in blocks[plain.scheme.scheme_id] if line.startswith("  reflections_and_echo | "))
        assert " | 評分條件不同；兩者計算涵蓋範圍不同" in reflection
    else:
        assert "兩者計算涵蓋範圍不同" not in text


@pytest.mark.parametrize("reverse", [False, True])
def test_plain_incompatible_compare_matches_every_mainline_output_line(
    tmp_path_factory: pytest.TempPathFactory, worker_id: str, tmp_path: Path,
    capsys: pytest.CaptureFixture[str], reverse: bool,
) -> None:
    plain = shared_control_result(tmp_path_factory, worker_id, "wall-1")
    moved = moved_primary_result(tmp_path_factory, worker_id)
    pair = (moved, plain) if reverse else (plain, moved)
    text = _compare_text(pair, tmp_path, capsys, "2026-09-27")
    expected = MOVED_THEN_PLAIN if reverse else PLAIN_THEN_MOVED
    assert text.splitlines(keepends=True) == expected.splitlines(keepends=True)


@pytest.mark.parametrize("reverse", [False, True])
def test_mixed_table_approximation_is_in_only_the_furnished_cell(
    scheme_pair: tuple[SchemeResult, ...], tmp_path: Path,
    capsys: pytest.CaptureFixture[str], reverse: bool,
) -> None:
    pair = tuple(reversed(scheme_pair)) if reverse else scheme_pair
    text = _compare_text(pair, tmp_path, capsys)
    plain, furnished = scheme_pair
    for category in ("timbre_balance", "listening_area_stability", "reflections_and_echo", "channel_matching"):
        cells = _table_cells(text, category)
        assert "近似" not in cells[plain.scheme.scheme_id]
        assert cells[furnished.scheme.scheme_id].endswith("；近似")


def test_reverberation_line_marks_only_the_furnished_result(
    scheme_pair: tuple[SchemeResult, ...], tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    blocks = _blocks(_compare_text(scheme_pair, tmp_path, capsys))
    for result in scheme_pair:
        line = next(line for line in blocks[result.scheme.scheme_id] if line.startswith("  reverberation | "))
        assert (" | 未包含家具吸音" in line) == bool(result.scheme.furniture)



def test_furnished_non_furniture_identity_difference_has_no_b1_reason(
    scheme_pair: tuple[SchemeResult, ...], tmp_path_factory: pytest.TempPathFactory, worker_id: str,
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    moved = shared_moved_furnished_result(tmp_path_factory, worker_id)
    text = _compare_text((scheme_pair[1], moved), tmp_path, capsys)
    assert "not_comparable" in text
    assert "評分條件不同" in text
    assert "兩者計算涵蓋範圍不同" not in text


def test_three_results_mark_only_the_incompatible_candidates_own_categories(
    tmp_path_factory: pytest.TempPathFactory, worker_id: str, scheme_pair: tuple[SchemeResult, ...],
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    plain = shared_control_result(tmp_path_factory, worker_id, "wall-1")
    moved = moved_primary_result(tmp_path_factory, worker_id)
    furnished = scheme_pair[1]
    text = _compare_text((plain, moved, furnished), tmp_path, capsys)
    blocks = _blocks(text)
    assert blocks[moved.scheme.scheme_id][0].endswith("：rankable")
    assert "評分條件不同" not in "\n".join(blocks[moved.scheme.scheme_id])
    for category in ("timbre_balance", "reflections_and_echo"):
        line = next(line for line in blocks[plain.scheme.scheme_id] if line.startswith(f"  {category} | "))
        assert "評分條件不同" not in line
        assert "兩者計算涵蓋範圍不同" not in line
        assert "評分條件不同" not in _table_cells(text, category)[plain.scheme.scheme_id]
    for category in ("listening_area_stability", "channel_matching"):
        line = next(line for line in blocks[plain.scheme.scheme_id] if line.startswith(f"  {category} | "))
        assert " | 評分條件不同" in line
        assert "兩者計算涵蓋範圍不同" not in line
    reflection = next(line for line in blocks[furnished.scheme.scheme_id] if line.startswith("  reflections_and_echo | "))
    assert " | 評分條件不同；兩者計算涵蓋範圍不同" in reflection
