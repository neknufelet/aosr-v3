"""選席答案直接從指定合分、峰高與距離推導。"""
from pathlib import Path
import random

import pytest

from aosr.config.paths import config_path
from aosr.search.store import SearchStore
from aosr.search.stage_one_selection_contract import StageOneSelection
from tests.engine._stage_one_refine_cases import pool


def select(store: SearchStore) -> StageOneSelection:
    from aosr.search.stage_one_selection import select_stage_one_candidates
    return select_stage_one_candidates(store, registry_path=config_path("quality_targets.toml"))


def test_top_order_ties_small_pool_and_worse_channel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = pool(tmp_path, monkeypatch, [(7, 2., 8., .5), (3, 1., 6., .6), (1, 1., 9., .7)], top_n=5)
    result = select(store)
    assert [s.trial_number for s in result.candidates] == [1, 3, 7]
    assert [s.search_rank for s in result.candidates] == [1, 2, 3]
    assert [s.big_peak_db for s in result.candidates] == [9., 6., 8.]
    assert result.actual_top_n == len(result.candidates)
    assert type(result).model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize("missing_payload", [False, True])
def test_unmeasurable_peak_is_not_a_number(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing_payload: bool) -> None:
    store = pool(tmp_path, monkeypatch, [(0, 1., None, .5), (1, 2., None, .6), (2, 3., 1., .7)],
                 missing_payload=missing_payload)
    selected = select(store)
    assert all(s.big_peak_db is None and s.peak_reason for s in selected.candidates)
    assert selected.big_peak_threshold_db is None
    assert selected.missing["big_peak_control"] == "前 N 名都量不到大峰"


@pytest.mark.parametrize("delta,control", [(0., 3), (1e-9, None)])
def test_big_peak_outside_top_and_inclusive_threshold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                     delta: float, control: int | None) -> None:
    store = pool(tmp_path, monkeypatch, [(0, 1., 5., .5), (1, 2., 9., .6), (2, 3., 4. + delta, .7),
        (3, 2.5, 4. + delta, .7)])
    selection = select(store)
    assert selection.big_peak_threshold_db == 4.
    assert [s.trial_number for s in selection.candidates if "big_peak_control" in s.roles] == ([] if control is None else [control])
    # 前 N 中的最小峰已低於另一個前 N 的峰減差距；仍不可搶走對照。
    assert "big_peak_control" not in selection.candidates[0].roles


@pytest.mark.parametrize("distances,expected", [([.8, .5, .9, 1.], 0), ([.5, .6, .8, 1.], 2),
    ([.5, .6, .7, .8], 3), ([.8 - 1e-9, .5, .6, .7], None)])
def test_far_control_uses_whole_pool_and_shared_roles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                    distances: list[float], expected: int | None) -> None:
    rows = [(i, float(i + 1), peak, distance) for i, (peak, distance) in enumerate(zip([5., 9., 4., 8.], distances, strict=True))]
    chosen = select(pool(tmp_path, monkeypatch, rows))
    assert [s.trial_number for s in chosen.candidates if "far_control" in s.roles] == ([] if expected is None else [expected])
    if expected == 0:
        assert chosen.candidates[0].roles == ("top_n", "far_control")
    if expected == 2:
        assert next(s for s in chosen.candidates if s.trial_number == expected).roles == ("big_peak_control", "far_control")
    if expected is None:
        assert chosen.missing["far_control"] == "本次未取得較遠對照"


def test_unreadable_top_consumes_slot_without_replacement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = pool(tmp_path, monkeypatch, [(0, 1., 5., .8), (1, 2., 7., .5), (2, 3., 8., .6)])
    store.candidate_path(0).write_text("broken")
    chosen = select(store)
    assert [s.trial_number for s in chosen.candidates] == [0, 1]
    assert chosen.candidates[0].error and chosen.candidates[0].big_peak_db is None
    assert "far_control" in chosen.missing


def test_margin_comes_from_registry_and_missing_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.search.stage_one_selection import select_stage_one_candidates
    text = '\n'.join(['schema_version = 1', '[[purpose]]', 'name = "dedicated_two_channel_listening_room"', 'setting = []', 'target = []', 'weight = []', 'qualification = []',
        '[[stage_one]]', 'name = "dedicated_two_channel_listening_room"',
        'target = []', 'weight = []', 'qualification = []', '[[stage_one.setting]]', 'key = "stage_one_selection.big_peak_control_margin_db"',
        'value = 2.0', 'unit = "dB"', 'status = "baseline"',
        'source_kind = "engineering_recommendation"', 'source = "test"'])
    path = tmp_path / "registry.toml"
    path.write_text(text)
    store = pool(tmp_path, monkeypatch, [(0, 1., 5., .5), (1, 2., 7., .6), (2, 3., 4., .6), (3, 4., 3., .6)])
    assert [s.trial_number for s in select(store).candidates if "big_peak_control" in s.roles] == [2]
    assert [s.trial_number for s in select_stage_one_candidates(store, registry_path=path).candidates if "big_peak_control" in s.roles] == [3]
    path.write_text(text.split('[[stage_one.setting]]')[0] + 'setting = []')
    with pytest.raises((KeyError, ValueError)):
        select_stage_one_candidates(store, registry_path=path)


def test_seeded_selection_invariants(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rng = random.Random(558)
    for batch in range(5):
        rows = [(i, rng.random(), rng.uniform(1., 10.), rng.uniform(.3, 1.)) for i in range(9)]
        chosen = select(pool(tmp_path / str(batch), monkeypatch, rows))
        assert len(chosen.candidates) <= chosen.top_n + 2
        assert sum("top_n" not in s.roles for s in chosen.candidates) <= 2
