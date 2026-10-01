"""真實擺位、報表、評估、排名與帳本串接；只替換昂貴物理控制組。"""

import logging
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import pytest

from aosr.config.capabilities import load_capabilities
from aosr.config.quality_targets import load_quality_targets
from aosr.geometry.shoebox import Point, Room, Wall
from aosr.physics import three_lane_report
from aosr.reporting.pipeline import run_scheme
from aosr.reporting.result import save_result
from aosr.scoring.contract import CandidateEvaluation
from aosr.scoring.ranking import RankingContext, comparison_identity_of
from tests.engine._directivity import DIRECTIVITY
from tests.engine._scoring_source_model_control import STAND_INS
from tests.engine._source_model_control import fake_fem_energy
from tests.engine._search_run_cases import ENGINE, RUN_DATE, make_store, next_params, rows


def _many_fem(*, room: Room, sources: Mapping[str, Point], receivers: Mapping[str, Point],
              wall_impedances: Mapping[Wall, float], frequencies_hz: tuple[float, ...],
              density_kg_m3: float, sound_speed_m_s: float) -> dict[tuple[str, str], tuple[float, ...]]:
    """把指定控制組的逐對替身接到管線使用的批次物理入口。"""
    return {(speaker, receiver): fake_fem_energy(
        room=room, source=source_point, receiver=receiver_point,
        wall_impedances=wall_impedances, frequencies_hz=frequencies_hz,
        density_kg_m3=density_kg_m3, sound_speed_m_s=sound_speed_m_s,
    ) for speaker, source_point in sources.items() for receiver, receiver_point in receivers.items()}


def test_two_batches_through_the_real_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """防真實候選全被報不能同表或原方案偷換聲源模型；舊題只查算完題數與檔案存在。"""
    from aosr.search.run import CandidateJob, ComputedCandidate, start_search

    # 一階反射、兩批各一個：真實串接只要證明「擺法→方案→物理→評估→排名→帳本→取樣器」接得起來，
    # 物理數值另有考卷；三階與四個候選要四分多鐘，一階與兩個候選約幾十秒（雲端時間吃緊，#586）。
    store, registry = make_store(tmp_path, budget=2, batch=1, scene_changes={"reflection_order_k": 1})
    source = next((Path(__file__).resolve().parents[2] / "src" / "aosr" / "config" / "data").glob("capabilities.*"))
    capabilities_path = tmp_path / "capabilities"
    capabilities_path.write_bytes(source.read_bytes())
    capabilities = load_capabilities(capabilities_path)
    for module, name, stand_in in STAND_INS:
        monkeypatch.setattr(module, name, stand_in)
    monkeypatch.setattr(three_lane_report, "_solve_fem_energies", _many_fem)
    candidates: dict[int | None, CandidateEvaluation] = {}

    def compute(jobs: Sequence[CandidateJob], workers: int) -> Iterator[ComputedCandidate]:
        assert workers == store.settings.max_workers
        for job in jobs:
            if job.trial_number is None:
                assert job.scheme.model_dump(exclude={"scheme_id"}) == store.project.model_dump(exclude={"scheme_id"})
            result = run_scheme(job.scheme, capabilities=capabilities, directivity=DIRECTIVITY,
                                quality_targets_path=registry, engine_commit="fixture",
                                program_fingerprint=ENGINE, physics_identity=store.identity.physics_identity,
                                run_date=RUN_DATE)
            save_result(result, job.result_path)
            candidates[job.trial_number] = result.candidate
            yield ComputedCandidate(job, result.candidate, result.timings.total_s)

    started = time.perf_counter()
    status = start_search(store, compute=compute, probe=lambda: store.identity,
                          registry_path=registry, run_date=RUN_DATE, engine_version=ENGINE)
    seconds = time.perf_counter() - started
    logging.getLogger(__name__).warning("真實搜尋串接耗時 %.6f 秒", seconds)
    assert status.baseline_outcome == "scored"
    assert status.state == "budget_exhausted"
    assert [row.trial_number for row in rows(store)] == list(range(store.settings.budget))
    assert {row.batch_index for row in rows(store)} == set(range(store.settings.budget // store.settings.batch_size))
    assert status.computed == store.settings.budget
    assert all(store.candidate_path(row.trial_number).is_file() for row in rows(store))
    assert store.baseline_path.is_file()
    assert next_params(store)
    assert any(row.outcome == "scored" for row in rows(store))
    context = RankingContext(purpose=store.project.purpose, receiver_set_fingerprint=store.project.receiver_set.fingerprint,
                             channel_group_fingerprint=store.project.channel_group.fingerprint,
                             run_date=RUN_DATE, engine_version=ENGINE)
    targets = load_quality_targets(registry)
    pinned = comparison_identity_of(candidates[None], targets, context)
    assert pinned is not None
    assert all(comparison_identity_of(candidate, targets, context) == pinned
               for number, candidate in candidates.items() if number is not None)
