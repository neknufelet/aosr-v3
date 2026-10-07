"""唯一真解：小房間搜尋附件的共用擺位快取可由選入後的網頁結果頁直接讀回。"""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

from starlette.testclient import TestClient

from aosr.gui.app import GuiSettings, create_app
from aosr.gui.modal_jobs import scheme_snapshot
from aosr.reporting.modal_diagnosis_model import ModalDiagnosisState
from aosr.reporting.modal_lookup import read_placement
from aosr.reporting.scheme import Scheme
from aosr.search.modal_attach import attach_modal
from aosr.search.outer import conclude
from aosr.search.run import CandidateJob, ComputedCandidate
from aosr.search.select import select_refined
from aosr.search.store import SearchStore
from tests.engine._gui_cache import gui_startup_identity_memo
from tests.engine._search_modal_cases import protected
from tests.engine._search_refine_cases import refine
from tests.engine._search_run_cases import make_store, run
from tests.engine._search_select_cases import ResultRefineCompute
from tests.engine.test_modal_diagnosis import C, RECEIVERS, RHO, ROOM, SOURCES, WALLS
from tests.engine.test_search_report import _ResultCompute


class _Schemes(_ResultCompute):
    def __call__(self, jobs: Sequence[CandidateJob], workers: int) -> Iterator[ComputedCandidate]:
        for job in jobs:
            SearchStore.scheme_path_for(job.result_path).write_text(job.scheme.model_dump_json())
        yield from super().__call__(jobs, workers)


def _small_store(tmp_path: Path) -> tuple[SearchStore, Path]:
    seed, registry = make_store(tmp_path / "seed", budget=2, refine={"budget": 2, "convergence_run": 30},
        layout_changes={"speaker_height_m": 0.5, "ear_height_m": 0.6,
                        "front_distance_m": {"low": 0.3, "high": 0.4},
                        "spacing_m": {"low": 0.6, "high": 0.8},
                        "listening_distance_m": {"low": 0.9, "high": 1.0}})
    document = seed.project.model_dump(mode="json")
    document["scene"].update(room_m={"Lx": ROOM.Lx, "Ly": ROOM.Ly, "Lz": ROOM.Lz},
        density_kg_m3=RHO, sound_speed_m_s=C,
        impedance_pa_s_per_m_by_wall={wall.wall_name(): value for wall, value in WALLS.items()})
    document["speakers"] = {name: {"x": point.x, "y": point.y, "z": point.z} for name, point in SOURCES.items()}
    main, side = document["receiver_set"]["points"][:2]
    document["receiver_set"]["points"] = [
        main | {"receiver_id": "main", "position_m": RECEIVERS["main"].as_tuple()},
        side | {"receiver_id": "side", "position_m": RECEIVERS["side"].as_tuple()}]
    store = SearchStore.create(tmp_path / "searches", project=Scheme.model_validate(document),
        settings=seed.settings, identity=seed.identity, versions=seed.versions)
    return store, registry


def test_small_true_solve_shared_cache_is_immediate_on_selected_result_page(tmp_path: Path) -> None:
    store, registry = _small_store(tmp_path)
    run(store, registry, _Schemes(store))
    compute = ResultRefineCompute(store, {None: 0.1, 0: 0.5, 1: 0.7})
    status = refine(store, registry, compute)
    for job in compute.jobs:
        store.scheme_path_for(job.result_path).write_text(job.scheme.model_dump_json())
    status = conclude(store, status, "refine_budget")
    before = protected(store)
    cache = tmp_path / "modal-cache"
    summary = attach_modal(store, status=status, cache_dir=cache)
    assert summary.completed and all(role.state == "diagnosed_not_scored" for role in summary.roles), summary
    assert protected(store) == before
    saved = read_placement(store.project, cache_dir=cache)
    assert saved is not None and saved.state is ModalDiagnosisState.DIAGNOSED_NOT_SCORED
    selected = select_refined(store, None, tmp_path)
    value = scheme_snapshot(selected.result_path)
    assert read_placement(value, cache_dir=cache) == saved
    with TestClient(create_app(GuiSettings(engine_commit="a" * 40, data_dir=tmp_path,
                                          modal_runner=("must-not-run",))), base_url="http://localhost") as client:
        response = client.get(f"/api/results/{selected.run_id}/modal")
        assert response.status_code == 200
        body = response.json()
        assert body["view"]["state_text"] == "已診斷不計分" and body.get("job") is None
        assert not tuple((tmp_path / "modal-jobs").rglob("job.json"))
