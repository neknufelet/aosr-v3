"""直接造帳本與結果；不用搜尋輸出決定選席答案。"""
from pathlib import Path
from collections.abc import Mapping, Sequence

import pytest

from aosr.reporting.result import ResultOrigin
from aosr.reporting.stage_one_result import run_stage_one, save_stage_one_result
from aosr.scoring.contract import Feature
from aosr.scoring.receiver_set import primary_cross
from aosr.scoring.stage_one_low_frequency_contract import StageOneReasonCode
from aosr.scoring.stage_one_score import score_stage_one
from aosr.search import ledger
from aosr.search.run import SearchStatus
from aosr.search.settings import SearchSettings
from aosr.search.store import SearchIdentity, SearchStore
from tests.engine import _stage_one_cases as stage
from tests.engine._search_room_loop_cases import layout_for
from tests.engine._search_store_cases import settings_document, purpose_settings


def pool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rows: Sequence[tuple[int, float, float | None, float]], *,
         top_n: int = 2, missing_payload: bool = False, enabled: bool = True,
         rooms: Mapping[int, tuple[float, float]] | None = None, area: tuple[float, float] | None = None) -> SearchStore:
    """rooms 給了就換成房間搜尋：帳本每題記自己的長寬（高固定 3 m），挑席的面積照帳本算。"""
    project = stage.scheme()
    thresholds = {} if area is None else {"area_control_max_spread": area[0], "area_control_min_difference": area[1]}
    settings = SearchSettings.model_validate(settings_document() | {"purpose": project.purpose,
        "stage": "stage_one_fem_only", **({"stage_one_refine": {"top_n": top_n,
        "far_control_min_front_distance_m": .8, **thresholds}} if enabled else {}),
        **({} if rooms is None else {"layout": layout_for().model_dump(mode="json")})})
    kwargs = stage.run_kwargs()
    store = SearchStore.create(tmp_path / "searches", project=project, settings=settings,
        identity=SearchIdentity(kwargs["physics_identity"], kwargs["program_fingerprint"], purpose_settings(project.purpose)),
        versions={"python": "test", "numpy": "test", "optuna": "test"})
    template = run_stage_one(project, shards=stage.shards(monkeypatch, project), **kwargs)
    book = ledger.create_for(store)
    x, y, z = project.receiver_set.primary.position_m
    for number, score, peak, distance in rows:
        # 每個試算的主位往旁邊挪一點：細算替身的能量跟著喇叭到座位的距離變，1 Hz 合分才會各不相同。
        scheme = project.model_copy(update={"scheme_id": f"{store.search_id}-trial-{number:06d}",
            "receiver_set": primary_cross(primary_position_m=(x + .03 * (number % 11), y, z), offset_m=.1)})
        speakers = []
        for index, speaker in enumerate(template.evaluation.speakers):
            assert speaker.payload is not None
            payload = speaker.payload.model_copy(update={"features": (Feature(kind="peak", center_frequency_hz=50.,
                depth_db=(peak or 1.) - (2. if index == 0 else 0.), width_octave=.2, flags=()),),
                "strongest_peak_index": None if peak is None and index == 0 else 0, "deepest_dip_index": None})
            changed = speaker.model_copy(update={"payload": payload})
            if missing_payload and peak is None and index == 0:
                changed = speaker.model_copy(update={"payload": None, "state": "unavailable",
                    "reason_codes": (StageOneReasonCode.NON_POSITIVE_ENERGY,)})
            speakers.append(changed)
        evaluation = template.evaluation.model_copy(update={"candidate_id": scheme.scheme_id, "speakers": tuple(speakers)})
        result = template.model_copy(update={"scheme": scheme, "evaluation": evaluation,
            "score": score_stage_one(evaluation, template.score_settings), "origin": ResultOrigin(
                kind="search_candidate", search_id=store.search_id, trial_number=number)})
        SearchStore.scheme_path_for(store.candidate_path(number)).write_text(scheme.model_dump_json())
        save_stage_one_result(result, store.candidate_path(number))
        room = {} if rooms is None else dict(zip(("room_Lx", "room_Ly", "room_Lz"), (*rooms[number], 3.), strict=True))
        book.append((ledger.LedgerRow if rooms is None else ledger.LedgerRowV2)(batch_index=0, trial_number=number,
            unit_params_hex={key: .5.hex() for key in ("front_distance", "spacing", "listening_distance", *room)},
            params_m={"front_distance": distance, "spacing": 1., "listening_distance": 2., **room}, outcome="scored",
            score=score, reason=None, violation_m=None, seconds=.25, result_file=f"candidates/trial-{number:06d}.json"))
    store.status_path.write_text(SearchStatus(state="budget_exhausted").model_dump_json())
    return store


def snapshot(store: SearchStore) -> dict[str, bytes]:
    return {p.relative_to(store.path).as_posix(): p.read_bytes() for p in store.path.rglob("*") if p.is_file()}
