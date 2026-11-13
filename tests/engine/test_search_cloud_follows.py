"""天雲中心跟著第二階段每個候選的喇叭與主位重算，入選後移位檢查時不動（#1147，#705 老闆 10-26 回 A）。

參考房 6×4×3 公尺、前牆 x0、面向 −x。沒鎖座位時兩喇叭與主位都在 y=2.0，喇叭 x＝離前牆、主位 x＝離前牆＋聆聽距離；
天雲水平中心＝兩喇叭水平中點與主位的中間，所以中心 x＝離前牆＋聆聽距離／2、y＝2.0。期望值都手算寫死，不呼叫產品的規則函式。
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from aosr.geometry.furniture import FurnitureKind
from aosr.geometry.shoebox import Point
from aosr.gui.input_setup import default_furniture
from aosr.reporting.scheme import RoomPlacement, Scheme
from aosr.search import layout
from aosr.search.placement_stability_geometry import PLACEMENT_SHIFT_M, generate_shifts
from aosr.search.refine_run import _job as refine_job
from aosr.search.store import SearchStore
from tests.engine._placement_stability_cases import scheme as hand_scheme, settings as hand_settings
from tests.engine._search_furniture_cases import enqueue_hand_placements
from tests.engine._search_refine_cases import SearchCompute
from tests.engine._search_run_cases import make_store, rows, run
from tests.engine._search_store_cases import reference_project

if TYPE_CHECKING:
    from aosr.search.run import CandidateJob, ComputedCandidate


def _cloud(scheme: Scheme) -> RoomPlacement:
    item, = (item for item in scheme.furniture or () if item.kind == FurnitureKind.CEILING_CLOUD)
    assert isinstance(item.placement, RoomPlacement)
    return item.placement


def _input_cloud(tmp_path: Path) -> dict[str, object]:
    """輸入頁勾天雲的真入口：參考房喇叭中點 (1.0,1.9)、主位 (3.2,1.9) → 中心 (2.1,1.9)，底面 3.0−0.15＝2.85。"""
    return default_furniture("ceiling_cloud", json.loads(reference_project(tmp_path).model_dump_json()))


def _rule_by_hand(scheme: Scheme) -> tuple[float, float]:
    left, right = scheme.speakers["left"], scheme.speakers["right"]
    px, py, _ = scheme.receiver_set.primary.position_m
    return ((left.x + right.x) / 2 + px) / 2, ((left.y + right.y) / 2 + py) / 2


class Recording(SearchCompute):
    """照試算編號記下送算的方案並寫方案檔（細算讀它）；計分照搜尋替身。"""

    def __init__(self, store: SearchStore) -> None:
        super().__init__(store)
        self.schemes: dict[int | None, Scheme] = {}

    def __call__(self, jobs: Sequence[CandidateJob], workers: int) -> Iterator[ComputedCandidate]:
        self.schemes.update({job.trial_number: job.scheme for job in jobs})
        yield from super().__call__(jobs, workers)


def test_pinned_batch_recomputes_cloud_per_candidate_and_cloud_outside_room_is_illegal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cloud = _input_cloud(tmp_path)
    assert cloud["placement"] == {"bottom_center_m": [2.1, 1.9, 2.85], "yaw_deg": 90}
    # 試算 0（離前牆 0.5、聆聽 0.5）：中心 x＝0.75，深 1.8 m 順面向軸 → x＝[−0.15,1.65]，伸出前牆 0.15 m；
    # 天雲若留在 (2.1,1.9) 不會出房，這一列是跟著走才有的剔除，要落帳成不合法、不送算、搜尋不中斷。
    # 試算 1（1.0、1.5）：中心 (1.75,2.0)；試算 2（1.5、1.75）：中心 (2.375,2.0)。
    enqueue_hand_placements(monkeypatch, placements=((0.5, 0.5), (1.0, 1.5), (1.5, 1.75)))
    store, registry = make_store(tmp_path / "search", budget=3, batch=3, furniture=[cloud])
    compute = Recording(store)
    status = run(store, registry, compute)
    # 原方案先算、釘住比較身分，三個試算走已釘批次（run.py 的 batch）。
    assert status.baseline_outcome == "scored" and status.state == "budget_exhausted"
    recorded = {row.trial_number: row for row in rows(store)}
    assert recorded[0].outcome == "illegal" and recorded[0].reason == "furniture_placement_invalid"
    assert recorded[0].violation_m == pytest.approx(0.15)
    assert 0 not in compute.schemes
    assert recorded[1].outcome == recorded[2].outcome == "scored"
    assert _cloud(compute.schemes[None]).bottom_center_m == (2.1, 1.9, 2.85)  # 原方案照存檔位置。
    assert _cloud(compute.schemes[1]).bottom_center_m == pytest.approx((1.75, 2.0, 2.85), abs=1e-12, rel=0)
    assert _cloud(compute.schemes[2]).bottom_center_m == pytest.approx((2.375, 2.0, 2.85), abs=1e-12, rel=0)
    original, = store.project.furniture or ()
    for number in (1, 2):
        moved, = compute.schemes[number].furniture or ()
        assert moved.model_dump(exclude={"placement"}) == original.model_dump(exclude={"placement"})
        assert _cloud(compute.schemes[number]).yaw_deg == 90


def test_locked_unpinned_batch_moves_blocking_cloud_with_speakers(tmp_path: Path) -> None:
    from tests.engine._search_blocked_cases import blocker
    from tests.engine._seat_locked_cases import LOCKED

    # 原方案：方塊 (2.8,1.85) 擋住左喇叭到主位，原方案不算、這一批還沒釘身分，走整批到齊那條路（run.py 的 unpinned_batch）。
    # 試算 0：喇叭 (1,0.9)、(1,2.9)，主位鎖在 (3.2,1.9) → 方塊中心 (2.1,1.9)，盒 x=[2.0,2.2]、y=[1.75,2.05]；
    # 左喇叭到主位在 x=2.0～2.2 時 y≈1.35～1.45，不再擋，候選照常送算；方塊留在 (2.8,1.85) 的話會被擋。
    store, registry = make_store(tmp_path, budget=1, batch=1, layout_changes=LOCKED, furniture=[blocker()])
    compute = Recording(store)
    status = run(store, registry, compute)
    assert status.baseline_outcome == "direct_path_blocked"
    first, = rows(store)
    assert first.params_m == {"front_distance": 1.0, "spacing": 2.0, "listening_distance": 2.2}
    assert first.outcome == "scored" and list(compute.schemes) == [0] and status.comparison_trial == 0
    assert _cloud(compute.schemes[0]).bottom_center_m == pytest.approx((2.1, 1.9, 1.1), abs=1e-12, rel=0)
    assert compute.schemes[0].receiver_set == store.project.receiver_set


def test_refined_scheme_and_all_shifts_keep_selected_candidate_cloud(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    enqueue_hand_placements(monkeypatch, placements=((1.0, 1.5),))
    store, registry = make_store(tmp_path / "search", budget=1, batch=1, furniture=[_input_cloud(tmp_path)])
    run(store, registry, Recording(store))
    # 細算讀候選存的方案：天雲在候選算好的 (1.75,2.0)，不彈回專案存的 (2.1,1.9)。
    refined = refine_job(store, 0).scheme
    assert _cloud(refined).bottom_center_m == pytest.approx((1.75, 2.0, 2.85), abs=1e-12, rel=0)
    shifts = generate_shifts(store.project, store.settings, refined)
    assert len(shifts) == 12
    # 兩喇叭一起前後、座位前後左右移 2 公分時，照規則重算的中心會差 1 公分；喇叭往外往內兩支對稱、中點不變，
    # 耳高與聲學中心只改高度，這六個重算也不會動，所以只拿前六個證明「天雲沒跟」是看得出來的。
    detectable = {"speakers_forward", "speakers_backward", "seat_forward", "seat_backward", "seat_left", "seat_right"}
    assert detectable <= {shift.name for shift in shifts}
    for shift in shifts:
        moved = shift.scheme
        assert _cloud(moved) == _cloud(refined), shift.name
        if shift.name in detectable:
            follow = _rule_by_hand(moved)
            gap = max(abs(a - b) for a, b in zip(follow, _cloud(refined).bottom_center_m[:2], strict=True))
            assert gap == pytest.approx(PLACEMENT_SHIFT_M / 2, abs=1e-9), shift.name


def test_input_page_and_candidates_share_one_center_rule(tmp_path: Path) -> None:
    project = reference_project(tmp_path)
    cloud = _input_cloud(tmp_path)
    furnished = Scheme.model_validate(project.model_dump() | {"furniture": [cloud]})
    # 候選照抄專案的擺位：兩邊同一條規則、同一串浮點運算，天雲逐位等於輸入頁算的，整份方案只差代號。
    same = layout.Placement(project.speakers["left"], project.speakers["right"], Point(*project.receiver_set.primary.position_m),
                            tuple((p.receiver_id, Point(*p.position_m)) for p in project.receiver_set.points), (-1.0, 0.0))
    copied = layout.to_scheme(furnished, same, "copied")
    assert _cloud(copied).bottom_center_m == (2.1, 1.9, 2.85)
    assert copied.model_dump(exclude={"scheme_id"}) == furnished.model_dump(exclude={"scheme_id"})
    # 勾選後在輸入頁挪了喇叭：天雲照存檔不動（輸入頁行為），搜尋候選照新擺位重算水平中心、高度照存檔。
    moved = Scheme.model_validate(furnished.model_dump() | {"speakers": {
        "left": {"x": 0.6, "y": 1.3, "z": 1.2}, "right": {"x": 0.6, "y": 2.5, "z": 1.2}}})
    assert _cloud(moved).bottom_center_m == (2.1, 1.9, 2.85)
    shifted = layout.Placement(moved.speakers["left"], moved.speakers["right"], same.primary, same.receivers, (-1.0, 0.0))
    assert _cloud(layout.to_scheme(moved, shifted, "moved")).bottom_center_m == pytest.approx((1.9, 1.9, 2.85), abs=1e-12, rel=0)
    assert _cloud(layout.to_scheme(moved, shifted, "kept", cloud_follows=False)).bottom_center_m == (2.1, 1.9, 2.85)


# 改動前主線（ad8b520f）照同一建法算的方案雜湊；沒放天雲的候選必須逐位不變。
_MAIN_DIGESTS = {
    ("desk_only", (1.0, 2.0, 2.0)): "f7dd902c8f7d3114450fb0875efe6e59219601460f1e80a4fc7e57cbe31bba68",
    ("desk_only", (1.5, 2.4, 2.5)): "e4449c860f49ef170f38a7c280d27b201640489e0fbcc81c915b84f46dec5608",
    ("no_furniture", (1.0, 2.0, 2.0)): "33564ed506714cc0b534ca518e95e957f0fbdecc89b4a1056f458349f8295ed7",
    ("no_furniture", (1.5, 2.4, 2.5)): "c619d3341985e068db8ce94a2823540fd0632c8259780ec81db0b6568ab81c19",
}


@pytest.mark.parametrize("name,params", list(_MAIN_DIGESTS))
def test_candidates_without_cloud_are_bit_identical_to_main(name: str, params: tuple[float, float, float]) -> None:
    keep = ("table",) if name == "desk_only" else ()
    full = hand_scheme("west")
    items = [item.model_dump() for item in full.furniture or () if item.furniture_id in keep]
    project = Scheme.model_validate(full.model_dump() | {"furniture": items or None})
    placement = layout.place(project, hand_settings(project).layout, layout.LayoutParams(*params))
    candidate = layout.to_scheme(project, placement, "control")
    assert hashlib.sha256(candidate.model_dump_json().encode()).hexdigest() == _MAIN_DIGESTS[name, params]
    assert candidate == layout.to_scheme(project, placement, "control", cloud_follows=False)
