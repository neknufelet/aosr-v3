"""座位鎖定專用設定；原三維夾具與答案不動。"""
from pathlib import Path

from aosr.search.settings import SearchSettings
from aosr.search.store import SearchStore
from tests.engine._search_store_cases import settings_document


LOCKED: dict[str, object] = {"seat_locked": True, "axis_offset_m": -0.1, "ear_height_m": 1.2,
          "listening_distance_m": None}
SEAT_LINE = "座位：鎖定在原方案主位，只搜離前牆與間距，聆聽距離由座位推出"


def locked_settings(**changes: object) -> SearchSettings:
    document = settings_document()
    original = SearchSettings.model_validate(document)
    return SearchSettings.model_validate(document | {"layout": original.layout.model_dump() | LOCKED | changes})


def locked_store(tmp_path: Path, *, workers: int = 1, budget: int = 17,
                 convergence: int = 100, feedback: dict[str, float] | None = None) -> tuple[SearchStore, Path]:
    from tests.engine._search_run_cases import make_store

    return make_store(tmp_path, layout_changes=LOCKED, workers=workers, budget=budget,
                      convergence=convergence, feedback=feedback)
