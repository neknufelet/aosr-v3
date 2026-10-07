"""近似窗保留牆面覆蓋證明；補算列仍只收牆面。"""
from __future__ import annotations

import pytest

from aosr.physics.reflection_window import ReflectionWindow
from aosr.physics.report_io import PathRow
from aosr.physics.room_paths import SUPPORTED_MAX_ORDER
from tests.engine import test_reflections as fixtures


@pytest.mark.parametrize("change, message", [
    ({"coverage": "approximate"}, "家具"),
    ({"coverage": "approximate", "furniture_ids": ("desk",),
      "computed_order_k": SUPPORTED_MAX_ORDER, "validation": "unvalidated",
      "next_uncomputed_earliest_relative_s": None}, "窗外證明"),
    ({"furniture_ids": ("desk",)}, "家具"),
])
def test_approximate_window_rejects_invalid_furniture_or_wall_proof(
    change: dict[str, object], message: str,
) -> None:
    window = fixtures._pair()[0].window
    assert window is not None
    document = window.model_dump(mode="python")
    document.update(change)
    with pytest.raises(ValueError, match=message):
        ReflectionWindow.model_validate(document)


@pytest.mark.parametrize("coverage", ["approximate", "not_provable"])
def test_furniture_window_accepts_wall_proof_and_unprovable_state(coverage: str) -> None:
    window = fixtures._pair()[0].window
    assert window is not None
    document = window.model_dump(mode="python")
    document.update(coverage=coverage, furniture_ids=("desk",))
    if coverage == "not_provable":
        document.update(computed_order_k=SUPPORTED_MAX_ORDER, validation="unvalidated",
                        next_uncomputed_earliest_relative_s=None)
    accepted = ReflectionWindow.model_validate(document)
    assert accepted.coverage == coverage
    assert accepted.furniture_ids == ("desk",)
    assert ReflectionWindow.model_validate(accepted.model_dump(mode="json")) == accepted


def test_window_rejects_furniture_rows_explicitly() -> None:
    record = fixtures._pair()[0]
    assert record.window is not None and record.report.path_table is not None
    row = next(row for row in record.report.path_table.rows if row.order > 0)
    document = row.model_dump(mode="python")
    document.update(order=1, wall_sequence=("furniture",), furniture_id="desk",
                    furniture_face="top", reflection_point_m=(2.0, 2.0, 0.8))
    furniture_row = PathRow.model_validate(document)
    with pytest.raises(ValueError, match="只收牆面"):
        ReflectionWindow.model_validate({**record.window.model_dump(mode="python"), "rows": (furniture_row,)})


def test_no_furniture_window_omits_optional_header() -> None:
    window = fixtures._pair()[0].window
    assert window is not None
    assert "furniture_ids" not in window.model_dump(mode="json")


@pytest.mark.parametrize("ids", [(), ("b", "a"), ("desk", "desk"), (" ",)])
@pytest.mark.parametrize("coverage", ["approximate", "not_provable"])
def test_window_rejects_malformed_furniture_ids(ids: tuple[str, ...], coverage: str) -> None:
    window = fixtures._pair()[0].window
    assert window is not None
    document = window.model_dump(mode="python")
    document.update(coverage=coverage, furniture_ids=ids)
    if coverage == "not_provable":
        document.update(computed_order_k=SUPPORTED_MAX_ORDER, validation="unvalidated",
                        next_uncomputed_earliest_relative_s=None)
    with pytest.raises(ValueError, match="家具代號清單必須非空、唯一且按代號排序"):
        ReflectionWindow.model_validate(document)
