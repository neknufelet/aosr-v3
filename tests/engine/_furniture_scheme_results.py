"""真家具方案共用樣本；第八步拆方案施工關時只需更換這一處。"""
from __future__ import annotations

import pytest

from aosr.reporting import validation
from aosr.reporting.result import SchemeResult
from aosr.reporting.scheme import Scheme
from tests.engine import _furniture_cases as schemes
from tests.engine._scheme_cache import shared_json
from tests.engine.test_scheme_pipeline import _run_control


def _open_only_construction_gate(patch: pytest.MonkeyPatch, swallowed: list[str]) -> None:
    original = validation._checked_furniture

    def checked(scheme: Scheme) -> None:
        try:
            original(scheme)
        except validation.SchemeValidationError as exc:
            if exc.problems != (validation.SchemeProblem("furniture", schemes.GATE_MESSAGE),):
                raise
            swallowed.append(scheme.scheme_id)

    patch.setattr(validation, "_checked_furniture", checked)


def _scheme_result(furnished: bool, furniture_id: str = "seat") -> str:
    content = schemes.document(schemes.relative_item(furniture_id=furniture_id)) if furnished else schemes.document()
    suffix = "" if furniture_id == "seat" else f"-{furniture_id}"
    content["scheme_id"] = f"step6-furnished{suffix}" if furnished else "step6-plain"
    scheme = Scheme.model_validate(content)
    swallowed: list[str] = []
    with pytest.MonkeyPatch.context() as patch:
        _open_only_construction_gate(patch, swallowed)
        result = _run_control(scheme)
    assert swallowed == ([scheme.scheme_id] if furnished else []), "施工關必須真的被吞過一次，拆關後此題要紅"
    return result.model_dump_json()


def shared_scheme_result(tmp_path_factory: pytest.TempPathFactory, worker_id: str,
                         furnished: bool, furniture_id: str = "seat") -> SchemeResult:
    suffix = "" if furniture_id == "seat" else f"-{furniture_id}"
    return SchemeResult.model_validate_json(shared_json(tmp_path_factory, worker_id,
        f"step6-scheme-{furnished}{suffix}", lambda: _scheme_result(furnished, furniture_id)))


def shared_scheme_pair(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> tuple[SchemeResult, SchemeResult]:
    return (shared_scheme_result(tmp_path_factory, worker_id, False),
            shared_scheme_result(tmp_path_factory, worker_id, True))


@pytest.fixture(scope="module")
def scheme_pair(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> tuple[SchemeResult, SchemeResult]:
    return shared_scheme_pair(tmp_path_factory, worker_id)
