"""真家具方案共用樣本；直接經正式方案入口計算。"""
from __future__ import annotations

import pytest

from aosr.reporting.result import SchemeResult
from aosr.reporting.scheme import Scheme
from tests.engine import _furniture_cases as schemes
from tests.engine._scheme_cache import shared_json
from tests.engine.test_scheme_pipeline import _run_control


def _scheme_result(furnished: bool, furniture_id: str = "seat") -> str:
    content = schemes.document(schemes.relative_item(furniture_id=furniture_id)) if furnished else schemes.document()
    suffix = "" if furniture_id == "seat" else f"-{furniture_id}"
    content["scheme_id"] = f"step6-furnished{suffix}" if furnished else "step6-plain"
    scheme = Scheme.model_validate(content)
    return _run_sample(scheme)


def _run_sample(scheme: Scheme) -> str:
    return _run_control(scheme).model_dump_json()


def shared_scheme_result(tmp_path_factory: pytest.TempPathFactory, worker_id: str,
                         furnished: bool, furniture_id: str = "seat") -> SchemeResult:
    suffix = "" if furniture_id == "seat" else f"-{furniture_id}"
    return SchemeResult.model_validate_json(shared_json(tmp_path_factory, worker_id,
        f"step6-scheme-{furnished}{suffix}", lambda: _scheme_result(furnished, furniture_id)))


def shared_scheme_pair(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> tuple[SchemeResult, SchemeResult]:
    return (shared_scheme_result(tmp_path_factory, worker_id, False),
            shared_scheme_result(tmp_path_factory, worker_id, True))


def _twodesks_k1_result() -> str:
    """審查 build_results2.py 的 twodesks-k1 配方：兩張書桌、木質天雲，有補算路徑。"""
    def desk(identifier: str, left: float, material: str) -> dict[str, object]:
        return schemes.relative_item(furniture_id=identifier, kind="desk", material=material,
            width_m=0.8, depth_m=0.6, height_m=0.05,
            placement={"forward_m": 1.0, "left_m": left, "bottom_height_m": 0.7, "yaw_deg": 0})
    cloud = schemes.cloud_item(width_m=2.0, depth_m=1.0,
        placement={"bottom_center_m": [3.0, 4.0, 2.5], "yaw_deg": 0})
    content = schemes.document(desk("desk-l", 0.5, "wood"), desk("desk-r", -0.5, "glass"), cloud)
    content["scheme_id"] = "twodesks-k1"
    scene = content["scene"]
    assert isinstance(scene, dict)
    scene["reflection_order_k"] = 1
    return _run_sample(Scheme.model_validate(content))


def shared_twodesks_k1_result(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> SchemeResult:
    return SchemeResult.model_validate_json(shared_json(tmp_path_factory, worker_id,
        "step7-twodesks-k1", _twodesks_k1_result))


def shared_moved_furnished_result(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> SchemeResult:
    """兩份都有家具、主位抬高而周圍點不動，僅座位相對佈局的比較身分不同。"""
    def produce() -> str:
        content = schemes.document(schemes.relative_item(), primary=(3.0, 3.0, 1.3))
        content["scheme_id"] = "moved-furnished"
        return _run_sample(Scheme.model_validate(content))
    return SchemeResult.model_validate_json(shared_json(tmp_path_factory, worker_id,
        "step7-moved-furnished", produce))


@pytest.fixture(scope="module")
def twodesks_k1_result(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> SchemeResult:
    return shared_twodesks_k1_result(tmp_path_factory, worker_id)


@pytest.fixture(scope="module")
def scheme_pair(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> tuple[SchemeResult, SchemeResult]:
    return shared_scheme_pair(tmp_path_factory, worker_id)
