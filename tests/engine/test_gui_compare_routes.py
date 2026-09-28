"""比較端點的成功與拒收路徑。"""
from __future__ import annotations

import json
from http import HTTPStatus
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from aosr.config.paths import config_path
from aosr.gui.app import GuiSettings, create_app
from aosr.reporting.result import SchemeResult, reevaluate, save_result
from tests.engine.test_scheme_pipeline import shared_control_result


@pytest.fixture(scope="module")
def pair(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> tuple[SchemeResult, SchemeResult]:
    return (shared_control_result(tmp_path_factory, worker_id, "wall-1"),
            shared_control_result(tmp_path_factory, worker_id, "wall-2"))


def _files(tmp_path: Path, result: SchemeResult, run_id: str) -> None:
    (tmp_path / "schemes" / f"{result.scheme.scheme_id}.json").write_text(result.scheme.model_dump_json())
    save_result(result, tmp_path / "results" / f"{run_id}.json")
    (tmp_path / "runs" / f"{run_id}.json").write_text(json.dumps({
        "run_id": run_id, "scheme_id": result.scheme.scheme_id,
        "status": "done", "started_at": 0, "pid": 0, "exit_code": 0,
        "result_path": str(tmp_path / "results" / f"{run_id}.json"),
        "stderr_path": str(tmp_path / "runs" / f"{run_id}.stderr"),
    }))


def _client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(GuiSettings(engine_commit="a" * 40, data_dir=tmp_path)),
                      base_url="http://localhost", raise_server_exceptions=False)


def _strings(value: object, field: str = "") -> bool:
    if isinstance(value, dict):
        return all(isinstance(key, str) and _strings(item, key) for key, item in value.items())
    if isinstance(value, list):
        return all(_strings(item, field) for item in value)
    if field.endswith("_text") or field in {"run_id", "scheme_id", "path", "label", "role",
                                              "speaker_id", "receiver_id", "receiver_role", "key",
                                              "category", "side", "run_date"}:
        return isinstance(value, str)
    return True


def _nulls_only_in_levels(value: object, field: str = "") -> bool:
    if isinstance(value, dict):
        return all(_nulls_only_in_levels(item, key) for key, item in value.items())
    if isinstance(value, list):
        return all(_nulls_only_in_levels(item, field) for item in value)
    return value is not None or field == "levels_db"


def test_compare_returns_both_sides(tmp_path: Path, pair: tuple[SchemeResult, SchemeResult]) -> None:
    a_id, b_id = "b" * 32, "c" * 32
    with _client(tmp_path) as client:
        _files(tmp_path, pair[0], a_id)
        _files(tmp_path, pair[1], b_id)
        response = client.get(f"/api/compare/{a_id}/{b_id}")
    assert response.status_code == HTTPStatus.OK
    data = response.json()
    assert _strings(data)
    assert _nulls_only_in_levels(data)
    assert data["a"]["scheme_id"] == "wall-1" and data["b"]["scheme_id"] == "wall-2"
    selected = [next(row for row in data["overlay"]["series"] if row["key"] == key)
                for key in data["overlay"]["default_keys"]]
    assert {(row["role"], row["receiver_role"]) for row in selected} == {("left", "primary")}
    assert "server-timing" in response.headers


def test_same_scheme_different_engine_lists_both_problems(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult]) -> None:
    a_id, b_id = "b" * 32, "c" * 32
    altered = pair[0].model_copy(update={"engine_commit": "e53bfae" + "f" * 33})
    altered = altered.model_copy(update={
        "candidate": reevaluate(altered, quality_targets_path=config_path("quality_targets.toml"))})
    with _client(tmp_path) as client:
        _files(tmp_path, pair[0], a_id)
        _files(tmp_path, altered, b_id)
        response = client.get(f"/api/compare/{a_id}/{b_id}")
    assert response.status_code == HTTPStatus.CONFLICT
    assert any("候選代號重複" in item for item in response.json()["problems"])
    assert any("engine_commit" in item for item in response.json()["problems"])
    assert set(response.json()["rerun_urls"]) == {"a", "b"}


def test_one_side_rejected_names_side_and_reason(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult]) -> None:
    a_id, b_id = "b" * 32, "c" * 32
    with _client(tmp_path) as client:
        _files(tmp_path, pair[0], a_id)
        _files(tmp_path, pair[1], b_id)
        path = tmp_path / "results" / f"{b_id}.json"
        document = json.loads(path.read_text())
        document["engine_commit"] = "forged"
        path.write_text(json.dumps(document))
        response = client.get(f"/api/compare/{a_id}/{b_id}")
    assert response.status_code == HTTPStatus.CONFLICT
    assert response.json()["side"] == "b" and response.json()["reason"]


def test_bad_missing_or_same_run_id(tmp_path: Path, pair: tuple[SchemeResult, SchemeResult]) -> None:
    a_id, b_id = "b" * 32, "c" * 32
    with _client(tmp_path) as client:
        _files(tmp_path, pair[0], a_id)
        assert client.get(f"/api/compare/invalid/{a_id}").status_code == HTTPStatus.BAD_REQUEST
        assert client.get(f"/api/compare/{a_id}/invalid").status_code == HTTPStatus.BAD_REQUEST
        assert client.get(f"/api/compare/{a_id}/{a_id}").status_code == HTTPStatus.CONFLICT
        missing_b = client.get(f"/api/compare/{a_id}/{b_id}")
        missing_a = client.get(f"/api/compare/{b_id}/{a_id}")
    assert missing_b.status_code == missing_a.status_code == HTTPStatus.NOT_FOUND
    assert "B" in str(missing_b.json()) and "A" in str(missing_a.json())
