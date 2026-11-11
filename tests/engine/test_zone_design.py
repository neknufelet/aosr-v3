"""設計設定（#1083）：每一區三種做法，跟著方案存、讀、封存、還原；對不上的寫入拒收、讀出照實列。"""
from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import cast

import httpx
import pytest
from starlette.requests import Request
from starlette.types import Message

from aosr.gui.app import GuiSettings, create_app
from aosr.gui.jobs import JobManager, ResultMoveConflict
from aosr.gui.scheme_archive import ArchivePackage, SchemeArchive
from aosr.gui.zone_design import DesignHandlers
from aosr.reporting.scheme import Scheme
from tests.engine._furniture_cases import reference_document
from tests.engine._surface_zone_cases import NEW, rectangle_document

NAME = "zoned"
Scenario = Callable[[httpx.AsyncClient], Awaitable[None]]


def _write_scheme(data_dir: Path, document: dict[str, object]) -> None:
    path = data_dir / "schemes" / f"{NAME}.json"
    path.write_text(Scheme.model_validate(document | {"scheme_id": NAME}).model_dump_json())


def _design(zones: dict[str, dict[str, dict[str, str]]], scheme_id: str = NAME) -> dict[str, object]:
    return {"schema_version": "aosr.zone_design.v1", "scheme_id": scheme_id, "zones": zones}


def _run(tmp_path: Path, scenario: Scenario) -> None:
    async def go() -> None:
        app = create_app(GuiSettings(engine_commit="a" * 40, data_dir=tmp_path))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost") as client:
            await scenario(client)
    asyncio.run(go())


def test_unlisted_zones_are_fixed_and_a_saved_last_zone_comes_back(tmp_path: Path) -> None:
    """沒存過：每一區都是固定；存了最後一區交給搜尋、地板（整面一區）保留原牆，讀回只有那兩格變。"""
    _run(tmp_path, lambda client: _saved_last_zone(tmp_path, client))


async def _saved_last_zone(tmp_path: Path, client: httpx.AsyncClient) -> None:
    _write_scheme(tmp_path, rectangle_document())
    first = (await client.get(f"/api/designs/{NAME}")).json()
    assert first["modes"]["ceiling"] == {"r0_c0": "fixed", "r0_c1": "fixed", "r1_c0": "fixed", "r1_c1": "fixed"}
    assert first["modes"]["floor"] == {"whole_wall": "fixed"} and first["problems"] == []
    saved = await client.put(f"/api/designs/{NAME}", json=_design(
        {"ceiling": {"r1_c1": {"mode": "search"}}, "floor": {"whole_wall": {"mode": "keep_original"}}}))
    assert saved.status_code == 200, saved.text
    again = (await client.get(f"/api/designs/{NAME}")).json()
    assert again["modes"]["ceiling"] == {"r0_c0": "fixed", "r0_c1": "fixed", "r1_c0": "fixed", "r1_c1": "search"}
    assert again["modes"]["floor"] == {"whole_wall": "keep_original"}
    assert again["modes"]["x0"]["r1_c1"] == "fixed" and again["problems"] == []


def test_writes_that_do_not_match_the_scheme_are_refused_and_not_saved(tmp_path: Path) -> None:
    _run(tmp_path, lambda client: _refused_writes(tmp_path, client))


async def _refused_writes(tmp_path: Path, client: httpx.AsyncClient) -> None:
    missing = await client.put(f"/api/designs/{NAME}", json=_design({}))
    assert missing.status_code == 404 and "要跟著方案存" in missing.text
    bad_name = await client.get("/api/designs/-zoned")
    assert bad_name.status_code == 400 and "方案代號只收" in bad_name.text
    _write_scheme(tmp_path, rectangle_document())
    cases = [(_design({"ceilling": {"r0_c0": {"mode": "search"}}}), "牆 ceilling 方案裡沒有"),
             (_design({"ceiling": {"r9_c9": {"mode": "search"}}}), "ceiling 分區 r9_c9 方案裡沒有"),
             (_design({"ceiling": {"r1_c1": {"mode": "guess"}}}), "格式不對"),
             (_design({}, scheme_id="other"), "跟網址不同")]
    for body, words in cases:
        response = await client.put(f"/api/designs/{NAME}", json=body)
        assert response.status_code == 400 and words in response.text, response.text
    assert not (tmp_path / "designs" / f"{NAME}.json").exists()


def test_six_number_scheme_has_six_whole_walls(tmp_path: Path) -> None:
    """六面整面固定阻抗的方案存檔會折回六個數字：六面各算整面一區，照樣存得了做法。"""
    _run(tmp_path, lambda client: _six_numbers(tmp_path, client))


async def _six_numbers(tmp_path: Path, client: httpx.AsyncClient) -> None:
    _write_scheme(tmp_path, reference_document())
    assert Scheme.model_validate(reference_document()).scene.surface_zones_by_wall is None
    modes = (await client.get(f"/api/designs/{NAME}")).json()["modes"]
    assert modes == {wall: {"whole_wall": "fixed"} for wall in ("x0", "xL", "y0", "yL", "floor", "ceiling")}
    saved = await client.put(f"/api/designs/{NAME}", json=_design({"floor": {"whole_wall": {"mode": "search"}}}))
    assert saved.status_code == 200, saved.text
    assert saved.json()["modes"]["floor"] == {"whole_wall": "search"}


def test_read_reports_zones_the_scheme_no_longer_has(tmp_path: Path) -> None:
    """存完設定後方案的最後一區改了代號：讀出照實列那一條，改名後那一區照預設固定。"""
    _run(tmp_path, lambda client: _stale_read(tmp_path, client))


async def _stale_read(tmp_path: Path, client: httpx.AsyncClient) -> None:
    _write_scheme(tmp_path, rectangle_document())
    assert (await client.put(f"/api/designs/{NAME}", json=_design({"ceiling": {"r1_c1": {"mode": "search"}}}))).status_code == 200
    document = rectangle_document()
    zones = cast(dict[str, list[dict[str, object]]], cast(dict[str, object], document["scene"])[NEW])
    zones["ceiling"][-1]["id"] = "renamed"
    _write_scheme(tmp_path, document)
    read = (await client.get(f"/api/designs/{NAME}")).json()
    assert read["problems"] == ["設計設定的 ceiling 分區 r1_c1 方案裡沒有"]
    assert read["modes"]["ceiling"]["renamed"] == "fixed"


def test_design_moves_with_archive_and_restore(tmp_path: Path) -> None:
    _run(tmp_path, lambda client: _archive_round_trip(tmp_path, client))


async def _archive_round_trip(tmp_path: Path, client: httpx.AsyncClient) -> None:
    _write_scheme(tmp_path, rectangle_document())
    body = _design({"ceiling": {"r1_c1": {"mode": "search"}}})
    assert (await client.put(f"/api/designs/{NAME}", json=body)).status_code == 200
    design = tmp_path / "designs" / f"{NAME}.json"
    saved = design.read_bytes()
    preview = (await client.get(f"/api/schemes/{NAME}/archive-preview")).json()
    assert f"designs/{NAME}.json" in preview["paths"]
    archived = await client.post(f"/api/schemes/{NAME}/archive", json={})
    assert archived.status_code == 200, archived.text
    assert not design.exists()
    assert (await client.get(f"/api/designs/{NAME}")).status_code == 404
    restored = await client.post(f"/api/archive/packages/{archived.json()['package_id']}/restore", json={})
    assert restored.status_code == 200, restored.text
    assert design.read_bytes() == saved
    again = await client.post(f"/api/schemes/{NAME}/archive", json={})
    package = again.json()["package_id"]
    assert (tmp_path / "archive" / "packages" / package / "designs" / f"{NAME}.json").read_bytes() == saved
    deleted = await client.post(f"/api/archive/packages/{package}/delete", json={})
    assert deleted.status_code == 200, deleted.text
    assert not (tmp_path / "archive" / "packages" / package).exists() and not design.exists()


def test_packages_made_before_designs_existed_still_validate() -> None:
    old = ArchivePackage.model_validate({
        "package_id": "a" * 32, "scheme_id": NAME, "archived_at": "2026-11-01T00:00:00+08:00", "status": "complete",
        "result_ids": [], "artifact_ids": [], "paths": [f"schemes/{NAME}.json"]})
    assert old.paths == [f"schemes/{NAME}.json"]


def _archive_tools(root: Path) -> tuple[JobManager, SchemeArchive]:
    for name in ("schemes", "designs"):
        (root / name).mkdir()
    _write_scheme(root, rectangle_document())
    jobs = JobManager(root, (), "a" * 40, root / "unused")
    return jobs, SchemeArchive(jobs)


def test_write_racing_an_archive_leaves_no_orphan(tmp_path: Path) -> None:
    """審查第 B 條：還在收請求內容時方案被封存走，寫入要拒收、不留包外的孤立設計檔。"""
    jobs, archive = _archive_tools(tmp_path)
    body = json.dumps(_design({"ceiling": {"r1_c1": {"mode": "search"}}})).encode()

    async def receive() -> Message:
        archive.archive(NAME, [])
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request({"type": "http", "method": "PUT", "path": f"/api/designs/{NAME}", "headers": [],
                       "path_params": {"name": NAME}}, receive)
    response = asyncio.run(DesignHandlers(tmp_path, jobs._lock).item(request))
    assert response.status_code == 404
    assert not (tmp_path / "designs" / f"{NAME}.json").exists()


def test_old_package_restores_and_a_same_name_design_blocks_restore(tmp_path: Path) -> None:
    """審查第 C 條：沒有設計設定的舊包照樣還原；原位已有同名設計檔時拒絕還原，兩邊原樣留著。"""
    _jobs, archive = _archive_tools(tmp_path)
    scheme = tmp_path / "schemes" / f"{NAME}.json"
    original = scheme.read_bytes()
    package = archive.archive(NAME, [])
    assert package.paths == [f"schemes/{NAME}.json"]
    archive.restore(package.package_id)
    assert scheme.read_bytes() == original
    package = archive.archive(NAME, [])
    design = tmp_path / "designs" / f"{NAME}.json"
    design.write_text(json.dumps(_design({})))
    saved = design.read_bytes()
    with pytest.raises(ResultMoveConflict):
        archive.restore(package.package_id)
    assert not scheme.exists() and design.read_bytes() == saved
    assert (archive.packages_dir / package.package_id / "schemes" / f"{NAME}.json").read_bytes() == original
