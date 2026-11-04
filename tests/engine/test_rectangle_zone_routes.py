"""矩形方案的未支援路徑明確拒收；只考 Python，不開瀏覽器。"""
from datetime import date
from pathlib import Path

import pytest

from aosr.geometry.shoebox import Wall
from aosr.gui.problem_text import checked_scheme
from aosr.gradient import endpoint
from aosr.materials.construction_models import ConstantImpedance
from aosr.physics import report_io, report_surfaces
from aosr.physics.report_measured import measured_seams
from aosr.physics.report_measured_text import construction_note_lines, measured_text
from aosr.reporting.modal_lookup import key_from_scheme, scheme_scope_reason
from aosr.reporting.pipeline import run_scheme
from aosr.reporting.scheme import Scheme
from aosr.reporting.stage_one_result import check_stage_one_prerequisites
from tests.engine._directivity import DIRECTIVITY
from tests.engine._furniture_cases import CAPABILITIES, pair
from tests.engine._surface_zone_cases import rectangle_document


@pytest.mark.parametrize("entry", ["scheme", "gradient-endpoint", "web-load", "seams", "text", "notes"])
def test_calculating_and_display_entries_refuse_rectangles(tmp_path: Path, entry: str) -> None:
    """抓任一路靜默取第一區；終點換材料也不能把分區覆蓋掉。"""
    scheme = Scheme.model_validate(rectangle_document())
    inputs = report_io.load_input_document(pair(scheme), CAPABILITIES, DIRECTIVITY)
    with pytest.raises(report_surfaces.RectangleZonesUnsupported, match=r"ceiling.*這一段還不支援矩形分區（#561 之後幾步）"):
        if entry == "scheme":
            run_scheme(scheme, capabilities=CAPABILITIES, directivity=DIRECTIVITY,
                       quality_targets_path=Path("src/aosr/config/data/quality_targets.toml"), engine_commit="unused",
                       program_fingerprint="unused", physics_identity="unused", run_date=date(2026, 11, 4))
        elif entry == "gradient-endpoint":
            endpoint.scene_with_materials(scheme.scene, {wall: ConstantImpedance(impedance_pa_s_m=1000.0) for wall in Wall.all()})
        elif entry == "web-load":
            checked_scheme(rectangle_document())
        elif entry == "seams":
            measured_seams(inputs.surface_zones_by_wall, density_kg_m3=1.2, sound_speed_m_s=343.0)
        elif entry == "text":
            measured_text(inputs)
        else:
            construction_note_lines(inputs, {})


def test_existing_whole_construction_guards_also_refuse_rectangles() -> None:
    """模態與第一階段原本整份拒收構造，保持原界線。"""
    scheme = Scheme.model_validate(rectangle_document())
    assert scheme_scope_reason(scheme) == "construction_walls"
    assert key_from_scheme(scheme) is None
    scheme = scheme.model_copy(update={"source_model": "omnidirectional", "furniture": None})
    with pytest.raises(ValueError, match="第一階段不收牆面分區"):
        check_stage_one_prerequisites(scheme)


def test_web_python_load_refuses_rectangle_file(tmp_path: Path) -> None:
    """讀磁碟上的矩形方案必須先拒收，不能送到只懂第一區的腳本。"""
    from aosr.gui.app import _read_scheme

    path = tmp_path / "rectangles.json"
    path.write_text(Scheme.model_validate(rectangle_document()).model_dump_json())
    with pytest.raises(report_surfaces.RectangleZonesUnsupported, match="ceiling.*這一段還不支援矩形分區"):
        _read_scheme(path)


def test_search_refuses_rectangles_before_creating_folder(tmp_path: Path) -> None:
    """搜尋建立時就拒收，不能先寫成等待計算的候選。"""
    from tests.engine.test_scheme_furniture import _store

    with pytest.raises(report_surfaces.RectangleZonesUnsupported, match="ceiling.*這一段還不支援矩形分區"):
        _store(tmp_path / "search", Scheme.model_validate(rectangle_document()))
    assert not (tmp_path / "search").exists()


@pytest.mark.parametrize("entry", ["preview", "edit", "validate"])
def test_web_form_python_entry_refuses_rectangles(entry: str) -> None:
    """表單載入／編輯／驗證都必須拒收，避免把矩形交給舊腳本。"""
    from aosr.gui.construction_input import web_scheme_problems
    from aosr.gui.input_setup import InputEdit, InputShapeError, edit_input, input_preview

    document = rectangle_document()
    if entry == "validate":
        problems = web_scheme_problems(document, capabilities=CAPABILITIES, directivity=DIRECTIVITY)
        assert len(problems) == 1
        assert problems[0].message == "牆面 ceiling：這一段還不支援矩形分區（#561 之後幾步）"
    else:
        with pytest.raises(InputShapeError, match="這一段還不支援矩形分區（#561 之後幾步）"):
            if entry == "preview":
                input_preview(document, CAPABILITIES, DIRECTIVITY)
            else:
                edit_input(InputEdit(scheme=document, action="shortcut", value="listening"), CAPABILITIES, DIRECTIVITY)


async def _web_requests(tmp_path: Path) -> None:
    """同一事件迴圈內使用正式 ASGI（網頁呼叫介面），不開 socket 或瀏覽器。"""
    import httpx
    from aosr.gui.app import GuiSettings, create_app

    app = create_app(GuiSettings(engine_commit="a" * 40, data_dir=tmp_path))
    document = rectangle_document()
    (tmp_path / "schemes" / "rectangles.json").write_text(Scheme.model_validate(document).model_dump_json())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost") as client:
        responses = [await client.get("/api/schemes/rectangles"),
                     await client.post("/api/input-preview", json=document),
                     await client.post("/api/input-edit", json={"scheme": document, "action": "shortcut", "value": "listening"}),
                     await client.post("/api/validate", json=document),
                     await client.put("/api/schemes/new", json=document | {"scheme_id": "new"})]
        for response in responses:
            assert response.status_code in (400, 422), response.text
            assert "這一段還不支援矩形分區（#561 之後幾步）" in response.text, response.text
            assert "ceiling" in response.text or "天花" in response.text
    assert not (tmp_path / "schemes" / "new.json").exists()


def test_web_requests_refuse_rectangles_without_browser(tmp_path: Path) -> None:
    """正式載入、預覽、編輯、驗證及保存端點都回白話錯誤且不保存。"""
    import asyncio

    asyncio.run(_web_requests(tmp_path))
