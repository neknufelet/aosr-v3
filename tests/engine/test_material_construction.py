"""#1243：構造完整存回材料庫，各種背後與放置方式由條目明寫。"""
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError
from starlette.applications import Starlette
from starlette.testclient import TestClient

from tests.engine._construction_cases import construction
from tests.engine._bonded_backing_cases import BONDED_VPR
from aosr.gui.material_editor import material_routes
from aosr.gui.material_names import registered_entry
from aosr.materials import library, search_library
from aosr.materials.construction_models import Assembly


_ENTRY: TypeAdapter[library.LibraryEntry] = TypeAdapter(library.LibraryEntry)


@pytest.mark.parametrize(("place", "surfaces"), [
    ("wall", ["x0", "xL", "y0", "yL", "ceiling"]), ("floor", ["floor"]), ("ceiling_build", ["ceiling"]), ("door_window", ["x0", "xL", "y0", "yL"]),
])
def test_construction_saved_and_listed_with_its_explicit_placement(tmp_path: Path, place: str, surfaces: list[str]) -> None:
    doc = construction(placement=place, product=True)
    with TestClient(Starlette(routes=material_routes(tmp_path))) as client:
        saved = client.put(f"/api/materials/{doc['id']}", json=doc)
        assert saved.status_code == 200, saved.text
    with TestClient(Starlette(routes=material_routes(tmp_path))) as reopened:
        listing = reopened.get("/api/materials").json()
        row = next(row for row in listing["materials"] if row["entry"]["id"] == doc["id"])
        assert row["entry"] == doc
        assert row["placement"]["group"] == place and row["placement"]["surfaces"] == surfaces
        assert listing["user_file_problem"] is None
        assert listing["entry_kind_labels"]["construction"] == "構造"


@pytest.mark.parametrize("backing", ["rigid", "air", "zone_wall"])
def test_fixed_surface_requires_the_wall_only_for_zone_wall(backing: str) -> None:
    doc = construction(backing)
    entry = _ENTRY.validate_python(doc)
    expected = None if backing == "zone_wall" else Assembly.model_validate({key: doc[key] for key in ("layers", "backing")})
    assert library.fixed_surface(entry) == expected


@pytest.mark.parametrize(("field", "bad"), [
    ("layers", []), ("layers", [{"kind": "air", "thickness_m": -0.01}]), ("backing", "other"),
    ("product", 1), ("placement", "guess"), ("variables", []),
])
def test_invalid_construction_rejected_at_its_field(field: str, bad: object) -> None:
    with pytest.raises(ValidationError) as error:
        _ENTRY.validate_python(construction() | {field: bad})
    assert any(field in item["loc"] for item in error.value.errors())


@pytest.mark.parametrize("backing", ["rigid", "air"])
def test_search_converts_only_requested_constructions_to_identical_v2_fixed_entries(backing: str) -> None:
    doc = construction(backing)
    entry = _ENTRY.validate_python(doc)
    legacy = library.FixedEntry.model_validate({**{key: doc[key] for key in ("id", "label", "source", "provenance", "links", "notes")},
        "kind": "fixed", "material": {"kind": "assembly", "layers": doc["layers"], "backing": backing}})
    other = _ENTRY.validate_python(construction("zone_wall", id="other"))
    fixed = search_library.user_fixed_entries([entry.id], [entry, other])
    assert fixed == (legacy,)
    assert search_library.snapshot_bytes([], fixed) == search_library.snapshot_bytes([], [legacy])
    candidate, = search_library.fixed_candidates([f"fixed:{entry.id}"], [], fixed)
    assert candidate.material == legacy.surface_material().model_dump(mode="json")


@pytest.mark.parametrize("backing", ["rigid", "air", "zone_wall"])
def test_construction_cannot_be_the_starting_family(backing: str) -> None:
    entry = _ENTRY.validate_python(construction(backing))
    with pytest.raises(search_library.MaterialsRefused, match="不能拿來當搜尋的家族或起跑樣板"):
        search_library.refuse_fixed_as_family([entry.id], [entry])


def test_zone_wall_search_refusal_names_the_zone_page() -> None:
    entry = _ENTRY.validate_python(construction("zone_wall"))
    with pytest.raises(search_library.MaterialsRefused, match="現在還不能整塊當候選.*分區頁直接指定"):
        search_library.user_fixed_entries([entry.id], [entry])


def test_registered_name_for_a_direct_replacement_matches_the_zone_picker() -> None:
    doc = construction("air")
    entry = _ENTRY.validate_python(doc)
    surface = library.fixed_surface(entry)
    assert isinstance(surface, Assembly)
    legacy = library.FixedEntry(kind="fixed", id="old_panel", label="舊面板", source="自編", provenance="model", material=surface)
    assert registered_entry(surface, [legacy, entry]) == entry
    assert registered_entry(surface, [legacy]) == legacy


def test_bonded_backing_in_a_construction_requires_rigid_load() -> None:
    with pytest.raises(ValidationError, match="背後負載必須是剛背"):
        _ENTRY.validate_python(construction("air", layers=BONDED_VPR["layers"]))


def test_construction_kind_field_has_a_chinese_description() -> None:
    assert "構造" in (library.ConstructionEntry.model_fields["kind"].description or "")


def test_candidate_lookup_accepts_only_bare_material_ids() -> None:
    entry = _ENTRY.validate_python(construction("zone_wall"))
    assert search_library.user_fixed_entries([f"fixed:{entry.id}"], [entry]) == ()
