"""#1243：構造完整存回材料庫，各種背後與放置方式由條目明寫。"""
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError
from starlette.applications import Starlette
from starlette.testclient import TestClient

from aosr.gui.material_editor import material_routes
from aosr.materials import library
from aosr.materials.construction_models import Assembly


_ENTRY: TypeAdapter[library.LibraryEntry] = TypeAdapter(library.LibraryEntry)


def construction(backing: str = "rigid", placement: str = "wall", **changes: object) -> dict[str, object]:
    return {"kind": "construction", "id": "demo109_floor_sample", "label": "天花自編樣品", "source": "考卷自編",
            "provenance": "literature", "links": ["https://example.org/test"], "notes": ["自編條件"],
            "layers": [{"kind": "fabric", "surface_mass_kg_m2": 0.19, "flow_resistance_pa_s_m": 180.0}, {"kind": "air", "thickness_m": 0.063}],
            "product": False, "placement": placement, "backing": backing, **changes}


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
