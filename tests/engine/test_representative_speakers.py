"""家具決策紙第 9 條的代表模型與既有 Cabinet 欄位契約。"""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from aosr.config.representative_speakers import RepresentativeSpeakers, load_representative_speakers
from aosr.search.layout_settings import Cabinet


DATA = Path(__file__).resolve().parents[2] / "src/aosr/config/data/representative_speakers.toml"


@pytest.mark.parametrize("name,body,base,center,datum,height", [
    ("bookshelf", 355.0, 0.0, 205.0, "cabinet_bottom", 0.355),
    ("floorstanding", 1060.0, 45.0, 800.0, "floor", 1.105),
])
def test_representatives_equal_decision_and_cabinet_fields(name: str, body: float, base: float,
                                                         center: float, datum: str, height: float) -> None:
    model = getattr(load_representative_speakers(DATA), name)
    assert model.width_mm == 210.0
    assert model.depth_mm == (280.0 if name == "bookshelf" else 380.0)
    assert model.body_height_mm == body
    assert model.base_height_mm == base
    assert model.acoustic_center_above_bottom_mm == center
    assert model.acoustic_center_reference == datum
    assert model.acoustic_center_behind_front_mm == 0.0
    cabinet = Cabinet(**model.cabinet_fields_m())
    assert cabinet.width_m == 0.210
    assert cabinet.depth_m == (0.280 if name == "bookshelf" else 0.380)
    assert cabinet.height_m == height
    assert cabinet.acoustic_center_above_bottom_m == (0.205 if name == "bookshelf" else 0.800)
    assert cabinet.acoustic_center_behind_front_m == 0.0
    assert "代表模型，非實際型號" in model.label
    assert model.provenance.status == "estimated"
    if name == "floorstanding":
        assert "資料只有 KEF 與 Arendal 兩家" in model.label


@pytest.mark.parametrize("bad", ["missing_center", "missing_provenance", "extra", "nonfinite",
                                "nonpositive", "boolean", "center_above_top", "center_behind_back",
                                "negative_base", "wrong_reference", "bookshelf_base"])
def test_rejects_each_invalid_speaker_shape(bad: str) -> None:
    with DATA.open("rb") as file:
        data = tomllib.load(file)
    speaker = data["bookshelf"]
    if bad == "missing_center":
        del speaker["acoustic_center_above_bottom_mm"]
    elif bad == "missing_provenance":
        del speaker["provenance"]["conditions"]
    else:
        key, value = {
            "extra": ("extra", 1), "nonfinite": ("width_mm", float("inf")),
            "nonpositive": ("width_mm", 0.0), "boolean": ("width_mm", True),
            "center_above_top": ("acoustic_center_above_bottom_mm", 400.0),
            "center_behind_back": ("acoustic_center_behind_front_mm", 280.0),
            "negative_base": ("base_height_mm", -1.0),
            "wrong_reference": ("acoustic_center_reference", "floor"),
            "bookshelf_base": ("base_height_mm", 45.0),
        }[bad]
        speaker[key] = value
    with pytest.raises(ValidationError):
        RepresentativeSpeakers.model_validate(data)


def test_speakers_are_frozen_and_loader_requires_explicit_path(tmp_path: Path) -> None:
    path = tmp_path / "speakers.toml"
    path.write_bytes(DATA.read_bytes())
    model = load_representative_speakers(path)
    with pytest.raises(ValidationError, match="frozen"):
        model.bookshelf.acoustic_center_above_bottom_mm = 177.5
    with pytest.raises(TypeError):
        load_representative_speakers(**{})
