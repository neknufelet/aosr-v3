"""構造設定考卷：抓原研究數字、推導範圍與出處遺失，以及壞資料被接受。"""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from aosr.config.paths import config_path
from aosr.config.porous_model_ranges import load_porous_model_ranges


def test_original_numbers_and_provenance() -> None:
    """抓把樣品範圍改成物理限制、四個原始數字或出處漏載。"""
    ranges = load_porous_model_ranges(config_path("porous_model_ranges.toml"))
    k = ranges.komatsu
    assert (k.sample_flow_resistivity_min_pa_s_m2, k.sample_flow_resistivity_max_pa_s_m2) == (6000, 72900)
    assert (k.measured_frequency_min_hz, k.measured_frequency_max_hz) == (100, 5000)
    assert k.fitted_material_class == "fibrous"
    assert set(k.fitted_materials) == {"glass wool", "rock wool"}
    assert "Komatsu, T. (2008)" in k.provenance.reference
    assert "29(2), 121–129" in k.provenance.reference
    assert k.provenance.doi == "https://doi.org/10.1250/ast.29.121"
    assert "Table 3" in k.provenance.flow_resistivity_source
    assert "glass wool 01" in k.provenance.flow_resistivity_source
    assert "glass wool 15" in k.provenance.flow_resistivity_source
    assert "§3" in k.provenance.frequency_source
    assert "100 Hz to 5 kHz" in k.provenance.frequency_source
    assert "two-cavity method" in k.provenance.frequency_source
    assert k.f_over_sigma_bounds_m3_kg == pytest.approx((100 / 72900, 5000 / 6000))


@pytest.mark.parametrize("old,new", [
    ("manual_flow_resistivity_min_pa_s_m2 = 4000", "manual_flow_resistivity_min_pa_s_m2 = 6000"),
    ("measured_frequency_min_hz = 100", ""),
    ("measured_frequency_min_hz = 100", "measured_frequency_min_hz = 100\nextra = 1"),
    ("measured_frequency_min_hz = 100", "measured_frequency_min_hz = 6000"),
    ("sample_flow_resistivity_min_pa_s_m2 = 6000", "sample_flow_resistivity_min_pa_s_m2 = 80000"),
    ("sample_flow_resistivity_min_pa_s_m2 = 6000", "sample_flow_resistivity_min_pa_s_m2 = nan"),
    ('doi = "https://doi.org/10.1250/ast.29.121"', 'doi = ""'),
    ('doi = "https://doi.org/10.1250/ast.29.121"', 'doi = "https://doi.org/10.1250/ast.29.121"\nextra = "bad"'),
])
def test_broken_file_rejected(tmp_path: Path, old: str, new: str) -> None:
    """抓缺欄位、額外欄位、顛倒上下界、非有限值與空出處被放行。"""
    source = config_path("porous_model_ranges.toml").read_text()
    assert old in source
    path = tmp_path / "ranges.toml"
    path.write_text(source.replace(old, new))
    with pytest.raises(ValidationError):
        load_porous_model_ranges(path)


def test_ranges_are_frozen() -> None:
    """抓載入後可改掉模型範圍或出處，導致同一設定變成兩份答案。"""
    ranges = load_porous_model_ranges(config_path("porous_model_ranges.toml"))
    with pytest.raises(ValidationError):
        ranges.komatsu.measured_frequency_min_hz = 200
    with pytest.raises(ValidationError):
        ranges.komatsu.provenance.doi = "other"
