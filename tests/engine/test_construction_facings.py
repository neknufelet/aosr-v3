"""面層接線與低流阻試算：直接矩陣、解析空腔與獨立邊界守門。"""
from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest

from aosr.config.paths import config_path
from aosr.config.physics_constants import load_physics_constants
from tests.engine.test_construction import _notes, _request, _run_requests, _value
from tests.engine.test_construction_facings_models import FACING_CASES


BACKING = {"backing_fabric_surface_mass_kg_m2": .063, "backing_fabric_flow_resistance_pa_s_m": 236.}
AIR = {"kind": "air", "thickness_m": .03}


def facing_requests() -> dict[str, dict[str, object]]:
    rows = {}
    for kind, base in FACING_CASES.items():
        for suffix, options in (("", {}), ("_angles", {"angles": [0., 37., 75.]}), ("_diffuse", {"mode": "diffuse"})):
            rows[kind + suffix] = _request([base, AIR], direct=True, **options)
        rows[kind + "_sheet_angles"] = _request([base], backing="air", angles=[0., 37., 75.], direct=True)
        rows[kind + "_sheet"] = _request([base], backing="air", direct=True)
        for field, value in base.items():
            if isinstance(value, float):
                rows[kind + "_" + field] = _request([base | {field: value * 1.01}, AIR])
        if kind in ("perforated", "slotted"):
            for suffix, options in (("", {}), ("_angles", {"angles": [0., 37., 75.]}), ("_diffuse", {"mode": "diffuse"})):
                rows[kind + "_backed" + suffix] = _request([base | BACKING, AIR], direct=True, **options)
            for field, value in BACKING.items():
                rows[kind + "_" + field] = _request([base | BACKING | {field: value * 1.01}, AIR])
    for kind, patterns in (("perforated", ["square", "staggered_60", "staggered_45", "rectangular"]), ("slotted", ["slotted"])):
        for pattern in patterns:
            base = {key: value for key, value in FACING_CASES[kind].items() if key != "open_area"}
            base |= {"slot_spacing_m": .04} if kind == "slotted" else {"hole_spacing_m": .02, "hole_pattern": pattern}
            if pattern == "rectangular":
                base["hole_spacing_2_m"] = .03
            rows["pattern_" + pattern] = _request([base, AIR], direct=True)
            for field in ("slot_spacing_m",) if kind == "slotted" else ("hole_spacing_m", "hole_spacing_2_m") if pattern == "rectangular" else ("hole_spacing_m",):
                rows["pattern_" + pattern + field] = _request([base | {field: cast(float, base[field]) * 1.01}, AIR])
    for sigma in (3999.99, 4000., 5999.99, 6000.):
        rows[str(sigma)] = _request([{"kind": "porous", "thickness_m": .05,
            "flow_resistivity_pa_s_m2": sigma, "material_class": "fibrous"}], frequencies=[100., 3000., 4000.])
    for source in ("measured", "density_table", "catalog", "unspecified"):
        rows[source] = _request([{"kind": "porous", "thickness_m": .05, "flow_resistivity_pa_s_m2": 6000.,
            "material_class": "fibrous", "flow_resistivity_source": source}], frequencies=[100., 3000., 4000.])
    rows["fabric_thickness"] = _request([FACING_CASES["fabric"], AIR], direct=True)
    return rows


@pytest.fixture(scope="module")
def results(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict[str, object]]:
    return _run_requests(facing_requests(), tmp_path_factory.mktemp("facings"))


@pytest.mark.parametrize("kind", FACING_CASES)
@pytest.mark.parametrize("suffix", ["", "_angles", "_diffuse"])
def test_facing_matches_direct_bitwise_and_double_precision(results: dict[str, dict[str, object]], kind: str, suffix: str) -> None:
    row = results[kind + suffix]
    assert "error" not in row, row
    assert row["value"] == row["direct"]
    assert cast(dict[str, object], row["value"])["dtype"] == ("float64" if suffix == "_diffuse" else "complex128")


@pytest.mark.parametrize("kind", FACING_CASES)
def test_every_facing_number_changes_impedance(results: dict[str, dict[str, object]], kind: str) -> None:
    for field, value in FACING_CASES[kind].items():
        if isinstance(value, float):
            assert np.any(_value(results[kind]) != _value(results[kind + "_" + field])), field


@pytest.mark.parametrize("kind", FACING_CASES)
def test_sheet_cavity_has_analytic_impedance(results: dict[str, dict[str, object]], kind: str) -> None:
    """直接薄片減掉空氣負載後，加剛背空腔解析解；抓層次、符號或角度錯接。"""
    air = load_physics_constants(config_path("physics_constants.toml"))
    f = np.array([100., 250., 500., 1000., 4000.])
    sheet = _value(results[kind + "_sheet"], "direct") - air.rho_c
    expected = sheet - 1j * air.rho_c / np.tan(2 * np.pi * f / air.sound_speed_m_s * .03)
    np.testing.assert_allclose(_value(results[kind]), expected, rtol=2e-12, atol=2e-10)


@pytest.mark.parametrize("kind", ["perforated", "slotted"])
def test_backing_is_resistance_divided_by_own_open_area_and_mass_is_record_only(results: dict[str, dict[str, object]], kind: str) -> None:
    difference = _value(results[kind + "_backed"]) - _value(results[kind])
    np.testing.assert_allclose(difference, 236. / cast(float, FACING_CASES[kind]["open_area"]), rtol=2e-12, atol=2e-10)
    assert results[kind + "_backed"]["value"] == results[kind + "_backing_fabric_surface_mass_kg_m2"]["value"]
    assert np.any(_value(results[kind + "_backed"]) != _value(results[kind + "_backing_fabric_flow_resistance_pa_s_m"]))
    for suffix in ("", "_angles", "_diffuse"):
        row = results[kind + "_backed" + suffix]
        assert row["value"] == row["direct"]


@pytest.mark.parametrize("pattern", ["square", "staggered_60", "staggered_45", "rectangular", "slotted"])
def test_spacing_is_used_in_core(results: dict[str, dict[str, object]], pattern: str) -> None:
    row = results["pattern_" + pattern]
    assert "error" not in row, row
    assert row["value"] == row["direct"]
    for field in ("slot_spacing_m",) if pattern == "slotted" else ("hole_spacing_m", "hole_spacing_2_m") if pattern == "rectangular" else ("hole_spacing_m",):
        assert np.any(_value(row) != _value(results["pattern_" + pattern + field]))


def test_manual_komatsu_boundaries_and_original_extrapolation_window(results: dict[str, dict[str, object]]) -> None:
    assert "4000" in cast(str, results["3999.99"]["error"])
    for sigma in (4000., 5999.99, 6000.):
        row = results[str(sigma)]
        notes = _notes(row)
        assert any(note["code"] == "komatsu_flow_below_samples" for note in notes) is (sigma < 6000)
    notes = _notes(results["4000.0"])
    assert next(note for note in notes if note["code"] == "komatsu_frequency_extrapolation")["frequencies_hz"] == [4000.]


def test_fabric_required_thickness_does_not_affect_tmm(results: dict[str, dict[str, object]]) -> None:
    """探針用另一個正厚度直接算同一片布，抓厚度意外進入公式。"""
    row = results["fabric_thickness"]
    assert row["value"] == row["direct"]


def test_flow_source_only_records_provenance(results: dict[str, dict[str, object]]) -> None:
    for source in ("measured", "density_table", "catalog", "unspecified"):
        assert results[source] == results["6000.0"]


@pytest.mark.parametrize("kind", FACING_CASES)
def test_facing_sheet_only_broadcasts_across_angles(results: dict[str, dict[str, object]], kind: str) -> None:
    air = load_physics_constants(config_path("physics_constants.toml"))
    sheet = _value(results[kind + "_sheet"], "direct") - air.rho_c
    expected = sheet[:, None] + air.rho_c / np.cos(np.deg2rad([0., 37., 75.]))
    np.testing.assert_allclose(_value(results[kind + "_sheet_angles"]), expected, rtol=2e-12, atol=2e-10)
