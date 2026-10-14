"""實測表的格式與算數契約；端點舍入相對容差 1e-12 只住考卷。"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
import subprocess
import sys

import pytest
from pydantic import ValidationError

from aosr.physics.report_surfaces import WholeWallZone
from tests.engine.test_construction_cli_end_to_end import environment


def measured() -> dict[str, object]:
    return {"kind": "measured_impedance", "points": [
        {"frequency_hz": 100., "real_pa_s_m": 400., "imag_pa_s_m": -200.},
        {"frequency_hz": 1000., "real_pa_s_m": 800., "imag_pa_s_m": 100.}],
        "base_assembly": {"kind": "assembly", "layers": [{"kind": "board", "surface_mass_kg_m2": .15,
            "resistance_pa_s_m": 80.}, {"kind": "air", "thickness_m": .03}], "backing": "rigid"},
        "source_and_conditions": "阻抗管，垂直入射，樣品厚 30 mm"}


def test_measured_zone_roundtrip_and_default_fill_rule() -> None:
    zone = WholeWallZone.model_validate({"area": "whole_wall", "material": measured()})
    assert zone.material.kind == "measured_impedance"
    assert "fill_rule" in zone.material.model_dump()
    from aosr.physics.report_surfaces import validation_content
    content = validation_content(zone.material)
    assert isinstance(content, dict) and content["fill_rule"] == zone.material.fill_rule
    assert WholeWallZone.model_validate_json(zone.model_dump_json()) == zone


@pytest.mark.parametrize("case,path", [
    ("missing-source", ("source_and_conditions",)), ("blank-source", ("source_and_conditions",)),
    ("missing-base", ("base_assembly",)), ("missing-base-kind", ("base_assembly",)), ("fixed-base", ("base_assembly", "kind")),
    ("nested-base", ("base_assembly", "kind")), ("short", ("points",)),
    ("duplicate", ("points", 1, "frequency_hz")), ("decreasing", ("points", 1, "frequency_hz")),
    ("negative-frequency", ("points", 0, "frequency_hz")), ("infinite-frequency", ("points", 0, "frequency_hz")),
    ("zero-real", ("points", 0, "real_pa_s_m")), ("negative-real", ("points", 0, "real_pa_s_m")),
    ("infinite-real", ("points", 0, "real_pa_s_m")), ("nan-imag", ("points", 0, "imag_pa_s_m")),
    ("missing-imag", ("points", 0, "imag_pa_s_m")), ("extra-point", ("points", 0, "unknown")),
    ("fill", ("fill_rule",)), ("extra", ("unknown",)),
])
def test_measured_format_rejects_at_the_row_and_field(case: str, path: tuple[str | int, ...]) -> None:
    data = deepcopy(measured())
    points = data["points"]
    assert isinstance(points, list)
    changes = {"duplicate": (1, "frequency_hz", 100.), "decreasing": (1, "frequency_hz", 50.),
        "negative-frequency": (0, "frequency_hz", -1.), "infinite-frequency": (0, "frequency_hz", float("inf")),
        "zero-real": (0, "real_pa_s_m", 0.), "negative-real": (0, "real_pa_s_m", -1.),
        "infinite-real": (0, "real_pa_s_m", float("inf")), "nan-imag": (0, "imag_pa_s_m", float("nan")),
        "extra-point": (0, "unknown", 1.)}
    if case == "missing-base-kind":
        base = data["base_assembly"]
        assert isinstance(base, dict)
        base.pop("kind")
    elif case in changes:
        index, field, value = changes[case]
        points[index][field] = value
    elif case.startswith("missing-"):
        if case == "missing-imag":
            points[0].pop("imag_pa_s_m")
        else:
            data.pop("source_and_conditions" if case == "missing-source" else "base_assembly")
    elif case.endswith("-base"):
        data["base_assembly"] = {"kind": "constant_impedance" if case == "fixed-base" else "measured_impedance"}
    elif case == "short":
        data["points"] = points[:1]
    else:
        data[{"blank-source": "source_and_conditions", "fill": "fill_rule", "extra": "unknown"}[case]] = " "
    with pytest.raises(ValidationError) as caught:
        WholeWallZone.model_validate({"area": "whole_wall", "material": data})
    assert ("material", "measured_impedance", *path) in {e["loc"] for e in caught.value.errors()}


@pytest.mark.parametrize("case", ("core", "notes", "seams", "context"))
def test_measured_arithmetic_in_precision_process(case: str, tmp_path: Path) -> None:
    probe = Path(__file__).with_name("_measured_probe.py")
    child = subprocess.run([sys.executable, "-B", str(probe), case, json.dumps(measured())], cwd=tmp_path,
                           env=environment(tmp_path), capture_output=True, text=True, timeout=120)
    assert child.returncode == 0, child.stdout + child.stderr


def test_measured_scene_validation_remains_cold(tmp_path: Path) -> None:
    from tests.engine.test_measured_reporting import document
    path = tmp_path / 'measured.json'
    path.write_text(json.dumps(document()))
    script = """
import json, sys
import aosr.reporting.physics_stage
from aosr.reporting.scheme import Scheme
from aosr.physics.report_surfaces import validation_content
with open(sys.argv[1], encoding='utf-8') as handle:
    scheme = Scheme.model_validate(json.load(handle))
assert scheme.scene.surface_zones_by_wall is not None
assert validation_content(scheme.scene.surface_zones_by_wall['floor'][0].material)['fill_rule']
assert 'jax' not in sys.modules
assert 'aosr.materials.construction' not in sys.modules
"""
    child = subprocess.run([sys.executable, '-B', '-c', script, str(path)], cwd=tmp_path,
        env=environment(tmp_path), capture_output=True, text=True, timeout=60)
    assert child.returncode == 0, child.stdout + child.stderr


@pytest.mark.parametrize("value", ("", 123))
@pytest.mark.parametrize("entry", ("scheme", "report", "search"))
def test_measured_source_errors_are_plain_chinese(value: object, entry: str, tmp_path: Path) -> None:
    """出處欄填空字串或數字：三個正式入口都要定位到欄位、說中文，不落回英文原句。"""
    import re
    from tests.engine._surface_zone_cases import NEW, zone_scene
    from tests.engine._zone_review_cases import reject_at
    scene = zone_scene(mixed=True)
    zones = scene[NEW]
    assert isinstance(zones, dict)
    material = measured()
    material["source_and_conditions"] = value
    zones["floor"][0]["material"] = material
    with pytest.raises(Exception) as caught:
        reject_at(entry, scene, tmp_path)
    text = str(caught.value)
    assert "source_and_conditions" in text, text
    assert ("必填" if value == "" else "要填文字") in text, text
    assert re.search(r"Input should|String should|at least 1 character", text) is None, text
