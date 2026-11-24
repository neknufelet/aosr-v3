"""固定區的選料紀錄失配時，照已存構造判深度；原牆、真選料與命令列各走原有入口。"""
import json
from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest
from starlette.testclient import TestClient

from aosr.config.frequency_axis import LowFrequencyAxis
from aosr.gui.app import GuiSettings, create_app
from aosr.gui.gradient_plan import GradientRequest, PlanRefused
from aosr.gui.zone_design import DesignSettings
from aosr.gui.zone_limits import zone_limits
from aosr.materials import library
from tests.engine._construction_cases import construction
from tests.engine._surface_zone_cases import zone_document
from tests.engine.test_gui_gradient_jobs import FakeLauncher, _jobs, _place_surrogate
from tests.engine.test_gui_zone_limits import Doc, _ENTRY, _lines, _scheme, _zones


def _panel_trial() -> tuple[Doc, Doc]:
    document = zone_document() | {"scheme_id": "panel_trial"}
    cast(Doc, document["scene"]).update(room_m={"Lx": 6.1, "Ly": 4.4, "Lz": 2.9},
                                       low_frequency_axis=LowFrequencyAxis.SEARCH.value)
    entry = construction("rigid", id="reed_panel", label="蘆葦試片", layers=[
        {"kind": "fabric", "surface_mass_kg_m2": 0.23, "flow_resistance_pa_s_m": 175.0},
        {"kind": "air", "thickness_m": 0.081}])
    return document, entry


def test_deleting_picked_panel_keeps_saved_depth_refusal(tmp_path: Path) -> None:
    document, entry = _panel_trial()
    name = "panel_trial"
    launcher = FakeLauncher()
    with TestClient(create_app(GuiSettings(engine_commit="a" * 40, data_dir=tmp_path, gradient_launcher=launcher)),
                    base_url="http://127.0.0.1") as client:
        url = f"/api/materials/{entry['id']}"
        assert client.put(url, json=entry).status_code == 200
        pick = client.post("/api/zone-material", json={"scheme": document, "wall": "x0", "zone": "whole_wall",
                           "choice": "fixed", "material_id": entry["id"]}).json()
        assert pick["picked_entry"] == entry["id"]
        _zones(document)["x0"][0]["material"] = pick["material"]
        design = {"schema_version": "aosr.zone_design.v1", "scheme_id": name, "zones": {
            "x0": {"whole_wall": {"mode": "fixed", "picked_entry": pick["picked_entry"], "max_depth_m": 0.05}},
            "xL": {"whole_wall": {"mode": "search"}}}}
        scheme = _scheme(document)
        (tmp_path / "schemes" / f"{name}.json").write_text(scheme.model_dump_json())
        (tmp_path / "designs" / f"{name}.json").write_text(json.dumps(design))
        _place_surrogate(tmp_path, scheme)
        request = {"scheme": document, "design": design}
        assert client.post("/api/zone-limits", json=request).json()["red"] == 1
        assert client.post(f"{url}/delete", json={}).status_code == 200
        page = client.post("/api/zone-limits", json=request).json()
        preview = client.post(f"/api/gradient/schemes/{name}/plan", json={})
        start = client.post(f"/api/gradient/schemes/{name}/start", json={})
        assert (page["red"], preview.status_code, start.status_code, len(launcher.launched)) == (1, 400, 400, 0)
        refusal, = page["refusals"]
        assert all(word in refusal for word in ("總深 0.081 公尺", "超過", "0.05 公尺", "這一塊"))
        assert refusal in preview.json()["error"] == start.json()["error"]
    jobs = _jobs(tmp_path, launcher)
    _, saved_scheme, saved_design = jobs.inputs(name)
    assert saved_design is not None and saved_design.zones["x0"]["whole_wall"].picked_entry == entry["id"]
    with pytest.raises(PlanRefused) as caught:
        jobs.plan(saved_scheme, saved_design, GradientRequest())
    assert refusal in str(caught.value)


@pytest.mark.parametrize(("case", "code"), [
    ("matched_fits", None), ("matched_over", "depth_over"), ("deleted_fits", None),
    ("deleted_original", None), ("deleted_changed", "depth_over"), ("deleted_unknown_depth", "depth_unknown"),
])
def test_saved_panel_depth_controls(tmp_path: Path, case: str, code: str | None) -> None:
    document, entry = _panel_trial()
    wall = _zones(document)["x0"][0]
    if case != "deleted_unknown_depth":
        wall["material"] = {"kind": "assembly", "backing": entry["backing"], "layers": entry["layers"]}
    cap = 0.09 if case.endswith("_fits") else 0.05
    original = None
    if case in ("deleted_original", "deleted_changed"):
        original = "panel_before"
        prior = deepcopy(document) if case == "deleted_original" else _panel_trial()[0]
        (tmp_path / "schemes").mkdir()
        (tmp_path / "schemes" / f"{original}.json").write_text(json.dumps(prior | {"scheme_id": original}))
    if case.startswith("matched"):
        (tmp_path / "materials.json").write_text(library.dump_user_library((_ENTRY.validate_python(entry),)))
    jobs = _jobs(tmp_path, FakeLauncher())
    design = DesignSettings.model_validate({"scheme_id": document["scheme_id"], "original_scheme_id": original, "zones": {
        "x0": {"whole_wall": {"mode": "fixed", "picked_entry": entry["id"], "max_depth_m": cap}},
        "xL": {"whole_wall": {"mode": "search"}}}})
    result = zone_limits(document, design.model_dump(mode="json"), depth_targets=jobs.depth_targets)
    red = [line for line in _lines(result, "x0", "whole_wall") if line["level"] == "red"]
    assert [line["code"] for line in red] == ([code] if code else [])
    if code:
        with pytest.raises(PlanRefused) as refused:
            jobs.plan(_scheme(document), design, GradientRequest())
        assert red[0]["text"] in str(refused.value)
        if code == "depth_unknown":
            assert all(word in red[0]["text"] for word in ("看不出", "這間房貼什麼", "重選"))
            (tmp_path / "schemes").mkdir(exist_ok=True)
            (tmp_path / "designs").mkdir(exist_ok=True)
            (tmp_path / "schemes" / "panel_trial.json").write_text(_scheme(document).model_dump_json())
            (tmp_path / "designs" / "panel_trial.json").write_text(design.model_dump_json())
            with pytest.raises(PlanRefused) as start:
                jobs.start("panel_trial", GradientRequest())
            assert red[0]["text"] in str(start.value)
    else:
        assert jobs.plan(_scheme(document), design, GradientRequest())[0].fixed_surfaces["x0"] == "scheme-material"


def test_cli_rejects_capped_fixed_scheme_panel_after_deletion(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from aosr.gradient import cli

    document, entry = _panel_trial()
    _zones(document)["x0"][0]["material"] = {"kind": "assembly", "backing": entry["backing"], "layers": entry["layers"]}
    scheme = tmp_path / "panel_trial.json"
    scheme.write_text(_scheme(document).model_dump_json())
    materials = tmp_path / "materials.json"
    materials.write_text(library.dump_user_library(()))
    root = tmp_path / "run"
    args = ["start", "--scheme", str(scheme), "--materials", str(materials), "--presets", ",".join(["none"] * 6),
            "--purpose", str(document["purpose"]), "--root", str(root), "--surrogate", str(tmp_path / "unused.npz"),
            "--fixed-surface", "x0=scheme-material", "--max-depth", "x0=0.05"]
    assert cli.main(args) == cli.EXIT_FAILED
    assert "固定面 x0 不准再給深度上限" in capsys.readouterr().err
    assert not root.exists()


def _panel_files(folder: Path, *, original: str, recorded: bool = True) -> tuple[Path, Path, Path]:
    document, entry = _panel_trial()
    scheme_dir, design_dir = folder / "schemes", folder / "designs"
    scheme_dir.mkdir(exist_ok=True)
    design_dir.mkdir(exist_ok=True)
    scheme_file, design_file = scheme_dir / "panel_trial.json", design_dir / "panel_trial.json"
    prior = deepcopy(document) | {"scheme_id": "panel_source"}
    wall = _zones(document)["x0"][0]
    wall["material"] = {"kind": "assembly", "backing": entry["backing"], "layers": entry["layers"]}
    if original == "same":
        _zones(prior)["x0"][0]["material"] = wall["material"]
    if original in ("same", "different"):
        (scheme_dir / "panel_source.json").write_text(_scheme(prior).model_dump_json())
    fixed: Doc = {"mode": "fixed", "max_depth_m": 0.05}
    if recorded:
        fixed["picked_entry"] = entry["id"]
    settings = {"scheme_id": "panel_trial", "original_scheme_id": None if original == "none" else "panel_source",
                "left_right_symmetry": "off", "zones": {"x0": {"whole_wall": fixed}, "xL": {"whole_wall": {"mode": "search"}}}}
    scheme_file.write_text(_scheme(document).model_dump_json())
    design_file.write_text(json.dumps(settings))
    catalog = folder / "materials.json"
    catalog.write_text(library.dump_user_library((_ENTRY.validate_python(entry),)))
    return scheme_file, design_file, catalog


def test_legacy_different_wall_stays_refused_after_library_deletion(tmp_path: Path) -> None:
    """沒有選料欄的舊檔，仍可由同位置的原牆證明構造不同；刪料不能解除既有拒收。"""
    scheme_file, design_file, catalog = _panel_files(tmp_path, original="different", recorded=False)
    before = scheme_file.read_bytes(), design_file.read_bytes()
    launcher = FakeLauncher()
    jobs = _jobs(tmp_path, launcher)
    request = GradientRequest(device="cpu")
    _, scheme, _ = jobs.inputs("panel_trial")
    _place_surrogate(tmp_path, scheme)
    with pytest.raises(PlanRefused, match="總深 0.081 公尺.*超過.*0.05 公尺"):
        jobs.start("panel_trial", request)
    catalog.write_text(library.dump_user_library(()))
    assert (scheme_file.read_bytes(), design_file.read_bytes()) == before
    with pytest.raises(PlanRefused, match="總深 0.081 公尺.*超過.*0.05 公尺"):
        jobs.start("panel_trial", request)
    assert launcher.launched == []


def _check_dispatched_panel(launcher: FakeLauncher, source: Path, design: DesignSettings) -> None:
    """把假派工收到的指令與快照交給真的命令列起跑資料檢查，停在匯出算子之前。"""
    from aosr.config.capabilities import load_capabilities
    from aosr.config.directivity_defaults import load_directivity_defaults
    from aosr.config.paths import config_path
    from aosr.gradient import cli
    from aosr.gradient.fixed_surfaces import parse_allowed_targets, parse_fixed_surfaces, parse_max_depths, parse_ranges
    from aosr.gradient.ties import parse_ties

    (_, argv, _, _), = launcher.launched
    args = cli._parser().parse_args(argv[argv.index("aosr.gradient.cli") + 1:])
    assert args.command == "start" and args.scheme.read_bytes() == source.read_bytes()
    assert DesignSettings.model_validate_json((args.scheme.parent / "design.json").read_bytes()) == design
    scheme = cli._start_scheme(args.root, args.scheme)
    fixed, caps = parse_fixed_surfaces(args.fixed_surface), parse_max_depths(args.max_depth)
    allowed = parse_allowed_targets(args.allowed)
    assert fixed["x0"] == "scheme-material" and "x0" not in caps
    materials, settings, _, _ = cli._start_materials(args, args.settings, allowed)
    designs, _ = cli.start_designs(scheme, settings, materials.presets, args.presets, mount=args.mount, fixed=fixed,
                                  caps=caps, allowed=allowed, capabilities=load_capabilities(args.capabilities),
                                  directivity=load_directivity_defaults(config_path("directivity_defaults.toml")),
                                  ties=parse_ties(args.tie), ranges=parse_ranges(args.range))
    assert scheme.scene.surface_zones_by_wall is not None
    kept = next(item for item in designs if item.surface_id == "x0").kept_material
    assert kept == scheme.scene.surface_zones_by_wall["x0"][0].material


@pytest.mark.parametrize("original", ["none", "missing", "different", "same"])
@pytest.mark.parametrize("deleted", [False, True], ids=["present", "deleted"])
def test_saved_fixed_depth_through_dispatch(tmp_path: Path, original: str, deleted: bool) -> None:
    scheme_file, design_file, catalog = _panel_files(tmp_path, original=original)
    before = scheme_file.read_bytes(), design_file.read_bytes()
    if deleted:
        catalog.write_text(library.dump_user_library(()))
    launcher = FakeLauncher()
    jobs = _jobs(tmp_path, launcher)
    name = scheme_file.stem
    _, scheme, design = jobs.inputs(name)
    assert design is not None and design.zones["x0"]["whole_wall"].max_depth_m == 0.05
    request = GradientRequest(device="cpu", max_steps=1)
    _place_surrogate(tmp_path, scheme)
    limits = zone_limits(scheme.model_dump(mode="json"), design.model_dump(mode="json"), depth_targets=jobs.depth_targets)
    if original == "same":
        assert limits["refusals"] == []
        assert jobs.plan(scheme, design, request)[0].fixed_surfaces["x0"] == "scheme-material"
        preview = jobs.preview(name, request)
        assert cast(Doc, preview["surrogate"])["found"] is True
        assert cast(Doc, preview["fixed_depths"])["x0"] == {"cap": {"value_m": 0.05, "level": "zone"}, "lines": []}
        assert jobs.start(name, request)["state"] == "checking"
        _check_dispatched_panel(launcher, scheme_file, design)
    else:
        refusal, = cast(list[str], limits["refusals"])
        assert all(word in refusal for word in ("總深 0.081 公尺", "超過", "0.05 公尺", "這一塊"))
        with pytest.raises(PlanRefused) as plan:
            jobs.plan(scheme, design, request)
        with pytest.raises(PlanRefused) as preview_error:
            jobs.preview(name, request)
        with pytest.raises(PlanRefused) as start:
            jobs.start(name, request)
        assert refusal in str(plan.value) == str(preview_error.value) == str(start.value)
        assert launcher.launched == [] and not list((tmp_path / "gradient").iterdir())
    assert (scheme_file.read_bytes(), design_file.read_bytes()) == before


@pytest.mark.parametrize("recorded", [False, True], ids=["no_pick", "deleted_pick"])
@pytest.mark.parametrize("save", [False, True], ids=["read", "save"])
def test_legacy_design_io_keeps_depth_evidence(tmp_path: Path, recorded: bool, save: bool) -> None:
    scheme_file, design_file, catalog = _panel_files(tmp_path, original="none", recorded=recorded)
    catalog.write_text(library.dump_user_library(()))
    initial = design_file.read_bytes()
    source = scheme_file.read_bytes()
    expected = json.loads(initial)["zones"]
    name = scheme_file.stem
    app = create_app(GuiSettings(engine_commit="a" * 40, data_dir=tmp_path, gradient_launcher=FakeLauncher()))
    with TestClient(app, base_url="http://127.0.0.1") as client:
        url = f"/api/designs/{name}"
        response = client.get(url)
        assert response.status_code == 200
        design = response.json()["design"]
        assert design["zones"] == expected and response.json()["problems"] == []
        if save:
            design["room_max_depth_m"] = 0.12
            assert client.put(url, json=design).status_code == 200
            written = json.loads(design_file.read_bytes())
            assert written["room_max_depth_m"] == 0.12 and written["zones"] == expected
            assert client.get(url).json()["design"] == design
        else:
            assert design_file.read_bytes() == initial
        preview = client.post(f"/api/gradient/schemes/{name}/plan", json={})
        if recorded:
            assert preview.status_code == 400 and "總深 0.081 公尺" in preview.json()["error"]
        else:
            assert preview.status_code == 200
            line, = preview.json()["fixed_depths"]["x0"]["lines"]
            assert line["level"] == "note" and "比不了" in line["text"]
    assert scheme_file.read_bytes() == source
