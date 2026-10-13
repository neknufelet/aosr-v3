"""材料流考卷的冷子行程；先走正式精度初始化，之後才載入物理核心。"""
from __future__ import annotations

import hashlib
import io
import sys
from collections.abc import Callable, Mapping
from contextlib import redirect_stdout
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

from aosr.runtime import configure_calculation_jax

configure_calculation_jax()

import jax
import jax.numpy as jnp

from aosr.config.capabilities import load_capabilities
from aosr.config.construction_environment import construction_environment
from aosr.config.paths import config_path
from aosr.geometry.shoebox import Wall
from aosr.materials.construction import surface_impedance
from aosr.materials.construction_models import AirLayer, Assembly, BoardLayer, SurfaceMaterial
from aosr.physics import geometric_lane as geo
from aosr.physics import reflection_screen, reflection_window, report_io, report_output
from aosr.physics import report_path_table as tables
from aosr.physics import three_lane_report as report
from aosr.physics import three_lane_report_batch as batch
from aosr.physics import three_lane_report_geometry as geometry
from aosr.physics.wall_reflections import wall_reflection_sources
from aosr.physics.wall_surfaces import wall_surface_response
from aosr.reporting.fem_slices import fem_inputs
from tests.engine._fem_shared_cases import small_scheme
from tests.engine._source_model_control import fake_fem_energy
from tests.engine._wall_reflections_probe import _input


def _inputs() -> tuple[report_io.ReportInput, report_io.SolverInputs, report_io.SolverInputs]:
    inputs = _input(order=1)
    fixed = report_io.solver_inputs(inputs)
    materials = dict(fixed.materials_by_wall)
    materials[Wall.X0] = Assembly(layers=(BoardLayer(surface_mass_kg_m2=.15, resistance_pa_s_m=80.),
                                         AirLayer(thickness_m=.43)), backing="rigid")
    return inputs, fixed, fixed._replace(materials_by_wall=materials)


def _normal(solved: report_io.SolverInputs, axis: tuple[float, ...]) -> tuple[complex, ...]:
    air, ranges = construction_environment(density_kg_m3=solved.density_kg_m3,
                                           sound_speed_m_s=solved.sound_speed_m_s)
    row, _ = surface_impedance(cast(Assembly, solved.materials_by_wall[Wall.X0]), jnp.asarray(axis), air=air, ranges=ranges)
    return tuple(complex(z) for z in row)


def _solve(solved: report_io.SolverInputs) -> report.ThreeLaneReport:
    with patch.object(report, "_solve_fem_energy", fake_fem_energy):
        return report.solve_three_lane_report(**solved._asdict())


def _fem() -> None:
    _, fixed, solved = _inputs()
    seen: list[dict[str, object]] = []

    def capture(**kwargs: object) -> tuple[float, ...]:
        seen.append(kwargs)
        return cast(Callable[..., tuple[float, ...]], fake_fem_energy)(**kwargs)

    with patch.object(report, "_solve_fem_energy", capture):
        report.solve_three_lane_report(**solved._asdict())
    assert seen
    for call in seen:
        axis = cast(tuple[float, ...], call["frequencies_hz"])
        row = cast(Mapping[Wall, tuple[complex, ...]], call["wall_impedances"])[Wall.X0]
        assert row == _normal(solved, axis)
        assert fixed.impedance_by_wall is not None
        assert row != (complex(fixed.impedance_by_wall[Wall.X0]),) * len(axis)
        assert any(z.imag != 0. for z in row)


def _late() -> None:
    _, fixed, solved = _inputs()
    seen: list[tuple[tuple[float, ...], Mapping[str, tuple[float, ...]]]] = []
    original = geo.solve_geometric_late_energy
    decay = report._solve_report_late_decay

    def capture(**kwargs: object) -> object:
        seen.append((cast(tuple[float, ...], kwargs["frequencies_hz"]),
                     cast(Mapping[str, tuple[float, ...]], kwargs["absorption_by_wall"])))
        return cast(Callable[..., object], original)(**kwargs)

    def capture_decay(**kwargs: object) -> object:
        from aosr.config.frequency_axis import LATE_DECAY_FREQUENCIES_HZ
        seen.append((LATE_DECAY_FREQUENCIES_HZ,
                     cast(Mapping[str, tuple[float, ...]], kwargs["absorption_by_wall"])))
        return cast(Callable[..., object], decay)(**kwargs)

    with patch.object(batch, "solve_geometric_late_energy", capture), \
            patch.object(report, "_solve_report_late_decay", capture_decay):
        changed = _solve(solved)
    plain = _solve(fixed)
    assert changed.eyring_t60_by_band_s != plain.eyring_t60_by_band_s
    assert changed.geometric_lane.late_energy != plain.geometric_lane.late_energy
    for axis, rows in seen:
        wanted = wall_surface_response(solved.materials_by_wall, axis,
            density_kg_m3=solved.density_kg_m3, sound_speed_m_s=solved.sound_speed_m_s).named_rows(axis)
        assert rows == wanted
        old = wall_surface_response(fixed.materials_by_wall, axis,
            density_kg_m3=solved.density_kg_m3, sound_speed_m_s=solved.sound_speed_m_s).named_rows(axis)
        assert rows["x0"] != old["x0"]


def _direct(solved: report_io.SolverInputs, axis: tuple[float, ...]) -> geo.GeometricEarlyResult:
    sources = wall_reflection_sources(solved.materials_by_wall, axis,
        density_kg_m3=solved.density_kg_m3, sound_speed_m_s=solved.sound_speed_m_s)
    return geo.solve_geometric_early_lane(source_model=solved.source_model, room=solved.room,
        source=solved.source, receiver=solved.receiver, sound_speed_m_s=solved.sound_speed_m_s,
        rho_c_pa_s_per_m=solved.density_kg_m3 * solved.sound_speed_m_s, frequencies_hz=axis,
        impedance_by_wall=sources.normal_rows(), reflection_sources=sources,
        scattering_by_wall=None if solved.scattering_by_wall is None else
        {w.wall_name(): s for w, s in solved.scattering_by_wall.items()},
        reflection_order_k=solved.reflection_order_k)


def _geometry() -> None:
    _, fixed, solved = _inputs()
    seen: list[geo.GeometricLaneResult | geo.GeometricEarlyResult] = []
    original = geometry._solve_both_geometric_report_lanes

    def capture(**kwargs: object) -> object:
        result = cast(Callable[..., tuple[geo.GeometricLaneResult, geo.GeometricEarlyResult]], original)(**kwargs)
        seen.extend(result)
        return result

    with patch.object(geometry, "_solve_both_geometric_report_lanes", capture):
        changed = _solve(solved)
    # 細軸與密軸兩條都要真的攔到；批次繞過攔截器時，回傳報表本身仍要等於直接構造的結果。
    assert {type(lane) for lane in seen} == {geo.GeometricLaneResult, geo.GeometricEarlyResult}
    for lane in (*seen, changed.geometric_lane):
        direct = _direct(solved, lane.frequencies_hz)
        old = _direct(fixed, lane.frequencies_hz)
        assert lane.reflected_energy == direct.reflected_energy
        assert lane.interference_energy == direct.interference_energy
        assert lane.reflected_energy != old.reflected_energy


def _slices() -> None:
    _, fixed, solved = _inputs()
    item = fem_inputs(small_scheme(), solved, materials_by_wall=solved.materials_by_wall)
    assert item.wall_impedances[Wall.X0] == _normal(solved, item.frequencies_hz)
    assert fixed.impedance_by_wall is not None
    assert item.wall_impedances[Wall.X0] != fixed.impedance_by_wall[Wall.X0]
    from aosr.config.directivity_defaults import load_directivity_defaults
    from aosr.reporting.fem_slices import _checked_fem_inputs
    with patch.object(report_io, "solver_inputs", lambda inputs: solved):
        _, checked = _checked_fem_inputs(small_scheme(), capabilities=load_capabilities(config_path("capabilities.toml")),
            directivity=load_directivity_defaults(config_path("directivity_defaults.toml")))
    assert checked.wall_impedances[Wall.X0] == _normal(solved, checked.frequencies_hz)


def _screen() -> None:
    inputs, fixed, solved = _inputs()
    axis = (125., 503., 2017.)
    changed = reflection_screen.build_reflection_screen(inputs, axis, materials_by_wall=solved.materials_by_wall)
    plain = reflection_screen.build_reflection_screen(inputs, axis, materials_by_wall=fixed.materials_by_wall)
    assert fixed.impedance_by_wall is not None
    normal: dict[Wall, float | tuple[complex, ...] | None] = {**fixed.impedance_by_wall, Wall.X0: _normal(solved, axis)}
    wanted = tuple(reflection_screen._wall_pair(inputs, axis, face, normal) for face in reflection_screen._FACES)
    assert changed.pairs == wanted and changed.pairs != plain.pairs


def _table() -> None:
    _, fixed, solved = _inputs()
    axis, scatter = (125., 503., 2017.), (.21,)*3
    raw = SimpleNamespace(geometric_lane=SimpleNamespace(frequencies_hz=axis, scattering=scatter))
    changed = tables.build_path_table_section(raw, solved)
    plain = tables.build_path_table_section(raw, fixed)
    direct = tables.build_path_table(room=solved.room, source=solved.source, receiver=solved.receiver,
        source_model=solved.source_model, sound_speed_m_s=solved.sound_speed_m_s,
        rho_c_pa_s_per_m=solved.density_kg_m3 * solved.sound_speed_m_s,
         density_kg_m3=solved.density_kg_m3,
        materials_by_wall=solved.materials_by_wall, frequencies_hz=axis,
        scattering_coefficient=scatter, reflection_order_k=solved.reflection_order_k, furniture=None)
    from dataclasses import asdict
    assert changed == report_io.PathTableSection.model_validate(asdict(direct))
    assert changed.rows != plain.rows


def _window() -> None:
    inputs, fixed, solved = _inputs()
    axis, scatter = (125., 503., 2017.), (.21,)*3
    changed = reflection_window.build_reflection_window(inputs, frequencies_hz=axis,
        scattering_coefficient=scatter, window_s=.025, materials_by_wall=solved.materials_by_wall)
    plain = reflection_window.build_reflection_window(inputs, frequencies_hz=axis,
        scattering_coefficient=scatter, window_s=.025, materials_by_wall=fixed.materials_by_wall)
    direct = tables.build_path_table(room=solved.room, source=solved.source, receiver=solved.receiver,
        source_model=solved.source_model, sound_speed_m_s=solved.sound_speed_m_s,
        rho_c_pa_s_per_m=solved.density_kg_m3 * solved.sound_speed_m_s,
         density_kg_m3=solved.density_kg_m3,
        materials_by_wall=solved.materials_by_wall, frequencies_hz=axis,
        scattering_coefficient=scatter, reflection_order_k=changed.computed_order_k, furniture=None)
    from dataclasses import asdict
    wanted = tuple(report_io.PathRow.model_validate(asdict(row)) for row in direct.rows
                   if inputs.reflection_order_k < row.order <= changed.computed_order_k
                   and row.delay_s - changed.direct_delay_s <= changed.window_s)
    assert changed.rows == wanted and changed.rows != plain.rows


def _capability() -> None:
    _, fixed, solved = _inputs()
    table = load_capabilities(config_path("capabilities.toml"))
    record = report_output.report_capability(table, fixed.materials_by_wall)
    assert record.materials == "real_frequency_independent_impedance"
    construction = report_output.report_capability(table, solved.materials_by_wall)
    assert construction.materials == "assembly_by_wall"
    assert construction.record is not None and construction.record.status == "experimental"


def _pair_window() -> None:
    from aosr.reporting.evaluation import build_pair_window
    inputs, fixed, solved = _inputs()
    raw = _solve(fixed)
    output = report_output.output_from_report(raw, inputs=inputs, with_points=True, path_table_inputs=fixed)
    changed = build_pair_window(inputs, output, .025, materials_by_wall=solved.materials_by_wall)
    plain = build_pair_window(inputs, output, .025, materials_by_wall=fixed.materials_by_wall)
    wanted = reflection_window.build_reflection_window(inputs, frequencies_hz=raw.geometric_lane.frequencies_hz,
        scattering_coefficient=raw.geometric_lane.scattering, window_s=.025,
        materials_by_wall=solved.materials_by_wall)
    assert changed == wanted and changed.rows != plain.rows


def _evaluate_material_windows(args: tuple[object, ...], kwargs: dict[str, object],
                              fixed: report_io.SolverInputs, solved: report_io.SolverInputs) -> object:
    """保留真正評估器與建窗器，直接核每一對補算路徑的材料響應。"""
    from aosr.reporting import evaluation
    original = evaluation.build_pair_window
    changed_rows: list[bool] = []

    def window(inputs: report_io.ReportInput, output: report_io.ReportOutput, window_s: float,
               *, materials_by_wall: Mapping[Wall, SurfaceMaterial]) -> reflection_window.ReflectionWindow:
        actual = original(inputs, output, window_s, materials_by_wall=materials_by_wall)
        wanted = original(inputs, output, window_s, materials_by_wall=solved.materials_by_wall)
        plain = original(inputs, output, window_s, materials_by_wall=fixed.materials_by_wall)
        assert actual == wanted, "evaluate_parts 建窗時丟掉構造材料"
        changed_rows.append(wanted.rows != plain.rows)
        return actual

    with patch.object(evaluation, "build_pair_window", window):
        result = cast(Callable[..., object], evaluation.evaluate_parts)(*args, **kwargs)
    assert any(changed_rows), "案例必須含受構造影響的補算路徑，不能只驗空窗"
    return result


def _batch_guard(solved: report_io.SolverInputs, seen: list[object]) -> Callable[..., object]:
    """攔真正的批次求解：方案層交下去的必須是整份方案共用的那一份材料。"""
    original = report.solve_three_lane_reports

    def solve(**kwargs: object) -> object:
        seen.append(kwargs["materials_by_wall"])
        assert kwargs["materials_by_wall"] is solved.materials_by_wall
        return cast(Callable[..., object], original)(**kwargs)
    return solve


def _stage() -> None:
    """能力列與輸出逐欄比對尚未開構造；替換這兩道門，守正式呼叫接線。"""
    from aosr.config.directivity_defaults import load_directivity_defaults
    from aosr.reporting import physics_stage, pipeline
    from aosr.reporting.validation import checked_inputs
    from aosr.scoring.contract import CandidateEvaluation
    from tests.engine.test_scheme_pipeline import _many_fem
    table = load_capabilities(config_path("capabilities.toml"))
    directivity = load_directivity_defaults(config_path("directivity_defaults.toml"))
    scheme, documents = checked_inputs(small_scheme().model_copy(update={"purpose": "dedicated_two_channel_listening_room"}),
                                        capabilities=table, directivity=directivity)
    inputs = next(iter(documents.values()))[1]
    fixed = report_io.solver_inputs(inputs)
    board = _inputs()[2].materials_by_wall[Wall.X0]
    solved = fixed._replace(materials_by_wall={**fixed.materials_by_wall, Wall.X0: board})
    seen: list[report_io.SolverInputs] = []

    def capability(table: object, materials: object) -> report.ReportCapability:
        assert materials is solved.materials_by_wall
        return report._unchecked_capability()

    def output(raw: report.ThreeLaneReport, *, inputs: report_io.ReportInput, with_points: bool,
               path_table_inputs: report_io.SolverInputs, contact_rel: float | None) -> report_io.ReportOutput:
        seen.append(path_table_inputs)
        assert path_table_inputs.materials_by_wall is solved.materials_by_wall
        plain = _solve(fixed._replace(source=inputs.source_m, receiver=inputs.receiver_m))
        lane = raw.geometric_lane
        direct = _direct(solved._replace(source=inputs.source_m, receiver=inputs.receiver_m), lane.frequencies_hz)
        assert lane.reflected_energy == direct.reflected_energy, "批次求解吃到的不是指定的構造"
        assert lane.interference_energy == direct.interference_energy
        assert lane.reflected_energy != plain.geometric_lane.reflected_energy
        result = report_output.output_from_report(plain, inputs=inputs, with_points=with_points)
        fixed_pair = fixed._replace(source=inputs.source_m, receiver=inputs.receiver_m)
        return result.model_copy(update={"path_table": tables.build_path_table_section(plain, fixed_pair)})

    original_fem = fem_inputs

    def fem(scheme: object, solved: report_io.SolverInputs, *, materials_by_wall: object) -> object:
        assert materials_by_wall is solved.materials_by_wall
        result = cast(Callable[..., object], original_fem)(scheme, solved, materials_by_wall=materials_by_wall)
        assert getattr(result, "wall_impedances")[Wall.X0] == _normal(solved, getattr(result, "frequencies_hz"))
        return result

    def evaluate(*args: object, **kwargs: object) -> CandidateEvaluation:
        assert kwargs["materials_by_wall"] == solved.materials_by_wall
        return cast(CandidateEvaluation, _evaluate_material_windows(args, kwargs, fixed, solved))

    batches: list[object] = []

    with patch.object(report_io, "solver_inputs", lambda inputs: solved), \
            patch.object(physics_stage, "fem_inputs", fem), \
            patch.object(pipeline, "evaluate_parts", evaluate), \
            patch.object(physics_stage, "report_capability", capability), \
            patch.object(physics_stage, "output_from_report", output), \
            patch.object(report, "solve_three_lane_reports", _batch_guard(solved, batches)), \
            patch.object(report, "_solve_fem_energies", _many_fem):
        result = pipeline.run_scheme(scheme, capabilities=table, directivity=directivity,
            quality_targets_path=config_path("quality_targets.toml"), engine_commit="control",
            program_fingerprint="calc-v1:" + "0" * 64, physics_identity="phys-v1:" + "0" * 64, run_date=date(2026, 10, 13))
    assert seen and batches
    for pair in result.pairs:
        inputs = documents[pair.speaker_id, pair.receiver_id][1]
        wanted = reflection_screen.build_reflection_screen(inputs, pair.screen.frequencies_hz,
                                                            materials_by_wall=solved.materials_by_wall)
        old = reflection_screen.build_reflection_screen(inputs, pair.screen.frequencies_hz,
                                                        materials_by_wall=fixed.materials_by_wall)
        assert pair.screen == wanted and pair.screen.pairs != old.pairs


def _constant() -> None:
    """主線 ae4dd68a 的命令列實跑答案，三個替身原封不動共用，不從被測函式算摘要裁判。"""
    from aosr.physics import three_lane_report_cli as cli
    from tests.engine import _source_model_control as control
    path = Path(sys.argv[2]) / "constant.json"
    import json
    document = {**control.SCENE_WITHOUT_SOURCE_MODEL, "source_model": {"kind": "omnidirectional"}}
    path.write_text(json.dumps(document), encoding="utf-8")
    stream = io.StringIO()
    with patch.object(report, "_solve_fem_energy", control.fake_fem_energy), \
            patch.object(report, "_solve_report_late_decay", control.fast_late_decay), \
            patch.object(batch, "solve_geometric_late_energy", control.fake_late_energy), redirect_stdout(stream):
        code = cli.main([str(path), "--format", "json", "--points", "--path-table",
                         "--capabilities", str(config_path("capabilities.toml"))])
    assert code == 0
    assert hashlib.sha256(stream.getvalue().encode()).hexdigest() == \
        "471bdd64d46368ce4f8f993c604ccb5fa06213d15e26f5e223864a748ca25dcf"


def _constant_zones() -> None:
    """主線 ae4dd68a 的命令列實跑答案，三個替身原封不動共用，不從被測函式算摘要裁判。"""
    from aosr.physics import three_lane_report_cli as cli
    from tests.engine import _source_model_control as control
    path = Path(sys.argv[2]) / "constant-zones.json"
    import json
    document = {**control.SCENE_WITHOUT_SOURCE_MODEL, "source_model": {"kind": "omnidirectional"}}
    walls = document.pop("impedance_pa_s_per_m_by_wall")
    assert isinstance(walls, dict)
    document["surface_zones_by_wall"] = {name: [{"area": "whole_wall", "material": {
        "kind": "constant_impedance", "impedance_pa_s_m": value}}] for name, value in walls.items()}
    path.write_text(json.dumps(document), encoding="utf-8")
    stream = io.StringIO()
    with patch.object(report, "_solve_fem_energy", control.fake_fem_energy), \
            patch.object(report, "_solve_report_late_decay", control.fast_late_decay), \
            patch.object(batch, "solve_geometric_late_energy", control.fake_late_energy), redirect_stdout(stream):
        code = cli.main([str(path), "--format", "json", "--points", "--path-table",
                         "--capabilities", str(config_path("capabilities.toml"))])
    assert code == 0
    assert hashlib.sha256(stream.getvalue().encode()).hexdigest() == \
        "471bdd64d46368ce4f8f993c604ccb5fa06213d15e26f5e223864a748ca25dcf"


def _formal_zones() -> None:
    """新格式走真解析與三路準備；有限元素沿用替身，構造沒有假的舊阻抗表。"""
    import json

    inputs, fixed, _ = _inputs()
    document = json.loads(inputs.model_dump_json())
    walls = document.pop("impedance_pa_s_per_m_by_wall")
    zones = {name: [{"area": "whole_wall", "material": {
        "kind": "constant_impedance", "impedance_pa_s_m": value}}] for name, value in walls.items()}
    zones["x0"] = [{"area": "whole_wall", "material": {
        "kind": "assembly", "backing": "rigid", "layers": [
            {"kind": "board", "surface_mass_kg_m2": 0.15, "resistance_pa_s_m": 80.0},
            {"kind": "air", "thickness_m": 0.43}]}}]
    document["surface_zones_by_wall"] = zones
    from tests.engine._directivity import DIRECTIVITY

    parsed = report_io.load_input_document(document, load_capabilities(config_path("capabilities.toml")), DIRECTIVITY)
    solved = report_io.solver_inputs(parsed)
    assert solved.impedance_by_wall is None
    changed, plain = _solve(solved), _solve(fixed)
    assert changed.capability.materials == "assembly_by_wall"
    assert changed.eyring_t60_by_band_s != plain.eyring_t60_by_band_s
    assert changed.geometric_lane.reflected_energy != plain.geometric_lane.reflected_energy
    assert changed.geometric_lane.late_energy != plain.geometric_lane.late_energy


def _cli() -> None:
    """正式格式仍固定；只替換解碼後的材料及能力／輸出門，守命令列到求解的接線。"""
    from aosr.physics import three_lane_report_cli as cli
    inputs, fixed, solved = _inputs()
    path = Path(sys.argv[2]) / "room.json"
    path.write_text(inputs.model_dump_json(), encoding="utf-8")
    plain = _solve(fixed)
    output = report_output.output_from_report(plain, inputs=inputs, with_points=True)
    seen: list[object] = []

    def capability(table: object, materials: object) -> report.ReportCapability:
        assert materials is solved.materials_by_wall
        return report._unchecked_capability()

    def solve(**kwargs: object) -> report.ThreeLaneReport:
        seen.append(kwargs["materials_by_wall"])
        assert kwargs["materials_by_wall"] is solved.materials_by_wall
        changed = cast(Callable[..., report.ThreeLaneReport], report.solve_three_lane_report)(**kwargs)
        assert changed.geometric_lane.reflected_energy != plain.geometric_lane.reflected_energy
        return changed

    with patch.object(report_io, "solver_inputs", lambda inputs: solved), \
            patch.object(report_output, "report_capability", capability), \
            patch.object(report_output, "output_from_report", lambda *args, **kwargs: output), \
            patch.object(cli, "solve_three_lane_report", solve), \
            patch.object(report, "_solve_fem_energy", fake_fem_energy), redirect_stdout(io.StringIO()):
        code = cli.main([str(path), "--format", "json", "--points",
                         "--capabilities", str(config_path("capabilities.toml"))])
    assert code == 0 and seen


if __name__ == "__main__":
    assert bool(getattr(jax.config, "x64_enabled")) and jax.default_backend() == "cpu"
    modes = {"fem": _fem, "late": _late, "geometry": _geometry, "slices": _slices,
             "screen": _screen, "table": _table, "window": _window, "capability": _capability,
             "stage": _stage, "pair-window": _pair_window, "cli": _cli, "constant": _constant, "constant-zones": _constant_zones, "formal-zones": _formal_zones}
    modes[sys.argv[1]]()
