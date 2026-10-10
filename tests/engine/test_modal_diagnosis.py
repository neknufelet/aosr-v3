"""#671 核心診斷：只解一次小房，獨立重算殘數雙支與群和，不釘模態個數。"""
from __future__ import annotations

import json
import math
import platform
import shutil
from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from aosr.config.paths import config_path
from aosr.config.physics_constants import load_physics_constants
from aosr.config.quality_targets import load_quality_targets
from aosr.geometry.shoebox import Point, Room, Wall
from aosr.physics.fem_helmholtz import assemble_p2_operators
from aosr.physics.fem_modal import ModalSolverOptions
from aosr.physics.fem_modal_participation import ModalExpansion, overlap_groups, prepare_modal_expansion
from aosr.physics.modal_convention import ModalKind
from aosr.reporting import import_closure as closure_api
from aosr.reporting import modal_diagnosis as api
from aosr.reporting import modal_diagnosis_cache as cache_api
from aosr.reporting.modal_diagnosis_model import (
    ModalDiagnosis, ModalDiagnosisState, ModalKey, PlacementPair,
)
from aosr.reporting.modal_diagnosis_readout import read_modal_diagnosis
from aosr.reporting.physics_identity import PHYSICS_ENTRY_MODULE, physics_import_closure
from aosr.scoring.low_frequency_decay_reference import LowFrequencyDecayReference, load_reference

CONSTANTS = load_physics_constants(config_path("physics_constants.toml"))
RHO = CONSTANTS.air_density_kg_m3
C = CONSTANTS.sound_speed_m_s
WALLS = {wall: 10 * CONSTANTS.rho_c for wall in Wall.all()}
ROOM = Room(2.0, 1.7, 1.3)
SOURCES = {"left": Point(0.32, 0.4, 0.5), "right": Point(0.36, 1.2, 0.5)}
RECEIVERS = {"main": Point(1.5, 0.8, 0.6), "side": Point(1.4, 1.05, 0.55)}


@dataclass(frozen=True)
class Sample:
    diagnosis: ModalDiagnosis
    cached: cache_api.CachedRoom
    expansion: ModalExpansion
    cache_dir: Path


def _key(room: Room = ROOM, *, walls: dict[Wall, float] = WALLS,
         density: float = RHO, speed: float = C,
         options: ModalSolverOptions = ModalSolverOptions()) -> ModalKey:
    return ModalKey.from_inputs(room=room, wall_impedances=walls, density_kg_m3=density,
                                sound_speed_m_s=speed, options=options)


def _diagnose(cache_dir: Path, *, room: Room = ROOM, walls: dict[Wall, float] = WALLS,
              sources: dict[str, Point] = SOURCES, receivers: dict[str, Point] = RECEIVERS) -> ModalDiagnosis:
    return api.diagnose_modes(room=room, wall_impedances=walls, density_kg_m3=RHO,
                               sound_speed_m_s=C, sources=sources, receivers=receivers,
                               cache_dir=cache_dir)


@pytest.fixture(scope="module")
def small(tmp_path_factory: pytest.TempPathFactory) -> Sample:
    # worksteal（工作偷取）會分題；共用本次 pytest 暫存根與產品鎖，跨工人仍只解一次。
    base = tmp_path_factory.getbasetemp()
    if base.name.startswith("popen-gw"):
        base = base.parent
    cache_dir = base / "modal-diagnosis-small-cache"
    diagnosis = _diagnose(cache_dir)
    assert diagnosis.state is ModalDiagnosisState.DIAGNOSED_NOT_SCORED, diagnosis.reason_text
    cached = cache_api.load_modal_cache(cache_dir=cache_dir, key=_key(), modal_identity=api.modal_identity())
    assert cached is not None
    mesh = api.mesh_from_key(_key())
    expansion = prepare_modal_expansion(cached.spectrum, assemble_p2_operators(mesh),
        wall_impedances=WALLS, density_kg_m3=RHO, sound_speed_m_s=C)
    return Sample(diagnosis, cached, expansion, cache_dir)


def _reference() -> LowFrequencyDecayReference:
    purpose = load_quality_targets(config_path("quality_targets.toml")).purpose("dedicated_two_channel_listening_room")
    return load_reference(purpose)


def test_diagnosis_entry_exists() -> None:
    assert find_spec("aosr.reporting.modal_diagnosis") is not None


def test_import_scan_accepts_an_entry_without_changing_physics() -> None:
    assert physics_import_closure(entry_module=PHYSICS_ENTRY_MODULE) == physics_import_closure()


def test_key_covers_official_mesh_and_all_solver_fields() -> None:
    from aosr.config.fem_lane import FEM_ELEMENTS_PER_WAVELENGTH, FEM_MESH_RANDOM_SEED
    from aosr.config.frequency_axis import FEM_GEOMETRIC_CROSSOVER_CAP_HZ
    key = _key()
    assert key == _key()
    assert key.room_hex == tuple(float(x).hex() for x in (ROOM.Lx, ROOM.Ly, ROOM.Lz))
    assert key.impedance_hex == tuple(WALLS[w].hex() for w in Wall.all())
    assert key.mesh_frequency_max_hex == float(FEM_GEOMETRIC_CROSSOVER_CAP_HZ).hex()
    assert key.elements_per_wavelength == FEM_ELEMENTS_PER_WAVELENGTH
    assert key.mesh_random_seed == FEM_MESH_RANDOM_SEED
    assert set(key.model_dump()["solver_options"]) == {f.name for f in fields(ModalSolverOptions)}
    assert {"frequency_hex", "frequencies_hz", "sources", "receivers"}.isdisjoint(ModalKey.model_fields)


@pytest.mark.parametrize("axis", range(3))
def test_key_changes_for_each_room_edge(axis: int) -> None:
    dimensions = [ROOM.Lx, ROOM.Ly, ROOM.Lz]
    dimensions[axis] += 0.1
    assert _key(Room(*dimensions)).digest != _key().digest


@pytest.mark.parametrize("wall", Wall.all())
def test_key_changes_for_each_wall(wall: Wall) -> None:
    assert _key(walls={**WALLS, wall: WALLS[wall] * 1.1}).digest != _key().digest


@pytest.mark.parametrize("field", ["density", "speed"])
def test_key_changes_for_medium(field: str) -> None:
    original = RHO if field == "density" else C
    changed = _key(density=original * 1.1) if field == "density" else _key(speed=original * 1.1)
    assert changed.digest != _key().digest


@pytest.mark.parametrize("field", [f.name for f in fields(ModalSolverOptions)])
def test_key_changes_for_every_solver_option(field: str) -> None:
    options = ModalSolverOptions()
    value = getattr(options, field)
    changed = 100 if value is None else value * 2
    assert _key(options=replace(options, **{field: changed})).digest != _key().digest


@pytest.mark.parametrize("field", ["mesh_frequency_max_hex", "elements_per_wavelength", "mesh_random_seed"])
def test_key_changes_for_every_mesh_parameter(field: str) -> None:
    key = _key()
    value = getattr(key, field)
    changed = float(float.fromhex(value) * 1.1).hex() if isinstance(value, str) else value + 1
    other = ModalKey.model_validate({**key.model_dump(), field: changed})
    assert other.digest != key.digest


def test_positions_reuse_same_room_cache_without_solving(small: Sample, monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("快取命中後不准再次求解")
    monkeypatch.setattr(api, "solve_fem_modes", forbidden)
    moved = _diagnose(small.cache_dir, sources={"moved": Point(0.6, 0.7, 0.5)},
                      receivers={"new": Point(1.6, 0.9, 0.6)})
    assert moved.state is ModalDiagnosisState.DIAGNOSED_NOT_SCORED, moved.reason_text
    assert moved.key == small.diagnosis.key
    assert moved.room_layer == small.diagnosis.room_layer
    assert moved.placement_layer != small.diagnosis.placement_layer


def test_modal_closure_contains_solver_and_excludes_scoring_registries() -> None:
    closure = api.modal_import_closure()
    assert {"aosr.reporting.modal_diagnosis", "aosr.physics.fem_modal",
            "aosr.physics.fem_modal_check", "aosr.physics.fem_modal_participation",
            "aosr.physics.modal_convention"} <= set(closure.modules)
    assert not any(name.startswith(("aosr.scoring", "aosr.search", "aosr.gui")) for name in closure.modules)
    assert {"aosr.config.quality_targets", "aosr.config.capabilities",
            "aosr.reporting.modal_diagnosis_readout"}.isdisjoint(closure.modules)
    assert closure.data_files == (config_path("fem_lane.toml").name,), (
        "模態閉包多讀了資料檔：決定它進鑰匙、進模態身分，還是在這裡明列放行（決策紙代價段）")
    assert "aosr.reporting.modal_diagnosis" not in physics_import_closure().modules
    assert api.modal_identity().startswith("modal-v1:")


@pytest.mark.parametrize("module", ["aosr.physics.fem_modal", "aosr.physics.fem_modal_participation", "aosr.reporting.modal_diagnosis"])
def test_modal_code_change_flips_identity(tmp_path: Path, module: str) -> None:
    root = Path(shutil.copytree(config_path("capabilities.toml").parents[2], tmp_path / "aosr"))
    before = api.modal_identity(package_root=root)
    path = root.joinpath(*module.split(".")[1:]).with_suffix(".py")
    path.write_text(path.read_text() + "\nMODAL_TEST_CHANGE = True\n")
    assert api.modal_identity(package_root=root) != before


def test_quality_registry_change_does_not_flip_modal_identity(tmp_path: Path) -> None:
    root = Path(shutil.copytree(config_path("capabilities.toml").parents[2], tmp_path / "aosr"))
    before = api.modal_identity(package_root=root)
    path = root / "config" / "data" / "quality_targets.toml"
    original = path.read_text()
    changed = original.replace("value = [0.51, 0.30, 0.12]", "value = [0.61, 0.40, 0.22]")
    assert changed != original
    path.write_text(changed)
    assert api.modal_identity(package_root=root) == before


def test_cache_round_trip_preserves_ledger_and_shapes(small: Sample, tmp_path: Path) -> None:
    cache_api.write_modal_cache(cache_dir=tmp_path, cached=small.cached)
    restored = cache_api.load_modal_cache(cache_dir=tmp_path, key=_key(), modal_identity=api.modal_identity())
    assert restored is not None
    assert restored.room_layer == small.cached.room_layer
    assert replace(restored.spectrum, solutions=()) == replace(small.cached.spectrum, solutions=())
    for got, want in zip(restored.spectrum.solutions, small.cached.spectrum.solutions, strict=True):
        assert replace(got, shape=None) == replace(want, shape=None)
        assert got.shape is not None and want.shape is not None
        assert np.array_equal(got.shape, want.shape)
    assert any(_key().digest in path.name for path in tmp_path.iterdir())


def test_rewriting_a_key_keeps_only_the_published_shape_file(small: Sample, tmp_path: Path) -> None:
    """同鑰匙重寫（例如程式身分變了要重解）：舊形狀檔沒有標頭指著，要清掉，不然每次重解多留一份。"""
    cache_api.write_modal_cache(cache_dir=tmp_path, cached=small.cached)
    first = {path.name for path in tmp_path.glob("*.npz")}
    cache_api.write_modal_cache(cache_dir=tmp_path, cached=small.cached)
    header = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert {path.name for path in tmp_path.glob("*.npz")} == {header["payload_name"]}
    assert header["payload_name"] not in first
    assert cache_api.load_modal_cache(cache_dir=tmp_path, key=_key(), modal_identity=api.modal_identity()) is not None


@pytest.mark.parametrize("field", ["key", "modal_identity", "schema_version"])
def test_cache_rejects_header_mismatch(small: Sample, tmp_path: Path, field: str) -> None:
    cache_api.write_modal_cache(cache_dir=tmp_path, cached=small.cached)
    header_path = next(tmp_path.glob("*.json"))
    header = json.loads(header_path.read_text())
    header[field] = (_key(density=RHO * 1.1).model_dump(mode="json") if field == "key"
                     else "modal-v1:" + "0" * 64 if field == "modal_identity" else "future")
    header_path.write_text(json.dumps(header))
    assert cache_api.load_modal_cache(cache_dir=tmp_path, key=_key(), modal_identity=api.modal_identity()) is None


def test_cache_rejects_payload_corruption(small: Sample, tmp_path: Path) -> None:
    cache_api.write_modal_cache(cache_dir=tmp_path, cached=small.cached)
    next(tmp_path.glob("*.npz")).write_bytes(b"unfinished")
    assert cache_api.load_modal_cache(cache_dir=tmp_path, key=_key(), modal_identity=api.modal_identity()) is None


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("at", ["npz_write", "header_publish"])
def test_interrupted_cache_never_exposes_half_a_pair(small: Sample, tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch, existing: bool, at: str) -> None:
    if existing:
        cache_api.write_modal_cache(cache_dir=tmp_path, cached=small.cached)
    original = Path.replace
    def interrupt_replace(path: Path, target: str | Path) -> Path:
        if path.suffix == ".json":
            raise KeyboardInterrupt("標頭發布前中斷")
        return original(path, target)
    def interrupt_npz(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt("形狀寫一半中斷")
    if at == "header_publish":
        monkeypatch.setattr(Path, "replace", interrupt_replace)
    else:
        monkeypatch.setattr(np, "savez_compressed", interrupt_npz)
    with pytest.raises(KeyboardInterrupt):
        cache_api.write_modal_cache(cache_dir=tmp_path, cached=small.cached)
    got = cache_api.load_modal_cache(cache_dir=tmp_path, key=_key(), modal_identity=api.modal_identity())
    assert (got is not None) == existing
    if got is not None:
        assert got.room_layer == small.cached.room_layer
    assert not any(path.is_dir() for path in tmp_path.iterdir())


def test_room_layer_copies_the_given_spectrum(small: Sample) -> None:
    """餵一份做過記號的頻譜給被測函式，再拿輸出比這份輸入；不拿房間層去比從房間層重建的快取（那是自己比自己）。"""
    base = small.cached.spectrum
    check = base.check
    assert check is not None
    marked_check = replace(check,
        static_count=check.static_count + 1,
        zero_mode_continuation_count=check.zero_mode_continuation_count + 2,
        overdamped_count=check.overdamped_count + 3,
        unconfirmed_decay_count=check.unconfirmed_decay_count + 4,
        returned_above_limit_count=check.returned_above_limit_count + 5,
        weyl_terms=check.weyl_terms + " [marked]",
        count_bands=tuple(replace(b, found_resonances=b.found_resonances + 6 + i)
                          for i, b in enumerate(check.count_bands)))
    source = replace(base, check=marked_check)
    fingerprint = "a" * 64
    layer = api.room_layer_from_spectrum(source, key=_key(), identity=api.modal_identity(),
                                         mesh_sha256=fingerprint, solve_seconds=small.cached.room_layer.solve_seconds)
    assert layer.mesh_sha256 == fingerprint
    assert tuple(g.member_indices for g in layer.groups) == overlap_groups(source)
    assert any(len(g.member_indices) > 1 for g in layer.groups)
    for mode in layer.modes:
        raw = source.solutions[mode.mode_index]
        assert mode.kind is raw.kind
        assert mode.frequency_hz == raw.frequency_hz
        assert (mode.omega_real_rad_s, mode.omega_imag_rad_s) == (raw.omega.real, raw.omega.imag)
        assert (mode.t60_s, mode.q) == (raw.t60_s, raw.q)
        if raw.kind is ModalKind.RESONANCE:
            assert mode.t60_s == 3 * math.log(10) / raw.omega.imag
            assert mode.q == raw.omega.real / (2 * raw.omega.imag)
            group = next(g for g in layer.groups if mode.mode_index in g.member_indices)
            assert mode.group_id == group.group_id
            assert (group.lower_hz, group.upper_hz) == (
                source.solutions[group.member_indices[0]].frequency_hz,
                source.solutions[group.member_indices[-1]].frequency_hz)
        else:
            assert mode.group_id is None
            if raw.kind is ModalKind.STATIC:
                assert mode.t60_s is None and mode.q is None
    assert layer.check_summary.model_dump() == {
        f.name: getattr(marked_check, f.name) if f.name != "count_bands" else tuple(
            {bf.name: getattr(b, bf.name) for bf in fields(b)} for b in marked_check.count_bands)
        for f in fields(marked_check)}


def test_diagnosis_layer_records_the_mesh_it_was_solved_on(small: Sample) -> None:
    layer = small.diagnosis.room_layer
    assert layer is not None
    assert layer.mesh_sha256 == api.mesh_fingerprint(api.mesh_from_key(_key()))


def test_mesh_fingerprint_changes_when_any_mesh_array_changes() -> None:
    mesh = api.mesh_from_key(_key())
    moved = mesh.nodes.copy()
    moved[-1, 0] = np.nextafter(moved[-1, 0], np.inf)
    renumbered = mesh.tetrahedra[:, [1, 0, 2, 3]]
    variants = [replace(mesh, nodes=moved), replace(mesh, tetrahedra=renumbered),
                replace(mesh, boundary_triangles=mesh.boundary_triangles[::-1].copy()),
                replace(mesh, boundary_wall_indices=mesh.boundary_wall_indices[::-1].copy())]
    prints = {api.mesh_fingerprint(m) for m in (mesh, *variants)}
    assert len(prints) == 1 + len(variants)


def test_cache_from_a_different_mesh_is_resolved_not_failed(small: Sample, tmp_path: Path,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    """同一把鑰匙、當初解的網格跟現在照鑰匙重建的不同（例如網格產生器升版）：當成沒有快取、重解覆寫，不是每次都判失敗。"""
    stale_layer = small.cached.room_layer.model_copy(update={"mesh_sha256": "0" * 64})
    cache_api.write_modal_cache(cache_dir=tmp_path, cached=cache_api.CachedRoom(stale_layer, small.cached.spectrum))
    calls: list[str] = []
    def resolve(*args: object, **kwargs: object) -> object:
        calls.append("solve")
        return small.cached.spectrum
    monkeypatch.setattr(api, "solve_fem_modes", resolve)
    got = _diagnose(tmp_path)
    assert calls == ["solve"]
    assert got.state is ModalDiagnosisState.DIAGNOSED_NOT_SCORED, got.reason_text
    assert got.room_layer is not None
    assert got.room_layer.mesh_sha256 == api.mesh_fingerprint(api.mesh_from_key(_key()))
    reread = cache_api.load_modal_cache(cache_dir=tmp_path, key=_key(), modal_identity=api.modal_identity())
    assert reread is not None and reread.room_layer.mesh_sha256 == got.room_layer.mesh_sha256


def test_solver_library_upgrade_flips_modal_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    before = api.modal_identity()
    real = closure_api.physics_dependency_versions()
    bumped = tuple((name, version + ".post1" if name == "gmsh" else version) for name, version in real)
    monkeypatch.setattr(closure_api, "physics_dependency_versions", lambda: bumped)
    assert api.modal_identity() != before


@pytest.mark.parametrize("part", ["python", "machine", "libc"])
def test_every_environment_part_flips_the_environment_digest(monkeypatch: pytest.MonkeyPatch, part: str) -> None:
    """模態身分與物理身分共用的環境摘要：Python 版本、平台、C 函式庫任一換了，摘要都要換。"""
    before = closure_api.environment_digest()
    if part == "python":
        monkeypatch.setattr(closure_api, "sys", SimpleNamespace(version_info=(3, 0, 0)))
    elif part == "machine":
        monkeypatch.setattr(platform, "machine", lambda: "other-machine")
    else:
        monkeypatch.setattr(platform, "libc_ver", lambda: ("otherlibc", "0.0"))
    assert closure_api.environment_digest() != before


def test_expansion_failure_after_a_fresh_solve_keeps_the_room_cache(small: Sample, tmp_path: Path,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    """展開丟錯：解完的房間層已經落地，同房同材料下一次不用重解，失敗結果也帶得出房間層。"""
    def broken(*args: object, **kwargs: object) -> None:
        raise ArithmeticError("重根沒有做 B 雙線性正交")
    def must_not_solve(*args: object, **kwargs: object) -> None:
        raise AssertionError("同房同材料第二次不該重解")
    monkeypatch.setattr(api, "prepare_modal_expansion", broken)
    monkeypatch.setattr(api, "solve_fem_modes", lambda *args, **kwargs: small.cached.spectrum)
    first = _diagnose(tmp_path)
    monkeypatch.setattr(api, "solve_fem_modes", must_not_solve)
    second = _diagnose(tmp_path)
    for got in (first, second):
        assert got.state is ModalDiagnosisState.FAILED
        assert got.reason_text == "ArithmeticError('重根沒有做 B 雙線性正交')"
    assert first.room_layer is not None and second.room_layer == first.room_layer


def test_placement_magnitudes_equal_independent_double_branch_sum(small: Sample) -> None:
    layer = small.diagnosis.placement_layer
    assert layer is not None
    lookup = small.expansion.at_positions(tuple(SOURCES.values()), tuple(RECEIVERS.values()))
    entries = {(r.source_index, r.receiver_index, r.mode_index): r for r in lookup.entries}
    source_indices = {name: i for i, name in enumerate(SOURCES)}
    receiver_indices = {name: i for i, name in enumerate(RECEIVERS)}
    for pair in layer.pairs:
        residues = {i: entries[source_indices[pair.speaker_id], receiver_indices[pair.receiver_id], i].residue_k
                    for i in pair.resonance_indices}
        for i, own, whole in zip(pair.resonance_indices, pair.resonance_magnitude, pair.group_magnitude, strict=True):
            mode = small.cached.spectrum.solutions[i]
            k = np.asarray([mode.omega.real / C], dtype=np.complex128)
            def term(index: int) -> np.complex128:
                pole = small.cached.spectrum.solutions[index].omega / C
                residue = residues[index]
                pressure = residue / (k - pole) - residue.conjugate() / (k + pole.conjugate())
                return np.complex128(pressure[0])
            expected_own = abs(term(i))
            group = next(g for g in overlap_groups(small.cached.spectrum) if i in g)
            expected_group = abs(sum(term(j) for j in group))
            row = entries[source_indices[pair.speaker_id], receiver_indices[pair.receiver_id], i]
            assert own == expected_own == row.resonance_magnitude
            assert whole == expected_group == row.group_magnitude
    assert all("db" not in f for f in PlacementPair.model_fields)


@pytest.mark.parametrize("state", list(ModalDiagnosisState))
def test_all_four_states_can_be_constructed(small: Sample, state: ModalDiagnosisState) -> None:
    if state is ModalDiagnosisState.DIAGNOSED_NOT_SCORED:
        got = small.diagnosis
    else:
        got = ModalDiagnosis(state=state, reason_text="求解原文" if state is ModalDiagnosisState.FAILED else None,
                             reason_code="unsupported_impedance" if state is ModalDiagnosisState.OUT_OF_SCOPE else None)
    assert got.state is state
    assert got.model_dump()["schema_version"] == "aosr.modal_diagnosis.v1"
    if state is not ModalDiagnosisState.DIAGNOSED_NOT_SCORED:
        assert got.room_layer is None and got.placement_layer is None


@pytest.mark.parametrize("payload", [
    {"state": "diagnosed_not_scored"}, {"state": "failed"},
    {"state": "failed", "reason_text": "   "}, {"state": "out_of_scope"},
    {"state": "out_of_scope", "reason_code": ""}, {"state": "not_computed", "extra": True},
    {"state": "not_computed", "schema_version": "future"},
])
def test_model_rejects_inconsistent_states_and_extra_fields(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ModalDiagnosis.model_validate(payload)


@pytest.mark.parametrize("change", ["state", "key", "modal_identity", "resonance_indices", "pairs"])
def test_model_rejects_mismatched_layers(small: Sample, change: str) -> None:
    payload = small.diagnosis.model_dump(mode="json")
    if change == "state":
        payload["state"] = "not_computed"
    elif change == "key":
        payload["key"] = _key(density=RHO * 1.1).model_dump(mode="json")
    elif change == "modal_identity":
        payload["modal_identity"] = "modal-v1:" + "0" * 64
    elif change == "resonance_indices":
        payload["placement_layer"]["pairs"][0]["resonance_indices"] = []
    else:
        payload["placement_layer"]["pairs"] = []
    with pytest.raises(ValidationError):
        ModalDiagnosis.model_validate(payload)


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_models_reject_nonfinite_values(small: Sample, value: float) -> None:
    payload = small.diagnosis.model_dump()
    payload["placement_layer"]["pairs"][0]["resonance_magnitude"] = (value,)
    with pytest.raises(ValidationError):
        ModalDiagnosis.model_validate(payload)
    with pytest.raises(ValidationError):
        ModalKey.model_validate({**_key().model_dump(), "density_hex": value.hex()})


def test_models_are_frozen_and_json_round_trip(small: Sample) -> None:
    assert ModalDiagnosis.model_validate_json(small.diagnosis.model_dump_json()) == small.diagnosis
    with pytest.raises(ValidationError):
        small.diagnosis.state = ModalDiagnosisState.NOT_COMPUTED


@pytest.mark.parametrize("impedance", [0.0, -1.0, math.inf, math.nan, 1.0 + 1.0j, None])
def test_unsupported_impedance_is_out_of_scope(tmp_path: Path, impedance: object) -> None:
    walls = {**WALLS, Wall.X0: impedance}
    got = api.diagnose_modes(room=ROOM, wall_impedances=walls, density_kg_m3=RHO,
        sound_speed_m_s=C, sources=SOURCES, receivers=RECEIVERS, cache_dir=tmp_path)
    assert got.state is ModalDiagnosisState.OUT_OF_SCOPE
    assert got.reason_code == "unsupported_impedance"
    assert got.room_layer is None and got.placement_layer is None
    assert not tuple(tmp_path.iterdir())


def test_solve_failure_preserves_original_without_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    errors = iter([ArithmeticError("原始零根錯誤：λ = (1+2j)")])
    def broken(*args: object, **kwargs: object) -> None:
        raise next(errors)
    monkeypatch.setattr(api, "solve_fem_modes", broken)
    got = _diagnose(tmp_path)
    assert got.state is ModalDiagnosisState.FAILED
    assert got.reason_text == "ArithmeticError('原始零根錯誤：λ = (1+2j)')"
    assert got.room_layer is None and got.placement_layer is None


def test_position_failure_preserves_original(small: Sample) -> None:
    got = _diagnose(small.cache_dir, receivers={"outside": Point(10.0, 10.0, 10.0)})
    assert got.state is ModalDiagnosisState.FAILED
    with pytest.raises(ValueError) as error:
        small.expansion.at_positions(tuple(SOURCES.values()), (Point(10.0, 10.0, 10.0),))
    assert got.reason_text == repr(error.value)


def test_undamped_infinite_magnitude_fails_whole_diagnosis(small: Sample, monkeypatch: pytest.MonkeyPatch) -> None:
    original = small.expansion.at_positions(tuple(SOURCES.values()), tuple(RECEIVERS.values()))
    infinite = replace(original, entries=tuple(replace(row, resonance_magnitude=math.inf)
        if row.mode.kind is ModalKind.RESONANCE else row for row in original.entries))
    monkeypatch.setattr(ModalExpansion, "at_positions", lambda *args: infinite)
    got = _diagnose(small.cache_dir)
    assert got.state is ModalDiagnosisState.FAILED
    assert got.reason_text is not None and "非有限" in got.reason_text
    assert got.placement_layer is None


def test_reference_outside_high_end_is_none_and_not_assessed(small: Sample) -> None:
    readout = read_modal_diagnosis(small.diagnosis, _reference())
    assert readout.resonances is not None
    outside = [m for m in readout.resonances if m.frequency_hz > 250.0]
    assert outside
    assert all(m.target_t60_s is None and m.excess_t60_s is None and m.target_origin == "not_assessed" for m in outside)
    assert small.diagnosis.state is ModalDiagnosisState.DIAGNOSED_NOT_SCORED


def test_reference_outside_low_end_is_none_and_not_assessed() -> None:
    from tests.engine._modal_cases import sample
    diagnosis = sample()[0]  # 明列 25 Hz，只考參考線讀回，不為這題另解一間房。
    assert diagnosis.state is ModalDiagnosisState.DIAGNOSED_NOT_SCORED, diagnosis.reason_text
    readout = read_modal_diagnosis(diagnosis, _reference())
    assert readout.resonances is not None
    rows = [m for m in readout.resonances if m.frequency_hz < 32.0]
    assert rows
    assert all(m.excess_t60_s is None and m.target_origin == "not_assessed" for m in rows)


def test_readout_uses_current_reference_without_resolving(small: Sample) -> None:
    reference = _reference()
    assert reference.target.extension is not None
    changed = replace(reference, target=replace(reference.target,
        t60_s=tuple(x * 2 for x in reference.target.t60_s),
        extension=(reference.target.extension[0], reference.target.extension[1], reference.target.extension[2] * 2)))
    before = small.diagnosis.model_dump_json()
    old = read_modal_diagnosis(small.diagnosis, reference)
    new = read_modal_diagnosis(small.diagnosis, changed)
    assert old != new
    assert small.diagnosis.model_dump_json() == before
    assert new.resonances is not None
    for row in new.resonances:
        target = changed.target_at(row.frequency_hz)
        assert row.target_t60_s == target.t60_s
        assert row.target_origin == target.origin.value
        assert row.excess_t60_s == (None if target.t60_s is None else max(0.0, row.t60_s - target.t60_s))


def test_group_summaries_and_profiles_have_no_representative(small: Sample) -> None:
    readout = read_modal_diagnosis(small.diagnosis, _reference())
    room = small.diagnosis.room_layer
    placement = small.diagnosis.placement_layer
    assert room is not None and placement is not None
    assert readout.resonances is not None and readout.groups is not None and readout.placements is not None
    modes = {m.mode_index: m for m in readout.resonances}
    for summary, group in zip(readout.groups, room.groups, strict=True):
        members = [modes[i] for i in group.member_indices]
        assert summary.member_count == len(members)
        assert summary.frequency_span_hz == group.upper_hz - group.lower_hz
        assert summary.t60_range_s == (min(m.t60_s for m in members), max(m.t60_s for m in members))
        assert summary.exceeding_member_count == sum(m.excess_t60_s is not None and m.excess_t60_s > 0 for m in members)
        assert summary.not_assessed_member_count == sum(m.target_origin == "not_assessed" for m in members)
        assert {"representative", "representative_member", "t60_s", "excess_t60_s"}.isdisjoint(type(summary).model_fields)
    for pair_view, pair in zip(readout.placements, placement.pairs, strict=True):
        values = dict(zip(pair.resonance_indices, pair.group_magnitude, strict=True))
        for profile, group in zip(pair_view.group_profiles, room.groups, strict=True):
            assert profile.mode_indices == group.member_indices
            assert profile.frequencies_hz == tuple(modes[i].frequency_hz for i in group.member_indices)
            assert profile.group_magnitude == tuple(values[i] for i in group.member_indices)


def test_ranking_and_db_are_relative_to_pair_strongest_resonance(small: Sample) -> None:
    readout = read_modal_diagnosis(small.diagnosis, _reference())
    placement = small.diagnosis.placement_layer
    assert placement is not None
    assert readout.placements is not None
    for view, pair in zip(readout.placements, placement.pairs, strict=True):
        expected = sorted(zip(pair.resonance_indices, pair.resonance_magnitude, strict=True), key=lambda row: (-row[1], row[0]))
        assert [(r.mode_index, r.resonance_magnitude) for r in view.sorted_resonances] == expected
        strongest = max(pair.resonance_magnitude)
        for row in view.sorted_resonances:
            assert row.db_relative_to_pair_strongest_resonance == 20 * math.log10(row.resonance_magnitude / strongest)


@pytest.mark.parametrize("all_zero", [False, True])
def test_zero_magnitude_db_is_explicitly_undefined(small: Sample, all_zero: bool) -> None:
    payload = small.diagnosis.model_dump()
    pair = payload["placement_layer"]["pairs"][0]
    pair["resonance_magnitude"] = tuple(0.0 if all_zero or i == 0 else x
        for i, x in enumerate(pair["resonance_magnitude"]))
    diagnosis = ModalDiagnosis.model_validate(payload)
    view = read_modal_diagnosis(diagnosis, _reference())
    assert view.placements is not None
    rows = view.placements[0].sorted_resonances
    zeros = [r for r in rows if r.resonance_magnitude == 0.0]
    assert zeros and all(r.db_relative_to_pair_strongest_resonance is None for r in zeros)


@pytest.mark.parametrize("state", [ModalDiagnosisState.NOT_COMPUTED, ModalDiagnosisState.FAILED, ModalDiagnosisState.OUT_OF_SCOPE])
def test_uncomputed_readout_does_not_fabricate_measurements(state: ModalDiagnosisState) -> None:
    diagnosis = ModalDiagnosis(state=state, reason_text="原文" if state is ModalDiagnosisState.FAILED else None,
                              reason_code="unsupported" if state is ModalDiagnosisState.OUT_OF_SCOPE else None)
    got = read_modal_diagnosis(diagnosis, _reference())
    assert got.state is state
    assert got.resonances is None and got.groups is None and got.placements is None


def test_huge_impedance_is_out_of_scope_without_throwing(tmp_path: Path) -> None:
    got = api.diagnose_modes(room=ROOM, wall_impedances={**WALLS, Wall.X0: 10**1000},
        density_kg_m3=RHO, sound_speed_m_s=C, sources=SOURCES, receivers=RECEIVERS, cache_dir=tmp_path)
    assert got.state is ModalDiagnosisState.OUT_OF_SCOPE
    assert got.reason_code == "unsupported_impedance"


def test_zero_guarantee_height_has_no_fabricated_minimum_t60(small: Sample) -> None:
    check = small.cached.spectrum.check
    assert check is not None
    no_guarantee = replace(check, guaranteed_decay_rate_rad_s=0.0, guaranteed_min_t60_s=math.inf)
    layer = api.room_layer_from_spectrum(replace(small.cached.spectrum, check=no_guarantee),
        key=_key(), identity=api.modal_identity(), mesh_sha256=small.cached.room_layer.mesh_sha256,
        solve_seconds=small.cached.room_layer.solve_seconds)
    assert layer.check_summary.guaranteed_min_t60_s is None
    assert layer.check_summary.guaranteed_decay_rate_rad_s == 0.0


@pytest.mark.parametrize("field", ["room", "mode", "group", "check", "band", "placement", "pair"])
def test_nested_models_reject_extra_fields(small: Sample, field: str) -> None:
    payload = small.diagnosis.model_dump()
    choices = {"room": payload["room_layer"], "mode": payload["room_layer"]["modes"][0],
               "group": payload["room_layer"]["groups"][0], "check": payload["room_layer"]["check_summary"],
               "band": payload["room_layer"]["check_summary"]["count_bands"][0],
               "placement": payload["placement_layer"], "pair": payload["placement_layer"]["pairs"][0]}
    choices[field]["unexpected"] = True
    with pytest.raises(ValidationError):
        ModalDiagnosis.model_validate(payload)


@pytest.mark.parametrize("field", ["t60_s", "q", "group_id"])
def test_resonance_model_does_not_accept_missing_quantities(small: Sample, field: str) -> None:
    payload = small.diagnosis.model_dump()
    mode = next(m for m in payload["room_layer"]["modes"] if m["kind"] is ModalKind.RESONANCE)
    mode[field] = None
    with pytest.raises(ValidationError):
        ModalDiagnosis.model_validate(payload)


def test_only_resonances_can_belong_to_groups(small: Sample) -> None:
    payload = small.diagnosis.model_dump()
    static = next(m for m in payload["room_layer"]["modes"] if m["kind"] is ModalKind.STATIC)
    static["group_id"] = payload["room_layer"]["groups"][0]["group_id"]
    with pytest.raises(ValidationError):
        ModalDiagnosis.model_validate(payload)


def test_missing_magnitude_fails_without_zero_substitution(small: Sample, monkeypatch: pytest.MonkeyPatch) -> None:
    original = small.expansion.at_positions(tuple(SOURCES.values()), tuple(RECEIVERS.values()))
    missing = replace(original, entries=tuple(replace(row, group_magnitude=None)
        if row.mode.kind is ModalKind.RESONANCE else row for row in original.entries))
    monkeypatch.setattr(ModalExpansion, "at_positions", lambda *args: missing)
    got = _diagnose(small.cache_dir)
    assert got.state is ModalDiagnosisState.FAILED
    assert got.reason_text is not None and "缺少絕對大小" in got.reason_text
    assert got.placement_layer is None


def test_cache_path_has_no_default() -> None:
    entry: Callable[..., ModalDiagnosis] = api.diagnose_modes
    with pytest.raises(TypeError, match="cache_dir"):
        entry(room=ROOM, wall_impedances=WALLS, density_kg_m3=RHO,
              sound_speed_m_s=C, sources=SOURCES, receivers=RECEIVERS)


def test_modal_identity_is_measured_for_real_in_this_module() -> None:
    # 其他引擎考卷的身分有記憶（tests/engine/_identity_memo.py）；考模態身分的這支要看到真的函式。
    assert Path(api.modal_identity.__code__.co_filename).name == "modal_diagnosis.py"
