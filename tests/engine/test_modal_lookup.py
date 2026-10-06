"""共用快取入口只查不解、擺位產物沿用與完整閉包守門。"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from aosr.geometry.shoebox import Point, Wall
from aosr.reporting import modal_diagnosis as core
from aosr.reporting import modal_lookup as api
from aosr.reporting.modal_diagnosis_cache import write_modal_cache
from aosr.reporting.modal_diagnosis_model import ModalDiagnosisState
from tests.engine._modal_cases import sample, scheme
from tests.engine.test_modal_diagnosis import Sample, small


@pytest.fixture(autouse=True)
def no_solve(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("只查快取不可求解")
    monkeypatch.setattr(core, "solve_fem_modes", forbidden)


def cached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    diagnosis, room = sample()
    write_modal_cache(cache_dir=tmp_path, cached=room)
    monkeypatch.setattr(api, "mesh_from_key", lambda key: object())
    monkeypatch.setattr(api, "mesh_fingerprint", lambda mesh: room.room_layer.mesh_sha256)
    monkeypatch.setattr(api, "assemble_p2_operators", lambda mesh: object())
    monkeypatch.setattr(api, "prepare_modal_expansion", lambda *args, **kwargs: object())
    monkeypatch.setattr(api, "placement_layer_from_expansion", lambda *args, **kwargs: diagnosis.placement_layer)


def test_hit_reuses_room_and_placement_across_entrypoints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cached(tmp_path, monkeypatch)
    first = api.diagnose_scheme(scheme(), cache_dir=tmp_path, cache_only=True)
    assert first.state is ModalDiagnosisState.DIAGNOSED_NOT_SCORED
    path = api.placement_path(scheme(), cache_dir=tmp_path)
    assert path is not None and path.is_file()
    before = path.stat().st_mtime_ns
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("擺位產物命中不可重新展開或解")
    monkeypatch.setattr(api, "prepare_modal_expansion", forbidden)
    monkeypatch.setattr(api, "diagnose_modes", forbidden)
    assert api.diagnose_scheme(scheme(), cache_dir=tmp_path, cache_only=False) == first
    assert path.stat().st_mtime_ns == before
    _, room = sample()
    write_modal_cache(cache_dir=tmp_path, cached=room)
    write_modal_cache(cache_dir=tmp_path, cached=room)
    assert path.is_file() and not path.name.startswith("modal-")


@pytest.mark.parametrize("mismatch", ["absent", "identity", "mesh"])
def test_cache_misses_never_solve(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mismatch: str) -> None:
    if mismatch != "absent":
        cached(tmp_path, monkeypatch)
        if mismatch == "mesh":
            monkeypatch.setattr(api, "mesh_fingerprint", lambda mesh: "b" * 64)
        else:
            header = next(tmp_path.glob("modal-*.json"))
            document = json.loads(header.read_text())
            document["modal_identity"] = "modal-v1:" + "0" * 64
            header.write_text(json.dumps(document))
    assert api.diagnose_scheme(scheme(), cache_dir=tmp_path, cache_only=True).state is ModalDiagnosisState.NOT_COMPUTED


@pytest.mark.parametrize("value", [0, -1, math.inf, math.nan, True, 1j, None, 1000])
@pytest.mark.parametrize("missing", [False, True])
def test_scope_agrees_with_core(value: object, missing: bool) -> None:
    walls = dict.fromkeys(Wall.all(), value)
    if missing:
        walls.pop(Wall.FLOOR)
    assert (api.scope_reason(scheme().scene.room_m, walls) is None) == core._supported(walls)
    assert api.scope_reason(object(), walls) == "unsupported_room"


def test_out_of_scope_and_position_error_preserve_reasons(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    value = scheme()
    scene = value.scene.model_copy(update={"impedance_pa_s_per_m_by_wall": {"floor": 0.0}})
    got = api.diagnose_scheme(value.model_copy(update={"scene": scene}), cache_dir=tmp_path, cache_only=True)
    assert got.state is ModalDiagnosisState.OUT_OF_SCOPE and got.reason_code == "unsupported_impedance"
    cached(tmp_path, monkeypatch)
    def broken(*args: object, **kwargs: object) -> None:
        raise ValueError("查位置原文：在邊界外")
    monkeypatch.setattr(api, "placement_layer_from_expansion", broken)
    got = api.diagnose_scheme(value, cache_dir=tmp_path, cache_only=True)
    assert got.state is ModalDiagnosisState.FAILED and got.reason_text == repr(ValueError("查位置原文：在邊界外"))


def test_key_uses_scene_medium_and_default_solver() -> None:
    from aosr.physics.fem_modal import ModalSolverOptions
    original = scheme()
    value = original.model_copy(update={"scene": original.scene.model_copy(update={
        "density_kg_m3": original.scene.density_kg_m3 * 1.1, "sound_speed_m_s": original.scene.sound_speed_m_s * 1.2})})
    key = api.key_from_scheme(value)
    assert key is not None
    assert key.density_hex == value.scene.density_kg_m3.hex()
    assert key.speed_hex == value.scene.sound_speed_m_s.hex()
    assert key.solver_options == ModalSolverOptions()
    changed = value.model_copy(update={"scheme_id": "another-entry"})
    assert api.placement_path(changed, cache_dir=Path("cache")) == api.placement_path(value, cache_dir=Path("cache"))


def test_concurrent_entrypoints_expand_one_placement_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from concurrent.futures import ThreadPoolExecutor
    import time
    cached(tmp_path, monkeypatch)
    calls: list[str] = []
    def expansion(*args: object, **kwargs: object) -> object:
        calls.append("expand")
        time.sleep(0.02)
        return object()
    monkeypatch.setattr(api, "prepare_modal_expansion", expansion)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(api.diagnose_scheme, scheme(), cache_dir=tmp_path, cache_only=True) for _ in range(2)]
        diagnoses = [future.result() for future in futures]
    assert all(d.state is ModalDiagnosisState.DIAGNOSED_NOT_SCORED for d in diagnoses)
    assert calls == ["expand"] and all(d == diagnoses[0] for d in diagnoses)


def test_atomic_placement_write_preserves_previous_document_on_failure(tmp_path: Path,
                                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    diagnosis, _ = sample()
    path = tmp_path / "placement.json"
    api.save_diagnosis(diagnosis, path)
    before = path.read_bytes()
    def broken(*args: object, **kwargs: object) -> None:
        raise OSError("發布中斷原文")
    monkeypatch.setattr(Path, "replace", broken)
    with pytest.raises(OSError, match="發布中斷原文"):
        api.save_diagnosis(diagnosis, path)
    assert path.read_bytes() == before
    assert not tuple(tmp_path.glob("placement-write-*"))


def test_compute_delegates_to_core_after_scope_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    value = scheme()
    diagnosis, _ = sample(value)
    calls: list[dict[str, object]] = []
    def calculate(**kwargs: object) -> object:
        calls.append(kwargs)
        return diagnosis
    monkeypatch.setattr(api, "diagnose_modes", calculate)
    assert api.diagnose_scheme(value, cache_dir=tmp_path, cache_only=False) == diagnosis
    assert calls and calls[0]["density_kg_m3"] == value.scene.density_kg_m3
    assert calls[0]["sound_speed_m_s"] == value.scene.sound_speed_m_s
    invalid = value.model_copy(update={"scene": value.scene.model_copy(update={"room_m": object()})})
    outside = api.diagnose_scheme(invalid, cache_dir=tmp_path, cache_only=False)
    assert outside.reason_code == "unsupported_room"


def test_native_small_cache_is_shared_with_cli_without_another_solve(small: Sample, tmp_path: Path,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.reporting.modal_diagnosis_model import ModalDiagnosis
    from aosr.reporting.scheme import Scheme
    from aosr.reporting.scheme_cli import main
    from tests.engine.test_modal_diagnosis import C, RECEIVERS, RHO, ROOM, SOURCES, WALLS
    document = scheme().model_dump(mode="json")
    document["scene"].update(room_m={"Lx": ROOM.Lx, "Ly": ROOM.Ly, "Lz": ROOM.Lz},
        density_kg_m3=RHO, sound_speed_m_s=C,
        impedance_pa_s_per_m_by_wall={w.wall_name(): z for w, z in WALLS.items()})
    document["speakers"] = {name: {"x": p.x, "y": p.y, "z": p.z} for name, p in SOURCES.items()}
    primary, surrounding = document["receiver_set"]["points"][:2]
    document["receiver_set"]["points"] = [
        {**primary, "receiver_id": "main", "position_m": RECEIVERS["main"].as_tuple()},
        {**surrounding, "receiver_id": "side", "position_m": RECEIVERS["side"].as_tuple()}]
    value = Scheme.model_validate(document)
    diagnosis = api.diagnose_scheme(value, cache_dir=small.cache_dir, cache_only=True)
    assert diagnosis.state is ModalDiagnosisState.DIAGNOSED_NOT_SCORED, diagnosis.reason_text
    assert diagnosis.room_layer == small.diagnosis.room_layer
    assert diagnosis.placement_layer == small.diagnosis.placement_layer
    source, out = tmp_path / "native-scheme.json", tmp_path / "native-diagnosis.json"
    source.write_text(value.model_dump_json())
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("跨入口已存擺位不可重新展開")
    monkeypatch.setattr(api, "prepare_modal_expansion", forbidden)
    exit_code = main(["modal", str(source), "--out", str(out), "--cache-dir", str(small.cache_dir), "--cache-only"])
    assert exit_code == 0
    assert ModalDiagnosis.model_validate_json(out.read_text()) == diagnosis


def test_entire_modal_closure_is_frozen_and_lookup_stays_out() -> None:
    expected = (
        "aosr", "aosr.config", "aosr.config.fem_lane", "aosr.config.frequency_axis", "aosr.config.paths",
        "aosr.geometry", "aosr.geometry.shoebox", "aosr.geometry.shoebox_mesh", "aosr.physics",
        "aosr.physics.fem_helmholtz", "aosr.physics.fem_modal", "aosr.physics.fem_modal_check",
        "aosr.physics.fem_modal_participation", "aosr.physics.modal_convention", "aosr.reporting",
        "aosr.reporting.import_closure", "aosr.reporting.modal_diagnosis", "aosr.reporting.modal_diagnosis_cache",
        "aosr.reporting.modal_diagnosis_model", "aosr.runtime",
    )
    assert core.modal_import_closure().modules == expected
    assert "aosr.reporting.modal_lookup" not in core.modal_import_closure().modules


def test_placement_file_holding_other_positions_is_not_used(tmp_path: Path) -> None:
    """擺位檔放錯了（內容是別組喇叭位置算的）：讀回要核對完整擺位，對不上就當沒有，不拿別人的報告冒充。"""
    original = scheme()
    name, point = next(iter(original.speakers.items()))
    moved_point = Point(*(c + d for c, d in zip(point.as_tuple(), (0.1, 0.0, 0.0), strict=True)))
    moved = original.model_copy(update={"speakers": {**original.speakers, name: moved_point}})
    planted, _room = sample(moved)
    path = api.placement_path(original, cache_dir=tmp_path)
    assert path is not None
    api.save_diagnosis(planted, path)
    assert api.read_placement(original, cache_dir=tmp_path) is None
    assert api.diagnose_scheme(original, cache_dir=tmp_path, cache_only=True).state is ModalDiagnosisState.NOT_COMPUTED
