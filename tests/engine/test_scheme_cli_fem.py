"""有限元素分片命令列：拒收不得退回求解，身分檢查必須早於合併與落盤。"""
from __future__ import annotations

import builtins
import importlib.util
import json
import os
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

import pytest

from aosr.config.capabilities import CapabilityTable
from aosr.config.directivity_defaults import DirectivityDefaults
from aosr.physics import three_lane_report
from aosr.reporting import fem_slices, scheme_cli
from aosr.reporting.fem_slices import FemShard
from aosr.reporting.scheme import Scheme
from tests.engine import _fem_cli_cases as cli
from tests.engine import _fem_shared_cases as cases
from tests.engine.test_scheme_pipeline import _many_fem


def test_slice_envelope_and_axis_union(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                       capsys: pytest.CaptureFixture[str]) -> None:
    paths, common = cli.prepare(tmp_path, monkeypatch)
    outputs = cli.parts(tmp_path, paths, common)
    shards = []
    for path in outputs:
        envelope = json.loads(path.read_text())
        assert set(envelope) == {"shard", "wall_s", "max_rss_kib"}
        assert all(isinstance(envelope[name], (int, float)) and envelope[name] >= 0
                   for name in ("wall_s", "max_rss_kib"))
        shard = FemShard.model_validate(envelope["shard"])
        assert set(shard.candidates) == {path.name for path in paths}
        shards.append(shard)
    assert {index for shard in shards for index in shard.indices} == set(range(len(cases.FREQUENCIES)))
    assert sum(len(shard.indices) for shard in shards) == len(cases.FREQUENCIES)
    assert capsys.readouterr() == ("", "")


def test_run_shuffled_parts_equals_regular_run_without_fem(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths, common = cli.prepare(tmp_path, monkeypatch)
    outputs = cli.parts(tmp_path, paths, common)
    # 保留完整主管線、評估與結果存檔，只換掉慢速物理求解。
    monkeypatch.setattr(three_lane_report, "_solve_fem_energies", _many_fem)
    expected, actual = tmp_path / "regular", tmp_path / "injected"
    code = scheme_cli.main(cli.run_args(paths[0], common, expected))
    assert code == 0
    calls = cli.forbid_fem(monkeypatch)
    code = scheme_cli.main(cli.run_args(paths[0], common, actual, list(reversed(outputs))))
    assert code == 0
    left, right = json.loads(expected.read_text()), json.loads(actual.read_text())
    left.pop("timings")
    right.pop("timings")
    assert left == right
    assert calls == []


@pytest.mark.parametrize("damage", ("missing", "absent", "truncated", "coordinates", "candidate",
                                    "key", "overlap", "envelope", "model"))
def test_run_bad_parts_refuses_without_fallback(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str], damage: str) -> None:
    paths, common = cli.prepare(tmp_path, monkeypatch)
    outputs = cli.parts(tmp_path, paths, common)
    if damage == "missing":
        outputs = outputs[1:]
    elif damage == "absent":
        outputs[0].unlink()
    elif damage == "truncated":
        outputs[0].write_text('{"shard":')
    elif damage == "overlap":
        outputs.append(outputs[0])
    else:
        envelope = json.loads(outputs[0].read_text())
        shard = FemShard.model_validate(envelope["shard"])
        if damage == "coordinates":
            candidate = shard.candidates[paths[0].name]
            point = next(iter(candidate.speakers))
            candidate.speakers[point] = (float(0.7).hex(), *candidate.speakers[point][1:])
        elif damage == "candidate":
            del shard.candidates[paths[0].name]
        elif damage == "key":
            shard = shard.model_copy(update={"fem_key": shard.fem_key.model_copy(
                update={"density_hex": float(1.3).hex()})})
        envelope["shard"] = shard.model_dump(mode="json")
        if damage == "envelope":
            del envelope["shard"]
        elif damage == "model":
            envelope["shard"] = {"unexpected": "invalid"}
        outputs[0].write_text(json.dumps(envelope))
    calls = cli.forbid_fem(monkeypatch)
    out = tmp_path / "result"
    code = scheme_cli.main(cli.run_args(paths[0], common, out, outputs))
    assert code != 0
    assert not out.exists()
    assert calls == []
    assert capsys.readouterr().err.strip()


def test_run_shard_identity_mismatch_uses_changed_exit(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    paths, common = cli.prepare(tmp_path, monkeypatch)
    outputs = cli.parts(tmp_path, paths, common)
    envelope = json.loads(outputs[0].read_text())
    envelope["shard"]["physics_identity"] = "another-program"
    outputs[0].write_text(json.dumps(envelope))
    calls = cli.forbid_fem(monkeypatch)
    out = tmp_path / "result"
    assert scheme_cli.main(cli.run_args(paths[0], common, out, outputs)) == scheme_cli.PROGRAM_CHANGED_EXIT
    assert not out.exists()
    assert calls == []
    stderr = capsys.readouterr().err
    assert scheme_cli.PROGRAM_CHANGED_MARKER in stderr.splitlines()
    assert "身分" in stderr


@pytest.mark.parametrize("identity", ("program", "physics"))
def test_slice_midrun_change_leaves_no_output_or_temp(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str], identity: str) -> None:
    paths, common = cli.prepare(tmp_path, monkeypatch)
    values = iter(("before", "after"))
    if identity == "program":
        monkeypatch.setattr(scheme_cli, "calculation_fingerprint", lambda **kwargs: next(values))
    else:
        monkeypatch.setattr("aosr.reporting.physics_identity.physics_identity", lambda **kwargs: next(values))
    out = tmp_path / "part"
    previous = set(tmp_path.iterdir())
    assert scheme_cli.main(cli.slice_args(paths, common, out, 0, 2)) == scheme_cli.PROGRAM_CHANGED_EXIT
    assert not out.exists()
    assert set(tmp_path.iterdir()) == previous
    assert scheme_cli.PROGRAM_CHANGED_MARKER in capsys.readouterr().err.splitlines()


@pytest.mark.parametrize("fail_replace", (False, True))
def test_slice_atomic_replace_preserves_existing_output_on_failure(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_replace: bool) -> None:
    paths, common = cli.prepare(tmp_path, monkeypatch)
    out = tmp_path / "part"
    out.write_text("previous complete output")
    previous = set(tmp_path.iterdir())
    replace = os.replace
    replaced: list[Path] = []
    def inspect(source: str | Path, target: str | Path) -> None:
        source, target = Path(source), Path(target)
        assert source.parent == out.parent and source != out and target == out
        assert out.read_text() == "previous complete output"
        FemShard.model_validate(json.loads(source.read_text())["shard"])
        replaced.append(source)
        if fail_replace:
            raise OSError("interrupted replacement")
        replace(source, target)
    monkeypatch.setattr(os, "replace", inspect)
    code = scheme_cli.main(cli.slice_args(paths, common, out, 0, 2))
    assert replaced
    assert set(tmp_path.iterdir()) == previous
    if fail_replace:
        assert code != 0
        assert out.read_text() == "previous complete output"
    else:
        assert code == 0
        FemShard.model_validate(json.loads(out.read_text())["shard"])


def test_slice_measures_before_import_solve_and_replace(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths, common = cli.prepare(tmp_path, monkeypatch)
    events: list[str] = []
    real_import = builtins.__import__
    def record_import(name: str, globals: dict[str, object] | None = None,
                      locals: dict[str, object] | None = None,
                      fromlist: Sequence[str] = (), level: int = 0) -> ModuleType:
        if name.startswith("aosr") and name != "aosr.reporting.calculation_fingerprint":
            events.append("import")
        return real_import(name, globals, locals, fromlist, level)
    def fingerprint(**kwargs: object) -> str:
        events.append("fingerprint")
        return "calc-v1:" + "0" * 64
    def physics(**kwargs: object) -> str:
        events.append("physics")
        return "physical"
    solve = fem_slices.solve_slice
    def recorded_solve(schemes: Sequence[Scheme], *, capabilities: CapabilityTable,
                       directivity: DirectivityDefaults, slices: int, slice_index: int,
                       physics_identity: str) -> FemShard:
        events.append("solve")
        return solve(schemes, capabilities=capabilities, directivity=directivity,
                     slices=slices, slice_index=slice_index, physics_identity=physics_identity)
    replace = os.replace
    def recorded_replace(source: str | Path, target: str | Path) -> None:
        events.append("replace")
        replace(source, target)
    monkeypatch.setattr("aosr.reporting.physics_identity.physics_identity", physics)
    monkeypatch.setattr(fem_slices, "solve_slice", recorded_solve)
    monkeypatch.setattr(os, "replace", recorded_replace)
    monkeypatch.setattr(builtins, "__import__", record_import)
    # 行程內重新載入入口：連模組最外層偷載計算程式都要抓到，不借既有 import 快取。
    spec = importlib.util.spec_from_file_location("fem_cli_probe", scheme_cli.__file__)
    assert spec is not None and spec.loader is not None
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    assert events == []
    monkeypatch.setattr(probe, "calculation_fingerprint", fingerprint)
    code = probe.main(cli.slice_args(paths, common, tmp_path / "part", 0, 2))
    assert code == 0
    assert events[0] == "fingerprint"
    assert [event for event in events if event != "import"] == [
        "fingerprint", "physics", "solve", "fingerprint", "physics", "replace"]


def test_run_measures_identity_before_reading_and_merging_parts(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths, common = cli.prepare(tmp_path, monkeypatch)
    outputs = cli.parts(tmp_path, paths, common)
    identity = FemShard.model_validate(json.loads(outputs[0].read_text())["shard"]).physics_identity
    events: list[str] = []
    def fingerprint(**kwargs: object) -> str:
        events.append("fingerprint")
        return "calc-v1:" + "0" * 64
    def physics(**kwargs: object) -> str:
        events.append("physics")
        return identity
    read_text = Path.read_text
    def read(path: Path, encoding: str | None = None, errors: str | None = None) -> str:
        if path in outputs:
            assert events[:2] == ["fingerprint", "physics"]
            events.append("read")
        return read_text(path, encoding=encoding, errors=errors)
    merge = fem_slices.energies_from_shards
    def recorded_merge(shards: Sequence[FemShard], *, scheme: Scheme,
                       capabilities: CapabilityTable, directivity: DirectivityDefaults,
                       physics_identity: str) -> dict[tuple[str, str], tuple[float, ...]]:
        assert events == ["fingerprint", "physics", *["read" for _ in outputs]]
        assert physics_identity == identity
        events.append("merge")
        return merge(shards, scheme=scheme, capabilities=capabilities,
                     directivity=directivity, physics_identity=physics_identity)
    monkeypatch.setattr(scheme_cli, "calculation_fingerprint", fingerprint)
    monkeypatch.setattr("aosr.reporting.physics_identity.physics_identity", physics)
    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(fem_slices, "energies_from_shards", recorded_merge)
    code = scheme_cli.main(cli.run_args(paths[0], common, tmp_path / "result", outputs))
    assert code == 0
    assert events == ["fingerprint", "physics", *["read" for _ in outputs],
                      "merge", "fingerprint", "physics"]


@pytest.mark.parametrize("identity", ("program", "physics"))
def test_run_with_parts_rechecks_identity_before_saving(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str], identity: str) -> None:
    paths, common = cli.prepare(tmp_path, monkeypatch)
    outputs = cli.parts(tmp_path, paths, common)
    shard = FemShard.model_validate(json.loads(outputs[0].read_text())["shard"])
    if identity == "program":
        values = iter(("calc-v1:" + "0" * 64, "calc-v1:" + "1" * 64))
        monkeypatch.setattr(scheme_cli, "calculation_fingerprint", lambda **kwargs: next(values))
    else:
        values = iter((shard.physics_identity, "phys-v1:" + "1" * 64))
        monkeypatch.setattr("aosr.reporting.physics_identity.physics_identity", lambda **kwargs: next(values))
    out = tmp_path / "result"
    assert scheme_cli.main(cli.run_args(paths[0], common, out, outputs)) == scheme_cli.PROGRAM_CHANGED_EXIT
    assert not out.exists()
    assert scheme_cli.PROGRAM_CHANGED_MARKER in capsys.readouterr().err.splitlines()
