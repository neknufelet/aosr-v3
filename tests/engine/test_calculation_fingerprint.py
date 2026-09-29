"""計算指紋的範圍與結果身分契約。"""
from __future__ import annotations

import json
import importlib.metadata
import re
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from aosr.reporting.calculation_fingerprint import short_fingerprint
from aosr.config.capabilities import load_capabilities
from aosr.config.directivity_defaults import load_directivity_defaults
from aosr.config.paths import config_path
from aosr.config.quality_targets import load_quality_targets
from aosr.gui.app import GuiSettings, create_app
from aosr.reporting import scheme_cli
from aosr.reporting.compare import compare_results, comparison_problems
from aosr.reporting.result import SchemeResult, load_result, save_result
from tests.engine.test_scheme_pipeline import _scheme, shared_control_result


FAKE = "calc-v1:" + "0" * 64
OTHER = "calc-v1:" + "1" * 64


def _copy_package(tmp_path: Path) -> tuple[Path, Path]:
    package = tmp_path / "aosr"
    shutil.copytree(Path(__file__).parents[2] / "src" / "aosr", package,
                    ignore=shutil.ignore_patterns("__pycache__"))
    capabilities = tmp_path / "capabilities.toml"
    capabilities.write_bytes(config_path("capabilities.toml").read_bytes())
    return package, capabilities


def test_fingerprint_ignores_gui_and_pycache_but_not_calculation_files(tmp_path: Path) -> None:
    from aosr.reporting.calculation_fingerprint import calculation_fingerprint

    package, capabilities = _copy_package(tmp_path)
    def measure() -> str:
        return calculation_fingerprint(capabilities_path=capabilities, package_root=package)

    baseline = measure()
    gui = package / "gui" / "result_list.py"
    gui.write_bytes(gui.read_bytes() + b"\n# visual change\n")
    cache = package / "reporting" / "__pycache__" / "x.pyc"
    cache.parent.mkdir()
    cache.write_bytes(b"cache")
    assert measure() == baseline
    for parts in (("physics", "report_io.py"), ("config", "data", "capabilities.toml"),
                  ("reporting", "pipeline.py"), ("scoring", "ranking.py")):
        path = package.joinpath(*parts)
        original = path.read_bytes()
        path.write_bytes(original + b"\n# changed\n")
        assert measure() != baseline, parts
        path.write_bytes(original)
        assert measure() == baseline


def test_fingerprint_follows_capabilities_content_not_path(tmp_path: Path) -> None:
    from aosr.reporting.calculation_fingerprint import calculation_fingerprint

    package = tmp_path / "aosr"
    package.mkdir()
    (package / "one.py").write_bytes(b"one")
    first, second = tmp_path / "first.toml", tmp_path / "second.toml"
    first.write_bytes(b"abc")
    second.write_bytes(b"abc")
    def measure(path: Path) -> str:
        return calculation_fingerprint(capabilities_path=path, package_root=package)

    baseline = measure(first)
    assert measure(second) == baseline
    second.write_bytes(b"abd")
    assert measure(second) != baseline


def test_fingerprint_format_and_versions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.reporting.calculation_fingerprint import calculation_fingerprint

    package = tmp_path / "aosr"
    package.mkdir()
    capabilities = tmp_path / "capabilities.toml"
    capabilities.write_bytes(b"value")
    def measure() -> str:
        return calculation_fingerprint(capabilities_path=capabilities, package_root=package)

    baseline = measure()
    assert re.fullmatch(r"calc-v1:[0-9a-f]{64}", baseline)
    real_version = importlib.metadata.version
    monkeypatch.setattr(importlib.metadata, "version",
                        lambda name: "changed" if name == "numpy" else real_version(name))
    assert measure() != baseline
    monkeypatch.setattr(importlib.metadata, "version", real_version)
    monkeypatch.setattr(sys, "version_info", (3, 99, 0))
    assert measure() != baseline


def test_short_fingerprint_skips_version_prefix() -> None:
    # 給人看的前 12 碼是指紋本身，不是「calc-v1:」加 4 碼。
    assert short_fingerprint("calc-v1:" + "0123456789ab" + "f" * 52) == "0123456789ab"


def test_calculation_entry_does_not_import_gui() -> None:
    command = ("import sys; import aosr.reporting.scheme_cli; "
               "import aosr.reporting.pipeline; "
               "print([name for name in sys.modules if name.startswith('aosr.gui')])")
    completed = subprocess.run([sys.executable, "-c", command], capture_output=True,
                               text=True, check=True)
    assert completed.stdout.strip() == "[]"


def test_compare_uses_fingerprint_not_engine_commit(
        tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> None:
    first = shared_control_result(tmp_path_factory, worker_id, "wall-1")
    second = shared_control_result(tmp_path_factory, worker_id, "wall-2")
    changed_commit = second.model_copy(update={"engine_commit": "other"})
    assert not any("計算指紋" in row for row in comparison_problems((first, changed_commit)))
    compare_results((first, changed_commit),
                    quality_targets=load_quality_targets(config_path("quality_targets.toml")),
                    run_date=date(2026, 9, 27))
    changed_fingerprint = changed_commit.model_copy(update={"calculation_fingerprint": OTHER})
    problems = comparison_problems((first, changed_fingerprint))
    assert any("計算指紋" in row and short_fingerprint(first.calculation_fingerprint) in row
               and short_fingerprint(OTHER) in row for row in problems)


def test_comparison_guard_uses_fingerprint_without_solver() -> None:
    first = SchemeResult.model_construct(
        scheme=_scheme("wall-1"), engine_commit="control", calculation_fingerprint=FAKE)
    second = SchemeResult.model_construct(
        scheme=_scheme("wall-2"), engine_commit="other", calculation_fingerprint=FAKE)
    assert comparison_problems((first, second)) == ()
    changed = second.model_copy(update={"calculation_fingerprint": OTHER})
    assert any("計算指紋" in row and short_fingerprint(FAKE) in row and short_fingerprint(OTHER) in row
               for row in comparison_problems((first, changed)))


def test_run_refuses_when_fingerprint_changes_mid_run(
        tmp_path: Path, tmp_path_factory: pytest.TempPathFactory, worker_id: str,
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    result = shared_control_result(tmp_path_factory, worker_id, "wall-1")
    scheme_path = tmp_path / "scheme.json"
    scheme_path.write_text(result.scheme.model_dump_json())
    out = tmp_path / "result.json"
    values = iter((FAKE, OTHER))
    monkeypatch.setattr(scheme_cli, "calculation_fingerprint", lambda **kwargs: next(values))
    monkeypatch.setattr(scheme_cli, "run_scheme", lambda *args, **kwargs: result)
    exit_code = scheme_cli.main(["run", str(scheme_path), "--out", str(out),
                                 "--capabilities", str(config_path("capabilities.toml")),
                                 "--engine-commit", "control"])
    assert exit_code != 0
    assert not out.exists()
    assert "這一跑不算" in capsys.readouterr().err


def test_midrun_guard_prevents_saving_before_result_is_used(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    scheme_path = tmp_path / "scheme.json"
    scheme_path.write_text(_scheme("wall-1").model_dump_json())
    out = tmp_path / "result.json"
    values = iter((FAKE, OTHER))
    monkeypatch.setattr(scheme_cli, "calculation_fingerprint", lambda **kwargs: next(values))
    monkeypatch.setattr(scheme_cli, "run_scheme", lambda *args, **kwargs: object())
    exit_code = scheme_cli.main(["run", str(scheme_path), "--out", str(out),
                                 "--capabilities", str(config_path("capabilities.toml")),
                                 "--engine-commit", "control"])
    assert exit_code
    assert not out.exists()
    assert "這一跑不算" in capsys.readouterr().err


def test_v2_result_is_rejected_as_old_format(
        tmp_path: Path, tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> None:
    result = shared_control_result(tmp_path_factory, worker_id, "wall-1")
    run_id = "a" * 32
    path = tmp_path / "results" / f"{run_id}.json"
    with TestClient(create_app(GuiSettings(engine_commit="a" * 40, data_dir=tmp_path)),
                    base_url="http://localhost") as client:
        save_result(result, path)
        document = json.loads(path.read_text())
        document["schema_version"] = "aosr.scheme_result.v2"
        document.pop("calculation_fingerprint")
        path.write_text(json.dumps(document))
        with pytest.raises(ValueError, match="舊版"):
            load_result(path, capabilities=load_capabilities(config_path("capabilities.toml")),
                        directivity=load_directivity_defaults(config_path("directivity_defaults.toml")),
                        quality_targets_path=config_path("quality_targets.toml"))
        response = client.get(f"/api/results/{run_id}")
        assert response.status_code == 409
        assert "舊版" in response.json()["reason"]
        assert response.json()["rerun_url"] == f"/api/results/{run_id}/rerun"


def test_v2_header_is_rejected_before_payload(tmp_path: Path) -> None:
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"schema_version": "aosr.scheme_result.v2"}))
    with pytest.raises(ValueError, match="舊版"):
        load_result(path, capabilities=load_capabilities(config_path("capabilities.toml")),
                    directivity=load_directivity_defaults(config_path("directivity_defaults.toml")),
                    quality_targets_path=config_path("quality_targets.toml"))
