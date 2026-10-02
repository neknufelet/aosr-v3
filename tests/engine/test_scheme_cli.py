"""命令列必填參數考卷；讀寫、日期與印字由 test_scheme_repair 守。"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from aosr.reporting import scheme_cli


def test_cli_parser_requires_capabilities(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        scheme_cli.main(["run", str(tmp_path / "scheme.json"), "--out",
                         str(tmp_path / "result.json"), "--engine-commit", "control",
                         "--run-date", date(2026, 9, 27).isoformat()])
    exit_code = exc.value.code
    assert exit_code == 2


@pytest.mark.parametrize(("extra", "kind", "number"), [
    ([], "run", None), (["--search-id", "search"], "search_baseline", None),
    (["--search-id", "search", "--trial-number", "5"], "search_candidate", 5),
])
def test_cli_writes_origin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                           extra: list[str], kind: str, number: int | None) -> None:
    from aosr.config.paths import config_path
    from aosr.physics import three_lane_report
    from aosr.reporting import physics_stage
    from aosr.reporting.result import SchemeResult
    from tests.engine import _scoring_source_model_control as control
    from tests.engine.test_scheme_pipeline import _many_fem, _scheme

    for module, name, fake in control.STAND_INS:
        monkeypatch.setattr(module, name, fake)
    monkeypatch.setattr(three_lane_report, "_solve_fem_energies", _many_fem)
    monkeypatch.setattr(physics_stage, "report_capability", lambda table: three_lane_report._unchecked_capability())
    scheme = tmp_path / "scheme"
    scheme.write_text(_scheme("wall-1").model_dump_json())
    out = tmp_path / "result"
    code = scheme_cli.main(["run", str(scheme), "--out", str(out), "--engine-commit", "test",
                            "--capabilities", str(config_path("capabilities.toml")), *extra])
    saved = SchemeResult.model_validate_json(out.read_bytes())
    assert code == 0 and saved.origin.kind == kind and saved.origin.trial_number == number
    assert saved.origin.search_id == ("search" if extra else None)


def test_cli_trial_number_requires_search_id(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        scheme_cli.main(["run", str(tmp_path / "scheme"), "--out", str(tmp_path / "result"),
                         "--engine-commit", "test", "--capabilities", str(tmp_path / "capabilities"),
                         "--trial-number", "0"])
    assert exc.value.code == 2


@pytest.mark.parametrize("extra", (["--search-id", "search", "--trial-number", "-1"],
                                  ["--search-id", "   "]))
def test_cli_rejects_invalid_search_origin(tmp_path: Path, extra: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        scheme_cli.main(["run", str(tmp_path / "scheme"), "--out", str(tmp_path / "result"),
                         "--engine-commit", "test", "--capabilities", str(tmp_path / "capabilities"), *extra])
    assert exc.value.code == 2


@pytest.mark.parametrize("identity", ["program", "physics"])
def test_cli_identity_change_has_machine_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                capsys: pytest.CaptureFixture[str], identity: str) -> None:
    from aosr.config.paths import config_path
    from tests.engine._search_worker_cases import prepared_result

    result = prepared_result(tmp_path)
    scheme = tmp_path / "scheme"
    scheme.write_text(result.scheme.model_dump_json())
    out = tmp_path / "result"
    monkeypatch.setattr("aosr.reporting.pipeline.run_scheme", lambda *args, **kwargs: result)
    if identity == "program":
        values = iter((result.program_fingerprint, "calc-v1:" + "f" * 64))
        monkeypatch.setattr(scheme_cli, "calculation_fingerprint", lambda **kwargs: next(values))
    else:
        values = iter((result.physics_identity, "phys-v1:" + "f" * 64))
        monkeypatch.setattr("aosr.reporting.physics_identity.physics_identity", lambda **kwargs: next(values))
    code = scheme_cli.main(["run", str(scheme), "--out", str(out), "--engine-commit", "test",
                            "--capabilities", str(config_path("capabilities.toml"))])
    assert code == scheme_cli.PROGRAM_CHANGED_EXIT
    assert scheme_cli.PROGRAM_CHANGED_MARKER in capsys.readouterr().err.splitlines()
    assert not out.exists()
