"""命令列讀寫與文字表頭的考卷；求解由固定結果替身隔開。"""
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
