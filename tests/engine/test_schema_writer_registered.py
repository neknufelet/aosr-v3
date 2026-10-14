"""把靜態規則的登記與產品寫入端綁定，防止改名後規則靜靜失效。"""
from __future__ import annotations

import tomllib
from pathlib import Path

from aosr.physics import report_io

REPO = Path(__file__).resolve().parents[2]


def test_card_registers_the_schema_writer_and_its_committed_dir() -> None:
    card: dict[str, object] = tomllib.loads(
        (REPO / "governance/rules/tests-isolated-from-real-env.toml").read_text(encoding="utf-8")
    )
    settings = card["settings"]
    assert isinstance(settings, dict)
    writers, constants, paths = (
        settings["product_writers"], settings["committed_constants"], settings["committed_paths"]
    )
    assert isinstance(writers, list) and writers
    assert isinstance(constants, list) and constants
    assert isinstance(paths, list) and paths
    assert report_io.regenerate_schema_files.__name__ in writers
    for name in constants:
        assert isinstance(name, str)
        assert hasattr(report_io, name)
        path: object = getattr(report_io, name)
        assert isinstance(path, Path)
        assert path.relative_to(REPO).as_posix() in paths
