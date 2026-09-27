"""收據考卷共用：全套正式跑的 pytest option 長相，與只帶 option 的假 session。"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

# 全套那一跑的 option 長相（2026-09-28 在 pytest 9.1.1／xdist 3.8.0 實量的預設值）。
CLEAN_OPTION: dict[str, object] = {
    "file_or_dir": [],
    "keyword": "",
    "markexpr": "",
    "deselect": None,
    "ignore": None,
    "ignore_glob": None,
    "lf": False,
    "stepwise": False,
    "maxfail": None,
    "collectonly": False,
    "setuponly": False,
    "setupplan": False,
    "showfixtures": False,
    "show_fixtures_per_test": False,
    "cacheshow": None,
    "override_ini": None,
    "inifilename": None,
}


def fake_session(
    root: Path, xmlpath: str | None, overrides: dict[str, object] | None = None, *, worker: bool = False,
    args_source: pytest.Config.ArgsSource = pytest.Config.ArgsSource.TESTPATHS,
    inipath: Path | None = None,
) -> pytest.Session:
    """只帶 conftest 會讀的那幾格：option、開跑目錄、收集清單的來源、設定檔（預設 root 底下的 pyproject.toml）。"""
    option = SimpleNamespace(**{**CLEAN_OPTION, "xmlpath": xmlpath, **(overrides or {})})
    config = SimpleNamespace(option=option, invocation_params=SimpleNamespace(dir=root),
                             args_source=args_source,
                             inipath=root / "pyproject.toml" if inipath is None else inipath)
    if worker:
        config.workerinput = {"workerid": "gw0"}
    return cast(pytest.Session, SimpleNamespace(config=config))
