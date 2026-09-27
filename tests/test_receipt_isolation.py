"""正式 junit 收據只有全套那一跑能丟、能寫：清點與局部跑不碰它，局部跑指名它就拒跑。"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from governance import repo_residue
from governance.status import mirror_receipts
from tests import conftest as suite_conftest

FORMAL = "governance/receipts/pytest.junit.xml"

# 開錄音目錄與鏡收據被呼叫的紀錄：拒跑必須發生在這兩件事之前。
CALLS: list[str] = []

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

PARTIAL = [
    pytest.param({"file_or_dir": ["tests/engine"]}, id="positional"),
    pytest.param({"keyword": "x"}, id="k"),
    pytest.param({"markexpr": "slow"}, id="m"),
    pytest.param({"deselect": ["tests/engine::t"]}, id="deselect"),
    pytest.param({"lf": True}, id="lf"),
    pytest.param({"collectonly": True}, id="collect-only"),
    pytest.param({"override_ini": ["testpaths=tests/engine"]}, id="o-testpaths"),
]


def _session(
    root: Path, xmlpath: str | None, overrides: dict[str, object] | None = None, *, worker: bool = False
) -> pytest.Session:
    option = SimpleNamespace(**{**CLEAN_OPTION, "xmlpath": xmlpath, **(overrides or {})})
    config = SimpleNamespace(option=option, invocation_params=SimpleNamespace(dir=root))
    if worker:
        config.workerinput = {"workerid": "gw0"}
    return cast(pytest.Session, SimpleNamespace(config=config))


@pytest.fixture
def formal_receipt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    receipt = tmp_path / FORMAL
    receipt.parent.mkdir(parents=True)
    receipt.write_text("old", encoding="utf-8")
    calls = CALLS
    monkeypatch.setattr(suite_conftest, "REPO", tmp_path)
    monkeypatch.setattr(suite_conftest, "_BASELINE", suite_conftest._BASELINE)
    monkeypatch.setattr(suite_conftest, "junit_paths", lambda: [FORMAL])
    monkeypatch.setattr(repo_residue, "injected_env", lambda _env: [])
    monkeypatch.setattr(repo_residue, "porcelain", lambda _root: "baseline")
    monkeypatch.setattr(suite_conftest, "_open_replay_dir", lambda: calls.append("replay"))
    monkeypatch.setattr(mirror_receipts, "mirror", lambda *_a: calls.append("mirror"))
    CALLS[:] = []
    monkeypatch.delenv(suite_conftest.XDIST_WORKER_ENV, raising=False)  # 考卷自己跑在工人裡，不准讓「我是工人」蓋掉判準
    return receipt


@pytest.mark.parametrize(
    "xmlpath", [FORMAL, f"./{FORMAL}", f"governance/../{FORMAL}", "ABS"], ids=["rel", "dot", "dotdot", "abs"]
)
def test_full_run_discards_the_old_formal_receipt(formal_receipt: Path, tmp_path: Path, xmlpath: str) -> None:
    written = str(formal_receipt) if xmlpath == "ABS" else xmlpath
    suite_conftest.pytest_sessionstart(_session(tmp_path, written))
    assert not formal_receipt.exists()


@pytest.mark.parametrize("overrides", PARTIAL)
def test_partial_run_writing_elsewhere_leaves_the_formal_receipt(
    formal_receipt: Path, tmp_path: Path, overrides: dict[str, object]
) -> None:
    for xmlpath in (None, str(tmp_path / "dev.xml")):
        suite_conftest.pytest_sessionstart(_session(tmp_path, xmlpath, overrides))
        assert formal_receipt.read_text(encoding="utf-8") == "old"


@pytest.mark.parametrize("overrides", PARTIAL)
def test_partial_run_naming_the_formal_receipt_is_refused_before_side_effects(
    formal_receipt: Path, tmp_path: Path, overrides: dict[str, object]
) -> None:
    with pytest.raises(pytest.UsageError, match="冒充"):
        suite_conftest.pytest_sessionstart(_session(tmp_path, FORMAL, overrides))
    assert formal_receipt.read_text(encoding="utf-8") == "old"
    assert "replay" not in CALLS
    assert "mirror" not in CALLS


def test_unknown_option_shape_is_refused(formal_receipt: Path, tmp_path: Path) -> None:
    session = _session(tmp_path, FORMAL)
    del session.config.option.lf
    with pytest.raises(pytest.UsageError, match="判不出"):
        suite_conftest.pytest_sessionstart(session)
    assert formal_receipt.read_text(encoding="utf-8") == "old"


def test_worker_never_discards(formal_receipt: Path, tmp_path: Path) -> None:
    suite_conftest.pytest_sessionstart(_session(tmp_path, FORMAL, worker=True))
    assert formal_receipt.read_text(encoding="utf-8") == "old"
    assert "mirror" not in CALLS


def test_partial_option_names_exist_on_the_real_config(request: pytest.FixtureRequest) -> None:
    known = vars(request.config.option)
    assert [dest for dest, _flag in suite_conftest.PARTIAL_RUN_OPTIONS if dest not in known] == []
    assert set(CLEAN_OPTION) == {dest for dest, _flag in suite_conftest.PARTIAL_RUN_OPTIONS}


def test_collect_only_does_not_mirror_cloud_receipts(formal_receipt: Path, tmp_path: Path) -> None:
    """只清點不跑考卷：不鏡雲端收據（鏡一次十幾秒），正式收據也不碰。"""
    suite_conftest.pytest_sessionstart(_session(tmp_path, None, {"collectonly": True}))
    assert "mirror" not in CALLS
    assert formal_receipt.read_text(encoding="utf-8") == "old"


def test_full_run_still_mirrors_before_discarding(formal_receipt: Path, tmp_path: Path) -> None:
    suite_conftest.pytest_sessionstart(_session(tmp_path, FORMAL))
    assert CALLS == ["replay", "mirror"]
    assert not formal_receipt.exists()
