"""正式 junit 收據只有全套那一跑能丟、能寫：清點與局部跑不碰它，局部跑指名它就拒跑。"""
from __future__ import annotations

from pathlib import Path

import pytest

from governance import repo_residue
from governance.status import mirror_receipts
from tests import conftest as suite_conftest
from tests._receipt_session import CLEAN_OPTION, fake_session as _session

FORMAL = "governance/receipts/pytest.junit.xml"

# 開錄音目錄與鏡收據被呼叫的紀錄：拒跑必須發生在這兩件事之前。
CALLS: list[str] = []

PARTIAL = [
    pytest.param({"file_or_dir": ["tests/engine"]}, id="positional"),
    pytest.param({"keyword": "x"}, id="k"),
    pytest.param({"markexpr": "slow"}, id="m"),
    pytest.param({"deselect": ["tests/engine::t"]}, id="deselect"),
    pytest.param({"lf": True}, id="lf"),
    pytest.param({"collectonly": True}, id="collect-only"),
    pytest.param({"override_ini": ["testpaths=tests/engine"]}, id="o-testpaths"),
]


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
    # 真的設定檔就是 repo 根的 pyproject.toml：全套正式跑的判準靠這一格（另放設定檔的跑法會被拒）。
    assert request.config.inipath is not None
    assert Path(request.config.inipath).resolve() == (suite_conftest.REPO / "pyproject.toml").resolve()
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


def test_formal_path_through_symlink_and_dotdot_is_judged_like_pytest(formal_receipt: Path, tmp_path: Path) -> None:
    """pytest 寫檔前字面消掉 ..、不先跟著符號連結走；判準要跟它一樣，繞一個連結再 .. 回來也算正式路徑。"""
    elsewhere = tmp_path / "elsewhere" / "deep"
    elsewhere.mkdir(parents=True)
    (tmp_path / "link").symlink_to(elsewhere)
    sneaky = f"link/../{FORMAL}"
    with pytest.raises(pytest.UsageError, match="冒充"):
        suite_conftest.pytest_sessionstart(_session(tmp_path, sneaky, {"keyword": "x"}))
    assert formal_receipt.read_text(encoding="utf-8") == "old"
    suite_conftest.pytest_sessionstart(_session(tmp_path, sneaky))
    assert not formal_receipt.exists()


def test_run_not_collected_from_repo_root_testpaths_cannot_write_formal(formal_receipt: Path, tmp_path: Path) -> None:
    """從子目錄開跑（收集清單不是照 testpaths 來的）卻指名正式收據：拒跑。"""
    session = _session(tmp_path, FORMAL, args_source=pytest.Config.ArgsSource.INVOCATION_DIR)
    with pytest.raises(pytest.UsageError, match="testpaths"):
        suite_conftest.pytest_sessionstart(session)
    assert formal_receipt.read_text(encoding="utf-8") == "old"


def test_run_with_another_config_file_cannot_write_formal(formal_receipt: Path, tmp_path: Path) -> None:
    """另放一份設定檔（例如 pytest.toml 改收集範圍）卻指名正式收據：拒跑。"""
    session = _session(tmp_path, FORMAL, inipath=tmp_path / "pytest.toml")
    with pytest.raises(pytest.UsageError, match="pyproject.toml"):
        suite_conftest.pytest_sessionstart(session)
    assert formal_receipt.read_text(encoding="utf-8") == "old"


def test_home_or_variable_in_junit_path_is_refused(formal_receipt: Path, tmp_path: Path) -> None:
    for raw in ("~" + "receipt", "$" + "HOME"):  # 拆開寫：整串像路徑會被引用檢查當成指到家目錄
        with pytest.raises(pytest.UsageError, match="判不出"):
            suite_conftest.pytest_sessionstart(_session(tmp_path, raw))
    assert formal_receipt.read_text(encoding="utf-8") == "old"


def test_formal_path_through_an_alias_directory_is_the_same_file(formal_receipt: Path, tmp_path: Path) -> None:
    """字串不同、檔案相同（經過指到 repo 的捷徑寫絕對路徑）：一樣是正式收據——局部跑要擋、全套要丟。"""
    (tmp_path / "alias").symlink_to(tmp_path)
    aliased = str(tmp_path / "alias" / FORMAL)
    with pytest.raises(pytest.UsageError, match="冒充"):
        suite_conftest.pytest_sessionstart(_session(tmp_path, aliased, {"keyword": "x"}))
    assert formal_receipt.read_text(encoding="utf-8") == "old"
    suite_conftest.pytest_sessionstart(_session(tmp_path, aliased))
    assert not formal_receipt.exists()
