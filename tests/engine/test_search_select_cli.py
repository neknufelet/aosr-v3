"""助理代按的選入入口：明確參數、白話輸出、狀態與快照不變。"""

from pathlib import Path

import pytest

from aosr.search.cli import main
from aosr.search.refine import RefineLedger
from tests.engine._search_select_cases import refined_store


@pytest.mark.parametrize("options", [[], ["--trial", "0"], ["--baseline"],
                                     ["--data-dir", "data"],
                                     ["--trial", "0", "--baseline", "--data-dir", "data"]])
def test_select_arguments_are_required_and_exclusive(tmp_path: Path, options: list[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(["select", str(tmp_path / "search"), *options])
    assert error.value.code == 2
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize("baseline", [False, True])
def test_select_cli_success_repeat_and_refusal_preserve_states(
    tmp_path: Path, baseline: bool, capsys: pytest.CaptureFixture[str],
) -> None:
    from aosr.search.select import SelectLedger, selection_ledger_path

    store = refined_store(tmp_path)
    before = {path: path.read_bytes() for path in store.path.iterdir() if path.is_file()}
    trial = next(row.trial_number for row in RefineLedger.read(store.refine_ledger_path)[1]
                 if row.trial_number is not None)
    choice = ["--baseline"] if baseline else ["--trial", str(trial)]
    argv = ["select", str(store.path), *choice, "--data-dir", str(tmp_path / "data")]
    code = main(argv)
    assert code == 0
    row, = SelectLedger.read(selection_ledger_path(store))
    captured = capsys.readouterr()
    assert captured.out == f"已放進結果清單：代號 {row.run_id}\n" and captured.err == ""
    code = main(argv)
    assert code == 0
    captured = capsys.readouterr()
    assert captured.out == f"已在結果清單：代號 {row.run_id}\n" and captured.err == ""
    code = main(["select", str(store.path), "--trial", "999999", "--data-dir", str(tmp_path / "data")])
    assert code == 1
    captured = capsys.readouterr()
    assert captured.out == "" and "沒細算過" in captured.err
    assert all(path.read_bytes() == content for path, content in before.items())


def test_select_cli_error_is_original_text_without_state_write(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = refined_store(tmp_path)
    before = store.status_path.read_bytes()

    def refuse(*args: object) -> None:
        raise ValueError("拒絕原文")

    monkeypatch.setattr("aosr.search.cli.select_refined", refuse)
    code = main(["select", str(store.path), "--baseline", "--data-dir", str(tmp_path / "data")])
    assert code == 1
    captured = capsys.readouterr()
    assert captured.err == "拒絕原文\n" and captured.out == ""
    assert store.status_path.read_bytes() == before
