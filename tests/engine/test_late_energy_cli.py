"""晚期混響能量命令列的人看輸出與離開碼考卷。"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from tests.engine._precision_contracts import REGISTRY_PATH, contract_value


_ROOT = Path(__file__).resolve().parents[2]
_FLAT_ANSWER = _ROOT / "blueprint" / "reference_art_flat.json"
_FREQUENCIES_HZ = (125, 250, 500, 1000, 2000, 4000)
_TOLERANCE_REL = contract_value("late_energy_vs_legacy")
_CONTRACT_ARGS = ("--contracts", str(REGISTRY_PATH))


def test_flat_answer_compare_prints_every_band_as_nonblocking_record(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """抓第二類比較漏頻帶、漏六面吸收率，或舊界外錯誤擋住命令列。"""
    from aosr.physics import late_energy_cli

    exit_code = late_energy_cli.main(
        [str(_FLAT_ANSWER), "--compare", *_CONTRACT_ARGS]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert all(f"\n{frequency} " in output for frequency in _FREQUENCIES_HZ)
    assert all(
        heading in output
        for heading in ("alpha_floor", "alpha_ceiling", "alpha_x0", "alpha_xL", "alpha_y0", "alpha_yL")
    )
    assert "第二類相容紀錄" in output
    assert "不作通過判決" in output


def test_compare_rows_print_same_unit_reference_limits_and_classification(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """抓絕對差與舊參考差不同單位，或界內外分類沒有跟數字一致。"""
    from aosr.physics import late_energy_cli

    exit_code = late_energy_cli.main(
        [str(_FLAT_ANSWER), "--compare", *_CONTRACT_ARGS]
    )
    lines = capsys.readouterr().out.splitlines()
    headings = lines[1].split()
    rows = [dict(zip(headings, line.split(), strict=True)) for line in lines[2:-1]]

    assert exit_code == 0
    assert lines[0] == "上一代答案是第二類相容紀錄；差距照量、照留，不作通過判決"
    assert headings.index("absolute_difference") + 1 == headings.index("allowed_difference")
    assert headings.index("relative_difference") + 1 == headings.index("relative_limit")
    assert rows
    assert all(
        float(row["relative_limit"]) == _TOLERANCE_REL
        for row in rows
    )
    assert all(
        (float(row["absolute_difference"]) <= float(row["allowed_difference"]))
        == (row["comparison"] == "舊界內")
        for row in rows
    )


def test_tampered_answer_band_is_named_but_does_not_block(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """抓 CLI 沒用答案檔的 late_rev_E，或第二類舊界外仍回非零。"""
    from aosr.physics import late_energy_cli

    tampered_path = tmp_path / "reference_art_flat.json"
    shutil.copyfile(_FLAT_ANSWER, tampered_path)
    with tampered_path.open(encoding="utf-8") as handle:
        document: dict[str, object] = json.load(handle)
    bands = document["bands"]
    assert isinstance(bands, list)
    first_band = bands[0]
    assert isinstance(first_band, dict)
    answer = first_band["late_rev_E"]
    assert isinstance(answer, dict)
    tampered = float.fromhex(str(answer["hex"])) * 1.01
    answer["hex"] = tampered.hex()
    answer["dec"] = repr(tampered)
    tampered_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")

    exit_code = late_energy_cli.main(
        [str(tampered_path), "--compare", *_CONTRACT_ARGS]
    )
    output = capsys.readouterr().out
    changed_row = next(line for line in output.splitlines() if line.startswith("125 "))

    assert exit_code == 0
    assert changed_row.endswith("舊界外")
    assert "相容紀錄：" in output
    assert "舊界外（不擋）" in output
    assert "最壞=125 Hz" in output


def test_missing_input_returns_two_and_prints_reason(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """抓讀檔失敗冒泡，或被錯報成契約超界。"""
    from aosr.physics import late_energy_cli

    missing = tmp_path / "missing.json"
    exit_code = late_energy_cli.main(
        [str(missing), "--compare", *_CONTRACT_ARGS]
    )
    output = capsys.readouterr().out

    assert exit_code == 2
    assert "晚期混響能量算不出來" in output
    assert "missing.json" in output


def test_compare_requires_contracts(capsys: pytest.CaptureFixture[str]) -> None:
    """比對模式沒有明給登記簿路徑時必須報錯，不准暗找預設。"""
    from aosr.physics import late_energy_cli

    exit_code = late_energy_cli.main([str(_FLAT_ANSWER), "--compare"])

    assert exit_code == 2
    assert "--compare 模式必須給 --contracts" in capsys.readouterr().out
