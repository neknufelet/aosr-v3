"""三路接合物理量報表命令列的輸出與離開碼考卷。"""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from aosr.config.frequency_axis import GEOMETRIC_REPORT_OCTAVE_CENTERS_HZ
from aosr.config.paths import config_path
from aosr.geometry.shoebox import Point, Room, Wall
from aosr.materials.catalog_absorption import (
    normalized_impedance_from_random_incidence_absorption,
)


_TABLE_PATH = config_path("capabilities.toml")


def _input_document(*, impedance_multiple: float = 4.0) -> dict[str, object]:
    wall_names = Wall.wall_names()
    rho_c_pa_s_per_m = 1.2 * 343.0
    return {
        "room_m": {"Lx": 6.0, "Ly": 4.0, "Lz": 3.0},
        "source_model": {"kind": "omnidirectional"},
        "source_m": {"x": 1.2, "y": 1.3, "z": 1.1},
        "receiver_m": {"x": 4.7, "y": 2.8, "z": 1.4},
        "sound_speed_m_s": 343.0,
        "density_kg_m3": 1.2,
        "impedance_pa_s_per_m_by_wall": {
            wall: impedance_multiple * rho_c_pa_s_per_m for wall in wall_names
        },
        "scattering_by_wall": {wall: 0.2 for wall in wall_names},
    }


def _fake_fem_energy(
    *,
    room: Room,
    source: Point,
    receiver: Point,
    wall_impedances: Mapping[Wall, float],
    frequencies_hz: tuple[float, ...],
    density_kg_m3: float,
    sound_speed_m_s: float,
) -> tuple[float, ...]:
    del room, source, receiver, wall_impedances
    assert density_kg_m3 == 1.2
    assert sound_speed_m_s == 343.0
    return tuple(0.000012345 + frequency * 1e-10 for frequency in frequencies_hz)


def test_cli_prints_top_bands_and_points_without_real_fem(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """抓 CLI 漏取樣說明、頂層、報表帶、--points，或偷跑正式 FEM。"""
    from aosr.physics import three_lane_report, three_lane_report_cli

    input_path = tmp_path / "room.json"
    input_path.write_text(json.dumps(_input_document()), encoding="utf-8")
    monkeypatch.setattr(three_lane_report, "_solve_fem_energy", _fake_fem_energy)

    exit_code = three_lane_report_cli.main(
        [str(input_path), "--points", "--capabilities", str(_TABLE_PATH)]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    sampling_note = (
        "頻帶取樣：直達／反射／干涉／s 欄為 0.5 Hz 密頻率點平均；"
        "晚期／權重欄為當次報表軸每八度等權平均，T20／T30 為正式細軸點平均，權重只供閱讀；"
        "請用 fem_contribution 與 geometric_contribution 驗算 total_energy。"
    )
    assert sampling_note in output
    assert output.index(sampling_note) < output.index("center_frequency_hz")
    assert all(
        heading in output
        for heading in (
            "f_s_hz",
            "crossover_lower_hz",
            "crossover_upper_hz",
            "capped_by_upper_limit",
            "eyring_t60_500_hz_s",
            "eyring_t60_1000_hz_s",
            "fem_point_count",
            "interference_energy_all_points",
            "fem_contribution_all_points",
            "geometric_energy_all_points",
            "geometric_contribution_all_points",
            "t20_s",
            "t30_s",
            "frequency_hz",
            "interference_energy",
            "total_energy",
        )
    )
    assert all(
        f"\n{center:g} " in output for center in GEOMETRIC_REPORT_OCTAVE_CENTERS_HZ
    )
    assert "1.234" in output


def test_cli_names_unavailable_decay_and_hard_cut_in_chinese(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """抓 CLI 把 None 印成 none，或硬切只印 boolean 沒有人話。"""
    from aosr.physics import three_lane_report, three_lane_report_cli

    input_path = tmp_path / "hard-room.json"
    input_path.write_text(
        json.dumps(
            _input_document(
                impedance_multiple=(
                    normalized_impedance_from_random_incidence_absorption(0.026)
                )
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(three_lane_report, "_solve_fem_energy", _fake_fem_energy)

    exit_code = three_lane_report_cli.main(
        [str(input_path), "--capabilities", str(_TABLE_PATH)]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "被上限硬切（f_s ≥ 300 Hz，在 300 Hz 硬切換）" in output
    assert "算不出：T30 擬合無效" in output
    assert "第 256 階最低" in output


def test_cli_read_failure_returns_two_and_prints_reason(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """抓讀檔錯誤冒泡或錯報成成功。"""
    from aosr.physics import three_lane_report_cli

    missing = tmp_path / "missing.json"
    exit_code = three_lane_report_cli.main(
        [str(missing), "--capabilities", str(_TABLE_PATH)]
    )
    output = capsys.readouterr().out

    assert exit_code == 2
    assert "三路接合報表算不出來" in output
    assert "missing.json" in output
