"""能力與驗證範圍表的載入、收據對帳與三個入口的接線考卷（票 #315）。

這一支咬四件事：

1. **載入器的形狀**——壞 status、validated 沒 evidence、重複組合、頻率範圍顛倒
   各要報錯；`path` 必給、模型凍結。
2. **表上每一條 evidence 都是真的**——測試節點 `--collect-only` 收得到、答案檔
   在版控樹裡真的存在。表最貴的錯法是它自己說謊。
3. **三個入口的輸出帶著同一組標記**——capability 節的 status 要跟表上那一條一致。
4. **輸入驗證會擋**——複數阻抗與逐頻阻抗進三路報表入口要明確報錯，訊息要提票號
   與「能力表標 unsupported」。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from aosr.config.capabilities import (
    CapabilityTable,
    evidence_for,
    load_capabilities,
    status_for,
)
from aosr.config.paths import config_path
from aosr.geometry.shoebox import Wall
from tests.conftest import GitSandbox


_REPO = Path(__file__).resolve().parents[2]
_TABLE_PATH = config_path("capabilities.toml")


def _entry_block(
    *,
    name: str = "example_entry",
    module: str = "aosr.physics.example",
    capability: str,
) -> str:
    return f'[[entry]]\nname = "{name}"\nmodule = "{module}"\nnote = "測試樣本"\n{capability}'


def _capability_block(
    *,
    room: str = "shoebox",
    materials: str = "real_impedance",
    frequency_hz: str = "[20.0, 20000.0]",
    status: str = "validated",
    evidence: str = 'evidence = ["tests/engine/test_capabilities.py::test_every_evidence_node_collects"]',
) -> str:
    return (
        "[[entry.capability]]\n"
        f'room = "{room}"\n'
        f'materials = "{materials}"\n'
        f"frequency_hz = {frequency_hz}\n"
        'outputs = ["energy"]\n'
        f'status = "{status}"\n'
        f"{evidence}\n"
        'note = "樣本組合"\n'
    )


def _write_table(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


# --- 1. 載入器的形狀 ---------------------------------------------------------


def test_bad_status_is_rejected(tmp_path: Path) -> None:
    """status 只准三個值：拼錯、留白、或寫成 validated! 都要在載入當下紅。"""
    path = _write_table(
        tmp_path / "capabilities.toml",
        _entry_block(capability=_capability_block(status="validated!")),
    )

    with pytest.raises(ValidationError, match="status"):
        load_capabilities(path)


def test_validated_without_evidence_is_rejected(tmp_path: Path) -> None:
    """標成 validated 卻沒有指名任何收據，等於一個沒有證據的綠。"""
    path = _write_table(
        tmp_path / "capabilities.toml",
        _entry_block(capability=_capability_block(evidence="evidence = []")),
    )

    with pytest.raises(ValidationError, match="validated.*evidence"):
        load_capabilities(path)


def test_experimental_without_evidence_is_accepted(tmp_path: Path) -> None:
    """其他兩個狀態可以沒有收據——experimental 這一版就是還沒量。"""
    path = _write_table(
        tmp_path / "capabilities.toml",
        _entry_block(
            capability=_capability_block(status="experimental", evidence="evidence = []")
        ),
    )

    table = load_capabilities(path)

    assert status_for(
        table, "example_entry", room="shoebox", materials="real_impedance"
    ) == "experimental"


def test_duplicate_room_material_pair_is_rejected(tmp_path: Path) -> None:
    """同一入口下同一個 (房型, 材料形式) 出現兩次，就不知道該信哪一條。"""
    block = _capability_block()
    path = _write_table(
        tmp_path / "capabilities.toml",
        _entry_block(capability=block + block),
    )

    with pytest.raises(ValidationError, match="重複的組合"):
        load_capabilities(path)


def test_duplicate_entry_name_is_rejected(tmp_path: Path) -> None:
    """入口名重複，就答不出「這次輸入落在哪一節」。"""
    first = _entry_block(capability=_capability_block())
    second = _entry_block(capability=_capability_block())
    path = _write_table(tmp_path / "capabilities.toml", first + second)

    with pytest.raises(ValidationError, match="同名入口"):
        load_capabilities(path)


@pytest.mark.parametrize(
    ("frequency_hz", "message"),
    (("[20000.0, 20.0]", "遞增"), ("[0.0, 20000.0]", "正數"), ("[-1.0, 20.0]", "正數")),
)
def test_frequency_range_must_be_positive_and_ascending(
    tmp_path: Path, frequency_hz: str, message: str
) -> None:
    """頻率範圍的兩端是正數而且遞增；頭尾顛倒或非正的界線沒有意義。"""
    path = _write_table(
        tmp_path / "capabilities.toml",
        _entry_block(capability=_capability_block(frequency_hz=frequency_hz)),
    )

    with pytest.raises(ValidationError, match=message):
        load_capabilities(path)


def test_extra_field_is_rejected(tmp_path: Path) -> None:
    """表不准夾帶沒登記的欄位——欄位是契約，多的那一格沒有人守。"""
    path = _write_table(
        tmp_path / "capabilities.toml",
        _entry_block(capability=_capability_block()) + "unregistered = true\n",
    )

    with pytest.raises(ValidationError, match="extra"):
        load_capabilities(path)


def test_capability_table_path_has_no_default() -> None:
    """路徑必給：這一支不替呼叫端決定能力表住哪裡。"""
    import inspect

    parameter = inspect.signature(load_capabilities).parameters["path"]

    assert parameter.default is inspect.Parameter.empty


def test_loaded_table_is_frozen(tmp_path: Path) -> None:
    """載入後不准就地改；能力表是資料，不是可變狀態。"""
    path = _write_table(
        tmp_path / "capabilities.toml",
        _entry_block(capability=_capability_block()),
    )
    table = load_capabilities(path)

    assert isinstance(table, CapabilityTable)
    with pytest.raises(ValidationError):
        table.entry[0].capability[0].status = "unsupported"


def test_unknown_combination_raises(tmp_path: Path) -> None:
    """查一條表上沒有的組合要大聲炸，不回一個看起來合理的狀態。"""
    path = _write_table(
        tmp_path / "capabilities.toml",
        _entry_block(capability=_capability_block()),
    )
    table = load_capabilities(path)

    with pytest.raises(KeyError, match="沒有這個組合"):
        status_for(table, "example_entry", room="hall", materials="real_impedance")


# --- 2. 表上的收據都是真的 ---------------------------------------------------


def test_real_table_loads_with_the_named_entries() -> None:
    """真表載得起來，四個入口都在，而且每個入口至少一條 validated。"""
    table = load_capabilities(_TABLE_PATH)

    assert {entry.name for entry in table.entry} >= {
        "three_lane_report",
        "fem_rigid",
        "late_energy",
        "catalog_absorption",
    }
    for entry in table.entry:
        assert any(item.status == "validated" for item in entry.capability), entry.name


def test_every_evidence_node_collects(git_sandbox: GitSandbox) -> None:
    """表上每一個測試節點都要真的收得到——編出來的節點名字在這裡紅。"""
    table = load_capabilities(_TABLE_PATH)
    nodes = [
        evidence
        for entry in table.entry
        for item in entry.capability
        for evidence in item.evidence
        if "::" in evidence
    ]
    assert nodes
    absolute_nodes = []
    for node in nodes:
        test_path, separator, test_name = node.partition("::")
        assert separator and test_name, node
        absolute_nodes.append(f"{_REPO / test_path}::{test_name}")
    env = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTHONPATH": os.pathsep.join((str(_REPO / "src"), str(_REPO))),
    }
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-o",
            "addopts=",
            "-p",
            "no:cacheprovider",
            *absolute_nodes,
        ],
        cwd=git_sandbox.root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_every_evidence_file_exists() -> None:
    """表上指名的答案檔與測試檔都要真的在版控樹裡。"""
    table = load_capabilities(_TABLE_PATH)
    files = [
        evidence.partition("::")[0]
        for entry in table.entry
        for item in entry.capability
        for evidence in item.evidence
    ]

    assert files
    for relative in files:
        assert (_REPO / relative).exists(), relative


def test_unsupported_combinations_have_no_evidence() -> None:
    """標 unsupported 就不准掛收據——它這一版根本沒被驗過。"""
    table = load_capabilities(_TABLE_PATH)
    unsupported = [
        item
        for entry in table.entry
        for item in entry.capability
        if item.status == "unsupported"
    ]

    assert unsupported
    assert all(not item.evidence for item in unsupported)


def test_three_lane_report_declares_complex_and_frequency_dependent_unsupported() -> None:
    """複數與逐頻阻抗要明確標 unsupported，而且指到票 #309。"""
    table = load_capabilities(_TABLE_PATH)

    for materials in ("complex_impedance_by_wall", "frequency_dependent_impedance"):
        assert status_for(
            table, "three_lane_report", room="shoebox", materials=materials
        ) == "unsupported"
        note = next(
            item.note
            for item in table.for_entry("three_lane_report").capability
            if item.materials == materials
        )
        assert "#309" in note


# --- 3. 三個入口的輸出帶著同一組標記 -----------------------------------------


def _fake_fem_energy(
    *,
    frequencies_hz: tuple[float, ...],
    **_ignored: object,
) -> tuple[float, ...]:
    return tuple(0.000012345 for _ in frequencies_hz)


def _three_lane_input() -> dict[str, object]:
    rho_c_pa_s_per_m = 1.2 * 343.0
    return {
        "room_m": {"Lx": 6.0, "Ly": 4.0, "Lz": 3.0},
        "source_m": {"x": 1.2, "y": 1.3, "z": 1.1},
        "receiver_m": {"x": 4.7, "y": 2.8, "z": 1.4},
        "sound_speed_m_s": 343.0,
        "density_kg_m3": 1.2,
        "impedance_pa_s_per_m_by_wall": {
            wall: 4.0 * rho_c_pa_s_per_m for wall in Wall.wall_names()
        },
        "scattering_by_wall": {wall: 0.2 for wall in Wall.wall_names()},
    }


def test_three_lane_report_cli_prints_capability_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """命令列要印出 capability 節，而且 status 跟表上那一條一致。"""
    from aosr.physics import three_lane_report, three_lane_report_cli

    input_path = tmp_path / "room.json"
    input_path.write_text(json.dumps(_three_lane_input()), encoding="utf-8")
    monkeypatch.setattr(three_lane_report, "_solve_fem_energy", _fake_fem_energy)

    exit_code = three_lane_report_cli.main([str(input_path)])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "capability entry=three_lane_report" in output
    assert "status=validated" in output
    expected = status_for(
        load_capabilities(_TABLE_PATH),
        "three_lane_report",
        room="shoebox",
        materials="real_frequency_independent_impedance",
    )
    assert f"status={expected}" in output
    assert output.index("capability ") < output.index("f_s_hz")


def test_three_lane_report_dataclass_carries_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """報表本身要帶能力的欄位，不是只有命令列印字時才知道。"""
    from aosr.geometry.shoebox import Point, Room
    from aosr.physics import three_lane_report

    monkeypatch.setattr(three_lane_report, "_solve_fem_energy", _fake_fem_energy)

    report = three_lane_report.solve_three_lane_report(
        room=Room(6.0, 4.0, 3.0),
        source=Point(1.2, 1.3, 1.1),
        receiver=Point(4.7, 2.8, 1.4),
        sound_speed_m_s=343.0,
        density_kg_m3=1.2,
        impedance_by_wall={wall: 4.0 * 411.6 for wall in Wall.all()},
    )

    assert report.capability.status == "unverified"
    assert report.capability.entry == "three_lane_report"


def test_late_energy_cli_prints_capability_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """晚期混響命令列的輸出開頭要有 capability 節，status 跟表一致。"""
    from aosr.physics import late_energy_cli

    exit_code = late_energy_cli.main(
        [
            str(_REPO / "blueprint" / "reference_art_flat.json"),
            "--compare",
            "--contracts",
            str(_REPO / "blueprint" / "precision_contracts.toml"),
        ]
    )
    first_line = capsys.readouterr().out.splitlines()[0]

    assert exit_code == 0
    assert first_line.startswith("capability entry=late_energy ")
    assert "status=validated" in first_line


def test_fem_rigid_capability_line_reports_the_rigid_combination() -> None:
    """剛性入口的 capability 節要指名 rigid_walls 那一條。"""
    from aosr.physics import fem_rigid

    line = fem_rigid.capability_line(_TABLE_PATH)

    assert line.startswith("capability entry=fem_rigid ")
    assert "materials=rigid_walls" in line
    assert "status=validated" in line
    assert "blueprint/reference_fem_rigid.json" in line


# --- 4. 輸入驗證：不支援的阻抗形式當場擋下 -------------------------------


@pytest.mark.parametrize(
    "cell",
    (
        {"real": 1646.4, "imag": -100.0},
        [1646.4, 1646.5, 1646.6],
        {"125.0": 1646.4, "250.0": 1646.5},
    ),
    ids=("complex_object", "frequency_list", "frequency_object"),
)
def test_three_lane_cli_rejects_unsupported_impedance_forms(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    cell: object,
) -> None:
    """複數或逐頻阻抗進報表入口要回 2，而且訊息提到票 #309 與 unsupported。"""
    from aosr.physics import three_lane_report_cli

    document = _three_lane_input()
    by_wall = document["impedance_pa_s_per_m_by_wall"]
    assert isinstance(by_wall, dict)
    by_wall[next(iter(by_wall))] = cell
    input_path = tmp_path / "unsupported.json"
    input_path.write_text(json.dumps(document), encoding="utf-8")

    exit_code = three_lane_report_cli.main([str(input_path)])
    output = capsys.readouterr().out

    assert exit_code == 2
    assert "#309" in output
    assert "unsupported" in output


def test_three_lane_cli_rejects_negative_real_impedance(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """負的實數阻抗不是這裡收的材料形式，一樣要擋。"""
    from aosr.physics import three_lane_report_cli

    document = _three_lane_input()
    by_wall = document["impedance_pa_s_per_m_by_wall"]
    assert isinstance(by_wall, dict)
    by_wall[next(iter(by_wall))] = -5.0
    input_path = tmp_path / "negative.json"
    input_path.write_text(json.dumps(document), encoding="utf-8")

    exit_code = three_lane_report_cli.main([str(input_path)])

    assert exit_code == 2
    assert "unsupported" in capsys.readouterr().out


def test_capability_lookup_rejects_unsupported(tmp_path: Path) -> None:
    """表上那條組合是 unsupported 時，查詢端要拒絕而不是放行。"""
    path = _write_table(
        tmp_path / "capabilities.toml",
        _entry_block(
            capability=_capability_block(
                status="unsupported", evidence="evidence = []"
            )
        ),
    )
    table = load_capabilities(path)

    assert status_for(
        table, "example_entry", room="shoebox", materials="real_impedance"
    ) == "unsupported"
    assert (
        evidence_for(
            table, "example_entry", room="shoebox", materials="real_impedance"
        )
        == ()
    )
