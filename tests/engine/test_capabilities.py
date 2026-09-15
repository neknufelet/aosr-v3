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
    Capability,
    CapabilityStatus,
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


def test_blank_evidence_item_is_rejected(tmp_path: Path) -> None:
    """evidence 的每一項都要非空——一格空白不是一份收據。"""
    path = _write_table(
        tmp_path / "capabilities.toml",
        _entry_block(capability=_capability_block(evidence='evidence = ["  "]')),
    )

    with pytest.raises(ValidationError, match="非空"):
        load_capabilities(path)


def test_duplicate_evidence_item_is_rejected(tmp_path: Path) -> None:
    """同一條組合裡兩次指名同一份收據，不是兩份收據。"""
    node = "tests/engine/test_capabilities.py::test_every_evidence_node_collects"
    path = _write_table(
        tmp_path / "capabilities.toml",
        _entry_block(capability=_capability_block(evidence=f'evidence = ["{node}", "{node}"]')),
    )

    with pytest.raises(ValidationError, match="重複"):
        load_capabilities(path)


def test_unsupported_with_evidence_is_rejected(tmp_path: Path) -> None:
    """標 unsupported 就不准掛收據——它這一版根本沒被驗過。"""
    path = _write_table(
        tmp_path / "capabilities.toml",
        _entry_block(capability=_capability_block(status="unsupported")),
    )

    with pytest.raises(ValidationError, match="unsupported.*evidence"):
        load_capabilities(path)


def test_duplicate_output_is_rejected(tmp_path: Path) -> None:
    """同一格輸出的欄名寫兩次，多不出第二個欄位。"""
    block = _capability_block().replace('outputs = ["energy"]', 'outputs = ["energy", "energy"]')
    path = _write_table(tmp_path / "capabilities.toml", _entry_block(capability=block))

    with pytest.raises(ValidationError, match="outputs"):
        load_capabilities(path)


def test_empty_output_list_is_rejected(tmp_path: Path) -> None:
    """輸出欄空的一條不准進表：那一條會安靜印成 outputs=，沒有欄位卻說得好像有。"""
    block = _capability_block().replace('outputs = ["energy"]', "outputs = []")
    path = _write_table(tmp_path / "capabilities.toml", _entry_block(capability=block))

    with pytest.raises(ValidationError, match="outputs"):
        load_capabilities(path)


def test_surrounding_whitespace_is_trimmed_before_validating(tmp_path: Path) -> None:
    """前後空白先去掉，跟規矩卡的做法一致；"validated " 不該溜過三個值的檢查。"""
    path = _write_table(
        tmp_path / "capabilities.toml",
        _entry_block(capability=_capability_block(status="validated ")),
    )

    table = load_capabilities(path)

    assert status_for(
        table, "example_entry", room="shoebox", materials="real_impedance"
    ) == "validated"


# --- 2. 表上的收據都是真的 ---------------------------------------------------


def test_real_table_loads_with_the_named_entries() -> None:
    """真表載得起來，四個入口都在，而且每個入口至少一條 validated 或 experimental。"""
    table = load_capabilities(_TABLE_PATH)

    assert {entry.name for entry in table.entry} >= {
        "three_lane_report",
        "fem_rigid",
        "late_energy",
        "catalog_absorption",
    }
    for entry in table.entry:
        assert any(
            item.status in ("validated", "experimental") for item in entry.capability
        ), entry.name


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


def _assert_line_carries_range_and_outputs(
    line: str,
    *,
    entry: str,
    materials: str,
) -> None:
    """咬住 capability 那一行真的帶了表上那一條的頻率範圍與輸出欄。

    只印 status 的話，validated 會蓋到整條頻率軸與所有輸出欄——剛性入口的軸到
    300 Hz、晚期能量印的欄位比驗過的多，卻掛著同一個 validated。這裡逐格對表，
    不是對程式裡抄的常數。
    """
    table = load_capabilities(_TABLE_PATH)
    record = next(
        item
        for item in table.for_entry(entry).capability
        if item.room == "shoebox" and item.materials == materials
    )
    lower, upper = record.frequency_hz
    assert f"frequency_hz=[{lower:g}, {upper:g}]" in line, line
    assert f"outputs={','.join(record.outputs)}" in line, line


def _make_record(
    *,
    status: CapabilityStatus = "validated",
    evidence: tuple[str, ...] = ("tests/engine/test_capabilities.py::test_x",),
) -> Capability:
    """造一條欄位齊全的 capability：輸出欄非空、頻率範圍正數遞增。

    這裡是「有沒有查證」的預設值，不是「驗過沒有」的宣告：預設的 evidence
    ``tests/engine/test_capabilities.py::test_x`` 只是一個佔位字串，這個節點不存在，
    這一支不宣稱它指得到任何考卷——真的節點對不對得上由
    :func:`test_every_evidence_node_collects` 對真表咬。直接走 :class:`Capability`
    而不是能力表檔案：能力表本身由別的考卷咬，這裡要驗的是「那一行只收本人」
    這條規則。欄位逐一寫出來，不靠 `**dict` 展開——那會讓型別警衛看不到每一個
    關鍵字參數的型別。
    """
    return Capability(
        room="shoebox",
        materials="m",
        frequency_hz=(20.0, 300.0),
        outputs=("spl_db", "t30_s"),
        status=status,
        evidence=evidence,
        note="考卷造的",
    )


def test_a_validated_record_without_evidence_cannot_be_built() -> None:
    """沒證據卻說驗過造不出來：validated 但 evidence 空，建構那一刻就 ValidationError，印不出那一行。"""
    with pytest.raises(ValidationError) as caught:
        Capability(
            room="shoebox",
            materials="m",
            frequency_hz=(20.0, 300.0),
            outputs=("spl_db",),
            status="validated",
            evidence=(),
            note="沒有收據的綠",
        )

    assert "evidence" in str(caught.value)


def test_capability_report_line_prints_unchecked_only_for_none() -> None:
    """這一跑沒查表＝record is None：印原本 unchecked 那一行，字串不變。"""
    from aosr.physics import capability_report

    line = capability_report.capability_line("x", "shoebox", "m", None)

    assert line == (
        "capability entry=x room=shoebox materials=m "
        "frequency_hz=none outputs=none status=unchecked evidence=none"
    )


def test_capability_report_line_prints_the_record_itself() -> None:
    """有 record 就照本人印四格：範圍、輸出欄、狀態都來自那一條，收據那一格印 `none`（這一題的 record 沒有 evidence）。"""
    from aosr.physics import capability_report

    record = _make_record(status="experimental", evidence=())
    line = capability_report.capability_line("x", "shoebox", "m", record)
    assert line == (
        "capability entry=x room=shoebox materials=m "
        "frequency_hz=[20, 300] outputs=spl_db,t30_s "
        "status=experimental evidence=none"
    )


def test_capability_report_line_rejects_a_mismatched_room_or_materials() -> None:
    """名字與那一條對不上就報錯：拿實數阻抗那一條卻印 materials=rigid_walls，等於用這一條的名字蓋另一條的證據。"""
    from aosr.physics import capability_report

    record = Capability(
        room="shoebox",
        materials="real_frequency_independent_impedance",
        frequency_hz=(20.0, 300.0),
        outputs=("spl_db", "t30_s"),
        status="experimental",
        evidence=(),
        note="實數阻抗那一條的形狀",
    )

    with pytest.raises(ValueError) as caught:
        capability_report.capability_line("x", "shoebox", "rigid_walls", record)

    message = str(caught.value)
    assert "rigid_walls" in message
    assert "real_frequency_independent_impedance" in message


def test_capability_report_line_takes_no_half_given_cells() -> None:
    """半套的憑據造不出來：這一支不再收 status／frequency_hz／outputs／evidence 那些格子，給了的格子也不會被安靜丟掉。"""
    import inspect

    from aosr.physics import capability_report

    parameters = set(inspect.signature(capability_report.capability_line).parameters)

    assert parameters == {"entry", "room", "materials", "record"}
    assert not hasattr(capability_report, "describe_missing_half")
    assert not hasattr(capability_report, "capability_line_from_record")


def test_report_capability_carries_the_record_or_nothing() -> None:
    """ReportCapability 不再有 status 那一格可單獨給：只收 record 本人或 None。

    「不敢單獨給 status」用 `dataclasses.fields` 咬，不靠丟一個 mypy 會抱怨的
    關鍵字——那要掛 `# type: ignore`，而每一筆放行都要理由跟到期日。
    """
    from dataclasses import fields

    from aosr.physics.three_lane_report import ReportCapability

    unchecked = ReportCapability(
        entry="three_lane_report",
        room="shoebox",
        materials="real_frequency_independent_impedance",
        record=None,
    )
    assert unchecked.record is None
    assert {field.name for field in fields(ReportCapability)} == {
        "entry",
        "room",
        "materials",
        "record",
    }


def test_report_capability_section_prints_range_and_outputs() -> None:
    """報表帶的那一格是表上那一條本人，印出來的行因此一定帶著範圍與輸出欄。"""
    from aosr.physics import three_lane_report_cli
    from aosr.physics.three_lane_report import ReportCapability

    record = _make_record()
    section = three_lane_report_cli._capability_section(
        ReportCapability(
            entry="three_lane_report",
            room="shoebox",
            materials="m",
            record=record,
        )
    )

    assert section.startswith("capability entry=three_lane_report room=shoebox materials=m ")
    assert f"frequency_hz=[{record.frequency_hz[0]:g}, {record.frequency_hz[1]:g}]" in section
    assert f"outputs={','.join(record.outputs)}" in section
    assert "status=validated" in section


def test_three_lane_report_cli_prints_capability_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """命令列要印出 capability 節，status 跟表上那一條一致，且帶著範圍與輸出欄。"""
    from aosr.physics import three_lane_report, three_lane_report_cli

    input_path = tmp_path / "room.json"
    input_path.write_text(json.dumps(_three_lane_input()), encoding="utf-8")
    monkeypatch.setattr(three_lane_report, "_solve_fem_energy", _fake_fem_energy)

    exit_code = three_lane_report_cli.main(
        [str(input_path), "--capabilities", str(_TABLE_PATH)]
    )
    output = capsys.readouterr().out
    first_line = output.splitlines()[0]

    assert exit_code == 0
    assert "capability entry=three_lane_report" in output
    expected = status_for(
        load_capabilities(_TABLE_PATH),
        "three_lane_report",
        room="shoebox",
        materials="real_frequency_independent_impedance",
    )
    assert f"status={expected}" in output
    _assert_line_carries_range_and_outputs(
        first_line,
        entry="three_lane_report",
        materials="real_frequency_independent_impedance",
    )
    assert output.index("capability ") < output.index("f_s_hz")


def test_three_lane_report_dataclass_carries_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """報表本身要帶能力的欄位；沒給能力表時 record 是 None，不自創第四個狀態。"""
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

    assert report.capability.record is None
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
            "--capabilities",
            str(_TABLE_PATH),
        ]
    )
    first_line = capsys.readouterr().out.splitlines()[0]

    assert exit_code == 0
    assert first_line.startswith("capability entry=late_energy ")
    assert "materials=real_frequency_independent_impedance" in first_line
    assert "status=validated" in first_line
    _assert_line_carries_range_and_outputs(
        first_line,
        entry="late_energy",
        materials="real_frequency_independent_impedance",
    )


def test_late_energy_cli_rejects_unsupported_material_form(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """入口要從輸入判斷材料形式：阻抗有虛部時去查表，查不到就回 2，不標 validated。"""
    from aosr.physics import late_energy_cli

    # 直接讀答案檔本人：varied 那一案的阻抗就帶虛部，抄一份到 tmp_path 不改內容
    # 只是多一次 I/O，證不了任何事。
    exit_code = late_energy_cli.main(
        [
            str(_REPO / "blueprint" / "reference_art_varied.json"),
            "--capabilities",
            str(_TABLE_PATH),
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 2
    assert "complex_impedance_by_wall" in output
    assert "不支援" in output


def test_late_energy_cli_requires_capabilities(capsys: pytest.CaptureFixture[str]) -> None:
    """--capabilities 必給，物理層不設隱含預設。"""
    from aosr.physics import late_energy_cli

    with pytest.raises(SystemExit) as caught:
        late_energy_cli.main([str(_REPO / "blueprint" / "reference_art_flat.json")])

    assert caught.value.code == 2
    assert "--capabilities" in capsys.readouterr().err


def test_fem_rigid_capability_line_reports_the_rigid_combination() -> None:
    """剛性入口的 capability 節要指名 rigid_walls 那一條，並帶它的範圍與輸出欄。"""
    from aosr.physics import fem_rigid

    line = fem_rigid.capability_line(_TABLE_PATH, "rigid_walls")

    assert line.startswith("capability entry=fem_rigid ")
    assert "materials=rigid_walls" in line
    assert "status=validated" in line
    assert "blueprint/reference_fem_rigid.json" in line
    _assert_line_carries_range_and_outputs(
        line, entry="fem_rigid", materials="rigid_walls"
    )


def _fake_fenics_contract(
    answer_path: Path,
    tolerance_rel: float,
) -> object:
    from aosr.physics.fem_rigid import FenicsContractReport, FenicsPointJudgment

    return FenicsContractReport(
        (
            FenicsPointJudgment(
                case_name="flat",
                frequency_hz=10.0,
                actual_pressure=1.0 + 0.0j,
                expected_pressure=1.0 + 0.0j,
                relative_error=0.0,
                within_contract=True,
            ),
        ),
        tolerance_rel,
    )


def test_fem_rigid_compare_prints_the_real_material_combination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--compare 解的是實數阻抗牆，印出來的 materials 不准是 rigid_walls。"""
    from aosr.physics import fem_rigid

    monkeypatch.setattr(fem_rigid, "_run_fenics_compare", _fake_fenics_contract)

    exit_code = fem_rigid.main(
        [
            "--compare",
            str(_REPO / "blueprint" / "fem_fenics_answers.json"),
            "--contracts",
            str(_REPO / "blueprint" / "precision_contracts.toml"),
            "--capabilities",
            str(_TABLE_PATH),
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    first_line = output.splitlines()[0]
    assert "materials=real_frequency_independent_impedance" in first_line
    assert "materials=rigid_walls" not in first_line
    assert "blueprint/fem_fenics_answers.json" in first_line
    _assert_line_carries_range_and_outputs(
        first_line,
        entry="fem_rigid",
        materials="real_frequency_independent_impedance",
    )


def test_fem_rigid_input_mode_prints_the_rigid_combination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """輸入模式解的是剛性邊界，印出來的 materials 是 rigid_walls。"""
    import numpy as np

    from aosr.physics import fem_rigid

    case = fem_rigid.load_rigid_reference_case(
        _REPO / "blueprint" / "reference_fem_rigid.json"
    )
    monkeypatch.setattr(
        fem_rigid,
        "solve_rigid_fem_lane_case",
        lambda _case: np.ones(len(case.fem_lane_frequencies_hz), dtype=np.complex128),
    )

    exit_code = fem_rigid.main(
        [str(_REPO / "blueprint" / "reference_fem_rigid.json"), "--capabilities", str(_TABLE_PATH)]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    first_line = output.splitlines()[0]
    assert "materials=rigid_walls" in first_line
    assert "status=validated" in first_line
    assert "blueprint/reference_fem_rigid.json" in first_line
    # 這一跑的解在整條 300 Hz 軸上，但剛性那一條驗過的只到 20 Hz；印出來的那一行
    # 要帶的是表上那一條的範圍（不是這一跑算了多寬）。
    _assert_line_carries_range_and_outputs(
        first_line, entry="fem_rigid", materials="rigid_walls"
    )


def test_fem_rigid_input_mode_rejects_a_non_rigid_boundary(tmp_path: Path) -> None:
    """輸入模式要讀 material.boundary；不是 rigid 就明確報錯，不安靜當剛性算。"""
    from aosr.physics import fem_rigid

    raw = json.loads(
        (_REPO / "blueprint" / "reference_fem_rigid.json").read_text(encoding="utf-8")
    )
    raw["parameters"]["material"]["boundary"] = "impedance"
    changed = tmp_path / "non-rigid.json"
    changed.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="boundary"):
        fem_rigid.load_rigid_reference_case(changed)


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

    exit_code = three_lane_report_cli.main(
        [str(input_path), "--capabilities", str(_TABLE_PATH)]
    )
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

    exit_code = three_lane_report_cli.main(
        [str(input_path), "--capabilities", str(_TABLE_PATH)]
    )

    assert exit_code == 2
    assert "unsupported" in capsys.readouterr().out


def test_three_lane_cli_accepts_zero_scattering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """散射係數 0 是合法輸入，不准被「正實數阻抗」那條擋掉（退步考卷）。"""
    from aosr.physics import three_lane_report, three_lane_report_cli

    document = _three_lane_input()
    document["scattering_by_wall"] = {wall: 0.0 for wall in Wall.wall_names()}
    input_path = tmp_path / "zero-scattering.json"
    input_path.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr(three_lane_report, "_solve_fem_energy", _fake_fem_energy)

    exit_code = three_lane_report_cli.main(
        [str(input_path), "--capabilities", str(_TABLE_PATH)]
    )

    assert exit_code == 0, capsys.readouterr().out


def test_three_lane_cli_rejects_scattering_out_of_range(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """散射係數落在 [0,1] 之外就擋，不管阻抗那條怎麼寫。"""
    from aosr.physics import three_lane_report_cli

    document = _three_lane_input()
    document["scattering_by_wall"] = {wall: 1.5 for wall in Wall.wall_names()}
    input_path = tmp_path / "bad-scattering.json"
    input_path.write_text(json.dumps(document), encoding="utf-8")

    exit_code = three_lane_report_cli.main(
        [str(input_path), "--capabilities", str(_TABLE_PATH)]
    )
    output = capsys.readouterr().out

    assert exit_code == 2
    assert "[0,1]" in output or "[0, 1]" in output


def test_capability_lookup_rejects_unsupported(tmp_path: Path) -> None:
    """表上那條組合是 unsupported 時，查到的狀態與收據都要是 unsupported、沒有收據。"""
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
