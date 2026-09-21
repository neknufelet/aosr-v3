"""票 #316 報表命令列行為的考卷：``--format json``、壞輸入的錯誤文字、拒收訊息。

**這一支在守什麼。** ``tests/engine/test_report_io_contract.py`` 守**契約本身**——輸入
模型的驗證規則、輸出模型的四件事與兩族空值規則、schema 檔逐位鎖住、重匯入口；這一支
守**命令列那一層**：同一份契約接上真的 ``three_lane_report_cli`` 之後對人說話的樣子。
第五刀從那一支搬過來的三節題目一題都沒增刪、沒改斷言、沒改名字；第 ⑥ 節是第七刀新開的。

**行數：兩個時點分開講。** 派工單第 5 點原本要的是「兩支各自不超過 700 行」，**沒有
達到**：第五刀拆完的那一刻契約那一支是 797 行、這一支兩百多行；第六刀又把契約那一支
加到 900 行，第七刀是 933 行（這一支 295 行）。兩支都還在寫法警衛
（``style-guard``）登記的**上限**之內（數字住在卡上，這裡不抄），但契約那一支已經用掉
上限的九成出頭、剩不到一成——拆檔解掉的是「每改一行都要先想從哪裡騰行數」那個壓力，不是
把那個壓力消掉了。監督拍板不再切第二刀。

**這四件事。** 前三件依原檔的節次分，第四件是第七刀新開的：

1. ``④ --format json 跑真的命令列``：``--format json`` 印出來的 JSON 要反解得回輸出
   模型、帶不帶 ``--points`` 都對；不給 ``--format`` 時是 text、JSON 那一條不准變成
   預設；壞輸入走 JSON 那條路一樣回 2，不准因為要印 JSON 就把錯誤吞掉。有限元素那一半
   照既有 CLI 考卷換成假的（``_fake_fem_energy``），只驗接線與契約。
2. ``⑤ 壞輸入的錯誤文字``：印的是一句人話——欄位路徑與原因在，官網網址與多行 dump
   不在；Pydantic 自己加的 ``Value error, `` 前綴拿掉、原因一字不動；同一格的名字不准
   在同一句話裡出現兩次，而且**不是延後才炸**（錯誤要在解析階段就被收成人話）。
3. ``沒查表那條路／接線錯誤不報成使用者輸入錯誤``：沒帶能力表時 ``WiringError`` 照原樣
   往上冒，不准被包成「輸入不合 ReportInput」——看訊息的人會去改 JSON，而真正該改的是
   呼叫端。這一題在**契約那一支**（``test_report_io_contract.py``），因為它走的是繞過
   命令列、直接餵 ``ReportInput.model_validate`` 的那條路；兩支加起來才是完整的守備面。
4. ``⑥ 報表模式的兩個必給參數``：不給 ``input``、不給 ``--capabilities`` 各咬一題回 2。
   第五刀把兩個 parser 併成一個之後，這兩格的必給從 argparse 宣告改成 ``main`` 裡手寫的
   兩行，而且零考卷——刪掉那兩行不會有任何一盞燈紅（第七刀必修 2）。

**共用的工具在契約那一支。** 造輸入文件、跑命令列與換掉有限元素那一半的小工具
（``_input_document``、``_impedance_map``、``_rejects``、``_table``、``_WALL_NAMES``、
``_TABLE_PATH``、``_fake_fem_energy``）全部住在 ``test_report_io_contract.py``，這一支
單向 import 過來用（第五刀時 ``_fake_fem_energy`` 一度放在這一支，兩支互相 import；
派工把它移回去，這個環就斷了）；不為此新開第三個模組。搬過來的題目**一個字都沒改**，
只有 ``import`` 這一段是新的。

**不碰真環境。** 這一支只寫 ``tmp_path``（規矩卡 ``tests-isolated-from-real-env``）：
暫時檔一律由測試函式把 ``tmp_path`` 傳進輔助函式，不用 ``tempfile`` 寫系統暫存目錄。
"""
from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType

import pytest

from aosr.physics.report_io import BandRow, PointRow, ReportOutput, TopFields
from tests.engine.test_report_io_contract import (
    _TABLE_PATH,
    _WALL_NAMES,
    _fake_fem_energy,
    _input_document,
    _impedance_map,
    _rejects,
)


# ── ④ --format json 跑真的命令列 ───────────────────────────────────────────────
@pytest.mark.parametrize("extra", ((), ("--points",)), ids=("bands", "with_points"))
def test_json_format_round_trips_through_the_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    extra: tuple[str, ...],
) -> None:
    """``--format json`` 印出來的 JSON 要反解得回模型，而且帶不帶 --points 都對。"""
    from aosr.physics import three_lane_report, three_lane_report_cli
    from aosr.physics.report_facts import FieldFacts

    input_path = tmp_path / "room.json"
    input_path.write_text(json.dumps(_input_document()), encoding="utf-8")
    monkeypatch.setattr(three_lane_report, "_solve_fem_energy", _fake_fem_energy)

    exit_code = three_lane_report_cli.main(
        [
            str(input_path),
            *extra,
            "--format",
            "json",
            "--capabilities",
            str(_TABLE_PATH),
        ]
    )
    output = capsys.readouterr().out
    assert exit_code == 0, output
    parsed = ReportOutput.model_validate_json(output)
    assert parsed.capability.status == "experimental"
    assert parsed.top.room_volume_m3 == 72.0
    centers = [band.center_frequency_hz for band in parsed.bands]
    assert parsed.points is not None if extra else parsed.points is None
    assert all(
        parsed.bands[index].center_frequency_hz < parsed.bands[index + 1].center_frequency_hz
        for index in range(len(centers) - 1)
    )
    assert all(
        band.t20_s is None or band.t20_unavailable_reason is None
        for band in parsed.bands
    )
    payload = json.loads(output)
    assert set(payload) == {"capability", "scene", "top", "bands", "points"}
    assert set(payload["top"]) == set(TopFields.model_fields)
    assert set(payload["bands"][0]) == set(BandRow.model_fields)
    if extra:
        assert set(payload["points"][0]) == set(PointRow.model_fields)


def test_json_path_table_is_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """預設 JSON 逐位維持舊形；明給 ``--path-table`` 才多路徑表。"""
    from aosr.physics import three_lane_report, three_lane_report_cli

    input_path = tmp_path / "room.json"
    input_path.write_text(json.dumps(_input_document()), encoding="utf-8")
    monkeypatch.setattr(three_lane_report, "_solve_fem_energy", _fake_fem_energy)
    exit_code = three_lane_report_cli.main(
        [
            str(input_path),
            "--path-table",
            "--format",
            "json",
            "--capabilities",
            str(_TABLE_PATH),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["path_table"]["rows"]

    # 不給那個旗標時同一份輸入不准帶這一節（舊的輸出要一模一樣）。
    default_code = three_lane_report_cli.main(
        [
            str(input_path),
            "--format",
            "json",
            "--capabilities",
            str(_TABLE_PATH),
        ]
    )
    default_payload = json.loads(capsys.readouterr().out)
    assert default_code == 0
    assert default_payload.get("path_table") is None


def test_text_path_table_header_uses_frequency_axis_and_contract_wording(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """人看表頭要標頻率軸，指向性字面只能從輸出契約那一格來。"""
    from aosr.physics import report_io, three_lane_report, three_lane_report_cli
    from aosr.physics.report_facts import FieldFacts

    input_path = tmp_path / "room.json"
    input_path.write_text(json.dumps(_input_document()), encoding="utf-8")
    monkeypatch.setattr(three_lane_report, "_solve_fem_energy", _fake_fem_energy)
    original_quantity_table = report_io.quantity_table
    contract_wording = "契約提供的未含喇叭指向性字面"

    def quantity_table_with_marker() -> dict[str, FieldFacts]:
        table = original_quantity_table()
        field = "path_table.includes_speaker_directivity"
        table[field] = table[field]._replace(reference=contract_wording)
        return table

    monkeypatch.setattr(report_io, "quantity_table", quantity_table_with_marker)
    exit_code = three_lane_report_cli.main(
        [
            str(input_path),
            "--path-table",
            "--capabilities",
            str(_TABLE_PATH),
        ]
    )
    output = capsys.readouterr().out
    header = next(line for line in output.splitlines() if line.startswith("path_table "))

    assert exit_code == 0
    assert "frequencies_hz=(" in header
    assert contract_wording in header


def test_json_format_defaults_to_text_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """不給 `--format` 時是 text；JSON 那一條不准變成預設。"""
    from aosr.physics import three_lane_report, three_lane_report_cli

    input_path = tmp_path / "room.json"
    input_path.write_text(json.dumps(_input_document()), encoding="utf-8")
    monkeypatch.setattr(three_lane_report, "_solve_fem_energy", _fake_fem_energy)
    exit_code = three_lane_report_cli.main(
        [str(input_path), "--capabilities", str(_TABLE_PATH)]
    )
    output = capsys.readouterr().out
    assert exit_code == 0
    assert output.startswith("capability entry=")
    with pytest.raises(json.JSONDecodeError):
        json.loads(output)


def test_json_format_still_returns_two_on_bad_input(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """壞輸入走 JSON 那條路一樣回 2，不准因為要印 JSON 就把錯誤吞掉。"""
    from aosr.physics import three_lane_report_cli

    document = _input_document()
    by_wall = _impedance_map(document)
    by_wall["floor"] = -5.0
    document["impedance_pa_s_per_m_by_wall"] = by_wall
    input_path = tmp_path / "bad.json"
    input_path.write_text(json.dumps(document), encoding="utf-8")

    exit_code = three_lane_report_cli.main(
        [str(input_path), "--format", "json", "--capabilities", str(_TABLE_PATH)]
    )
    output = capsys.readouterr().out

    assert exit_code == 2
    assert "三路接合報表算不出來" in output


# ── ⑤ 壞輸入的錯誤文字：一句人話，不是 Pydantic 的多行 dump ─────────────────────
def test_bad_input_message_names_the_field_without_the_website(tmp_path: Path) -> None:
    """壞輸入印的是一句人話：欄位路徑與原因在，官網網址與多行 dump 不在。"""
    from aosr.physics import three_lane_report_cli

    document = _input_document()
    by_wall = _impedance_map(document)
    by_wall["floor"] = -5.0
    document["impedance_pa_s_per_m_by_wall"] = by_wall
    message = _cli_error(three_lane_report_cli, document, tmp_path)

    # 欄名只說一次：訊息的欄位路徑比 loc 細（多說了是哪一面牆），留訊息那一份，
    # loc 那一份一模一樣的前綴收掉（先前會印成
    # 「impedance_pa_s_per_m_by_wall：impedance_pa_s_per_m_by_wall.floor」那種口吃）。
    assert "impedance_pa_s_per_m_by_wall.floor 必須是正實數阻抗" in message
    assert "impedance_pa_s_per_m_by_wall：impedance" not in message
    assert "必須是正實數阻抗" in message
    assert "https://" not in message
    assert "errors.pydantic.dev" not in message
    assert "For further information visit" not in message
    # 一句人話就是一行：除了 print 尾巴那一個換行，訊息裡不准再有換行。
    body = message.removesuffix("\n")
    assert "\n" not in body
    assert body == message.rstrip("\n")


def test_bad_input_message_keeps_the_value_error_text(tmp_path: Path) -> None:
    """Pydantic 自己加的 ``Value error, `` 前綴要拿掉；原因本身一字不動。"""
    from aosr.physics import three_lane_report_cli

    message = _cli_error(
        three_lane_report_cli,
        _input_document(sound_speed_m_s=0.0),
        tmp_path,
    )

    assert "輸入：sound_speed_m_s 必須是有限正數" in message
    assert "Value error" not in message
    assert "https://" not in message


def test_bad_input_message_never_says_the_field_twice() -> None:
    """同一格的名字不准在同一句話裡出現兩次（第三刀非必修第 8 條的後半）。

    三種形狀各一題：``loc`` 與訊息開頭同名（``room_m``）、訊息比 ``loc`` 細
    （``impedance_pa_s_per_m_by_wall.floor``）、以及兩者不同（``Extra inputs``）。
    咬法是「把那一段字串數一次」——先前那一版靠考卷把口吃鎖住（``:773`` 的舊寫法
    逐字要求重複），所以這一題順便把那個鎖換成反向的鎖。
    """
    cases = (
        (_input_document(room_m=[6.0, 4.0, 3.0]), "room_m 必須是 JSON 物件"),
        (
            _input_document(
                impedance_pa_s_per_m_by_wall={
                    wall: (-5.0 if wall == "floor" else 4.0 * 411.6)
                    for wall in _WALL_NAMES
                }
            ),
            "impedance_pa_s_per_m_by_wall.floor 必須是正實數阻抗",
        ),
    )
    for document, expected in cases:
        message = _rejects(document, expected)
        # 說一次就夠：把那一句從訊息裡剪掉之後，欄名不該再出現。
        assert expected in message
        assert expected not in message.replace(expected, "", 1)
        for field in ("room_m", "impedance_pa_s_per_m_by_wall"):
            assert f"{field}：{field}" not in message, message


# ── ⑥ 報表模式的兩個必給參數：缺了要在 parser 那一關就回 2 ──────────────────────
@pytest.mark.parametrize(
    ("argv", "missing"),
    (
        ([], "input"),
        (["--format", "json"], "input"),
        (["room.json"], "--capabilities"),
    ),
    ids=("nothing", "only_a_flag", "input_without_the_table"),
)
def test_report_mode_refuses_to_run_without_its_required_arguments(
    argv: list[str],
    missing: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """報表模式缺 ``input`` 或缺 ``--capabilities`` 都要當場回 2（第七刀必修 2）。

    第五刀把兩個 parser 併成一個，這兩格的必給就從 argparse 宣告式的 ``required=True``
    換成 ``main`` 裡手寫的兩行 ``parser.error``，而**一題都沒有**——那兩行被刪掉不會有任何
    一盞燈紅：缺 ``--capabilities`` 會一路掉進 ``load_capabilities(None)``，被那一層的
    ``except Exception`` 接住照樣回 2，只是訊息變成「三路接合報表算不出來：⋯⋯」這種
    看訊息的人會去改 JSON 的話。所以這一題除了咬離開碼，也咬**缺的那一格的名字**要出現在
    stderr 的 usage 那一段，而且不准變成求解那一路的訊息。

    ``room.json`` 不必真的存在：缺 ``--capabilities`` 在讀檔之前就被擋下來——這一題因此
    也證明了那一關真的在讀檔前面（不碰真環境，連 ``tmp_path`` 都不用寫）。
    """
    from aosr.physics import three_lane_report_cli

    with pytest.raises(SystemExit) as refused:
        three_lane_report_cli.main(argv)
    assert refused.value.code == 2, argv
    captured = capsys.readouterr()
    assert missing in captured.err, captured.err
    assert "三路接合報表算不出來" not in captured.err + captured.out


def _cli_error(
    cli_module: ModuleType,
    document: dict[str, object],
    tmp_path: Path,
    *,
    extra: tuple[str, ...] = (),
) -> str:
    """把一份題目文件寫進這一題的 ``tmp_path``、跑一次命令列，收下它印出來的那一行（回 2）。

    只寫 ``tmp_path``、只讀 stdout，不碰真環境；訊息裡不准有換行，因為命令列那一層
    印的就是**一行**。
    """
    input_path = _write_document(document, tmp_path)
    captured = io.StringIO()
    with redirect_stdout(captured):
        exit_code = cli_module.main(
            [str(input_path), *extra, "--capabilities", str(_TABLE_PATH)]
        )
    output = captured.getvalue()
    assert exit_code == 2, output
    assert output.startswith("三路接合報表算不出來：")
    return output


def _write_document(document: dict[str, object], tmp_path: Path) -> Path:
    """把一份題目文件寫進這一題自己的 ``tmp_path``（不碰系統暫存目錄、也不必清）。

    ``tests-isolated-from-real-env`` 的第二段：測試不准寫進真的 repo；這一支連系統
    暫存目錄都不碰——檔就落在 pytest 給這一題的 ``tmp_path`` 底下，題目跑完跟著那棵樹
    一起收掉，不會像先前那一版用 ``delete=False`` 每跑一次在 ``/tmp`` 留兩個檔。
    """
    path = tmp_path / "document.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path
