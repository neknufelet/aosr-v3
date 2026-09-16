"""票 #316 報表輸入／輸出契約的考卷：schema 檔、驗證規則、值空必有原因。

**這一支在守什麼。** 票 #316 把三路報表的輸入與輸出收成凍結的 Pydantic 模型，並匯出兩份
正式 JSON schema 檔。這一支守**契約本身**（票 #316 第五刀把它拆成兩支，理由見下），四件事：

1. ``blueprint/schemas/`` 底下兩份 schema 檔**等於**模型現算出來的結果——檔過期就紅
   （改了模型忘了重匯，機器看得出來）。重匯的入口是
   ``uv run python -m aosr.physics.three_lane_report_cli --regenerate-schemas <目錄>``
   （那一題紅掉時的訊息裡也寫著同一行），而且重匯出來的檔要與版控那一份逐位元組相同。
2. 輸入模型每一條驗證規則各一題。題目形狀照既有的
   ``tests/engine/test_three_lane_report_cli.py`` 與 ``test_capabilities.py`` 補齊，
   不重複造同一種壞輸入。繞過命令列那一條（沒查表＝接線錯誤、``WiringError`` 不被包成
   使用者輸入錯誤）也在這一支，因為它走的是直接 ``ReportInput.model_validate`` 的路。
3. 輸出模型「值空必須有原因」的兩個方向各一題：值空沒原因要炸、值有又給原因也要炸。
4. 兩族「空」分得開：``fem_energy`` 空＝這一帶沒有有限元素頻點，**不必**帶原因；
   T20／T30 空＝值算不出來，**必須**帶原因（票 #316 第二刀監督拍板第 7 條）。

**另一支在守什麼。** ``tests/engine/test_report_io_cli.py`` 守**命令列那一層**：
``--format json``、壞輸入的錯誤文字、拒收訊息。**文字那一路「跟改之前逐位元組相同」
沒有機器在守**——既有 CLI 考卷只比子字串（``test_three_lane_report_cli.py`` 的
``in output`` 那幾條），沒有基準檔可比；監督在 2026-09-16 拿同一份參考房輸入加
``--points`` 各跑一次、兩份輸出逐位相同，那是一次性實測。

**為什麼拆。** 這一支原本剛好頂在寫法警衛（``style-guard``）登記的檔案行數**上限**上
（數字住在卡上，這裡不抄）：再多一行就紅，往後每次改都要先想從哪裡騰行數。票 #316 第五刀把命令列行為
那一半搬去 ``test_report_io_cli.py``，**題目一題都沒增刪、沒改斷言、沒改名字**；兩支
各自有自己的檔頭說明。造輸入文件、跑命令列與換掉有限元素那一半的小工具全部住在
**這一支**（``_input_document``、``_impedance_map``、``_rejects``、``_table``、
``_WALL_NAMES``、``_TABLE_PATH``、``_fake_fem_energy``），由另一支單向 import
過去用——照派工，不為此新開第三個模組；這一支不從另一支 import 任何東西。

**列類別是有欄名的物件。** ``TopFields``／``BandRow``／``PointRow`` 是 Pydantic 模型，
所以 JSON 出去是有欄名的物件、schema 檔裡是 ``properties``；考卷咬住這件事——位置陣列
等於沒有契約，前端讀不出哪一格是什麼。

**不碰真環境。** 這一支只寫 ``tmp_path``（規矩卡 ``tests-isolated-from-real-env``）：
暫時檔一律由測試函式把 ``tmp_path`` 傳進輔助函式，不用 ``tempfile`` 寫系統暫存目錄。
"""
from __future__ import annotations

import io
import json
from collections.abc import Callable, Mapping
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from aosr.config.capabilities import CapabilityTable, load_capabilities
from aosr.config.paths import config_path
from aosr.geometry.shoebox import Point, Room, Wall
from aosr.physics import report_facts, report_io
from aosr.physics.report_io import (
    BandRow,
    CapabilitySection,
    PointRow,
    ReportInput,
    ReportOutput,
    TopFields,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA_DIR = _REPO_ROOT / "blueprint" / "schemas"
# 重匯入口住在既有的命令列那一支（物理層的 report_io 只寫檔、不對人說話）。
_SCHEMA_CLI = "aosr.physics.three_lane_report_cli"
_REGENERATE_FLAG = "--regenerate-schemas"
_TABLE_PATH = config_path("capabilities.toml")
_WALL_NAMES = tuple(wall.wall_name() for wall in Wall.all())


def _table() -> CapabilityTable:
    """產品能力表；輸入驗證要收的那一張（跟命令列 ``--capabilities`` 同一個檔）。"""
    return load_capabilities(_TABLE_PATH)


def _input_document(**overrides: object) -> dict[str, object]:
    """一份合法的輸入；``overrides`` 換掉頂層某一格，壞輸入題目就從這裡長出來。"""
    rho_c_pa_s_per_m = 1.2 * 343.0
    document: dict[str, object] = {
        "room_m": {"Lx": 6.0, "Ly": 4.0, "Lz": 3.0},
        "source_m": {"x": 1.2, "y": 1.3, "z": 1.1},
        "receiver_m": {"x": 4.7, "y": 2.8, "z": 1.4},
        "sound_speed_m_s": 343.0,
        "density_kg_m3": 1.2,
        "impedance_pa_s_per_m_by_wall": {
            wall: 4.0 * rho_c_pa_s_per_m for wall in _WALL_NAMES
        },
        "scattering_by_wall": {wall: 0.2 for wall in _WALL_NAMES},
    }
    document.update(overrides)
    return document


def _impedance_map(document: dict[str, object]) -> dict[str, object]:
    """題目文件裡那一格阻抗表；型別在題目這一側是刻意寬的（壞輸入要放得進去）。"""
    by_wall = document["impedance_pa_s_per_m_by_wall"]
    assert isinstance(by_wall, dict)
    return {str(key): value for key, value in by_wall.items()}


def _rejects(document: object, expected: str) -> str:
    """驗一份壞輸入要炸，而且訊息裡要有預期的字樣；回傳訊息給題目自己再咬。"""
    with pytest.raises(ValueError) as caught:
        report_io.load_input_document(document, _table())
    message = str(caught.value)
    assert expected in message, message
    return message


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
    """有限元素那一半換成假的（照既有 CLI 考卷），只驗接線與契約。

    這是兩支考卷共用的工具，所以住在這一支（工具的家）；``test_report_io_cli.py``
    從這裡 import 過去用。第五刀把它暫放在命令列那一支，兩支於是互相 import、這一支
    得在題目裡延遲 import 才不會撞到循環；搬回來之後那個環就斷了。
    """
    del room, source, receiver, wall_impedances
    assert density_kg_m3 == 1.2
    assert sound_speed_m_s == 343.0
    return tuple(0.000012345 + frequency * 1e-10 for frequency in frequencies_hz)


# ── ① schema 檔等於模型現算出來的結果 ────────────────────────────────────────────
@pytest.mark.parametrize(
    ("file_name", "computed"),
    (
        ("three_lane_report_input.schema.json", report_io.input_schema),
        ("three_lane_report_output.schema.json", report_io.output_schema),
    ),
)
def test_schema_files_match_the_models(
    file_name: str,
    computed: Callable[[], dict[str, object]],
) -> None:
    """schema 檔過期（模型改了沒重匯）本題必須紅。"""
    on_disk = json.loads((_SCHEMA_DIR / file_name).read_text(encoding="utf-8"))
    assert on_disk == computed(), (
        f"{file_name} 跟模型現算的不一樣：跑 `uv run python -m {_SCHEMA_CLI} "
        f"{_REGENERATE_FLAG} {_SCHEMA_DIR}` 重匯（見 report_io 檔頭）"
    )


def test_regenerating_schema_files_reproduces_them_byte_for_byte(
    tmp_path: Path,
) -> None:
    """重匯入口真的跑得起來，寫出來的檔與版控那一份逐位元組相同，而且跑兩次一樣。

    票 #316 第三刀非必修第 10 條：先前紅了只知道「過期了」，不知道怎麼重生。
    ``--regenerate-schemas <目錄>`` 就是那個入口；這一題跑真的命令列把檔寫進
    ``tmp_path``（不碰真樹），再跟版控那一份對位元組——不一樣就等於入口不能用。
    第七刀非必修第 9 條：``regenerate_schema_files`` 的說明宣稱「同一份模型跑兩次得到
    同一個檔」，先前沒有任何一題跑過第二次；這一題現在真的跑兩次、兩份互相對位元組。
    """
    from aosr.physics import three_lane_report_cli

    captured = io.StringIO()
    with redirect_stdout(captured):
        exit_code = three_lane_report_cli.main([_REGENERATE_FLAG, str(tmp_path)])
    assert exit_code == 0, captured.getvalue()
    before = {name: (_SCHEMA_DIR / name).read_bytes() for name in report_io.SCHEMA_FILES}
    for file_name in report_io.SCHEMA_FILES:
        assert (tmp_path / file_name).read_bytes() == before[file_name], file_name
        assert file_name in captured.getvalue()
    # 同一份模型再跑一次寫到另一個目錄，兩次的檔逐位元組相同（第七刀非必修 9）。
    again = tmp_path / "again"
    second = io.StringIO()
    with redirect_stdout(second):
        again_code = three_lane_report_cli.main([_REGENERATE_FLAG, str(again)])
    assert again_code == 0, second.getvalue()
    for file_name in report_io.SCHEMA_FILES:
        assert (again / file_name).read_bytes() == (tmp_path / file_name).read_bytes()
    # 模組自己算得出 ``blueprint/schemas``（跑的人從哪個 cwd 進來都一樣）。
    assert report_io._SCHEMA_DIR == _SCHEMA_DIR
    # 第四刀非必修第 4／6 條加第七刀必修 1：不帶目錄不准洗版控那兩份（argparse 當場擋），
    # 多給的位置參數與報表模式那三支旗標也都不准被靜靜吃掉——併成一個 parser 之後那幾支
    # 旗標在重匯模式底下也合法登記著，不手動擋就會被收下來然後忽略。
    for argv in (
        [_REGENERATE_FLAG],
        [_REGENERATE_FLAG, str(tmp_path), "多給的"],
        [_REGENERATE_FLAG, str(tmp_path), "--points"],
        [_REGENERATE_FLAG, str(tmp_path), "--format", "json"],
        [_REGENERATE_FLAG, str(tmp_path), "--format", "text"],
        [_REGENERATE_FLAG, str(tmp_path), "--capabilities", str(_TABLE_PATH)],
    ):
        with pytest.raises(SystemExit) as refused:
            three_lane_report_cli.main(argv)
        assert refused.value.code == 2, argv
    # 舊版那個「重匯模式自己的 parser」連同它的 ``directory`` 位置參數都沒了，所以反過來
    # 咬說明裡不准再出現 ``directory``；旗標本人則要在（頂層 `--help` 那一題另外咬全貌）。
    assert "directory" not in three_lane_report_cli._report_parser().format_help()
    assert _REGENERATE_FLAG in three_lane_report_cli._report_parser().format_help()
    for name, committed in before.items():
        assert (_SCHEMA_DIR / name).read_bytes() == committed, name


def test_regenerate_flag_is_visible_in_the_top_level_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """頂層 ``--help`` 要看得到 ``--regenerate-schemas``（第五刀非必修 3）。

    CLI 檔頭叫人「要看用法跑 ``--help``」，先前那支旗標卻只住在一段字串分流裡、報表
    那個 parser 沒登記它，於是照著跑的人找不到入口。這一題跑真的頂層 ``--help``
    （不帶任何位置參數），咬三件事：旗標自己在說明裡、模組裡只剩一個 parser 工廠、
    命令列那個常數的字面沒被改掉（第七刀必修 6：先前這兩句話考卷都沒咬）。
    """
    from aosr.physics import three_lane_report_cli

    with pytest.raises(SystemExit) as exited:
        three_lane_report_cli.main(["--help"])
    assert exited.value.code == 0
    top_help = capsys.readouterr().out
    assert _REGENERATE_FLAG in top_help
    # 「只有這一個 parser」真的咬：模組裡造 parser 的工廠只准有 ``_report_parser`` 一個，
    # 再長出第二個（像先前那個只管重匯模式的）就紅——頂層 ``--help`` 看不到的那一半
    # 正是從第二個 parser 長出來的。
    factories = sorted(
        name
        for name, value in vars(three_lane_report_cli).items()
        if callable(value) and name.endswith("_parser")
    )
    assert factories == ["_report_parser"]
    # 這一行咬的是「那個字面沒被改掉」：考卷這一份（``:67``）是刻意的第二份，改了名字
    # 兩邊都得動。它**不**證明全樹只有一個來源——散文裡還抄著好幾份，那幾份沒有機器在守
    # （命令列那一支 ``_REGENERATE_FLAG`` 上面的註解把那幾處列了出來）。
    assert three_lane_report_cli._REGENERATE_FLAG == _REGENERATE_FLAG


def test_capability_frequency_range_is_empty_or_exactly_two() -> None:
    """``capability.frequency_hz`` 只有兩種合法形狀：空，或剛好兩個端點（第五刀必修 1）。

    ``tuple[float, ...]`` 那一版把長度放寬成 0 到無限大，schema 掉了
    ``minItems``／``maxItems``／``prefixItems``，可是同一格的說明與 reference 還寫
    「兩個端點」——話說 0 或 2、機器守 0 到無限大。這一題兩邊都咬：模型層三個端點要炸、
    schema 層要自己把那件事說出來。
    """
    section = report_io.CapabilitySection(
        frequency_hz=(20.0, 20000.0), outputs=(), status="unchecked", evidence=()
    )
    assert section.frequency_hz == (20.0, 20000.0)
    assert (
        report_io.CapabilitySection(
            frequency_hz=(), outputs=(), status="unchecked", evidence=()
        ).frequency_hz
        == ()
    )
    for wrong in ((500.0,), (500.0, 1000.0, 2000.0), (500.0, 1000.0, 2000.0, 4000.0)):
        with pytest.raises(Exception) as caught:
            report_io.CapabilitySection(
                frequency_hz=wrong,  # type: ignore[arg-type]  # expires=2026-12-08 reason=這一題就是要餵壞形狀進去看它炸
                outputs=(),
                status="unchecked",
                evidence=(),
            )
        assert "frequency_hz" in str(caught.value)
    # schema 自己說出「空的，或剛好兩個」——不是靠上游另一個檔的型別擋。
    cell = _field_definitions("CapabilitySection")["frequency_hz"]
    branches = cell["anyOf"]
    assert isinstance(branches, list)
    by_length = {
        (branch["minItems"], branch["maxItems"]): branch
        for branch in branches
        if isinstance(branch, dict)
    }
    assert set(by_length) == {(0, 0), (2, 2)}
    # 「兩個端點」那一支要自己說出端點是兩個數字，不是靠 items 的鬆散寫法。
    assert by_length[(2, 2)]["prefixItems"] == [{"type": "number"}, {"type": "number"}]
    # 空那一支不准帶 items／prefixItems——空的陣列就是空的，不是「0 個數字」以外的東西。
    assert "items" not in by_length[(0, 0)] and "prefixItems" not in by_length[(0, 0)]
    description = cell["description"]
    assert isinstance(description, str) and "兩個端點" in description


def test_capability_frequency_range_reference_speaks_the_no_basis_family() -> None:
    """``frequency_hz`` 的 reference 跟其他「不是量測」的欄同一族寫法（第五刀非必修 8）。

    其他非量測欄寫「沒有基準（只是…）」；這一格先前自成一套寫「報告頻率軸上的兩個端點」，
    同一個病兩種說法。咬住它用同一族的前綴，而且不准再自稱量測值。
    """
    table = report_io.quantity_table()
    reference = table["frequency_hz"].reference
    assert reference.startswith("沒有基準（只是")
    assert "不是量測值" in reference
    # 同一族的其他格一個都不准被順手改掉。
    for name in ("outputs", "status", "evidence"):
        assert table[name].reference.startswith("沒有基準（只是"), name


@pytest.mark.parametrize(
    ("file_name", "computed"),
    (
        ("three_lane_report_input.schema.json", report_io.input_schema),
        ("three_lane_report_output.schema.json", report_io.output_schema),
    ),
)
def test_schema_files_declare_the_draft(
    file_name: str,
    computed: Callable[[], dict[str, object]],
) -> None:
    """兩份 schema 檔都要宣告 draft 2020-12（``$ref`` 旁邊那四件事的兄弟鍵只有它認得）。"""
    on_disk = json.loads((_SCHEMA_DIR / file_name).read_text(encoding="utf-8"))
    assert on_disk["$schema"] == report_io.JSON_SCHEMA_DRAFT
    assert computed()["$schema"] == "https://json-schema.org/draft/2020-12/schema"


def _row_definition(model_name: str) -> dict[str, object]:
    """輸出 schema 裡某個列模型的定義本人（``$defs`` 那一格）。"""
    definitions = report_io.output_schema()["$defs"]
    assert isinstance(definitions, dict)
    definition = definitions[model_name]
    assert isinstance(definition, dict)
    return definition


def _field_definitions(model_name: str) -> dict[str, dict[str, object]]:
    """輸出 schema 裡某個列模型的欄位定義（``properties``；有欄名的物件）。"""
    properties = _row_definition(model_name)["properties"]
    assert isinstance(properties, dict)
    return {str(name): cell for name, cell in properties.items() if isinstance(cell, dict)}


@pytest.mark.parametrize("model_name", ("BandRow", "PointRow", "TopFields"))
def test_output_rows_are_named_objects_not_arrays(model_name: str) -> None:
    """三張列是**有欄名的物件**：加一欄不會靜靜改掉既有消費者讀到的意思。"""
    definition = _row_definition(model_name)
    assert definition["type"] == "object" and "prefixItems" not in definition
    fields = {
        "TopFields": set(TopFields.model_fields),
        "BandRow": set(BandRow.model_fields),
        "PointRow": set(PointRow.model_fields),
    }[model_name]
    assert set(_field_definitions(model_name)) == fields


def test_json_payload_carries_field_names() -> None:
    """``--format json`` 的形狀（``model_dump``）是有欄名的物件，不是位置陣列。"""
    payload = json.loads(_output().model_dump_json())
    assert isinstance(payload["bands"], list)
    first = payload["bands"][0]
    assert isinstance(first, dict)
    assert "geometric_energy" in first
    assert isinstance(payload["top"], dict)
    assert "room_volume_m3" in payload["top"]


def test_schema_files_carry_the_four_facts_per_field() -> None:
    """每一欄的 schema 都要帶物理量、單位、參考基準、有效狀態四格。"""
    band_row = _field_definitions("BandRow")
    assert [cell["unit"] for cell in band_row.values()] == [
        "Hz", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1",
        "Hz", "1", "s", "1", "s", "1",
    ]
    for cell in band_row.values():
        assert cell["quantity"]
        assert cell["reference"]
        assert cell["validity"]


def test_reference_column_carries_a_basis_not_a_unit() -> None:
    """「參考基準」那一欄不准塞單位：有量綱的欄要寫相對於什麼或絕對值。

    這一題是票 #316 第二刀必修第 2 條咬住的那件事——``unit`` 已經有一格了，
    ``reference`` 再寫一次「無因次」等於同一句話說兩遍，而且對有量綱的欄是錯的。
    """
    table = report_io.quantity_table()
    for facts in table.values():
        assert facts.reference != "無因次", facts
    for name in (
        "sound_speed_m_s",
        "density_kg_m3",
        "impedance_pa_s_per_m_by_wall",
        "top.room_volume_m3",
    ):
        assert table[name].reference != table[name].unit, name
        assert "絕對值" in table[name].reference, name
    for name in (
        "outputs",
        "status",
        "evidence",
        "top.capped_by_upper_limit",
        "top.schroeder_band_count",
        "bands.t20_unavailable_reason",
        "bands.t30_unavailable_reason",
        "bands.fem_point_count",
    ):
        assert "沒有基準" in table[name].reference, name
    # ``capability`` 那一格是能力表整條記錄，不是欄名清單（票 #316 第三刀非必修第 6 條）。
    assert "不是量測值" in table["capability"].reference
    assert "欄名清單" not in table["capability"].reference


def test_validity_column_does_not_call_a_string_estimable() -> None:
    """``validity`` 那一欄不准把不是量測的格子寫成「可估」（第三刀非必修第 7 條）。

    同一個病只治了 ``reference`` 那一半：``status`` 是表上那一條、``evidence`` 是收據、
    兩個原因欄是文字、``capped_by_upper_limit``／計數是旗標與整數，``capability``／
    ``top``／``bands``／``points`` 只是容器，``frequency_hz`` 是表上宣告的範圍（第四刀
    必修第 2 條）。它們沒有一個是估出來的數字。
    """
    table = report_io.quantity_table()
    for name in (
        "outputs",
        "status",
        "evidence",
        "capability",
        "top",
        "bands",
        "points",
        "top.capped_by_upper_limit",
        "bands.capped_by_upper_limit",
        "top.schroeder_band_count",
        "bands.fem_point_count",
        "bands.t20_unavailable_reason",
        "bands.t30_unavailable_reason",
        "frequency_hz",
    ):
        assert table[name].validity == report_facts.NOT_MEASURED, name
    # 反過來：真的量到的那些欄位不准被掃進去（這不是「全部改掉」就過）。
    for name in ("bands.direct_energy", "bands.t20_s", "top.room_volume_m3", "points.frequency_hz"):
        assert table[name].validity != report_facts.NOT_MEASURED, name
    # 「不是數字」那半句對兩個計數欄是錯的（第四刀必修第 1 條）：計數的 quantity 就是「計數」，
    # 措辭要涵蓋計數與表上宣告的值。
    counted = table["top.schroeder_band_count"].validity
    assert counted == report_facts.NOT_MEASURED
    assert "計數" in counted and "宣告" in counted and "不是數字" not in counted


def test_facts_default_validity_is_the_estimable_constant() -> None:
    """``_facts`` 的預設值就是 ``_ESTIMABLE``，而且那一格真的會落到欄位上。

    先前 ``_facts`` 的說明宣稱「兩處同一個字串，由考卷咬住」而那一題不存在。這一題讀
    簽章（不是讀字面），所以預設值被改掉、或常數被改掉，兩邊對不上就紅。
    """
    import inspect

    default = inspect.signature(report_facts.facts).parameters["validity"].default
    assert default == report_facts.ESTIMABLE
    # 那一格真的會落到欄位上，不是只有簽章好看。
    assert report_io.quantity_table()["bands.direct_energy"].validity == default


def test_summed_columns_say_alignment_is_unverified_not_a_different_yardstick() -> None:
    """相加而來的那幾欄要寫實話：基準名義上共用，沒對過的是跨方法的數值對齊。

    ``src/aosr/config/source_reference.py`` 的檔頭寫著整包共用單位振幅點源基準，晚期那一路的
    4π 正是換算到那個基準的因子。所以這幾欄不准寫成「兩路基準不是同一把尺」——那跟
    專案自己的聲源定義矛盾；要寫的是「對齊未驗」。
    """
    table = report_io.quantity_table()
    for name in (
        "bands.geometric_energy",
        "bands.geometric_contribution",
        "bands.total_energy",
        "points.geometric_energy",
        "points.total_energy",
    ):
        reference = table[name].reference
        assert "相加" in reference, name
        assert "對齊" in reference and "沒有對過" in reference, name
        assert "不是同一把尺" not in reference, name
    for name in ("bands.geometric_energy", "bands.geometric_contribution"):
        assert "4π" in table[name].reference, name
    for name in ("bands.total_energy", "points.total_energy"):
        assert "絕對聲壓級" in table[name].reference, name


def test_quantity_table_covers_every_declared_field() -> None:
    """對照表要蓋到輸入、輸出與兩種列的每一欄，一格都不准漏。"""
    table = report_io.quantity_table()
    expected = (
        set(ReportInput.model_fields)
        | set(CapabilitySection.model_fields)
        | set(ReportOutput.model_fields)
        | {f"bands.{name}" for name in BandRow.model_fields}
        | {f"points.{name}" for name in PointRow.model_fields}
        | {f"top.{name}" for name in TopFields.model_fields}
    )
    assert set(table) == expected
    facts = table["bands.t20_s"]
    assert facts.quantity == "時間" and facts.unit == "s"
    assert facts.validity.startswith("可估（空＝值算不出來")
    assert "視窗" in facts.reference


def test_quantity_table_separates_estimated_from_unestimated() -> None:
    """兩族「空」在對照表上要看得出差別，不是跟可估的混在一起。"""
    table = report_io.quantity_table()
    assert table["bands.fem_energy"].validity.startswith("這一格可能是空的")
    assert table["points.fem_energy"].validity.startswith("這一格可能是空的")
    for name in ("bands.t20_s", "bands.t30_s"):
        assert table[name].validity.startswith("可估（空＝值算不出來")
    # 說明與 validity 同一句話：那一欄的空值該不該帶原因寫在同一格裡。
    assert "必須帶原因" in table["bands.t20_s"].validity
    assert "有值就不准再給原因" in table["bands.t20_s"].validity
    assert "不必帶原因" in table["bands.fem_energy"].validity
    # 第一句不准自打嘴巴（第三刀非必修第 5 條）。
    assert not table["bands.fem_energy"].validity.startswith("不可估")
    assert "不是值算不出來" not in table["bands.fem_energy"].validity


def test_schema_facts_match_the_quantity_table() -> None:
    """schema 檔裡的四件事與對照表逐格相同——同一件事不准有兩個版本。"""
    band_row = _field_definitions("BandRow")
    table = report_io.quantity_table()
    for name in BandRow.model_fields:
        for key in ("quantity", "unit", "reference", "validity"):
            assert band_row[name][key] == getattr(table[f"bands.{name}"], key), name


def test_late_energy_reference_names_the_eyring_ratio() -> None:
    """``_LATE`` 要說實話：欄上是 raw × ``_eyring_ratio(alpha_bar)``，不是 raw 本人。

    票 #316 第三刀必修第 1 條。回原始碼對：``late_energy.py`` 的 ``_exact_raw_energy``
    只到 ``4π × 4 × mean_reflected``；``:438`` 才乘 ``_eyring_ratio`` 得到
    ``late_reverberant_energy``，``geometric_lane.py:389`` 拿的是那一格。這一題咬住
    「比值有被提到、而且欄上不是那個本人」，光寫「4π」不算過。
    """
    reference = report_io.quantity_table()["bands.late_energy"].reference
    assert "_eyring_ratio" in reference
    assert "不是 raw_energy 本人" in reference
    assert "4π" in reference
    # 舊的兩句錯話都不准留：多乘一次 4π、以及「就是這個值本身」。
    assert "raw_energy × 4π" not in reference
    assert "回傳的 raw_energy 本人" not in reference


def test_late_energy_reference_matches_the_source_of_truth() -> None:
    """咬住那個比值真的住在他的那份來源裡：欄上的字樣回原始碼對得上。

    字面比對會被「換一句很像真的話」繞過去，所以這一題自己去讀 ``late_energy.py``：
    ``_exact_raw_energy`` 裡的乘積、以及 ``late_reverberant_energy=raw * ratio`` 那一行
    都要在。原始碼改了這一句說明就得跟著改。
    """
    source = (
        Path(report_io.__file__).resolve().parents[1] / "physics" / "late_energy.py"
    ).read_text(encoding="utf-8")
    assert "DIFFUSE_MONOPOLE_4PI * 4.0 * mean_reflected" in source
    assert "late_reverberant_energy=raw * ratio" in source
    assert "_eyring_ratio(alpha_bar)" in source


def test_reference_strings_take_band_numbers_from_product_config() -> None:
    """reference 字串裡的頻帶數字不准從產品設定抄一份進來（會漂進 schema 檔）。"""
    from aosr.config.three_lane_crossover import SCHROEDER_T60_BANDS_HZ

    reference = report_io.quantity_table()["top.f_s_hz"].reference
    for frequency in SCHROEDER_T60_BANDS_HZ:
        assert f"{frequency:g}" not in reference
    assert "SCHROEDER_T60_BANDS_HZ" in reference
    for file_name in (
        "three_lane_report_input.schema.json",
        "three_lane_report_output.schema.json",
    ):
        text = (_SCHEMA_DIR / file_name).read_text(encoding="utf-8")
        for frequency in SCHROEDER_T60_BANDS_HZ:
            assert f"{frequency:g}" not in text, (file_name, frequency)


# ── ② 輸入模型每一條驗證規則各一題 ──────────────────────────────────────────────
def test_accepts_a_complete_document() -> None:
    """合法輸入要收得下，而且收回來的是求解層吃的形狀。"""
    inputs = report_io.load_input_document(_input_document(), _table())
    assert inputs.room_m == Room(6.0, 4.0, 3.0)
    assert inputs.source_m == Point(1.2, 1.3, 1.1)
    solved = report_io.solver_inputs(inputs)
    assert set(solved.impedance_by_wall) == set(Wall.all())
    assert set(solved.scattering_by_wall or {}) == set(Wall.all())


def test_scattering_may_be_omitted_entirely() -> None:
    """散射整格可省略，省略時是 None（由幾何路套既有預設值）。"""
    document = _input_document()
    del document["scattering_by_wall"]
    assert report_io.load_input_document(document, _table()).scattering_by_wall is None


def test_scattering_may_be_zero_on_every_wall() -> None:
    """散射 0 是合法輸入，不准被「正實數」那條擋掉。"""
    inputs = report_io.load_input_document(
        _input_document(scattering_by_wall={wall: 0.0 for wall in _WALL_NAMES}),
        _table(),
    )
    assert inputs.scattering_by_wall == {wall: 0.0 for wall in _WALL_NAMES}


def test_missing_room_component_is_named() -> None:
    """房少一軸要指名哪一軸（Pydantic 的欄位路徑帶出來，不是只說「輸入錯」）。"""
    _rejects(
        _input_document(room_m={"Lx": 6.0, "Ly": 4.0}),
        "Lz",
    )


def test_room_must_be_a_json_object() -> None:
    """房不是物件就擋，訊息跟原本一樣。"""
    _rejects(_input_document(room_m=[6.0, 4.0, 3.0]), "room_m 必須是 JSON 物件")


def test_point_must_be_a_json_object() -> None:
    """聲源不是物件就擋，訊息指名那一格。"""
    _rejects(_input_document(source_m="1.2,1.3,1.1"), "source_m 必須是 JSON 物件")


def test_point_component_must_be_a_finite_number() -> None:
    """點的座標要有限數字；布林不算數字。"""
    _rejects(
        _input_document(receiver_m={"x": True, "y": 2.8, "z": 1.4}),
        "必須是有限數字",
    )


@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")), ids=str)
def test_non_finite_numbers_are_rejected(value: float) -> None:
    """非有限值一律拒收（變異考卷：三個方向都要炸）。"""
    _rejects(_input_document(sound_speed_m_s=value), "必須是有限數字")


def test_non_positive_sound_speed_is_rejected() -> None:
    """聲速必須是有限正數。"""
    _rejects(_input_document(sound_speed_m_s=0.0), "sound_speed_m_s 必須是有限正數")


def test_non_positive_density_is_rejected() -> None:
    """密度必須是有限正數。"""
    _rejects(_input_document(density_kg_m3=-1.2), "density_kg_m3 必須是有限正數")


def test_non_positive_room_axis_is_rejected() -> None:
    """房三軸長都必須是有限正數。"""
    _rejects(
        _input_document(room_m={"Lx": 6.0, "Ly": 4.0, "Lz": -3.0}),
        "必須是有限正數",
    )


def test_missing_wall_impedance_is_rejected() -> None:
    """六個牆名固定；缺一面要指名那一面。"""
    by_wall = _impedance_map(_input_document())
    del by_wall["yL"]
    _rejects(_input_document(impedance_pa_s_per_m_by_wall=by_wall), "yL")


def test_negative_impedance_is_rejected_with_the_unsupported_hint() -> None:
    """阻抗要正；訊息照原本帶上能力表那兩條 unsupported 的 note。"""
    by_wall = _impedance_map(_input_document())
    by_wall["floor"] = -5.0
    message = _rejects(
        _input_document(impedance_pa_s_per_m_by_wall=by_wall),
        "必須是正實數阻抗",
    )
    assert "unsupported" in message
    assert "#309" in message


@pytest.mark.parametrize(
    "cell",
    (
        {"real": 1646.4, "imag": -100.0},
        [1646.4, 1646.5, 1646.6],
        {"125.0": 1646.4, "250.0": 1646.5},
    ),
    ids=("complex_object", "frequency_list", "frequency_object"),
)
def test_complex_or_frequency_dependent_impedance_is_rejected(cell: object) -> None:
    """複數或逐頻阻抗（物件或陣列）當場拒收，訊息提到票 #309 與 unsupported。"""
    by_wall = _impedance_map(_input_document())
    by_wall["floor"] = cell
    message = _rejects(
        _input_document(impedance_pa_s_per_m_by_wall=by_wall),
        "unsupported",
    )
    assert "#309" in message


@pytest.mark.parametrize("value", (-0.01, 1.01, 5.0), ids=str)
def test_scattering_outside_unit_interval_is_rejected(value: float) -> None:
    """散射係數落在 [0,1] 之外就擋。"""
    _rejects(
        _input_document(scattering_by_wall={wall: value for wall in _WALL_NAMES}),
        "[0,1]",
    )


def test_missing_scattering_wall_is_rejected() -> None:
    """散射六面固定；物件裡缺一面就擋並指名那一面。"""
    scattering = {wall: 0.2 for wall in _WALL_NAMES if wall != "ceiling"}
    _rejects(_input_document(scattering_by_wall=scattering), "ceiling")


def test_unknown_extra_field_is_rejected() -> None:
    """多一個沒登記的欄位就擋（extra=forbid）。"""
    _rejects(_input_document(materials="wood"), "materials")


def test_frozen_input_cannot_be_mutated() -> None:
    """輸入模型是凍結的：改一格要炸，不准出現第二種真相。"""
    inputs = report_io.load_input_document(_input_document(), _table())
    with pytest.raises(Exception):
        setattr(inputs, "density_kg_m3", 1.0)


def test_missing_capability_table_is_a_wiring_error_not_bad_input() -> None:
    """沒帶能力表是**接線錯誤**：不准被收成「輸入不合 ReportInput」。

    票 #316 第三刀非必修第 11 條。繞過命令列直接餵一份不合的材料形式（``_table_of``
    才會被叫到），此時驗證脈絡裡沒有表；先前那條路丟 ``ValueError``，被
    ``load_input_document`` 包成「輸入不合 ReportInput：…」——看訊息的人會去改 JSON，
    而真正該改的是呼叫端。現在它照原樣以 :class:`~aosr.physics.report_io.WiringError`
    往上冒。
    """
    document = _input_document()
    by_wall = _impedance_map(document)
    by_wall["floor"] = {"real": 1646.4, "imag": -100.0}
    document["impedance_pa_s_per_m_by_wall"] = by_wall

    with pytest.raises(report_io.WiringError) as caught:
        ReportInput.model_validate(document, context={})

    assert "能力表" in str(caught.value)
    assert "輸入不合 ReportInput" not in str(caught.value)


# ── ③ 輸出模型：值空必須有原因（兩個方向）────────────────────────────────────────
def _band(**overrides: object) -> BandRow:
    defaults: dict[str, object] = {
        "center_frequency_hz": 1000.0,
        "fem_energy": 1.0,
        "fem_point_count": 1,
        "direct_energy": 1.0,
        "reflected_energy": 1.0,
        "interference_energy": 0.0,
        "late_energy": 1.0,
        "geometric_energy": 1.0,
        "fem_contribution": 1.0,
        "geometric_contribution": 1.0,
        "total_energy": 1.0,
        "w_fem": 0.5,
        "w_geo": 0.5,
        "f_s_hz": 200.0,
        "capped_by_upper_limit": False,
        "t20_s": 1.0,
        "t20_unavailable_reason": None,
        "t30_s": 1.0,
        "t30_unavailable_reason": None,
    }
    defaults.update(overrides)
    return BandRow(**defaults)  # type: ignore[arg-type]  # expires=2026-12-08 reason=這一支刻意用寬鬆的 kwargs 組列，欄名由上面的 defaults 決定


def _point(**overrides: object) -> PointRow:
    defaults: dict[str, object] = {
        "frequency_hz": 1000.0,
        "fem_energy": 1.0,
        "direct_energy": 1.0,
        "reflected_energy": 1.0,
        "interference_energy": 0.0,
        "late_energy": 1.0,
        "scattering": 0.2,
        "geometric_energy": 1.0,
        "w_fem": 0.5,
        "w_geo": 0.5,
        "total_energy": 1.0,
    }
    defaults.update(overrides)
    return PointRow(**defaults)  # type: ignore[arg-type]  # expires=2026-12-08 reason=這一支刻意用寬鬆的 kwargs 組列，欄名由上面的 defaults 決定


def _output(**overrides: object) -> ReportOutput:
    defaults: dict[str, object] = {
        "capability": CapabilitySection(
            frequency_hz=(20.0, 5583.0),
            outputs=("total_energy",),
            status="experimental",
            evidence=(),
        ),
        "top": TopFields(
            f_s_hz=200.0,
            crossover_lower_hz=200.0,
            crossover_upper_hz=300.0,
            capped_by_upper_limit=False,
            eyring_t60_by_band_s={"500.0": 1.0},
            room_volume_m3=72.0,
            schroeder_band_count=2,
        ),
        "bands": (_band(),),
        "points": None,
    }
    defaults.update(overrides)
    return ReportOutput(**defaults)  # type: ignore[arg-type]  # expires=2026-12-08 reason=同上


def test_empty_value_without_a_reason_is_rejected() -> None:
    """方向一：值空而沒有原因要報錯。"""
    with pytest.raises(Exception) as caught:
        _output(bands=(_band(t20_s=None, t20_unavailable_reason=None),))
    assert "必須帶不可估原因" in str(caught.value)


def test_value_with_a_reason_is_rejected() -> None:
    """方向二：值有而又給原因也要報錯。"""
    with pytest.raises(Exception) as caught:
        _output(bands=(_band(t20_s=1.0, t20_unavailable_reason="擬合無效"),))
    assert "不准再給不可估原因" in str(caught.value)


def test_empty_fem_energy_needs_no_reason() -> None:
    """另一族：``fem_energy`` 空＝這一帶沒有有限元素頻點，不必帶原因。

    票 #316 第二刀監督拍板第 7 條：它跟 T20 的「算不出來」不同族，所以空的
    ``fem_energy``（``fem_point_count`` 是 0）要收得下，不准被那條規則擋掉。
    """
    output = _output(
        bands=(
            _band(fem_energy=None, fem_point_count=0),
            _band(center_frequency_hz=2000.0),
        )
    )
    assert output.bands[0].fem_energy is None
    assert output.bands[0].fem_point_count == 0
    assert output.bands[0].t20_s is not None


def test_empty_decay_value_in_the_same_row_still_needs_a_reason() -> None:
    """同一列上兩族分得開：``fem_energy`` 空不必原因，T20 空一樣要原因。"""
    with pytest.raises(Exception) as caught:
        _output(
            bands=(
                _band(fem_energy=None, fem_point_count=0, t20_s=None),
                _band(center_frequency_hz=2000.0),
            )
        )
    message = str(caught.value)
    assert "t20_s 是空的就必須帶不可估原因" in message
    assert "fem_energy" not in message


def test_empty_point_fem_energy_needs_no_reason() -> None:
    """逐點表那一欄同一族：空的 ``fem_energy`` 是「這一格沒有值」，不必帶原因。"""
    output = _output(points=(_point(fem_energy=None),))
    assert output.points is not None
    assert output.points[0].fem_energy is None
    assert output.points[0].total_energy == 1.0


def test_empty_value_with_a_reason_is_accepted() -> None:
    """合法的形狀：值空、原因在，其他結果照留。"""
    output = _output(
        bands=(
            _band(
                t20_s=None,
                t20_unavailable_reason="T20 擬合無效：第 256 階最低 -20 dB",
                t30_s=None,
                t30_unavailable_reason="T30 擬合無效：第 256 階最低 -20 dB",
            ),
        )
    )
    assert output.bands[0].total_energy == 1.0
    assert output.bands[0].t20_s is None
    assert output.bands[0].t20_unavailable_reason is not None


def test_unchecked_capability_does_not_invent_a_column_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """沒查表時 ``outputs`` 與 ``frequency_hz`` 都是空的，不准造假的欄名與頻率範圍。

    ``capability_report.capability_line`` 那一層印的是 ``outputs=none`` 給人看的一行；
    契約這一層是資料，空就是空——造一個字串欄名會讓前端以為真有那一欄，造一個
    ``(0.0, 0.0)``（第四刀非必修第 8 條）會讓前端以為宣告了 0 到 0 Hz。
    """
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
    section = report_io.output_from_report(
        report, room=Room(6.0, 4.0, 3.0), with_points=False
    ).capability
    assert section.outputs == ()
    assert section.frequency_hz == ()
    assert section.evidence == ()
    assert section.status == "unchecked"


def test_empty_bands_are_rejected() -> None:
    """報表一定有頻帶列；空的要炸。"""
    with pytest.raises(Exception) as caught:
        _output(bands=())
    assert "不可為空" in str(caught.value)


def test_out_of_order_band_centers_are_rejected() -> None:
    """頻帶列的中心頻率必須遞增。"""
    with pytest.raises(Exception) as caught:
        _output(
            bands=(
                _band(center_frequency_hz=1000.0),
                _band(center_frequency_hz=500.0),
            )
        )
    assert "遞增" in str(caught.value)


def test_descending_crossover_range_is_rejected() -> None:
    """交接下端不准大於上端。"""
    with pytest.raises(Exception):
        _output(
            top=TopFields(
                f_s_hz=400.0,
                crossover_lower_hz=400.0,
                crossover_upper_hz=300.0,
                capped_by_upper_limit=False,
                eyring_t60_by_band_s={"500.0": 1.0},
                room_volume_m3=72.0,
                schroeder_band_count=2,
            )
        )


def test_frozen_output_cannot_be_mutated() -> None:
    """輸出模型是凍結的：改一格要炸。"""
    output = _output()
    with pytest.raises(Exception):
        setattr(output, "bands", ())


