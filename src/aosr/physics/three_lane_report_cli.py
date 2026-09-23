"""三路接合物理量報表的人看命令列輸出層。

幾何能量含干涉項（票 #302），不是上一代定義。幾何路與晚期混響按反射階數分工
（票 #337）：反射與干涉兩欄已經含散射留存，晚期那一欄是晚期混響交給幾何路的那一份，
四欄相加等於幾何能量；當次用的交接階數 K 印在頂層 ``reflection_order_k`` 那一行。

輸入 JSON 格式如下；六個牆名固定是 ``floor``、``ceiling``、``x0``、``xL``、
``y0``、``yL``。``scattering_by_wall`` 整格可省略，省略時由幾何路套既有預設值。
``reflection_order_k``（交接階數 K）也可省略，省略時用產品設定 ``REFLECTION_ORDER_K``；
**沒有對應的命令列旗標**——輸入檔是宣告過的邊界，開第二道門就會變成兩份來源（票 #341）。
這份輸入由 :mod:`aosr.physics.report_io` 的輸入模型驗（票 #316）；欄位形狀與
每一欄的物理量／單位／參考基準／有效狀態匯出在 ``blueprint/schemas/``。

.. code-block:: json

   {
     "room_m": {"Lx": 6.0, "Ly": 4.0, "Lz": 3.0},
     "source_m": {"x": 1.2, "y": 1.3, "z": 1.1},
     "receiver_m": {"x": 4.7, "y": 2.8, "z": 1.4},
     "sound_speed_m_s": 343.0,
     "density_kg_m3": 1.2,
     "impedance_pa_s_per_m_by_wall": {
       "floor": 1646.4, "ceiling": 1646.4,
       "x0": 1646.4, "xL": 1646.4, "y0": 1646.4, "yL": 1646.4
     },
     "scattering_by_wall": {
       "floor": 0.1, "ceiling": 0.1,
       "x0": 0.1, "xL": 0.1, "y0": 0.1, "yL": 0.1
     }
   }

執行 ``uv run python -m aosr.physics.three_lane_report_cli input.json``；加
``--points`` 會在頂層與六頻帶表後再印完整細軸逐點表。加 ``--format json``
改印輸出契約的 JSON（印之前用模型自己反解一次，證明它真的合那份契約）。

``--regenerate-schemas <目錄>`` 是另一種模式：把該目錄底下那兩份匯出檔重寫成
:mod:`aosr.physics.report_io` 模型現算的內容（目錄必給，覆寫版控那兩份就寫
``blueprint/schemas``；要看用法跑 ``--help``），考卷
``test_report_io_contract.py::test_schema_files_match_the_models`` 紅掉時跑這一個。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from aosr.config.capabilities import (
    CapabilityTable,
    capability_for,
    load_capabilities,
)
from aosr.geometry.shoebox import Wall
from aosr.physics import capability_report, report_io, report_output
from aosr.physics.report_io import ReportOutput
from aosr.physics.three_lane_report import (
    ReportCapability,
    ThreeLaneReport,
    solve_three_lane_report,
)


# 這一節在能力表上的名字，以及這次輸入的材料形式：六面各一個與頻率無關的實數阻抗。
_MATERIALS = "real_frequency_independent_impedance"
_ROOM = "shoebox"
_ENTRY = "three_lane_report"
# 重匯 schema 那一支旗標。**程式裡**只有這一份字面：底下 parser 登記它用的就是這一格，
# 分流讀的是 argparse 收出來的 ``args.regenerate_schemas``（屬性名，不碰字面）。散文另外
# 抄了好幾份（這個檔的檔頭、``report_io`` 的檔頭與 ``regenerate_schema_files``、兩支考卷
# 的檔頭），那幾份沒有機器在守；考卷那一份是刻意的第二份，由
# ``test_regenerate_flag_is_visible_in_the_top_level_help`` 咬住它跟這一格相等。
_REGENERATE_FLAG = "--regenerate-schemas"

# 收到複數或逐頻阻抗時，拒收訊息要提的兩條組合。名單、訊息與「從哪一張表讀」全部住在
# report_io：命令列只把 --capabilities 那一張表傳進去，不在這裡拼第二份。


def _capability_section(capability: ReportCapability) -> str:
    """把這份報表落在哪一條能力組合、什麼狀態、憑什麼、範圍與欄位，印成一行人話。"""
    return capability_report.capability_line(
        capability.entry,
        capability.room,
        capability.materials,
        capability.record,
    )


def _capability_for(table: CapabilityTable) -> ReportCapability:
    """從能力表查這條組合本人；查不到或標 unsupported 就報錯。

    拿的是整條組合（狀態、收據、頻率範圍、輸出欄），不是只有狀態字串——
    印出來的那一行要能讓人看出 validated 蓋到哪裡為止。
    """
    record = capability_for(table, _ENTRY, room=_ROOM, materials=_MATERIALS)
    if record.status == "unsupported":
        hint = report_io.unsupported_materials_hint(table)
        raise ValueError(f"{_ENTRY} × {_ROOM} × {_MATERIALS}：{hint}")
    return ReportCapability(
        entry=_ENTRY,
        room=_ROOM,
        materials=_MATERIALS,
        record=record,
    )


def _value(value: float | None) -> str:
    return "none" if value is None else f"{value:.17g}"


def _top_table(report: ThreeLaneReport) -> str:
    rows = (
        ("f_s_hz", _value(report.f_s_hz)),
        ("low_frequency_axis", report.low_frequency_axis.value),
        ("crossover_lower_hz", _value(report.crossover_lower_hz)),
        ("crossover_upper_hz", _value(report.crossover_upper_hz)),
        ("capped_by_upper_limit", str(report.capped_by_upper_limit).lower()),
        ("reflection_order_k", str(report.reflection_order_k)),
        ("eyring_t60_500_hz_s", _value(report.eyring_t60_by_band_s[500.0])),
        ("eyring_t60_1000_hz_s", _value(report.eyring_t60_by_band_s[1000.0])),
    )
    lines = [f"{name} {value}" for name, value in rows]
    if report.capped_by_upper_limit:
        lines.append("被上限硬切（f_s ≥ 300 Hz，在 300 Hz 硬切換）")
    return "\n".join(lines)


def _decay_value(value: float | None, reason: str | None) -> str:
    if reason is not None:
        return f"算不出：{reason}"
    return _value(value)


def _band_table(report: ThreeLaneReport) -> str:
    sampling_note = (
        "頻帶取樣：直達／反射／干涉／s 欄為 0.5 Hz 密頻率點平均；"
        "晚期／權重欄為當次報表軸每八度等權平均，T20／T30 為正式細軸點平均，權重只供閱讀；"
        "請用 fem_contribution 與 geometric_contribution 驗算 total_energy。"
    )
    headings = (
        "center_frequency_hz fem_energy_fem_points_only fem_point_count "
        "direct_energy_all_points reflected_energy_all_points "
        "interference_energy_all_points "
        "late_energy_all_points geometric_energy_all_points "
        "fem_contribution_all_points geometric_contribution_all_points "
        "total_energy w_fem w_geo t20_s t30_s"
    )
    rows = []
    for band in report.bands:
        values = (
            band.center_frequency_hz,
            band.fem_energy,
            band.fem_point_count,
            band.direct_energy,
            band.reflected_energy,
            band.interference_energy,
            band.late_energy,
            band.geometric_energy,
            band.fem_contribution,
            band.geometric_contribution,
            band.total_energy,
            band.w_fem,
            band.w_geo,
        )
        cells = [_value(value) for value in values]
        cells.append(_decay_value(band.t20_s, band.t20_unavailable_reason))
        cells.append(_decay_value(band.t30_s, band.t30_unavailable_reason))
        rows.append(" ".join(cells))
    return "\n".join((sampling_note, headings, *rows))


def _point_table(report: ThreeLaneReport) -> str:
    headings = (
        "frequency_hz fem_energy direct_energy reflected_energy "
        "interference_energy late_energy scattering geometric_energy "
        "w_fem w_geo total_energy"
    )
    rows = []
    for point in report.points:
        values = (
            point.frequency_hz,
            point.fem_energy,
            point.direct_energy,
            point.reflected_energy,
            point.interference_energy,
            point.late_energy,
            point.scattering,
            point.geometric_energy,
            point.w_fem,
            point.w_geo,
            point.total_energy,
        )
        rows.append(" ".join(_value(value) for value in values))
    return "\n".join((headings, *rows))


def _scene_section(inputs: report_io.ReportInput) -> str:
    """把共享場景身分與這一份自己的兩個座標印成人話一節。"""
    return (
        f"scene scene_fingerprint={report_io.scene_fingerprint(inputs)} "
        f"source_m={inputs.source_m!r} receiver_m={inputs.receiver_m!r}"
    )


def _path_table(
    report: ThreeLaneReport,
    report_inputs: report_io.ReportInput,
    inputs: report_io.SolverInputs,
) -> str:
    section = report_output.output_from_report(
        report, inputs=report_inputs, with_points=False, path_table_inputs=inputs
    ).path_table
    if section is None:
        raise ValueError("要求路徑表卻沒有產生路徑表")
    directivity = report_io.quantity_table()[
        "path_table.includes_speaker_directivity"
    ].reference
    header = (
        f"path_table reflection_order_k={section.reflection_order_k} "
        f"frequencies_hz={section.frequencies_hz!r} "
        f"scattering_coefficient={section.scattering_coefficient!r} "
        f"includes_speaker_directivity={str(section.includes_speaker_directivity).lower()} "
        f"({directivity})"
    )
    headings = (
        "order wall_sequence delay_s direction_vector direction_angles "
        "relative_direct_energy"
    )
    rows = (
        f"{row.order} {row.wall_sequence!r} {_value(row.delay_s)} "
        f"{row.direction_vector!r} {row.direction_angles.model_dump()!r} "
        f"{row.relative_direct_energy!r}"
        for row in section.rows
    )
    return "\n".join((header, headings, *rows))


def _text_sections(
    report: ThreeLaneReport,
    *,
    inputs: report_io.ReportInput,
    with_points: bool,
    path_table_inputs: report_io.SolverInputs | None = None,
) -> list[str]:
    # 能力那一行照舊排第一（既有消費者與考卷認第一行）；場景一節接在它後面。
    sections = [
        _capability_section(report.capability),
        _scene_section(inputs),
        _top_table(report),
        _band_table(report),
    ]
    if with_points:
        sections.append(_point_table(report))
    if path_table_inputs is not None:
        sections.append(_path_table(report, inputs, path_table_inputs))
    return sections


def _regenerate_schemas(directory: Path) -> int:
    """``--regenerate-schemas <目錄>``：把兩份 schema 檔重匯成模型現算的內容。

    票 #316 第三刀非必修第 10 條：考卷 ``test_schema_files_match_the_models`` 紅掉時
    （模型改了沒重匯），跑的入口就是這一個。實作在 :mod:`aosr.physics.report_io`
    （那裡算得出「版控那份住哪」給考卷對，但寫哪個目錄由呼叫端必給——覆寫版控那兩份
    要把 ``blueprint/schemas`` 自己寫出來）；對人報告寫了哪幾個檔由這一層印——物理層
    那一支不對人說話（style-guard 的輸出層就是命令列這一類）。

    回 0 是寫完了、2 是寫的時候炸了——收的是 ``Exception``，跟報表那一條路（下面
    ``main`` 的 ``except Exception``）同一個寬度，模型自己炸掉不會變成 traceback。
    """
    try:
        written = report_io.regenerate_schema_files(directory)
    except Exception as exc:
        print(f"兩份 schema 檔寫不出來：{exc}")
        return 2
    for path in written:
        print(f"寫入 {path}")
    return 0


def _report_parser() -> argparse.ArgumentParser:
    """唯一的 parser：報表模式與重匯模式都走這一個，頂層 ``--help`` 兩種都看得到。

    第五刀非必修第 3 條：``--regenerate-schemas`` 先前只住在一段字串分流裡（第二個
    parser 只管重匯模式），使用者跑 ``--help`` 找不到那支旗標等於那個入口沒有說明；
    現在旗標登記在這裡，重匯模式的目錄就是它的值。兩種模式的必給參數不同，所以
    ``input`` 與 ``--capabilities`` 只在這裡宣告、required 由 ``main`` 分流之後自己檢查。

    因為共用一個 parser，``--help`` 的 usage 行會把 ``input`` 印成 ``[input]``（看起來可
    省）——那是 ``nargs="?"`` 的樣子，不是實話；實話寫在 ``input`` 自己的 help 那一句裡。
    手寫一行 usage 蓋掉它做得到，但那一行沒有機器在守、加旗標就會漂掉，所以不寫。
    """
    parser = argparse.ArgumentParser(
        prog="aosr.physics.three_lane_report_cli",
        description="三路接合物理量報表；也可以重匯 blueprint/schemas 那兩份 schema",
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        help="輸入 JSON；印報表時必給，重匯模式不吃它（給了當場回 2）",
    )
    parser.add_argument("--points", action="store_true", help="另印完整細軸逐點表")
    parser.add_argument(
        "--path-table",
        action="store_true",
        help="另印逐路徑表（大表；預設不帶，能量未含喇叭指向性）",
    )
    # ``--format`` 的預設是 ``None``＝「這一跑沒給過」，不是 ``"text"``：重匯模式要分得出
    # 「沒給」與「明著給了 --format text」，預設寫 "text" 的話後者看起來跟沒給一樣。
    # 報表那一路只問它等不等於 "json"，所以 None 照樣走人看的表格那一條。
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default=None,
        help="不給就是 text（人看的表格）；json 印輸出契約的 JSON",
    )
    parser.add_argument(
        "--capabilities",
        type=Path,
        default=None,
        help="能力與驗證範圍表 TOML；印報表時必給，物理層不設隱含預設",
    )
    parser.add_argument(
        _REGENERATE_FLAG,
        type=Path,
        default=None,
        metavar="目錄",
        help="另一個模式：把該目錄底下那兩份 schema 檔重匯成模型現算的內容",
    )
    return parser


def _refuse_report_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """重匯模式只吃那一個目錄：報表那幾格只要給過任何一格就當場回 2（第七刀必修 1）。

    併成一個 parser 之前，重匯模式有自己那一個只登記位置參數的 parser，所以
    ``--regenerate-schemas 目錄 --points`` 會被 argparse 判成 unrecognized；併完之後
    每一支旗標都合法登記在同一個 parser 裡，不在這裡擋就會被靜靜吃掉、使用者以為自己
    給的那幾格有作用。判「有沒有給過」靠的是各自的預設值：``input``／``--format``／
    ``--capabilities`` 沒給是 ``None``，``--points`` 沒給是 ``False``（store_true 給不出
    「明著給了 false」這種狀態）。
    """
    given = [
        name
        for name, was_given in (
            ("input", args.input is not None),
            ("--points", bool(args.points)),
            ("--path-table", bool(args.path_table)),
            ("--format", args.format is not None),
            ("--capabilities", args.capabilities is not None),
        )
        if was_given
    ]
    if given:
        parser.error(f"unrecognized arguments: {' '.join(given)}")


def main(argv: list[str]) -> int:
    """印報表；成功回 0，讀檔、輸入或求解失敗回 2。

    ``--regenerate-schemas <目錄>`` 是第二種模式：它不讀輸入、也不算報表，只把目錄底下
    那兩份匯出檔重寫成模型現算的內容（見 :func:`_regenerate_schemas`），所以在必給參數
    那一關之前就先分流——它仍走 argparse：旗標與位置參數登記在上面那一個 parser 裡
    （``--help`` 看得到），報表模式那幾格（多給的位置參數、``--points``、``--format``、
    ``--capabilities``）由 :func:`_refuse_report_arguments` 擋，不是靠 argparse 自己的
    unrecognized 那條路。

    兩種模式的必給參數不同（報表要輸入檔與能力表、重匯只要目錄），所以 ``input`` 與
    ``--capabilities`` 不在 parser 那一層宣告 required，改在分流之後自己檢查；缺了就用
    argparse 自己的 ``error()`` 回 2，跟先前那一版同一個離開碼。這兩行手寫的必給檢查由
    ``tests/engine/test_report_io_cli.py`` 的第 ⑥ 節咬（第七刀必修 2）。
    """
    parser = _report_parser()
    args = parser.parse_args(argv)
    if args.regenerate_schemas is not None:
        _refuse_report_arguments(parser, args)
        return _regenerate_schemas(args.regenerate_schemas)
    if args.input is None:
        parser.error("the following arguments are required: input")
    if args.capabilities is None:
        parser.error("the following arguments are required: --capabilities")
    try:
        table = load_capabilities(args.capabilities)
        capability = _capability_for(table)
        inputs = report_io.load_input(args.input, table)
        solved = report_io.solver_inputs(inputs)
        report = solve_three_lane_report(
            room=solved.room,
            source=solved.source,
            receiver=solved.receiver,
            sound_speed_m_s=solved.sound_speed_m_s,
            density_kg_m3=solved.density_kg_m3,
            impedance_by_wall=solved.impedance_by_wall,
            scattering_by_wall=solved.scattering_by_wall,
            capability=capability,
            reflection_order_k=solved.reflection_order_k,
            low_frequency_axis=solved.low_frequency_axis,
        )
        if args.format == "json":
            parsed = report_output.output_from_report(
                report,
                inputs=inputs,
                with_points=args.points,
                path_table_inputs=solved if args.path_table else None,
            )
            payload = parsed.model_dump_json()
            if ReportOutput.model_validate_json(payload) != parsed:
                raise ValueError("輸出契約反解回來的結果跟收成的結果不同")
            print(json.dumps(json.loads(payload), ensure_ascii=False, sort_keys=True))
            return 0
        print(
            "\n".join(
                _text_sections(
                    report,
                    inputs=inputs,
                    with_points=args.points,
                    path_table_inputs=solved if args.path_table else None,
                )
            )
        )
        return 0
    except Exception as exc:
        print(f"三路接合報表算不出來：{exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
