"""把三路接合的報表物件組裝成輸出契約。

這一層只做「求解後的報表物件 → :class:`ReportOutput`」轉換；模型、驗證規則與場景
指紋仍住 :mod:`aosr.physics.report_io`。獨立成單向依賴的新模組，是為了保留契約裡的
設計理由，同時讓每支產品檔維持在寫法警衛的行數上限內。
"""

from __future__ import annotations

from aosr.geometry.shoebox import Room
from aosr.physics.report_io import (
    BandRow,
    CapabilitySection,
    PointRow,
    ReportInput,
    ReportOutput,
    SceneSection,
    SolverInputs,
    TopFields,
    scene_fingerprint,
    solver_inputs,
)


def _capability_section(report: object) -> CapabilitySection:
    """報表那一格能力：表上那一條本人；沒查表時狀態是 unchecked、範圍與收據是空的。"""
    record = report.capability.record  # type: ignore[attr-defined]  # expires=2026-12-08 reason=呼叫端已驗過型別，這一支只收報告物件的那一格
    return CapabilitySection(
        frequency_hz=record.frequency_hz if record is not None else (),
        outputs=record.outputs if record is not None else (),
        status=record.status if record is not None else "unchecked",
        evidence=record.evidence if record is not None else (),
    )


def _top_fields(report: object, room: Room) -> TopFields:
    """頂層那幾格；Schroeder 帶數從產品設定讀，不寫死第二份。"""
    from aosr.config.three_lane_crossover import SCHROEDER_T60_BANDS_HZ

    return TopFields(
        f_s_hz=report.f_s_hz,  # type: ignore[attr-defined]  # expires=2026-12-08 reason=同上
        crossover_lower_hz=report.crossover_lower_hz,  # type: ignore[attr-defined]  # expires=2026-12-08 reason=同上
        crossover_upper_hz=report.crossover_upper_hz,  # type: ignore[attr-defined]  # expires=2026-12-08 reason=同上
        capped_by_upper_limit=report.capped_by_upper_limit,  # type: ignore[attr-defined]  # expires=2026-12-08 reason=同上
        reflection_order_k=report.reflection_order_k,  # type: ignore[attr-defined]  # expires=2026-12-08 reason=同上
        low_frequency_axis=report.low_frequency_axis,  # type: ignore[attr-defined]  # expires=2026-12-08 reason=同上
        eyring_t60_by_band_s={
            str(frequency): value
            for frequency, value in report.eyring_t60_by_band_s.items()  # type: ignore[attr-defined]  # expires=2026-12-08 reason=同上
        },
        room_volume_m3=room.Lx * room.Ly * room.Lz,
        schroeder_band_count=len(SCHROEDER_T60_BANDS_HZ),
    )


def _band_rows(report: object) -> tuple[BandRow, ...]:
    """頻帶列；宣告順序不是印出來的順序（見 :class:`BandRow`）。"""
    return tuple(
        BandRow(
            center_frequency_hz=band.center_frequency_hz,
            fem_energy=band.fem_energy,
            fem_point_count=band.fem_point_count,
            direct_energy=band.direct_energy,
            reflected_energy=band.reflected_energy,
            interference_energy=band.interference_energy,
            late_energy=band.late_energy,
            geometric_energy=band.geometric_energy,
            fem_contribution=band.fem_contribution,
            geometric_contribution=band.geometric_contribution,
            total_energy=band.total_energy,
            w_fem=band.w_fem,
            w_geo=band.w_geo,
            f_s_hz=band.f_s_hz,
            capped_by_upper_limit=band.capped_by_upper_limit,
            t20_s=band.t20_s,
            t20_unavailable_reason=band.t20_unavailable_reason,
            t30_s=band.t30_s,
            t30_unavailable_reason=band.t30_unavailable_reason,
        )
        for band in report.bands  # type: ignore[attr-defined]  # expires=2026-12-08 reason=同上
    )


def _point_rows(report: object) -> tuple[PointRow, ...]:
    """細軸逐點列；宣告順序不是印出來的順序（見 :class:`PointRow`）。"""
    return tuple(
        PointRow(
            frequency_hz=point.frequency_hz,
            fem_energy=point.fem_energy,
            direct_energy=point.direct_energy,
            reflected_energy=point.reflected_energy,
            interference_energy=point.interference_energy,
            late_energy=point.late_energy,
            scattering=point.scattering,
            geometric_energy=point.geometric_energy,
            w_fem=point.w_fem,
            w_geo=point.w_geo,
            total_energy=point.total_energy,
        )
        for point in report.points  # type: ignore[attr-defined]  # expires=2026-12-08 reason=同上
    )


def _scene_section(inputs: ReportInput) -> SceneSection:
    """由輸入組出共享場景身分與這一份自己的聲源／接收點座標。"""
    return SceneSection(
        scene_fingerprint=scene_fingerprint(inputs),
        source_m=inputs.source_m,
        receiver_m=inputs.receiver_m,
    )


def output_from_report(
    report: object,
    *,
    inputs: ReportInput,
    with_points: bool,
    path_table_inputs: SolverInputs | None = None,
) -> ReportOutput:
    """把 :class:`~aosr.physics.three_lane_report.ThreeLaneReport` 收成 :class:`ReportOutput`。

    ``with_points`` 決定帶不帶細軸逐點表。``inputs`` 必須是產出這份報表的那一份輸入：
    場景一節從它算。報表物件自己不記輸入，這裡可對反射階數與報表軸；完整的綁定要等求解結果
    自己帶輸入指紋（票 #415 留言）。
    """
    from aosr.physics.report_path_table import build_path_table_section as _build_path_table_section

    from aosr.physics.three_lane_report import ThreeLaneReport

    if not isinstance(report, ThreeLaneReport):
        raise ValueError(f"report 不是 ThreeLaneReport：{type(report).__name__}")
    if report.reflection_order_k != inputs.reflection_order_k:
        # 報表物件自己不記輸入，能對的只有兩邊都有的這一格；對不上就是拿錯輸入來組輸出。
        raise ValueError(
            f"inputs 的反射階數 {inputs.reflection_order_k} 跟 report 的 "
            f"{report.reflection_order_k} 不一樣：這一份輸入不是產出這份報表的那一份"
        )
    if report.low_frequency_axis is not inputs.low_frequency_axis:
        raise ValueError("inputs 的低頻軸跟 report 使用的低頻軸不同")
    if path_table_inputs is not None:
        expected = solver_inputs(inputs)
        for field in SolverInputs._fields:
            if getattr(path_table_inputs, field) != getattr(expected, field):
                raise ValueError(
                    f"path_table_inputs 的 {field} 跟 inputs 的 {field} 不同："
                    "路徑表輸入不是這份報表輸入的同一份"
                )
    return ReportOutput(
        scene=_scene_section(inputs),
        capability=_capability_section(report),
        top=_top_fields(report, inputs.room_m),
        bands=_band_rows(report),
        points=_point_rows(report) if with_points else None,
        path_table=(
            _build_path_table_section(report, path_table_inputs)
            if path_table_inputs is not None
            else None
        ),
    )
