"""材料編輯頁的吸音曲線、數字來源標語與可信範圍提醒（#1082 段 2-2b）。

只呼叫現成的物理（:func:`aosr.materials.construction.random_incidence_absorption`，Paris 角度平均的無規入射吸音），
不改任何物理。送來什麼材料內文就畫什麼——表單還沒存的值也能先看；不讀也不寫使用者材料庫。

- 頻率軸跟房間計算同一條細軸（``GEOMETRIC_LANE_FREQUENCIES_HZ``：20 Hz 起、每八度 24 點、到 8000 Hz 八度帶上緣以內）；
  不為了讓提醒消失而縮短：搜尋最在意的低頻正好落在小松模型的外推區。實測表的頻點另外併進軸，表內表外分得出來。
- 可最佳化材料是一族（搜尋會調厚度與空腔），這裡畫的是初值排法（``template_assembly``），圖旁寫明。
- 提醒分兩級：參數出界（流阻低於原研究樣品、泡棉不在擬合範圍）醒目；頻率外推與實測表外的模型補值照實寫頻段、圖上標底色。
- 只有小松模型與實測表登記了可信範圍；其他多孔模型照實寫「沒有範圍可查」，沒有提醒不等於在可信範圍內。
- 表單沒填齊、參數算不出來（例如流阻超出手動試算範圍）都是使用者會走到的正常狀態：回 200 加 ``status`` 一格，不回錯誤碼。
- 行程沒開雙精度（正式入口一定開；pytest 的行程刻意不開）另給一個狀態，講明是開機設定、不怪材料參數。
"""
from __future__ import annotations

from typing import Final, Literal, cast

import jax.numpy as jnp
import numpy as np
from pydantic import TypeAdapter, ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from aosr.config.frequency_axis import GEOMETRIC_LANE_FREQUENCIES_HZ, V3_AXIS_POINTS_PER_OCTAVE
from aosr.config.paths import config_path
from aosr.config.physics_constants import load_physics_constants
from aosr.config.porous_model_ranges import PorousModelRanges, load_porous_model_ranges
from aosr.gui.material_editor import plain_error
from aosr.gui.problem_text import material_form_spec
from aosr.materials.construction import random_incidence_absorption
from aosr.materials.construction_models import (
    Assembly,
    ConstantImpedance,
    ConstructionNote,
    MeasuredImpedance,
    PorousLayer,
    SurfaceMaterial,
)
from aosr.materials.library import (
    PROVENANCE_LABELS,
    LibraryEntry,
    OptimizableEntry,
    basis_label,
    impedance_basis,
    template_assembly,
)
from aosr.physics.wall_surfaces import extrapolated_frequency_text

SpanKind = Literal["extrapolation", "model_fill"]
RANGE_SCOPE: Final = ("只有小松纖維經驗式與實測阻抗表登記了可信範圍；其他多孔模型，以及板、孔板、長縫板、布這類層，"
                      "沒有範圍可查——沒有提醒不代表在可信範圍內。")
_STRONG: Final = frozenset({"komatsu_flow_below_samples", "komatsu_material_class_outside_fit"})
_ENTRY: Final[TypeAdapter[LibraryEntry]] = TypeAdapter(LibraryEntry)


def _model_names() -> dict[str, str]:
    """多孔模型的中文名，跟表單同一份。"""
    fields = cast(dict[str, dict[str, object]], material_form_spec()["porous"]["fields"])
    return cast(dict[str, str], fields["model"]["options"])


def _plain(message: str) -> str:
    """物理訊息裡的模型名換成表單上的中文名，畫面上同一個模型只有一個叫法。"""
    return message.replace("Komatsu（小松）纖維模型", _model_names()["komatsu"])


def _physics_material(entry: LibraryEntry) -> tuple[SurfaceMaterial, str | None]:
    """要算的牆面材料；可最佳化材料取初值排法，另回一句寫明是哪一種排法。"""
    if not isinstance(entry, OptimizableEntry):
        return entry.surface_material(), None
    cavity = None if entry.cavity is None else entry.cavity.initial_m
    layers = f"多孔層 {entry.porous.thickness_m:g} m" + (f"＋空腔 {cavity:g} m" if cavity else "、沒有空腔")
    return template_assembly(entry), (f"這條是初值排法：{layers}，背後是硬牆。可最佳化材料是一族，"
                                      "搜尋會在產品厚度與空腔範圍裡調，這條只是搜尋的起點。")


def _axis(material: SurfaceMaterial) -> tuple[float, ...]:
    if not isinstance(material, MeasuredImpedance):
        return GEOMETRIC_LANE_FREQUENCIES_HZ
    return tuple(sorted({*GEOMETRIC_LANE_FREQUENCIES_HZ, *(point.frequency_hz for point in material.points)}))


def _runs(axis: tuple[float, ...], flagged: frozenset[float]) -> list[tuple[float, float]]:
    """被標的頻點在軸上連成幾段（頭、尾）；中間照常算的頻點把段切開。"""
    runs: list[tuple[float, float]] = []
    start: float | None = None
    previous = 0.0
    for frequency in axis:
        if frequency in flagged:
            start = frequency if start is None else start
            previous = frequency
        elif start is not None:
            runs.append((start, previous))
            start = None
    if start is not None:
        runs.append((start, previous))
    return runs


def _layer_place(material: SurfaceMaterial, index: int | None) -> str:
    return f"{'底構造' if isinstance(material, MeasuredImpedance) else ''}第 {index} 層"


def _note_lines(material: SurfaceMaterial, axis: tuple[float, ...], notes: tuple[ConstructionNote, ...],
                ranges: PorousModelRanges) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """物理回的提醒 → 畫面上的提醒（醒目／一般）與圖上要標底色的頻段。"""
    warnings: list[dict[str, object]] = []
    spans: list[dict[str, object]] = []
    limits = ranges.komatsu
    for note in notes:
        if note.code in _STRONG:
            warnings.append({"level": "strong", "code": note.code, "text": _plain(note.message)})
            continue
        kind: SpanKind
        if note.code == "komatsu_frequency_extrapolation":
            kind = "extrapolation"
            text = (f"{_layer_place(material, note.layer_index)}（{_model_names()['komatsu']}）在 {extrapolated_frequency_text(note.frequencies_hz)} "
                    f"是材料模型外推：原研究只量過 {limits.measured_frequency_min_hz:g}–{limits.measured_frequency_max_hz:g} Hz、"
                    f"流阻 {limits.sample_flow_resistivity_min_pa_s_m2:g}–{limits.sample_flow_resistivity_max_pa_s_m2:g} Pa·s/m² 的樣品；"
                    "這幾段照算、不宣稱準確（圖上橘底）。")
        else:
            measured = cast(MeasuredImpedance, material)
            kind = "model_fill"
            low, high = measured.points[0].frequency_hz, measured.points[-1].frequency_hz
            text = (f"實測表只涵蓋 {low:g}–{high:g} Hz；"
                    f"{extrapolated_frequency_text(note.frequencies_hz, table_bounds_hz=(low, high))} 是用底構造算的模型補值，不是實測（圖上灰底）。")
        warnings.append({"level": "note", "code": note.code, "text": text})
        spans += [{"low_hz": low_hz, "high_hz": high_hz, "kind": kind}
                  for low_hz, high_hz in _runs(axis, frozenset(note.frequencies_hz))]
    return warnings, spans


def _unchecked_layers(material: SurfaceMaterial) -> list[dict[str, object]]:
    """用了沒有登記可信範圍的多孔模型的那幾層，照實寫出來。"""
    names = _model_names()
    structure = material.base_assembly if isinstance(material, MeasuredImpedance) else material
    if not isinstance(structure, Assembly):
        return []
    return [{"level": "note", "code": "model_range_not_registered",
             "text": f"{_layer_place(material, index)}用{names[layer.model]}：這個模型沒有登記可信範圍，這頁沒有範圍可查。"}
            for index, layer in enumerate(structure.layers, start=1)
            if isinstance(layer, PorousLayer) and layer.model != "komatsu"]


def _basis_detail(entry: LibraryEntry) -> str:
    """標語底下那一行：這條曲線的阻抗是怎麼來的。"""
    if isinstance(entry, OptimizableEntry) or isinstance(entry.material, Assembly):
        return "由各層參數用模型算的，之後要拿實測對照。"
    if isinstance(entry.material, ConstantImpedance):
        return "每個頻率都同一個實數阻抗，是近似。"
    table = entry.material
    basis = impedance_basis(entry)
    coverage = f"實測表涵蓋 {table.points[0].frequency_hz:g}–{table.points[-1].frequency_hz:g} Hz，表內假設與入射角無關；表外用底構造算。"
    return ("實測表沒有虛部，虛部當 0 算，所以是近似；" if basis == "approximate" else "") + coverage


def _labels(entry: LibraryEntry) -> dict[str, object]:
    return {"basis": basis_label(entry), "basis_code": impedance_basis(entry), "basis_detail": _basis_detail(entry),
            "provenance": PROVENANCE_LABELS[entry.provenance], "source": entry.source, "links": list(entry.links)}


def material_curve(document: object) -> dict[str, object]:
    """一份材料內文 → 吸音曲線、標語、提醒；算不出來或沒填齊時照實講原因。"""
    try:
        entry = _ENTRY.validate_python(document)
    except ValidationError as exc:
        return {"status": "incomplete", "message": f"表單還沒填齊，曲線畫不出來：{plain_error(exc)}", "range_scope": RANGE_SCOPE}
    material, arrangement = _physics_material(entry)
    axis = _axis(material)
    air = load_physics_constants(config_path("physics_constants.toml"))
    ranges = load_porous_model_ranges(config_path("porous_model_ranges.toml"))
    labels = _labels(entry)
    if jnp.asarray(0.0).dtype != jnp.float64:
        # 正式網頁入口開機就開雙精度；沒開時物理層會拒算，那不是材料參數的錯，不能說成「這組參數算不出」。
        return {"status": "no_precision", **labels, "arrangement": arrangement, "range_scope": RANGE_SCOPE,
                "message": "這個網頁行程沒有開雙精度，算不出吸音曲線；這是網頁開機設定的問題，不是材料參數的問題。"}
    try:
        absorption, notes = random_incidence_absorption(material, jnp.asarray(axis), air=air, ranges=ranges)
    except ValueError as exc:
        return {"status": "uncomputable", **labels, "arrangement": arrangement, "range_scope": RANGE_SCOPE,
                "message": f"這組參數算不出吸音曲線：{_plain(str(exc))}"}
    warnings, spans = _note_lines(material, axis, notes, ranges)
    warnings.sort(key=lambda line: line["level"] != "strong")
    measured_range = None if not isinstance(material, MeasuredImpedance) else [
        material.points[0].frequency_hz, material.points[-1].frequency_hz]
    return {"status": "drawn", **labels, "arrangement": arrangement, "range_scope": RANGE_SCOPE,
            "frequencies_hz": list(axis), "absorption": [float(value) for value in np.asarray(absorption)],
            "measured_range_hz": measured_range, "spans": spans, "warnings": [*warnings, *_unchecked_layers(material)],
            "axis_text": (f"頻率軸跟房間計算同一條細軸：{GEOMETRIC_LANE_FREQUENCIES_HZ[0]:g} Hz 起、每八度 {V3_AXIS_POINTS_PER_OCTAVE} 點，"
                          f"到 {GEOMETRIC_LANE_FREQUENCIES_HZ[-1]:.0f} Hz" + ("；實測表的頻點另外併進去。" if measured_range else "。"))}


def material_curve_routes() -> list[Route]:
    """曲線介面：POST 一份材料內文、回曲線與提醒；不讀也不寫使用者材料庫。

    路徑不放在 ``/api/materials/`` 底下：那裡的下一段是材料代號，``curve`` 也是合法的代號。
    """

    async def curve(request: Request) -> Response:
        document = cast(object, await request.json())
        return JSONResponse(await run_in_threadpool(material_curve, document))

    return [Route("/api/material-curve", curve, methods=["POST"])]
