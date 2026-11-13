"""准用清單裡的固定材料進第三階段搜尋前的關卡（#1082 段 2-3b）：物理算得出、替身留出驗證過了才開。

替身的驗證收據只證明它對建模那時候挑的留出材料準；使用者實測的阻抗表（或任何固定材料）不在那批裡，收據過了不等於
它也準——以前光牆（≥75ρc）就是收據過了、換上去逐頻差 2.3～2.8 dB（#990）。所以開跑前，准用清單裡每一個固定材料
各做一組留出驗證：把它放到准用它的每一片、其他片照起跑配置（樣板片取下降起點那一點），整條頻率軸完整解對替身，
兩把尺從精度契約登記簿讀（跟准用閘門同兩把，不另立尺）。任何一個沒過就拒收、照實寫出數字，不放寬、不偷偷補訓練。

#1082 段 2-8：每一個候選都核完才判（不是第一個沒過就停），沒過的一次全講；替身訓練說明記了它（訓練過）就說
「訓練過、但在哪個頻率不夠準」，沒記才說沒學過。建替身那一側（:mod:`aosr.gradient.surrogate_build`）裝進格子前用同一組
起跑配置、同一支 :func:`judge_candidates` 先核，沒過就補訓練再核（完整解只解一次，之後只重算替身那一側）。

阻抗一律走搜尋同一條材料鏈：固定材料是 surface_impedance（實測表在表內對數內插、表外用它的底構造補值），
跟終點 material_impedances 同一支函式、同一份材料，所以搜尋頻點與終點取到的是同一份表。
"""
from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Self

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from aosr.config.material_gradient import fixed_target_id
from aosr.gradient.boundary import patch_key
from aosr.gradient.material_gradient import SurfaceDesign, fixed_candidate_design
from aosr.materials.construction_models import MeasuredImpedance, SurfaceMaterial
from aosr.materials.search_library import MaterialsRefused, PresetMaterial
from aosr.physics.fem_reduced import FullOperators, ReducedModel
from aosr.physics.fem_reduced_boundary import FrequencyBoundaryImpedances
from aosr.gradient.surrogate_fixed_checks import FixedCandidateInaccurate

if TYPE_CHECKING:
    from aosr.physics.fem_reduced_check import ValidationReport

_FROZEN = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class FixedCandidateCheck(BaseModel):
    """准用清單裡一個固定材料的替身留出驗證（#1082 段 2-3b）：開跑時把它放到准用它的每一片（其他片照起跑配置）、
    整條頻率軸完整解對替身，兩把尺照登記簿判；記每個座位跨喇叭與頻點最大的逐頻差與帶能量差、完整解次數與秒數。"""

    model_config = _FROZEN
    target: str = Field(pattern=r"^fixed:[a-z][a-z0-9_]*$")
    surfaces: tuple[str, ...] = Field(min_length=1)
    frequency_db: float = Field(ge=0.0)
    band_db: float = Field(ge=0.0)
    frequency_ruler: float = Field(gt=0.0)
    band_ruler: float = Field(gt=0.0)
    full_solves: int = Field(ge=1, strict=True)
    seconds: float = Field(ge=0.0)
    material_digest: str = Field(pattern=r"^[0-9a-f]{64}$",
                                 description="驗的那份材料的身分摘要（endpoint.material_digest）；接續讀回的材料要逐位同一份")

    @model_validator(mode="after")
    def _passed(self) -> Self:
        if self.frequency_db > self.frequency_ruler or self.band_db > self.band_ruler:
            raise ValueError(f"固定材料 {self.target} 的替身留出驗證沒過，不會記成可以開跑的搜尋")
        return self


def candidate_designs(designs: Sequence[SurfaceDesign], target: str, allowed: Mapping[str, Sequence[str]],
                      table: Sequence[PresetMaterial]) -> tuple[SurfaceDesign, ...]:
    """起跑設計裡、准用清單列了 ``target`` 的每一片換成那份固定材料（整塊、沒有變數），其他片原樣。"""
    by_id = {preset.id: preset for preset in table}
    return tuple(fixed_candidate_design(design.surface_id, target, by_id, design.mount)
                 if target in allowed.get(design.surface_id, ()) else design for design in designs)


def design_impedances(designs: Sequence[SurfaceDesign], frequencies_hz: Sequence[float], *,
                      density_kg_m3: float, sound_speed_m_s: float) -> FrequencyBoundaryImpedances:
    """設計清單在指定頻點的每片阻抗：保留原牆、固定材料與樣板全用搜尋同一條材料鏈；樣板片的 θ 取下降起點
    （投影到盒界與總深度內）。空氣照方案場景（跟替身建模、搜尋同一組）。要在雙精度已開的行程裡叫。"""
    import jax.numpy as jnp
    import numpy as np

    from aosr.config.construction_environment import construction_environment
    from aosr.gradient.material_gradient import (
        baseline_theta, depth_limits, descent_start, design_bounds, design_templates, kept_impedance_rows, surface_parameters,
    )
    from aosr.materials.construction_diff import surface_impedance_from_parameters

    air, ranges = construction_environment(density_kg_m3=density_kg_m3, sound_speed_m_s=sound_speed_m_s)
    theta = descent_start(baseline_theta(designs), design_bounds(designs), depth_limits(designs))
    axis = jnp.asarray(np.asarray(frequencies_hz, dtype=np.float64))
    templates = design_templates(designs, axis, air=air, ranges=ranges)
    parameters = surface_parameters(designs, theta)
    kept = kept_impedance_rows(designs, frequencies_hz, air=air, ranges=ranges)
    rows = {}
    for design, template, values, original in zip(designs, templates, parameters, kept, strict=True):
        impedance = (original if original is not None else None if template is None else
                     surface_impedance_from_parameters(template, values, axis, air=air))
        rows[patch_key(design.surface_id)] = None if impedance is None else tuple(complex(value) for value in np.asarray(impedance))
    return rows


def table_range_notes(targets: Sequence[str], table: Sequence[PresetMaterial],
                      frequencies_hz: Sequence[float]) -> tuple[str, ...]:
    """實測阻抗表沒蓋到的搜尋頻點（表外用它的底構造補值，之七紙）：每一筆照實一行。"""
    adapter: TypeAdapter[SurfaceMaterial] = TypeAdapter(SurfaceMaterial)
    by_id = {preset.id: preset for preset in table}
    notes = []
    low, high = min(frequencies_hz), max(frequencies_hz)
    for target in targets:
        material = adapter.validate_python(by_id[target].material)
        if not isinstance(material, MeasuredImpedance):
            continue
        first, last = material.points[0].frequency_hz, material.points[-1].frequency_hz
        if low < first or high > last:
            notes.append(f"固定材料 {fixed_target_id(target)} 的實測表只到 {first:g}～{last:g} Hz，搜尋頻點 {low:g}～{high:g} Hz "
                         "在表外的那幾點用它的底構造補值（跟終點同一條規則）")
    return tuple(notes)


@dataclass(frozen=True)
class CandidateVerdict:
    """一個固定材料候選在起跑配置的一次判決：完整解對替身的報告（含兩側聲壓，補訓練後重判用）、兩個差、最差的頻點。"""

    target: str
    surfaces: tuple[str, ...]
    report: ValidationReport
    frequency_db: float
    band_db: float
    worst_hz: float
    passed: bool
    seconds: float


def candidate_configurations(designs: Sequence[SurfaceDesign], allowed: Mapping[str, Sequence[str]],
                             table: Sequence[PresetMaterial], frequencies_hz: Sequence[float], *, density_kg_m3: float,
                             sound_speed_m_s: float) -> dict[str, FrequencyBoundaryImpedances]:
    """准用清單裡每一個固定材料的核對配置：起跑設計裡准用它的片換成它、其他片照起跑配置，整條頻率軸的每片阻抗。
    算不出阻抗（物理拒算、不是有限數）就拒收、點名哪一個。"""
    import numpy as np

    from aosr.gradient.fixed_surfaces import fixed_targets_of

    found = {}
    for target in fixed_targets_of(allowed):
        swapped = candidate_designs(designs, target, allowed, table)
        try:
            impedances = design_impedances(swapped, frequencies_hz, density_kg_m3=density_kg_m3,
                                           sound_speed_m_s=sound_speed_m_s)
        except ValueError as error:
            raise MaterialsRefused(f"固定材料 {fixed_target_id(target)} 在搜尋頻點物理拒算，搜尋不開：{error}") from error
        if not all(np.all(np.isfinite(np.asarray(row))) for row in impedances.values() if row is not None):
            raise MaterialsRefused(f"固定材料 {fixed_target_id(target)} 在搜尋頻點的阻抗不是有限數，搜尋不開")
        found[target] = impedances
    return found


def _offset(progress: Callable[[int, int], None], done: int, total: int) -> Callable[[int, int], None]:
    """單一候選的「已解幾點」換成全部候選合計的進度。"""
    def call(solved: int, _points: int) -> None:
        progress(done + solved, total)
    return call


def judge_candidates(model: ReducedModel, operators: FullOperators, configurations: Mapping[str, FrequencyBoundaryImpedances],
                     allowed: Mapping[str, Sequence[str]], *, workers: int = 1,
                     previous: Mapping[str, CandidateVerdict] | None = None,
                     progress: Callable[[int, int], None] | None = None) -> dict[str, CandidateVerdict]:
    """每一個候選都判（不在第一個沒過就停）：沒給 ``previous`` 就整條頻率軸完整解對替身；給了（同一批配置、補訓練之後）
    完整解聲壓照上一次的、只重算替身那一側。兩把尺從登記簿讀。

    ``progress`` 給了就照「全部候選合計核到第幾個頻點、共幾點」叫（#1128：建替身那一道核 14 分鐘，進度一直停在 0）；
    只讀、不改判決。"""
    import time

    from aosr.gradient.run import holdout_rulers
    from aosr.physics import fem_reduced_check

    rulers = holdout_rulers()
    if set(operators.sources) != set(model.source_names) or set(operators.receivers) != set(model.receiver_names):
        raise ValueError("完整模型的喇叭或座位跟替身不同")
    operators = replace(operators, sources={name: operators.sources[name] for name in model.source_names},
                        receivers={name: operators.receivers[name] for name in model.receiver_names})
    found = {}
    points, total = len(model.frequencies_hz), len(model.frequencies_hz) * len(configurations)
    for index, (target, impedances) in enumerate(configurations.items()):
        clock = time.perf_counter()
        done = index * points
        if previous is None:
            report = fem_reduced_check.validate_reduced(
                model, operators, [impedances], workers=workers,
                progress=None if progress is None else _offset(progress, done, total))
        else:
            before = previous[target].report
            report = fem_reduced_check.revalidate_reduced(model, operators, [impedances], before.full_pressure,
                                                          full_solves=before.full_solves)
            if progress is not None:
                progress(done + points, total)
        seconds = time.perf_counter() - clock
        frequency_db, band_db = max(report.frequency_db_by_seat.values()), max(report.band_db_by_seat.values())
        passed = fem_reduced_check.within_reduced_contract(report, frequency_db=rulers.frequency.value,
                                                           band_db=rulers.band.value)
        worst = (float("nan") if passed else
                 fem_reduced_check.worst_frequency_hz(report, tuple(float(value) for value in model.frequencies_hz)))
        found[target] = CandidateVerdict(
            target=target, surfaces=tuple(name for name, targets in sorted(allowed.items()) if target in targets),
            report=report, frequency_db=frequency_db, band_db=band_db, worst_hz=worst, passed=passed, seconds=seconds)
    return found


def inaccurate_text(verdict: CandidateVerdict, *, trained: bool) -> str:
    """一個沒過的候選怎麼講：數字、尺、最差的頻點；訓練過（替身訓練說明記了它）就說訓練過但不夠準，沒記才說沒學過。"""
    from aosr.gradient.run import holdout_rulers

    rulers = holdout_rulers()
    numbers = (f"逐頻最大差 {verdict.frequency_db:g} dB（尺 {rulers.frequency.name}={rulers.frequency.value:g} dB）、"
               f"帶能量最大差 {verdict.band_db:g} dB（尺 {rulers.band.name}={rulers.band.value:g} dB）")
    name = fixed_target_id(verdict.target)
    if trained:
        # 逐頻那一把沒超才只講帶能量：最差頻點是逐頻差最大的那一點，帶能量超了不一定在那一點。
        spot = (f"在 {verdict.worst_hz:g} Hz 不夠準" if verdict.frequency_db > rulers.frequency.value
                else "帶能量不夠準")
        return f"替身對固定材料 {name} 沒過留出驗證：替身訓練過這種材料，但開跑這一組配置{spot}：{numbers}"
    return f"替身對固定材料 {name} 沒過留出驗證：{numbers}。這份替身沒有學過這種材料"


def check_fixed_candidates(model: ReducedModel, operators: FullOperators, designs: Sequence[SurfaceDesign],
                           allowed: Mapping[str, Sequence[str]], table: Sequence[PresetMaterial], *,
                           density_kg_m3: float, sound_speed_m_s: float, workers: int = 1,
                           trained: Collection[str] = ()) -> tuple[FixedCandidateCheck, ...]:
    """准用清單裡每一個固定材料各一組留出驗證（整條頻率軸、兩把尺）；全部過了回每一筆的紀錄，有沒過的就全部核完再拒收、
    一次寫出每一個沒過的數字與尺，不放寬。``trained`` 是替身訓練說明記過的固定材料（措辭用：訓練過但不夠準、或沒學過）。
    算不出阻抗（物理拒算、不是有限數）也拒收、點名哪一個。"""
    from aosr.gradient.endpoint import material_digest

    adapter: TypeAdapter[SurfaceMaterial] = TypeAdapter(SurfaceMaterial)
    by_id = {preset.id: preset for preset in table}
    axis = tuple(float(value) for value in model.frequencies_hz)
    configurations = candidate_configurations(designs, allowed, table, axis, density_kg_m3=density_kg_m3,
                                              sound_speed_m_s=sound_speed_m_s)
    verdicts = judge_candidates(model, operators, configurations, allowed, workers=workers)
    failed = [verdict for verdict in verdicts.values() if not verdict.passed]
    frequency_ruler, band_ruler = holdout_ruler_values()
    if failed:
        words = "；".join(inaccurate_text(verdict, trained=verdict.target in trained) for verdict in failed)
        advice = ("要重建替身（重建時會先用這一組起跑配置核，不夠準就補訓練）" if all(item.target in trained for item in failed)
                  else "要把它放進訓練材料重建替身")
        raise FixedCandidateInaccurate(f"{words}。{advice}；這次不開，也不放寬尺")
    return tuple(FixedCandidateCheck(
        target=verdict.target, surfaces=verdict.surfaces, frequency_db=verdict.frequency_db, band_db=verdict.band_db,
        frequency_ruler=frequency_ruler, band_ruler=band_ruler,
        full_solves=verdict.report.full_solves, seconds=verdict.seconds,
        material_digest=material_digest(adapter.validate_python(by_id[verdict.target].material)))
        for verdict in verdicts.values())


def holdout_ruler_values() -> tuple[float, float]:
    """登記簿那兩把留出尺的值（逐頻、帶能量）。"""
    from aosr.gradient.run import holdout_rulers

    rulers = holdout_rulers()
    return rulers.frequency.value, rulers.band.value
