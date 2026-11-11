"""准用清單裡的固定材料進第三階段搜尋前的關卡（#1082 段 2-3b）：物理算得出、替身留出驗證過了才開。

替身的驗證收據只證明它對建模那時候挑的留出材料準；使用者實測的阻抗表（或任何固定材料）不在那批裡，收據過了不等於
它也準——以前光牆（≥75ρc）就是收據過了、換上去逐頻差 2.3～2.8 dB（#990）。所以開跑前，准用清單裡每一個固定材料
各做一組留出驗證：把它放到准用它的每一片、其他片照起跑配置（樣板片取下降起點那一點），整條頻率軸完整解對替身，
兩把尺從精度契約登記簿讀（跟准用閘門同兩把，不另立尺）。任何一個沒過就拒收、照實寫出數字，不放寬、不偷偷補訓練。

阻抗一律走搜尋同一條材料鏈：固定材料是 surface_impedance（實測表在表內對數內插、表外用它的底構造補值），
跟終點 material_impedances 同一支函式、同一份材料，所以搜尋頻點與終點取到的是同一份表。
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from aosr.config.material_gradient import fixed_target_id
from aosr.gradient.boundary import patch_key
from aosr.gradient.material_gradient import SurfaceDesign, fixed_candidate_design
from aosr.materials.construction_models import MeasuredImpedance, SurfaceMaterial
from aosr.materials.search_library import MaterialsRefused, PresetMaterial
from aosr.physics.fem_reduced import FullOperators, ReducedModel
from aosr.physics.fem_reduced_boundary import FrequencyBoundaryImpedances

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


def check_fixed_candidates(model: ReducedModel, operators: FullOperators, designs: Sequence[SurfaceDesign],
                           allowed: Mapping[str, Sequence[str]], table: Sequence[PresetMaterial], *,
                           density_kg_m3: float, sound_speed_m_s: float, workers: int = 1) -> tuple[FixedCandidateCheck, ...]:
    """准用清單裡每一個固定材料各一組留出驗證（整條頻率軸、兩把尺）；全部過了回每一筆的紀錄，任何一個沒過就拒收、
    寫出它的數字與尺，不放寬。算不出阻抗（物理拒算、不是有限數）也拒收、點名哪一個。"""
    import time

    import numpy as np

    from aosr.gradient.fixed_surfaces import fixed_targets_of
    from aosr.gradient.run import holdout_rulers
    from aosr.physics.fem_reduced_check import validate_reduced, within_reduced_contract

    rulers = holdout_rulers()
    axis = tuple(float(value) for value in model.frequencies_hz)
    if set(operators.sources) != set(model.source_names) or set(operators.receivers) != set(model.receiver_names):
        raise ValueError("完整模型的喇叭或座位跟替身不同")
    operators = replace(operators, sources={name: operators.sources[name] for name in model.source_names},
                        receivers={name: operators.receivers[name] for name in model.receiver_names})
    found = []
    for target in fixed_targets_of(allowed):
        swapped = candidate_designs(designs, target, allowed, table)
        try:
            impedances = design_impedances(swapped, axis, density_kg_m3=density_kg_m3, sound_speed_m_s=sound_speed_m_s)
        except ValueError as error:
            raise MaterialsRefused(f"固定材料 {fixed_target_id(target)} 在搜尋頻點物理拒算，搜尋不開：{error}") from error
        if not all(np.all(np.isfinite(np.asarray(row))) for row in impedances.values() if row is not None):
            raise MaterialsRefused(f"固定材料 {fixed_target_id(target)} 在搜尋頻點的阻抗不是有限數，搜尋不開")
        clock = time.perf_counter()
        report = validate_reduced(model, operators, [impedances], workers=workers)
        seconds = time.perf_counter() - clock
        frequency_db, band_db = max(report.frequency_db_by_seat.values()), max(report.band_db_by_seat.values())
        if not within_reduced_contract(report, frequency_db=rulers.frequency.value, band_db=rulers.band.value):
            raise MaterialsRefused(
                f"替身對固定材料 {fixed_target_id(target)} 沒過留出驗證：逐頻最大差 {frequency_db:g} dB"
                f"（尺 {rulers.frequency.name}={rulers.frequency.value:g} dB）、帶能量最大差 {band_db:g} dB"
                f"（尺 {rulers.band.name}={rulers.band.value:g} dB）。這份替身沒有學過這種材料，"
                "要把它放進訓練材料重建替身；這次不開，也不放寬尺")
        found.append(FixedCandidateCheck(
            target=target, surfaces=tuple(name for name, targets in sorted(allowed.items()) if target in targets),
            frequency_db=frequency_db, band_db=band_db, frequency_ruler=rulers.frequency.value,
            band_ruler=rulers.band.value, full_solves=report.full_solves, seconds=seconds))
    return tuple(found)
