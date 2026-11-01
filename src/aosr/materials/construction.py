"""牆面構造計算核心：給具體值、在即時編譯之外驗證，不接進三路。

時間慣例沿用傳遞矩陣 e^{+jωt}（正時間指數），層從房間側往負載連乘。
Komatsu（纖維多孔模型）依注入設定拒收手動試算流阻範圍外值，低於樣品下限另標註，按 f/σ 記頻率外推，
泡棉另記類別不在擬合範圍；樣品範圍不是作者訂的物理有效界線。
Miki（纖維經驗模型）、Wilson 2015（鬆弛模型）、Mechel 1976（低頻延伸模型）尚未設範圍。
所有入口檢查最終表面阻抗的每個頻點、每個角度；非有限或實部負值拒收，沒有模型備援。

結果是 JAX（陣列算數函式庫）的 complex128（雙精度複數）或 float64（雙精度實數）。
本模組只查陣列型別，不讀寫全域設定。執行環境層須在載入陣列函式庫前開雙精度。
三個入口共用 _layer_matrix（層轉矩陣函式），集中連續物理參數，留給後續梯度工作。
固定阻抗的無規吸音沿用型錄既有換算；該函式實際使用高階積分，這裡不另造閉式。
貼在原牆上的構造（#990）：先用同一頻率軸、同一組角度求原牆的阻抗，當新增層的有限負載 (A·Z+B)/(C·Z+D)。
"""
from __future__ import annotations

from typing import NotRequired, TypedDict

import jax
import jax.numpy as jnp
import numpy as np

from aosr.config.physics_constants import PhysicsConstants
from aosr.config.porous_model_ranges import PorousModelRanges
from aosr.materials import transfer_matrix as tmm
from aosr.materials.catalog_absorption import complex_random_incidence_absorption
from aosr.materials.construction_models import AirLayer, BoardLayer, ConstantImpedance, ConstructionNote, FabricLayer, Layer, MeasuredImpedance, MicroPerforatedLayer, MountedOnWall, PerforatedLayer, PorousLayer, SlottedLayer, SurfaceMaterial, layer_open_area


# 布的兩個物性明給後不讀厚度；正數只滿足傳遞矩陣的必填位置。
FABRIC_THICKNESS_PLACEHOLDER_M = 0.001


class _PorousParameters(TypedDict):
    porosity: NotRequired[jax.Array]
    tortuosity: NotRequired[jax.Array]


def _context(material: SurfaceMaterial) -> str:
    """表面結果不能歸罪單層；列出完整構造，讓拒收訊息可定位輸入。"""
    if isinstance(material, ConstantImpedance):
        return f"固定阻抗（整面）{material.impedance_pa_s_m} Pa·s/m"
    if isinstance(material, MeasuredImpedance):
        return f"{material.coverage_text()}；底構造：{_context(material.base_assembly)}"
    descriptions = []
    for index, layer in enumerate(material.layers, start=1):
        prefix = f"第 {index} 層"
        if isinstance(layer, PorousLayer):
            descriptions.append(f"{prefix}：{layer.model} 多孔模型，厚度 {layer.thickness_m} m、流阻 {layer.flow_resistivity_pa_s_m2} Pa·s/m²、孔隙率 {layer.porosity}、曲折度 {layer.tortuosity}")
        elif isinstance(layer, AirLayer):
            descriptions.append(f"{prefix}：空氣層，厚度 {layer.thickness_m} m")
        elif isinstance(layer, BoardLayer):
            descriptions.append(f"{prefix}：板層，面密度 {layer.surface_mass_kg_m2} kg/m²、面阻力 {layer.resistance_pa_s_m} Pa·s/m")
        elif isinstance(layer, FabricLayer):
            descriptions.append(f"{prefix}：布層，面密度 {layer.surface_mass_kg_m2} kg/m²、面流阻 {layer.flow_resistance_pa_s_m} Pa·s/m")
        elif isinstance(layer, MicroPerforatedLayer):
            descriptions.append(f"{prefix}：微孔板，厚度 {layer.thickness_m} m、孔徑 {layer.hole_diameter_m} m")
        elif isinstance(layer, PerforatedLayer):
            descriptions.append(f"{prefix}：一般圓孔板，厚度 {layer.thickness_m} m、孔徑 {layer.hole_diameter_m} m")
        elif isinstance(layer, SlottedLayer):
            descriptions.append(f"{prefix}：長縫板，厚度 {layer.thickness_m} m、縫寬 {layer.slot_width_m} m")
        else:
            raise ValueError("不認識的構造層，無法描述；請修正層種類")
    if isinstance(material, MountedOnWall):
        return "；".join(descriptions) + f"；背後負載為原牆（{_context(material.wall)}）"
    return "；".join(descriptions) + f"；背後負載為{'剛背' if material.backing == 'rigid' else '空氣'}"


def _require_precision(value: jax.Array, dtype: str, context: str) -> None:
    if str(value.dtype) != dtype:
        raise ValueError(f"{context}：結果型別為 {value.dtype}，需要雙精度（{dtype}）；請由執行環境層在載入算數函式庫前開啟雙精度")


def _axis(values: jax.Array, *, angles: bool = False) -> jax.Array:
    label = "角度（度）" if angles else "頻率（Hz）"
    array = jnp.asarray(values)
    if isinstance(array, jax.core.Tracer):
        raise ValueError(f"{label}須給具體值，請在即時編譯之外呼叫構造計算")
    if array.ndim != 1 or array.size == 0 or jnp.iscomplexobj(array):
        raise ValueError(f"{label}須為非空一維實數陣列，收到形狀 {array.shape}、型別 {array.dtype}；請修正取樣軸")
    valid = jnp.isfinite(array) & ((array >= 0) & (array < 90) if angles else array > 0)
    if not bool(jnp.all(valid)):
        raise ValueError(f"{label}收到不合法值 {np.asarray(array).tolist()}；請給有限{'0≤角度<90' if angles else '正頻率'}")
    return array.astype(jnp.float64)


def _komatsu_notes(material: SurfaceMaterial, frequencies: jax.Array, ranges: PorousModelRanges) -> tuple[ConstructionNote, ...]:
    # 實測表不經這裡：它的底構造只在表外子軸求值，註記由 _fill_notes 加前綴（_evaluate_impedance 先分流）。
    if isinstance(material, (ConstantImpedance, MeasuredImpedance)):
        return ()
    notes = []
    limits = ranges.komatsu
    lower, upper = limits.f_over_sigma_bounds_m3_kg
    for index, layer in enumerate(material.layers, start=1):
        if not isinstance(layer, PorousLayer) or layer.model != "komatsu":
            continue
        sigma = layer.flow_resistivity_pa_s_m2
        if not limits.manual_flow_resistivity_min_pa_s_m2 <= sigma <= limits.sample_flow_resistivity_max_pa_s_m2:
            raise ValueError(f"第 {index} 層：Komatsu（小松）纖維模型流阻 {sigma} Pa·s/m² 超出手動試算範圍 [{limits.manual_flow_resistivity_min_pa_s_m2}, {limits.sample_flow_resistivity_max_pa_s_m2}]；原研究樣品範圍另記，這不是作者訂的物理有效界線。請改填實測阻抗，或明確指定別的多孔模型；不自動換模型")
        if sigma < limits.sample_flow_resistivity_min_pa_s_m2:
            notes.append(ConstructionNote(code="komatsu_flow_below_samples", layer_index=index,
                material_class=layer.material_class,
                message=f"第 {index} 層：流阻低於原研究樣品範圍，保留輸入 {sigma:g} Pa·s/m² 照算；手動試算下限 {limits.manual_flow_resistivity_min_pa_s_m2:g} 是工程界線，不是經實測證明的準確性界線"))
        outside = tuple(float(f) for f in np.asarray(frequencies) if not lower <= float(f) / sigma <= upper)
        if outside:
            notes.append(ConstructionNote(code="komatsu_frequency_extrapolation", layer_index=index,
                frequencies_hz=outside, material_class=layer.material_class,
                message=f"第 {index} 層：Komatsu（小松）纖維模型的材料模型外推；f/σ（頻率除流阻）超出量測涵蓋區間 [{lower}, {upper}] m³/kg，頻點照算"))
        if layer.material_class != limits.fitted_material_class:
            notes.append(ConstructionNote(code="komatsu_material_class_outside_fit", layer_index=index,
                material_class=layer.material_class, message=f"第 {index} 層：泡棉材料類別不在 Komatsu（小松）纖維模型擬合範圍，照算但不宣稱準確性"))
    return tuple(notes)


def _layer_matrix(layer: Layer, frequencies: jax.Array, air: PhysicsConstants,
                  angles: jax.Array | None) -> jax.Array:
    """唯一層到矩陣入口；連續參數不轉成普通浮點數，保留陣列計算路徑。"""
    if isinstance(layer, PorousLayer):
        # 只轉交模型真的讀的參數；不用未提供值冒充材料物性。
        parameters: _PorousParameters = {}
        if layer.porosity is not None:
            parameters["porosity"] = jnp.asarray(layer.porosity)
        if layer.tortuosity is not None:
            parameters["tortuosity"] = jnp.asarray(layer.tortuosity)
        return tmm.porous_layer(frequencies, jnp.asarray(layer.thickness_m),
            jnp.asarray(layer.flow_resistivity_pa_s_m2), air=air, model=layer.model,
            incidence_angles_deg=angles, **parameters)
    if isinstance(layer, AirLayer):
        return tmm.air_layer(frequencies, jnp.asarray(layer.thickness_m), air=air, incidence_angles_deg=angles)
    if isinstance(layer, BoardLayer):
        impedance = jnp.asarray(layer.resistance_pa_s_m) + 1j * 2 * jnp.pi * frequencies * jnp.asarray(layer.surface_mass_kg_m2)
        return tmm._sheet_layer(tmm._broadcast_angles(impedance, angles))
    if isinstance(layer, FabricLayer):
        return tmm.porous_facing_layer(frequencies, jnp.asarray(FABRIC_THICKNESS_PLACEHOLDER_M), air=air,
            method="pieren_free", surface_mass_kg_m2=jnp.asarray(layer.surface_mass_kg_m2),
            resistance_pa_s_m=jnp.asarray(layer.flow_resistance_pa_s_m), incidence_angles_deg=angles)
    if isinstance(layer, MicroPerforatedLayer):
        return tmm.perforated_panel_layer(frequencies, jnp.asarray(layer.thickness_m), jnp.asarray(layer.hole_diameter_m),
            air=air, open_area=_open_area(layer), surface_mass_kg_m2=_panel_mass(layer), method="barrier_mpp", incidence_angles_deg=angles)
    if isinstance(layer, PerforatedLayer):
        matrix = tmm.perforated_panel_layer(frequencies, jnp.asarray(layer.thickness_m), jnp.asarray(layer.hole_diameter_m),
            air=air, open_area=_open_area(layer), surface_mass_kg_m2=_panel_mass(layer), method="barrier", end_correction="jb", incidence_angles_deg=angles)
        return _backed_sheet(matrix, layer, frequencies, angles)
    if isinstance(layer, SlottedLayer):
        matrix = tmm.slotted_panel_layer(frequencies, jnp.asarray(layer.thickness_m), jnp.asarray(layer.slot_width_m),
            air=air, open_area=_open_area(layer), surface_mass_kg_m2=_panel_mass(layer), method="barrier", incidence_angles_deg=angles)
        return _backed_sheet(matrix, layer, frequencies, angles)
    raise ValueError("不認識的構造層，無法計算；請修正層種類")


def _panel_mass(layer: MicroPerforatedLayer | PerforatedLayer | SlottedLayer) -> jax.Array | None:
    return None if layer.surface_mass_kg_m2 is None else jnp.asarray(layer.surface_mass_kg_m2)


def _open_area(layer: MicroPerforatedLayer | PerforatedLayer | SlottedLayer) -> jax.Array:
    """核心只把算好的開孔率交給傳遞矩陣；不讓它再用方形式孔距推導。"""
    if layer.open_area is not None:
        return jnp.asarray(layer.open_area)
    if isinstance(layer, SlottedLayer):
        return jnp.asarray(layer_open_area(jnp.asarray(layer.slot_width_m), jnp.asarray(layer.slot_spacing_m), "slotted"))
    second = None if layer.hole_spacing_2_m is None else jnp.asarray(layer.hole_spacing_2_m)
    assert layer.hole_pattern is not None
    return jnp.asarray(layer_open_area(jnp.asarray(layer.hole_diameter_m), jnp.asarray(layer.hole_spacing_m), layer.hole_pattern, second))


def _backed_sheet(matrix: jax.Array, layer: PerforatedLayer | SlottedLayer, frequencies: jax.Array,
                  angles: jax.Array | None) -> jax.Array:
    """工程近似：板後串聯布面流阻除板開孔率，布不動，面密度只記錄。"""
    if layer.backing_fabric_flow_resistance_pa_s_m is None:
        return matrix
    impedance = jnp.zeros_like(frequencies, dtype=jnp.complex128) + jnp.asarray(layer.backing_fabric_flow_resistance_pa_s_m) / _open_area(layer)
    backing = tmm._sheet_layer(tmm._broadcast_angles(impedance, angles))
    return matrix @ backing


def _check_surface_impedance(impedance: jax.Array, frequencies: jax.Array, angles: jax.Array | None,
                             *, context: str) -> None:
    """逐頻逐角查最終阻抗；零實部可接受，負實部不裁切也不換模型。"""
    _require_precision(impedance, "complex128", context)
    values = np.asarray(impedance)
    invalid = ~np.isfinite(values) | (values.real < 0)
    if np.any(invalid):
        location = tuple(int(i) for i in np.argwhere(invalid)[0])
        frequency = float(frequencies[location[0]])
        angle = f"、角度 {float(angles[location[1]])} 度" if angles is not None else ""
        value = complex(values[location])
        raise ValueError(f"{context}：最終表面阻抗在頻率 {frequency} Hz{angle} 為 {value} Pa·s/m，實部 {value.real} 非有限或小於零；請檢查各層參數與背後負載，改填實測阻抗或明確指定別的多孔模型；不自動換模型")


def _evaluate_impedance(material: SurfaceMaterial, frequencies_hz: jax.Array, incidence_angles_deg: jax.Array | None,
                        air: PhysicsConstants, ranges: PorousModelRanges) -> tuple[jax.Array, tuple[ConstructionNote, ...]]:
    context = _context(material)
    # 不查全域開關；實際建出的陣列精度不足就拒收，也避免降精度轉型警告。
    _require_precision(jnp.asarray(0.), "float64", context)
    frequencies = _axis(frequencies_hz)
    angles = _axis(incidence_angles_deg, angles=True) if incidence_angles_deg is not None else None
    if isinstance(material, MeasuredImpedance):
        impedance, notes = _measured_impedance(material, frequencies, angles, air, ranges)
        _check_surface_impedance(impedance, frequencies, angles, context=context)
        return impedance, notes
    notes = _komatsu_notes(material, frequencies, ranges)
    if isinstance(material, ConstantImpedance):
        impedance = tmm.constant_z(frequencies, jnp.asarray(material.impedance_pa_s_m, dtype=jnp.complex128), incidence_angles_deg=angles)
    elif isinstance(material, MountedOnWall):
        load, wall_notes = _wall_load(material, frequencies, angles, air, ranges)
        matrices = [_layer_matrix(layer, frequencies, air, angles) for layer in material.layers]
        impedance = tmm.compute(matrices, air=air, load_impedance_pa_s_m=load, incidence_angles_deg=angles)[0]
        notes = (*notes, *wall_notes)
    else:
        matrices = [_layer_matrix(layer, frequencies, air, angles) for layer in material.layers]
        impedance = tmm.compute(matrices, air=air, backing=material.backing, incidence_angles_deg=angles)[0]
    _check_surface_impedance(impedance, frequencies, angles, context=context)
    return impedance, notes


def _wall_load(material: MountedOnWall, frequencies: jax.Array, angles: jax.Array | None, air: PhysicsConstants,
               ranges: PorousModelRanges) -> tuple[jax.Array, tuple[ConstructionNote, ...]]:
    """原牆在同一頻率軸、同一組角度的表面阻抗，當新增層的有限負載；拒收訊息與註記加「原牆」前綴。

    固定阻抗與實測表照局部反應攤成角度形狀，多層構造用它自己的角度求；同一個入口、同一套檢查。
    """
    try:
        impedance, notes = _evaluate_impedance(material.wall, frequencies, angles, air, ranges)
    except ValueError as exc:
        raise ValueError(f"原牆{exc}") from exc
    return impedance, tuple(note.model_copy(update={"message": "原牆" + note.message}) for note in notes)


def surface_impedance(material: SurfaceMaterial, frequencies_hz: jax.Array, *, air: PhysicsConstants,
                      ranges: PorousModelRanges) -> tuple[jax.Array, tuple[ConstructionNote, ...]]:
    """垂直入射表面阻抗：一維頻率 [Hz] → 雙精度複數陣列 (F,) [Pa·s/m] 與註記。"""
    return _evaluate_impedance(material, frequencies_hz, None, air, ranges)


def surface_impedance_at_angles(material: SurfaceMaterial, frequencies_hz: jax.Array, incidence_angles_deg: jax.Array,
                                *, air: PhysicsConstants, ranges: PorousModelRanges) -> tuple[jax.Array, tuple[ConstructionNote, ...]]:
    """各角度表面阻抗：一維頻率 [Hz]、一維角度 [度] → 雙精度複數陣列 (F,Nθ) 與註記。"""
    return _evaluate_impedance(material, frequencies_hz, incidence_angles_deg, air, ranges)


def random_incidence_absorption(material: SurfaceMaterial, frequencies_hz: jax.Array, *, air: PhysicsConstants,
                                ranges: PorousModelRanges) -> tuple[jax.Array, tuple[ConstructionNote, ...]]:
    """無規入射吸音：Paris（角度加權平均）預設完整角度格，固定阻抗沿用既有型錄換算。"""
    if isinstance(material, MeasuredImpedance):
        absorption, notes = _measured_absorption(material, frequencies_hz, air, ranges)
    elif isinstance(material, ConstantImpedance):
        impedance, notes = surface_impedance(material, frequencies_hz, air=air, ranges=ranges)
        alpha = complex_random_incidence_absorption(material.impedance_pa_s_m / air.rho_c)
        absorption = jnp.full(impedance.shape, alpha)
    else:
        _require_precision(jnp.asarray(0.), "float64", _context(material))
        angles = tmm.diffuse_incidence_angles_deg(method="paris")
        impedance, notes = surface_impedance_at_angles(material, frequencies_hz, angles, air=air, ranges=ranges)
        absorption = tmm.diffuse_absorption_coefficient(impedance, angles, air=air)
    _require_precision(absorption, "float64", _context(material))
    return absorption, notes


def _outside_table(material: MeasuredImpedance, frequencies: jax.Array) -> np.ndarray:
    """嚴格表外遮罩；端點屬於實測段，不重算底構造。"""
    axis = np.asarray(frequencies)
    return (axis < material.points[0].frequency_hz) | (axis > material.points[-1].frequency_hz)


def _table_impedance(material: MeasuredImpedance, frequencies: jax.Array) -> jax.Array:
    """對數頻率線性內插；明確覆寫命中列，包含端點，逐位保留輸入表值。"""
    axis = np.asarray(frequencies)
    knots = np.asarray([point.frequency_hz for point in material.points])
    real = np.asarray([point.real_pa_s_m for point in material.points])
    imag = np.asarray([point.imag_pa_s_m for point in material.points])
    values = np.empty(axis.shape, dtype=np.complex128)
    values.real = np.interp(np.log10(axis), np.log10(knots), real)
    values.imag = np.interp(np.log10(axis), np.log10(knots), imag)
    for point in material.points:
        values[axis == point.frequency_hz] = complex(point.real_pa_s_m, point.imag_pa_s_m)
    return jnp.asarray(values)


def _fill_notes(material: MeasuredImpedance, frequencies: jax.Array,
                notes: tuple[ConstructionNote, ...]) -> tuple[ConstructionNote, ...]:
    """只記實際求值的表外頻點；底構造訊息加前綴，原構造訊息不改。"""
    if not frequencies.size:
        return ()
    fill = ConstructionNote(code="measured_table_model_fill", frequencies_hz=tuple(sorted(set(np.asarray(frequencies).tolist()))),
        message=f"模型補值：實測表涵蓋 {material.points[0].frequency_hz:g}–{material.points[-1].frequency_hz:g} Hz，表外以底構造計算")
    return (fill, *(note.model_copy(update={"message": "底構造" + note.message}) for note in notes))


def _measured_impedance(material: MeasuredImpedance, frequencies: jax.Array, angles: jax.Array | None,
                        air: PhysicsConstants, ranges: PorousModelRanges) -> tuple[jax.Array, tuple[ConstructionNote, ...]]:
    impedance = tmm._broadcast_angles(_table_impedance(material, frequencies), angles)
    indices = np.flatnonzero(_outside_table(material, frequencies))
    if not indices.size:
        return impedance, ()
    outside = frequencies[indices]
    try:
        values, notes = _evaluate_impedance(material.base_assembly, outside, angles, air, ranges)
    except ValueError as exc:
        raise ValueError(f"{material.coverage_text()}；底構造{exc}") from exc
    return impedance.at[indices].set(values), _fill_notes(material, outside, notes)


def _measured_absorption(material: MeasuredImpedance, frequencies_hz: jax.Array, air: PhysicsConstants,
                         ranges: PorousModelRanges) -> tuple[jax.Array, tuple[ConstructionNote, ...]]:
    _require_precision(jnp.asarray(0.), "float64", _context(material))
    frequencies = _axis(frequencies_hz)
    outside = _outside_table(material, frequencies)
    inside_indices, outside_indices = np.flatnonzero(~outside), np.flatnonzero(outside)
    result = jnp.zeros(frequencies.shape, dtype=jnp.float64)
    values = _table_impedance(material, frequencies[inside_indices])
    absorption = jnp.asarray([complex_random_incidence_absorption(complex(z) / air.rho_c) for z in np.asarray(values)], dtype=jnp.float64)
    result = result.at[inside_indices].set(absorption)
    if not outside_indices.size:
        return result, ()
    try:
        base, notes = random_incidence_absorption(material.base_assembly, frequencies[outside_indices], air=air, ranges=ranges)
    except ValueError as exc:
        raise ValueError(f"{material.coverage_text()}；底構造{exc}") from exc
    return result.at[outside_indices].set(base), _fill_notes(material, frequencies[outside_indices], notes)
