"""牆面構造計算核心：給具體值、在即時編譯之外驗證，不接進三路。

時間慣例沿用傳遞矩陣 e^{+jωt}（正時間指數），層從房間側往負載連乘。
Komatsu（纖維多孔模型）依注入設定拒收樣品流阻範圍外值，按 f/σ 記頻率外推，
泡棉另記類別不在擬合範圍；樣品範圍不是作者訂的物理有效界線。
Miki（纖維經驗模型）、Wilson 2015（鬆弛模型）、Mechel 1976（低頻延伸模型）尚未設範圍。
所有入口檢查最終表面阻抗的每個頻點、每個角度；非有限或實部負值拒收，沒有模型備援。

結果是 JAX（陣列算數函式庫）的 complex128（雙精度複數）或 float64（雙精度實數）。
本模組只查陣列型別，不讀寫全域設定。執行環境層須在載入陣列函式庫前開雙精度。
三個入口共用 _layer_matrix（層轉矩陣函式），集中連續物理參數，留給後續梯度工作。
固定阻抗的無規吸音沿用型錄既有換算；該函式實際使用高階積分，這裡不另造閉式。
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
from aosr.materials.construction_models import AirLayer, ConstantImpedance, ConstructionNote, Layer, PorousLayer, SurfaceMaterial


class _PorousParameters(TypedDict):
    porosity: NotRequired[jax.Array]
    tortuosity: NotRequired[jax.Array]


def _context(material: SurfaceMaterial) -> str:
    """表面結果不能歸罪單層；列出完整構造，讓拒收訊息可定位輸入。"""
    if isinstance(material, ConstantImpedance):
        return f"固定阻抗（整面）{material.impedance_pa_s_m} Pa·s/m"
    descriptions = []
    for index, layer in enumerate(material.layers, start=1):
        prefix = f"第 {index} 層"
        if isinstance(layer, PorousLayer):
            descriptions.append(f"{prefix}：{layer.model} 多孔模型，厚度 {layer.thickness_m} m、流阻 {layer.flow_resistivity_pa_s_m2} Pa·s/m²、孔隙率 {layer.porosity}、曲折度 {layer.tortuosity}")
        elif isinstance(layer, AirLayer):
            descriptions.append(f"{prefix}：空氣層，厚度 {layer.thickness_m} m")
        else:
            descriptions.append(f"{prefix}：板層，面密度 {layer.surface_mass_kg_m2} kg/m²、面阻力 {layer.resistance_pa_s_m} Pa·s/m")
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
    if isinstance(material, ConstantImpedance):
        return ()
    notes = []
    limits = ranges.komatsu
    lower, upper = limits.f_over_sigma_bounds_m3_kg
    for index, layer in enumerate(material.layers, start=1):
        if not isinstance(layer, PorousLayer) or layer.model != "komatsu":
            continue
        sigma = layer.flow_resistivity_pa_s_m2
        if not limits.sample_flow_resistivity_min_pa_s_m2 <= sigma <= limits.sample_flow_resistivity_max_pa_s_m2:
            raise ValueError(f"第 {index} 層：Komatsu（小松）纖維模型流阻 {sigma} Pa·s/m² 超出原研究樣品範圍 [{limits.sample_flow_resistivity_min_pa_s_m2}, {limits.sample_flow_resistivity_max_pa_s_m2}]；這不是作者訂的物理有效界線。請改填實測阻抗，或明確指定別的多孔模型；不自動換模型")
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
    impedance = layer.resistance_pa_s_m + 1j * 2 * jnp.pi * frequencies * layer.surface_mass_kg_m2
    return tmm._sheet_layer(tmm._broadcast_angles(impedance, angles))


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
    notes = _komatsu_notes(material, frequencies, ranges)
    if isinstance(material, ConstantImpedance):
        impedance = tmm.constant_z(frequencies, jnp.asarray(material.impedance_pa_s_m, dtype=jnp.complex128), incidence_angles_deg=angles)
    else:
        matrices = [_layer_matrix(layer, frequencies, air, angles) for layer in material.layers]
        impedance = tmm.compute(matrices, air=air, backing=material.backing, incidence_angles_deg=angles)[0]
    _check_surface_impedance(impedance, frequencies, angles, context=context)
    return impedance, notes


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
    if isinstance(material, ConstantImpedance):
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
