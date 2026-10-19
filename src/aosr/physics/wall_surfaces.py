"""有限元素與晚期家族唯一的牆面響應來源；每條頻率軸直接求值，不內插。

固定阻抗只用普通複數與既有 Paris（無規入射積分）算式，不建立 JAX 陣列。
構造轉交材料核心，保留行程雙精度拒收與逐面 ConstructionNote（構造註記）。
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from aosr.config.construction_environment import construction_environment
from aosr.config.paths import config_path
from aosr.config.porous_model_ranges import load_porous_model_ranges
from aosr.geometry.shoebox import Wall
from aosr.materials.catalog_absorption import complex_random_incidence_absorption
from aosr.materials.construction_models import Assembly, ConstantImpedance, ConstructionNote, MeasuredImpedance, SurfaceMaterial


@dataclass(frozen=True)
class WallSurfaceResponse:
    """逐面逐頻吸音率與註記；頻率順序與呼叫端的取樣軸一致。"""

    frequencies_hz: tuple[float, ...]
    absorption_by_wall: dict[Wall, tuple[float, ...]]
    notes_by_wall: dict[Wall, tuple[ConstructionNote, ...]]

    def named_rows(self, frequencies_hz: tuple[float, ...]) -> dict[str, tuple[float, ...]]:
        """按精確頻點選列，交給晚期能量或衰減，不做內插。"""
        indices = {f: i for i, f in enumerate(self.frequencies_hz)}
        return {wall.wall_name(): tuple(row[indices[f]] for f in frequencies_hz)
                for wall, row in self.absorption_by_wall.items()}

    def band_rows(self, frequencies_hz: tuple[float, ...]) -> dict[Wall, dict[float, float]]:
        """以帶中心取值，交給逐帶艾林公式。"""
        return {Wall.from_name(wall): dict(zip(frequencies_hz, row, strict=True))
                for wall, row in self.named_rows(frequencies_hz).items()}


def impedance_absorption_rows(
    impedance_by_wall: Mapping[str, Sequence[complex]], rho_c_pa_s_per_m: float,
) -> dict[str, tuple[float, ...]]:
    """舊答案與固定阻抗共用的換算；保持 complex(Z)/ρc 的逐位精度契約。"""
    if not math.isfinite(rho_c_pa_s_per_m) or rho_c_pa_s_per_m <= 0:
        raise ValueError("rho_c 必須是有限正數")
    return {wall: tuple(complex_random_incidence_absorption(complex(z) / rho_c_pa_s_per_m)
                        for z in row) for wall, row in impedance_by_wall.items()}


def impedance_materials_code(impedance_by_wall: Mapping[str, Sequence[complex]]) -> str:
    """保留舊載入器逐面判斷次序與既有材料代號；吸音率不能反推出阻抗形式。"""
    for row in impedance_by_wall.values():
        if any(value.imag != 0.0 for value in row):
            return "complex_impedance_by_wall"
        if len(set(row)) > 1:
            return "frequency_dependent_impedance"
    return "real_frequency_independent_impedance"


def merge_construction_notes(
    into: dict[Wall, tuple[ConstructionNote, ...]] | None,
    wall: Wall, notes: tuple[ConstructionNote, ...],
) -> None:
    """保留當次材料註記物件，同面同層同代號只合併實際求值的頻點。"""
    if into is None or not notes:
        return
    merged = {(note.layer_index, note.code): note for note in into.get(wall, ())}
    for note in notes:
        key = note.layer_index, note.code
        previous = merged.get(key)
        frequencies = tuple(sorted(set(note.frequencies_hz) |
                                   set(() if previous is None else previous.frequencies_hz)))
        merged[key] = note if frequencies == note.frequencies_hz else note.model_copy(
            update={"frequencies_hz": frequencies})
    into[wall] = tuple(merged[key] for key in sorted(merged, key=lambda key: (key[0] or 0, key[1])))


def extrapolated_frequency_text(frequencies_hz: tuple[float, ...], *,
                                table_bounds_hz: tuple[float, float] | None = None) -> str:
    """外推頻點寫成低、高兩段（「20–27 Hz、8400–11313.5 Hz」），中間照常算的頻率不連進去。

    Komatsu 的 f/σ 界線由量測頻率與樣品流阻推導，手動流阻另允許低於樣品下限的工程試算。
    外推窗仍由樣品範圍推得；低段在量測頻率下限以下，高段可能低於量測上限，仍用下限切成兩段。
    """
    if table_bounds_hz is not None:
        lower, upper = table_bounds_hz
        spans = [part for part in (tuple(f for f in frequencies_hz if f < lower),
                                   tuple(f for f in frequencies_hz if f > upper)) if part]
        return "、".join(f"{part[0]:g} Hz" if part[0] == part[-1] else f"{part[0]:g}–{part[-1]:g} Hz"
                        for part in spans)
    lowest = load_porous_model_ranges(config_path("porous_model_ranges.toml")).komatsu.measured_frequency_min_hz
    spans = [part for part in (tuple(f for f in frequencies_hz if f < lowest),
                               tuple(f for f in frequencies_hz if f >= lowest)) if part]
    return "、".join(f"{part[0]:g} Hz" if part[0] == part[-1] else f"{part[0]:g}–{part[-1]:g} Hz"
                    for part in spans)


def wall_surface_response(
    materials_by_wall: Mapping[Wall, SurfaceMaterial], frequencies_hz: tuple[float, ...],
    *, density_kg_m3: float, sound_speed_m_s: float,
) -> WallSurfaceResponse:
    """六面材料直接求逐頻無規吸音；構造用場景ρ、c，其餘空氣常數讀產品設定。"""
    if set(materials_by_wall) != set(Wall.all()):
        raise ValueError("materials_by_wall 必須恰好包含六面牆")
    if not frequencies_hz or any(not math.isfinite(f) or f <= 0 for f in frequencies_hz):
        raise ValueError("頻率軸必須非空且全為有限正數")
    if any(not math.isfinite(v) or v <= 0 for v in (density_kg_m3, sound_speed_m_s)):
        raise ValueError("場景密度與聲速必須是有限正數")
    alpha: dict[Wall, tuple[float, ...]] = {}
    notes: dict[Wall, tuple[ConstructionNote, ...]] = {}
    environment = None
    for wall in Wall.all():
        material = materials_by_wall[wall]
        if isinstance(material, ConstantImpedance):
            row = impedance_absorption_rows({wall.wall_name(): (complex(material.impedance_pa_s_m),)},
                                           density_kg_m3 * sound_speed_m_s)[wall.wall_name()]
            alpha[wall], notes[wall] = row * len(frequencies_hz), ()
        else:
            # 只有構造分支載入算數函式庫；固定阻抗不呼叫任何 jnp 函式。
            import jax.numpy as jnp
            from aosr.materials.construction import random_incidence_absorption

            if environment is None:
                environment = construction_environment(density_kg_m3=density_kg_m3, sound_speed_m_s=sound_speed_m_s)
            air, ranges = environment
            try:
                values, notes[wall] = random_incidence_absorption(material, jnp.asarray(frequencies_hz),
                                                                 air=air, ranges=ranges)
            except ValueError as exc:
                if "需要雙精度" in str(exc):
                    raise ValueError(f"{wall.wall_name()}：行程的 JAX 精度設定不足；{exc}") from exc
                if isinstance(material, MeasuredImpedance):
                    # 既有構造牆的訊息不動；實測牆的底構造可能第一次在這條軸求值，要說出牆名。
                    raise ValueError(f"{wall.wall_name()}：{exc}") from exc
                raise
            alpha[wall] = tuple(float(value) for value in values)
    return WallSurfaceResponse(frequencies_hz, alpha, notes)


def wall_normal_impedances(
    materials_by_wall: Mapping[Wall, SurfaceMaterial | None], frequencies_hz: tuple[float, ...],
    *, density_kg_m3: float, sound_speed_m_s: float,
    notes_by_wall: dict[Wall, tuple[ConstructionNote, ...]] | None = None,
) -> dict[Wall, float | None | tuple[complex, ...]]:
    """六面垂直入射 Z(f,0°)：固定值保留原 float，None 表整軸剛性。

    構造直接在此軸雙精度求值、不內插；密度與聲速來自場景，其餘空氣常數來自產品設定。
    """
    if set(materials_by_wall) != set(Wall.all()):
        raise ValueError("materials_by_wall 必須恰好包含六面牆")
    if not frequencies_hz or any(not math.isfinite(f) or f <= 0 for f in frequencies_hz):
        raise ValueError("頻率軸必須非空且全為有限正數")
    if any(not math.isfinite(v) or v <= 0 for v in (density_kg_m3, sound_speed_m_s)):
        raise ValueError("場景密度與聲速必須是有限正數")
    result: dict[Wall, float | None | tuple[complex, ...]] = {}
    environment = None
    for wall in Wall.all():
        material = materials_by_wall[wall]
        if material is None:
            result[wall] = None
        elif isinstance(material, ConstantImpedance):
            result[wall] = material.impedance_pa_s_m
        else:
            import jax.numpy as jnp
            from aosr.materials.construction import surface_impedance

            if environment is None:
                environment = construction_environment(density_kg_m3=density_kg_m3, sound_speed_m_s=sound_speed_m_s)
            air, ranges = environment
            try:
                values, notes = surface_impedance(material, jnp.asarray(np.asarray(frequencies_hz, dtype=np.float64)),
                                                   air=air, ranges=ranges)
            except ValueError as exc:
                raise ValueError(f"{wall.wall_name()}：{exc}") from exc
            merge_construction_notes(notes_by_wall, wall, notes)
            # 先整排轉成 numpy 再交出 Python 複數：位元相同，免得一格一格向 JAX 取值。
            result[wall] = tuple(np.asarray(values, dtype=np.complex128).tolist())
    return result


def wall_path_impedances(
    material: SurfaceMaterial, frequencies_hz: tuple[float, ...], cosines: tuple[float, ...],
    *, wall_name: str, density_kg_m3: float, sound_speed_m_s: float,
    notes_by_wall: dict[Wall, tuple[ConstructionNote, ...]] | None = None,
) -> dict[float, NDArray[np.complex128]]:
    """已驗反射來源的單面逐路徑 Z；構造核心仍只從本模組的延後匯入分支進來。

    固定材料不碰 JAX（陣列算數函式庫）；構造直接求角，補到八欄桶、補欄結果丟掉。
    """
    if isinstance(material, ConstantImpedance):
        row = np.full(len(frequencies_hz), complex(material.impedance_pa_s_m), dtype=np.complex128)
        return {cos: row for cos in cosines}
    import jax.numpy as jnp
    from aosr.materials.construction import surface_impedance_at_angles

    angles = tuple(math.degrees(math.acos(cos)) for cos in cosines)
    padded = angles + (angles[-1],) * (-len(angles) % 8)
    air, ranges = construction_environment(density_kg_m3=density_kg_m3, sound_speed_m_s=sound_speed_m_s)
    try:
        values, notes = surface_impedance_at_angles(material, jnp.asarray(np.asarray(frequencies_hz, dtype=np.float64)),
                                                    jnp.asarray(padded), air=air, ranges=ranges)
    except ValueError as exc:
        raise ValueError(f"{wall_name}：{exc}") from exc
    merge_construction_notes(notes_by_wall, Wall.from_name(wall_name), notes)
    array = np.asarray(values, dtype=np.complex128)
    return {cos: array[:, i] for i, cos in enumerate(cosines)}


def assembly_endpoint_responses(
    material: Assembly, frequencies_hz: tuple[float, ...], *, density_kg_m3: float, sound_speed_m_s: float,
) -> tuple[tuple[complex, ...], tuple[float, ...], float]:
    """實測表接縫用：底構造在指定頻點的 Z(f,0°)、Paris 無規入射 α 與場景 ρc。

    構造核心仍只從本模組的延後匯入分支進來（TMM 進物理閉包只准這一個入口）。
    """
    import jax.numpy as jnp
    from aosr.materials.construction import random_incidence_absorption, surface_impedance

    air, ranges = construction_environment(density_kg_m3=density_kg_m3, sound_speed_m_s=sound_speed_m_s)
    axis = jnp.asarray(np.asarray(frequencies_hz, dtype=np.float64))
    normal, _ = surface_impedance(material, axis, air=air, ranges=ranges)
    diffuse, _ = random_incidence_absorption(material, axis, air=air, ranges=ranges)
    return (tuple(np.asarray(normal, dtype=np.complex128).tolist()),
            tuple(float(value) for value in np.asarray(diffuse, dtype=np.float64)), air.rho_c)
