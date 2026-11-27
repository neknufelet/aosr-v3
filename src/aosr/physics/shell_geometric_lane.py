"""自訂房間（房間殼）的高音那一路：鏡像法早期反射＋晚期能量交換，接成跟長方體同樣的報表四欄。

**兩步。** :func:`prepare_shell_acoustics` 每間房（每組喇叭、聽位）算一次，只看幾何：面、
路徑表（含遮擋）、晚期分格與形狀因子。:func:`solve_shell_geometric_lane` 每個候選算一次，
只吃各材料圖層的阻抗與散射係數，所以最佳化時不會重算幾何。

**四欄的定義跟長方體那一路相同**（:mod:`aosr.physics.geometric_lane` 模組說明的那一條式子）：
直達 ``|p_d|²``、反射 ``|Σ_k p̃_k|²``、干涉 ``2·Re(p_d·conj(Σ_k p̃_k))``、晚期
``Σ_{k≤K}[1−(1−s̄)^k]·A_k + 尾巴``。差別只在散射跟著材料走：

- 鏡像法那一邊每條路徑乘 ``Π_b √(1−s_b)``——每一次反彈用打到那一面材料自己的散射係數；
  全部材料同一個 s 時就是長方體那一路的 ``(1−s)^{k/2}``。
- 晚期那一邊的 ``s̄`` 用長方體那一路同一條加權（面積 × 垂直入射反射能量）把各材料的 s 合成一個；
  ``A_k`` 是整間房平均的逐階能量，分不出是哪幾面，所以只能用合成的那一個（已知近似）。

材料沒填散射係數就用 0.1（:data:`aosr.materials.scattering_defaults.MATERIAL_SCATTERING_DEFAULT_S`）。
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from aosr.config.room_shell_import import RoomShellImportSettings
from aosr.config.shell_lane import SHELL_FORM_FACTOR_SPLITS, SHELL_GEOMETRY_TOL_M, SHELL_PATCH_EDGE_M
from aosr.config.three_lane_crossover import REFLECTION_ORDER_K
from aosr.geometry.room_shell import RoomShell, Vec3
from aosr.geometry.shell_faces import FloatArray, ShellFaces, shell_faces
from aosr.geometry.shoebox import Point
from aosr.materials.scattering_defaults import MATERIAL_SCATTERING_DEFAULT_S
from aosr.physics.geometric_lane import (
    GeometricLaneResult,
    _geometric_energy,
    _interference_energy,
    _late_share_energy,
    _reflected_energy,
)
from aosr.physics.report_source import SourceModelSpec, directivity_to_apply
from aosr.physics.shell_late_energy import ShellLateGeometry, shell_late_geometry, solve_shell_late_energy_by_order
from aosr.physics.shell_paths import ComplexArray, ShellPathTable, path_pressures, shell_path_table
from aosr.physics.source_directivity import SourceModel, pressure_factor_for_direction, two_parameter_power_ratio
from aosr.physics.wall_surfaces import impedance_absorption_rows

LayerImpedance = complex | float | Sequence[complex]
LayerScattering = float | Sequence[float]


@dataclass(frozen=True)
class ShellAcoustics:
    """一間房、一組喇叭與聽位只看幾何的準備：面、路徑表、晚期分格。跟材料無關，可以重複用。"""

    faces: ShellFaces
    paths: ShellPathTable
    late: ShellLateGeometry

    @property
    def reflection_order_k(self) -> int:
        """路徑表算到第幾階。"""
        return self.paths.max_order


def prepare_shell_acoustics(
    room: RoomShell,
    source: Vec3,
    receiver: Vec3,
    settings: RoomShellImportSettings,
    *,
    reflection_order_k: int = REFLECTION_ORDER_K,
    patch_edge_m: float = SHELL_PATCH_EDGE_M,
    splits: int = SHELL_FORM_FACTOR_SPLITS,
    tol_m: float = SHELL_GEOMETRY_TOL_M,
) -> ShellAcoustics:
    """每間房一次：同平面合面（用匯入檢查同一組共平面界線）、路徑表、晚期分格與形狀因子。"""
    faces = shell_faces(room, settings.coplanar_angle_deg, settings.coplanar_offset_m, tol_m)
    paths = shell_path_table(faces, source, receiver, max_order=reflection_order_k, tol_m=tol_m)
    late = shell_late_geometry(faces, patch_edge_m=patch_edge_m, splits=splits, tol_m=tol_m)
    return ShellAcoustics(faces, paths, late)


def _rows(values: Mapping[str, LayerImpedance] | Mapping[str, LayerScattering], layers: tuple[str, ...],
          count: int, what: str, default: float | None) -> list[list[complex]]:
    """每個圖層一列、每個頻點一格；單一值照頻點數排開。沒填的圖層用預設，沒有預設就報出缺哪些。"""
    missing = [layer for layer in layers if layer not in values]
    if missing and default is None:
        raise ValueError(f"這些材料圖層還沒有{what}：{'、'.join(missing)}")
    rows: list[list[complex]] = []
    for layer in layers:
        value = values.get(layer, default)
        if value is None:
            raise ValueError(f"圖層「{layer}」的{what}是空的")
        cells = np.ravel(np.asarray(value))
        row = [complex(cells[0])] * count if np.ndim(value) == 0 else [complex(v) for v in cells]
        if len(row) != count:
            raise ValueError(f"圖層「{layer}」的{what}有 {len(row)} 格，頻點有 {count} 個")
        rows.append(row)
    return rows


def _impedance_table(prepared: ShellAcoustics, impedance_by_layer: Mapping[str, LayerImpedance],
                     count: int) -> ComplexArray:
    table = np.asarray(_rows(impedance_by_layer, prepared.faces.layers, count, "阻抗", None), dtype=np.complex128)
    if not np.isfinite(table).all():
        raise ValueError("阻抗必須是有限數")
    return table


def _scattering_table(prepared: ShellAcoustics, scattering_by_layer: Mapping[str, LayerScattering] | None,
                      count: int) -> FloatArray:
    rows = _rows(scattering_by_layer or {}, prepared.faces.layers, count, "散射係數", MATERIAL_SCATTERING_DEFAULT_S)
    table = np.asarray([[value.real for value in row] for row in rows], dtype=np.float64)
    if (any(value.imag != 0.0 for row in rows for value in row) or not np.isfinite(table).all()
            or bool(((table < 0) | (table > 1)).any())):
        raise ValueError("散射係數必須是 0 到 1 的實數")
    return table


def _room_scattering(areas: FloatArray, impedance: ComplexArray, scattering: FloatArray, rho_c: float) -> FloatArray:
    """同長方體那一路 ``_scattering_at`` 的加權與判斷順序：權重和為 0 回 0、全部一樣回那一個值、其餘照
    面積 × 垂直入射反射能量加權。面積用晚期那邊留下來的（被板子蓋住、看不到的不算）。"""
    weights = areas[:, None] * np.abs((impedance - rho_c) / (impedance + rho_c)) ** 2
    denominator = weights.sum(axis=0)
    blended = np.divide((weights * scattering).sum(axis=0), denominator,
                        out=np.zeros_like(denominator), where=denominator != 0.0)
    same = (scattering == scattering[:1]).all(axis=0)
    chosen = np.where(same, scattering[0], blended)
    return np.asarray(np.where(denominator == 0.0, 0.0, chosen), dtype=np.float64)


def _retained_by_path(paths: ShellPathTable, scattering: FloatArray) -> FloatArray:
    """每條路徑每個頻點留在鏡面的壓力比例 ``Π_b √(1−s_b)``；直達是 1。"""
    kept = np.ones((paths.path_count, scattering.shape[1]), dtype=np.float64)
    for bounce in range(paths.max_order):
        hit = paths.bounce_layer[:, bounce] >= 0
        kept[hit] *= 1.0 - scattering[paths.bounce_layer[hit, bounce]]
    return np.asarray(np.sqrt(kept), dtype=np.float64)


def _directivity(prepared: ShellAcoustics, source_model: SourceModelSpec,
                 frequencies_hz: tuple[float, ...]) -> tuple[FloatArray, FloatArray]:
    """（每條路徑每個頻點的聲壓倍率, 每個頻點的功率比）；全向兩者都是 1。"""
    paths = prepared.paths
    chosen = directivity_to_apply(source_model)
    if chosen is None:
        return np.ones((paths.path_count, len(frequencies_hz))), np.ones(len(frequencies_hz))
    curve, aim = chosen
    source = Point(*paths.source)
    factors = np.stack([
        pressure_factor_for_direction(tuple(float(x) for x in direction), frequencies_hz, SourceModel.TWO_PARAMETER,
                                      source=source, aim=aim, params=curve)
        for direction in paths.departure
    ])
    return np.asarray(factors, dtype=np.float64), np.asarray(two_parameter_power_ratio(frequencies_hz, curve))


def solve_shell_geometric_lane(
    *,
    prepared: ShellAcoustics,
    source_model: SourceModelSpec,
    sound_speed_m_s: float,
    rho_c_pa_s_per_m: float,
    frequencies_hz: tuple[float, ...],
    impedance_by_layer: Mapping[str, LayerImpedance],
    scattering_by_layer: Mapping[str, LayerScattering] | None = None,
) -> GeometricLaneResult:
    """每個候選一次：各材料圖層的阻抗與散射 → 報表四欄與幾何能量（細軸逐頻）。"""
    if not frequencies_hz:
        raise ValueError("frequencies_hz 不可為空")
    count = len(frequencies_hz)
    impedance = _impedance_table(prepared, impedance_by_layer, count)
    scattering = _scattering_table(prepared, scattering_by_layer, count)
    paths = prepared.paths
    pressure = path_pressures(paths, impedance, rho_c_pa_s_per_m, sound_speed_m_s, np.asarray(frequencies_hz))
    factors, power_ratio = _directivity(prepared, source_model, frequencies_hz)
    pressure = pressure * factors * _retained_by_path(paths, scattering)
    direct = tuple(complex(value) for value in pressure[0])
    reflected = tuple(complex(value) for value in pressure[1:].sum(axis=0))
    absorption_rows = impedance_absorption_rows(
        {layer: tuple(complex(z) for z in row) for layer, row in zip(prepared.faces.layers, impedance, strict=True)},
        rho_c_pa_s_per_m)
    absorption = np.asarray([absorption_rows[layer] for layer in prepared.faces.layers], dtype=np.float64)
    late_result = solve_shell_late_energy_by_order(prepared.late, absorption, frequencies_hz,
                                                   max_order=paths.max_order)
    room_s = _room_scattering(prepared.late.layer_areas(), impedance, scattering, rho_c_pa_s_per_m)
    s_tuple = tuple(float(value) for value in room_s)
    late = tuple(value * float(ratio) for value, ratio in
                 zip(_late_share_energy(late_result.bands, s_tuple), power_ratio, strict=True))
    direct_energy = tuple(abs(value) ** 2 for value in direct)
    reflected_energy = _reflected_energy(reflected)
    interference = _interference_energy(direct, reflected)
    return GeometricLaneResult(
        frequencies_hz=frequencies_hz,
        direct_energy=direct_energy,
        reflected_energy=reflected_energy,
        interference_energy=interference,
        late_energy=late,
        scattering=s_tuple,
        geometric_energy=_geometric_energy(direct_energy, reflected_energy, interference, late),
        reflection_order_k=paths.max_order,
        source_model=source_model,
    )
