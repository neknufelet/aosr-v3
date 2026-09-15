"""鞋盒房間晚期混響能量的雙精度精確解。

名字用 ``late_energy``，因為它描述這段交付的物理輸出，不把上一代實作名 ART
變成新家的公開概念。每個 patch 由複數阻抗算 Paris 無規入射吸音率；中心點形狀
因子的交換矩陣先對稱化，再以雙邊對稱縮放同時守面積加權互易與列和。逐頻以
:func:`numpy.linalg.solve` 直接解 ``(I-R)y=E``，再算 reflected-only 的 ``Ry``。
線性解失敗時例外原樣往外丟，不截尾、不迭代，也不換解法。
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from aosr.config.art_lane import guard_art_patch_count
from aosr.config.source_reference import DIFFUSE_MONOPOLE_4PI
from aosr.geometry.shoebox import Room, Wall
from aosr.materials.catalog_absorption import complex_random_incidence_absorption


# 精度契約門檻不住這裡：晚期能量對上一代、五項物理性質兩條界線都住在
# ``blueprint/precision_contracts.toml``，由呼叫端讀進來當參數（#312）。


@dataclass(frozen=True)
class LateEnergyInputs:
    """足以重建一組鞋盒晚期能量題目的輸入，不含答案能量。"""

    room: Room
    rho_c_pa_s_per_m: float
    frequencies_hz: tuple[float, ...]
    impedance_by_wall: dict[str, tuple[complex, ...]]
    n_per_wall: int
    domain_alpha_bar_max: float


@dataclass(frozen=True)
class LateEnergyBand:
    """單一頻帶的吸收率、中介量與修正後晚期能量。"""

    frequency_hz: float
    alpha_by_wall: dict[str, float]
    alpha_bar: float
    in_domain: bool
    raw_reverberant_energy: float
    eyring_ratio: float
    late_reverberant_energy: float


@dataclass(frozen=True)
class LateEnergyResult:
    """同一組材料所有頻帶的晚期能量。"""

    bands: tuple[LateEnergyBand, ...]


@dataclass(frozen=True)
class LateEnergyBandJudgment:
    """單一頻帶相對上一代答案的第二類相容紀錄。"""

    frequency_hz: float
    actual_energy: float
    expected_energy: float
    absolute_difference: float
    allowed_difference: float
    relative_difference: float
    contract_fraction: float
    within_contract: bool


@dataclass(frozen=True)
class LateEnergyContractReport:
    """一組材料逐頻帶的上一代差距；是否在舊界內只供閱讀。"""

    points: tuple[LateEnergyBandJudgment, ...]
    tolerance_rel: float

    @property
    def max_relative_difference(self) -> float:
        """全部頻帶中最大的相對差。"""
        return max(point.relative_difference for point in self.points)

    @property
    def max_contract_fraction(self) -> float:
        """全部頻帶中用掉契約界線最多的比例。"""
        return max(point.contract_fraction for point in self.points)

    @property
    def within_contract(self) -> bool:
        """只有所有頻帶都沒有越界才為真。"""
        return all(point.within_contract for point in self.points)


@dataclass(frozen=True)
class _Patches:
    """六面等分格的中心、面積、內法向量與牆索引。"""

    centroids: NDArray[np.float64]
    areas: NDArray[np.float64]
    normals: NDArray[np.float64]
    wall_index: NDArray[np.int64]


@dataclass(frozen=True)
class _ReflectionProblem:
    """晚期能量與晚期衰減共用的分格、吸收率與反射算子。"""

    patches: _Patches
    alpha_by_wall: NDArray[np.float64]
    transfer: NDArray[np.float64]


def _mapping(value: object, where: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{where} 不是一層表")
    return {str(key): item for key, item in value.items()}


def _sequence(value: object, where: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{where} 不是一串值")
    return value


def _hex_float(value: object, where: str) -> float:
    cell = _mapping(value, where)
    hexadecimal = cell.get("hex")
    if not isinstance(hexadecimal, str):
        raise ValueError(f"{where}.hex 不是字串")
    result = float.fromhex(hexadecimal)
    if not math.isfinite(result):
        raise ValueError(f"{where} 不是有限數")
    return result


def _hex_integer(value: object, where: str) -> int:
    cell = _mapping(value, where)
    hexadecimal = cell.get("hex")
    if not isinstance(hexadecimal, str):
        raise ValueError(f"{where}.hex 不是字串")
    return int(hexadecimal, 16)


def _room(parameters: Mapping[str, object]) -> Room:
    values = _mapping(parameters.get("room_m"), "parameters.room_m")
    return Room(
        Lx=_hex_float(values.get("Lx"), "room_m.Lx"),
        Ly=_hex_float(values.get("Ly"), "room_m.Ly"),
        Lz=_hex_float(values.get("Lz"), "room_m.Lz"),
    )


def _impedance_row(value: object, where: str) -> tuple[complex, ...]:
    cells = _sequence(value, where)
    result = []
    for index, value_cell in enumerate(cells):
        cell = _mapping(value_cell, f"{where}[{index}]")
        real = _hex_float(cell.get("real"), f"{where}[{index}].real")
        imag = _hex_float(cell.get("imag"), f"{where}[{index}].imag")
        result.append(complex(real, imag))
    return tuple(result)


def load_late_energy_inputs(path: Path) -> LateEnergyInputs:
    """只從答案檔讀 ``parameters``，不讀任何答案能量。"""
    with path.open(encoding="utf-8") as handle:
        loaded: object = json.load(handle)
    root = _mapping(loaded, "答案檔")
    parameters = _mapping(root.get("parameters"), "parameters")
    material = _mapping(parameters.get("material"), "parameters.material")
    by_wall = _mapping(material.get("impedance_by_wall"), "material.impedance_by_wall")
    frequencies = tuple(
        _hex_float(cell, f"frequencies_hz[{index}]")
        for index, cell in enumerate(
            _sequence(parameters.get("frequencies_hz"), "parameters.frequencies_hz")
        )
    )
    impedances = {
        wall: _impedance_row(by_wall.get(wall), f"impedance_by_wall.{wall}")
        for wall in Wall.wall_names()
    }
    domain = _hex_float(
        parameters.get("ART_RADIOSITY_ALPHA_BAR_MAX"),
        "ART_RADIOSITY_ALPHA_BAR_MAX",
    )
    return LateEnergyInputs(
        room=_room(parameters),
        rho_c_pa_s_per_m=_hex_float(parameters.get("rho_c_pa_s_per_m"), "rho_c"),
        frequencies_hz=frequencies,
        impedance_by_wall=impedances,
        n_per_wall=_hex_integer(parameters.get("n_per_wall"), "n_per_wall"),
        domain_alpha_bar_max=domain,
    )


def load_legacy_late_energies(
    path: Path,
    *,
    frequencies_hz: Sequence[float] | None = None,
) -> tuple[float, ...]:
    """從答案檔讀 ``bands[].late_rev_E``，並可核對頻帶順序。"""
    with path.open(encoding="utf-8") as handle:
        loaded: object = json.load(handle)
    root = _mapping(loaded, "答案檔")
    rows = _sequence(root.get("bands"), "bands")
    if not rows:
        raise ValueError("bands 不可為空")
    if frequencies_hz is not None and len(rows) != len(frequencies_hz):
        raise ValueError("v3 結果與上一代答案的頻帶數不同")

    energies = []
    for index, value in enumerate(rows):
        row = _mapping(value, f"bands[{index}]")
        if frequencies_hz is not None:
            expected_frequency = _hex_float(
                row.get("frequency_hz"), f"bands[{index}].frequency_hz"
            )
            actual_frequency = frequencies_hz[index]
            if actual_frequency != expected_frequency:
                raise ValueError(
                    f"頻帶沒有對齊：parameters 是 {actual_frequency:g} Hz，"
                    f"bands 答案是 {expected_frequency:g} Hz"
                )
        energies.append(_hex_float(row.get("late_rev_E"), f"bands[{index}].late_rev_E"))
    return tuple(energies)


def _axis_centers(length: float, n_per_wall: int) -> NDArray[np.float64]:
    return length * (np.arange(n_per_wall, dtype=np.float64) + 0.5) / float(n_per_wall)


def _wall_grid(
    const_axis: int,
    const_value: float,
    u_axis: int,
    u_values: NDArray[np.float64],
    v_axis: int,
    v_values: NDArray[np.float64],
) -> NDArray[np.float64]:
    uu, vv = np.meshgrid(u_values, v_values, indexing="ij")
    points = np.zeros((u_values.size * v_values.size, 3), dtype=np.float64)
    points[:, const_axis] = const_value
    points[:, u_axis] = uu.reshape(-1)
    points[:, v_axis] = vv.reshape(-1)
    return points


def _patch_geometry(room: Room, n_per_wall: int) -> _Patches:
    guard_art_patch_count(n_per_wall)
    if not all(math.isfinite(length) and length > 0.0 for length in (room.Lx, room.Ly, room.Lz)):
        raise ValueError("房間三軸必須是有限正數")
    x = _axis_centers(room.Lx, n_per_wall)
    y = _axis_centers(room.Ly, n_per_wall)
    z = _axis_centers(room.Lz, n_per_wall)
    centroids = np.concatenate(
        (
            _wall_grid(2, 0.0, 0, x, 1, y),
            _wall_grid(2, room.Lz, 0, x, 1, y),
            _wall_grid(0, 0.0, 1, y, 2, z),
            _wall_grid(0, room.Lx, 1, y, 2, z),
            _wall_grid(1, 0.0, 0, x, 2, z),
            _wall_grid(1, room.Ly, 0, x, 2, z),
        )
    )
    wall_areas = _wall_areas(room) / float(n_per_wall * n_per_wall)
    areas = np.repeat(wall_areas, n_per_wall * n_per_wall)
    normals = np.repeat(
        np.asarray(
            (
                (0.0, 0.0, 1.0),
                (0.0, 0.0, -1.0),
                (1.0, 0.0, 0.0),
                (-1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, -1.0, 0.0),
            ),
            dtype=np.float64,
        ),
        n_per_wall * n_per_wall,
        axis=0,
    )
    wall_index = np.repeat(np.arange(len(Wall.all()), dtype=np.int64), n_per_wall**2)
    return _Patches(centroids, areas, normals, wall_index)


def _wall_areas(room: Room) -> NDArray[np.float64]:
    return np.asarray(
        (
            room.Lx * room.Ly,
            room.Lx * room.Ly,
            room.Ly * room.Lz,
            room.Ly * room.Lz,
            room.Lx * room.Lz,
            room.Lx * room.Lz,
        ),
        dtype=np.float64,
    )


def _form_factors(patches: _Patches) -> NDArray[np.float64]:
    c_i = patches.centroids[:, None, :]
    c_j = patches.centroids[None, :, :]
    displacement = c_j - c_i
    distance_squared = np.sum(displacement * displacement, axis=-1)
    valid_distance = distance_squared > 1e-12
    safe_distance_squared = np.where(valid_distance, distance_squared, 1.0)
    inverse_distance = np.where(valid_distance, 1.0 / np.sqrt(safe_distance_squared), 0.0)
    direction = displacement * inverse_distance[..., None]
    cosine_i = np.einsum("ik,ijk->ij", patches.normals, direction)
    cosine_j = np.einsum("jk,ijk->ij", patches.normals, -direction)
    raw = cosine_i * cosine_j * patches.areas[None, :] / (np.pi * safe_distance_squared)
    raw = np.where(valid_distance & (raw > 0.0), raw, 0.0)
    raw_row_sum = np.sum(raw, axis=1)
    if np.any(raw_row_sum <= 0.0):
        raise ValueError("形狀因子有無法正規化的空列")
    original = raw / raw_row_sum[:, None]
    exchange = patches.areas[:, None] * original
    balanced = (exchange + exchange.T) / 2.0
    deviation = float(
        np.max(np.abs(np.sum(balanced, axis=1) - patches.areas) / patches.areas)
    )
    while True:
        row_sum = np.sum(balanced, axis=1)
        if np.any(row_sum <= 0.0):
            raise ValueError("交換矩陣有無法平衡的空列")
        scale = np.sqrt(patches.areas / row_sum)
        candidate = balanced * np.multiply.outer(scale, scale)
        candidate_deviation = float(
            np.max(
                np.abs(np.sum(candidate, axis=1) - patches.areas)
                / patches.areas
            )
        )
        if not candidate_deviation < deviation:
            break
        balanced = candidate
        deviation = candidate_deviation
    return np.asarray(balanced / patches.areas[:, None], dtype=np.float64)


def _wall_absorption(inputs: LateEnergyInputs) -> NDArray[np.float64]:
    """逐面逐頻以正規化複數阻抗計算 Paris 無規入射吸音率。"""
    if set(inputs.impedance_by_wall) != set(Wall.wall_names()):
        raise ValueError("impedance_by_wall 必須恰好包含六面牆")
    rows = []
    for wall in Wall.wall_names():
        impedance = np.asarray(inputs.impedance_by_wall[wall], dtype=np.complex128)
        if impedance.shape != (len(inputs.frequencies_hz),):
            raise ValueError(f"{wall} 的阻抗頻帶數與 frequencies_hz 不同")
        rows.append(
            np.asarray(
                [
                    complex_random_incidence_absorption(
                        complex(value) / inputs.rho_c_pa_s_per_m
                    )
                    for value in impedance
                ],
                dtype=np.float64,
            )
        )
    alpha = np.asarray(rows, dtype=np.float64)
    if not np.all(np.isfinite(alpha)):
        raise ValueError("無規入射吸收率含非有限值")
    return alpha


def _exact_raw_energy(
    transfer: NDArray[np.float64],
    patch_areas: NDArray[np.float64],
) -> NDArray[np.float64]:
    patch_count, _, frequency_count = transfer.shape
    source = np.full(patch_count, 1.0 / np.sum(patch_areas), dtype=np.float64)
    identity = np.eye(patch_count, dtype=np.float64)
    energies = []
    for frequency_index in range(frequency_count):
        operator = transfer[:, :, frequency_index]
        field = np.linalg.solve(identity - operator, source)
        reflected = operator @ field
        mean_reflected = np.dot(patch_areas, reflected) / np.sum(patch_areas)
        energies.append(DIFFUSE_MONOPOLE_4PI * 4.0 * mean_reflected)
    return np.asarray(energies, dtype=np.float64)


def _reflection_problem(inputs: LateEnergyInputs) -> _ReflectionProblem:
    """以唯一一份形狀因子實作建出雙精度反射問題。"""
    if not inputs.frequencies_hz:
        raise ValueError("frequencies_hz 不可為空")
    if not math.isfinite(inputs.rho_c_pa_s_per_m) or inputs.rho_c_pa_s_per_m <= 0.0:
        raise ValueError("rho_c 必須是有限正數")
    patches = _patch_geometry(inputs.room, inputs.n_per_wall)
    form_factors = _form_factors(patches)
    alpha_by_wall = _wall_absorption(inputs)
    alpha_patch = alpha_by_wall[patches.wall_index, :]
    transfer = np.asarray(
        (1.0 - alpha_patch)[:, None, :] * form_factors[:, :, None],
        dtype=np.float64,
    )
    return _ReflectionProblem(patches, alpha_by_wall, transfer)


def _eyring_ratio(alpha_bar: float) -> float:
    guarded = min(max(alpha_bar, 0.0), 1.0 - float(np.finfo(np.float32).eps))
    if guarded == 0.0:
        return 1.0
    return guarded / -math.log1p(-guarded)


def solve_late_energy(inputs: LateEnergyInputs) -> LateEnergyResult:
    """以六面分格形狀因子與 float64 直接解回傳逐頻晚期能量。"""
    problem = _reflection_problem(inputs)
    raw_energy = _exact_raw_energy(problem.transfer, problem.patches.areas)
    wall_areas = _wall_areas(inputs.room)
    total_wall_area = float(np.sum(wall_areas))
    bands = []
    for index, frequency in enumerate(inputs.frequencies_hz):
        wall_alpha = {
            wall: float(problem.alpha_by_wall[wall_index, index])
            for wall_index, wall in enumerate(Wall.wall_names())
        }
        alpha_bar = float(
            np.dot(wall_areas, problem.alpha_by_wall[:, index]) / total_wall_area
        )
        ratio = _eyring_ratio(alpha_bar)
        raw = float(raw_energy[index])
        bands.append(
            LateEnergyBand(
                frequency_hz=frequency,
                alpha_by_wall=wall_alpha,
                alpha_bar=alpha_bar,
                in_domain=alpha_bar <= inputs.domain_alpha_bar_max,
                raw_reverberant_energy=raw,
                eyring_ratio=ratio,
                late_reverberant_energy=raw * ratio,
            )
        )
    return LateEnergyResult(bands=tuple(bands))


def judge_late_energy(
    result: LateEnergyResult,
    expected_energies: Sequence[float],
    tolerance_rel: float,
) -> LateEnergyContractReport:
    """逐頻量 ``|E3-E2|``，套用呼叫端給定的相對容差供相容紀錄閱讀。"""
    if not result.bands or len(result.bands) != len(expected_energies):
        raise ValueError("v3 結果與上一代答案的頻帶數不同或為空")
    points = []
    for band, expected_value in zip(result.bands, expected_energies, strict=True):
        expected = float(expected_value)
        difference = abs(band.late_reverberant_energy - expected)
        allowed = tolerance_rel * expected
        relative = difference / expected if expected > 0.0 else math.inf
        fraction = difference / allowed if allowed > 0.0 else math.inf
        finite = math.isfinite(band.late_reverberant_energy) and math.isfinite(expected)
        points.append(
            LateEnergyBandJudgment(
                frequency_hz=band.frequency_hz,
                actual_energy=band.late_reverberant_energy,
                expected_energy=expected,
                absolute_difference=difference,
                allowed_difference=allowed,
                relative_difference=relative,
                contract_fraction=fraction,
                within_contract=finite and expected >= 0.0 and difference <= allowed,
            )
        )
    return LateEnergyContractReport(points=tuple(points), tolerance_rel=tolerance_rel)


def solve_late_energy_contract(
    inputs: LateEnergyInputs,
    expected_energies: Sequence[float],
    tolerance_rel: float,
) -> LateEnergyContractReport:
    """跑正式精確解，再把同一批修正後能量交給上一代差距量測。"""
    return judge_late_energy(
        solve_late_energy(inputs), expected_energies, tolerance_rel
    )
