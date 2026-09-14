"""剛性參考房的正式 FEM 契約、題目載入與人看命令列。

``src/aosr`` 不載入 ``blueprint``：這支只從呼叫端指定的 JSON 讀房間、音源、
接收點與考點。解析模態值不在現有答案檔 schema，因此本段刻意沒有假裝提供
``--rigid-compare``；解析值由考卷在 blueprint 側獨立算好後餵給本模組的裁判。

契約界線依既有引擎慣例住在擁有裁判的物理模組：如 ``amplitude.py`` 擁有振幅
常數與 tolerance 函式，這裡的 :data:`RIGID_MODAL_CONTRACT_REL` 與
:func:`judge_rigid_modal_pressures` 也同住。獨立 oracle 仍保有自己的同值常數，
兩邊都錨定 ``docs/decisions/precision-contract-fem-two-layers.md``。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

from aosr.config.fem_lane import (
    FEM_ELEMENTS_PER_WAVELENGTH,
    FEM_FMAX_CAP_HZ,
    FEM_MESH_RANDOM_SEED,
)
from aosr.config.paths import config_path
from aosr.config.physics_constants import load_physics_constants
from aosr.geometry.shoebox import Point, Room, Wall
from aosr.geometry.shoebox_mesh import generate_shoebox_mesh
from aosr.physics.fem_helmholtz import solve_fem_helmholtz


RIGID_MODAL_CONTRACT_REL: Final[float] = 2.0**-10
RIGID_MODAL_FMAX_HZ: Final[float] = 20.0
RIGID_POINT_SETS: Final[frozenset[str]] = frozenset(("A", "B"))
THIRD_OCTAVE_CENTERS_HZ: Final[tuple[float, ...]] = (
    12.5,
    16.0,
    20.0,
    25.0,
    31.5,
    40.0,
    50.0,
    63.0,
    80.0,
    100.0,
    125.0,
    160.0,
    200.0,
    250.0,
)


@dataclass(frozen=True)
class RigidReferenceCase:
    """答案檔中足以重建正式剛性題目的資料，不含任何答案壓力。"""

    room: Room
    source: Point
    receiver: Point
    sound_speed_m_s: float
    density_kg_m3: float
    set_names: tuple[str, ...]
    frequencies_hz: tuple[float, ...]
    formal_frequencies_hz: tuple[float, ...]


@dataclass(frozen=True)
class RigidPointJudgment:
    """一個剛性考點的 v3 壓力、解析壓力與相對契約判決。"""

    set_name: str
    frequency_hz: float
    actual_pressure: complex
    expected_pressure: complex
    relative_error: float
    within_contract: bool


@dataclass(frozen=True)
class RigidContractReport:
    """整批剛性考點的逐點判決。"""

    points: tuple[RigidPointJudgment, ...]

    @property
    def max_relative_error(self) -> float:
        """整批考點最大的複數壓力相對差。"""
        return max(point.relative_error for point in self.points)

    @property
    def max_contract_fraction(self) -> float:
        """最大相對差用掉契約界線的比例。"""
        return self.max_relative_error / RIGID_MODAL_CONTRACT_REL

    @property
    def within_contract(self) -> bool:
        """只有逐點全數在界線內才為真。"""
        return all(point.within_contract for point in self.points)


@dataclass(frozen=True)
class ThirdOctaveBandEnergy:
    """一個固定 1/3 八度帶收進的頻點及其平均能量 dB。"""

    center_hz: float
    frequencies_hz: tuple[float, ...]
    mean_energy: float | None
    energy_db: float | None


def _mapping(value: object, where: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{where} 不是一層表")
    return {str(key): item for key, item in value.items()}


def _number(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{where} 不是數字")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{where} 不是有限值")
    return result


def _xyz(value: object, where: str) -> tuple[float, float, float]:
    if not isinstance(value, list):
        raise ValueError(f"{where} 不是三軸座標")
    try:
        x_value, y_value, z_value = value
    except ValueError as exc:
        raise ValueError(f"{where} 不是三軸座標") from exc
    return (
        _number(x_value, f"{where}.x"),
        _number(y_value, f"{where}.y"),
        _number(z_value, f"{where}.z"),
    )


def _reference_points(
    value: object,
) -> tuple[tuple[str, ...], tuple[float, ...], tuple[float, ...]]:
    if not isinstance(value, list):
        raise ValueError("points 不是一串考點")
    chosen: list[tuple[str, float]] = []
    formal: list[float] = []
    for index, value_point in enumerate(value):
        point = _mapping(value_point, f"points[{index}]")
        set_name = point.get("set")
        frequency = _number(point.get("frequency_hz"), f"points[{index}].frequency_hz")
        if set_name == "A" and frequency <= FEM_FMAX_CAP_HZ:
            formal.append(frequency)
        if isinstance(set_name, str) and set_name in RIGID_POINT_SETS:
            if frequency <= RIGID_MODAL_FMAX_HZ:
                chosen.append((set_name, frequency))
    if not chosen:
        raise ValueError("答案檔沒有剛性解析契約考點")
    if not formal:
        raise ValueError("答案檔沒有正式 A 軸頻點")
    return (
        tuple(set_name for set_name, _frequency in chosen),
        tuple(frequency for _set_name, frequency in chosen),
        tuple(formal),
    )


def _validated_parameters(root: dict[str, object]) -> tuple[Room, Point, Point, float, float]:
    params = _mapping(root.get("parameters"), "parameters")
    room_values = _mapping(params.get("room_m"), "parameters.room_m")
    room = Room(
        _number(room_values.get("Lx"), "room_m.Lx"),
        _number(room_values.get("Ly"), "room_m.Ly"),
        _number(room_values.get("Lz"), "room_m.Lz"),
    )
    source = Point(*_xyz(params.get("source_xyz_m"), "source_xyz_m"))
    receiver = Point(*_xyz(params.get("receiver_xyz_m"), "receiver_xyz_m"))
    physics = load_physics_constants(config_path("physics_constants.toml"))
    mesh = _mapping(params.get("mesh"), "parameters.mesh")
    expected_header = (
        physics.sound_speed_m_s,
        FEM_FMAX_CAP_HZ,
        float(FEM_ELEMENTS_PER_WAVELENGTH),
    )
    _number(params.get("rho_c_pa_s_per_m"), "parameters.rho_c_pa_s_per_m")
    actual_header = (
        _number(params.get("c_m_s"), "parameters.c_m_s"),
        _number(mesh.get("f_max_cap_hz"), "parameters.mesh.f_max_cap_hz"),
        _number(
            mesh.get("elements_per_wavelength"),
            "parameters.mesh.elements_per_wavelength",
        ),
    )
    if actual_header != expected_header:
        raise ValueError("答案檔的聲速或正式網格設定跟 config 不同")
    return room, source, receiver, physics.sound_speed_m_s, physics.air_density_kg_m3


def load_rigid_reference_case(path: Path) -> RigidReferenceCase:
    """只讀答案檔的題目欄，選出 A/B 集合中不高於物理上限的考點。"""
    with path.open(encoding="utf-8") as handle:
        loaded: object = json.load(handle)
    root = _mapping(loaded, "答案檔")
    room, source, receiver, sound_speed, density = _validated_parameters(root)
    set_names, frequencies, formal_frequencies = _reference_points(root.get("points"))
    return RigidReferenceCase(
        room=room,
        source=source,
        receiver=receiver,
        sound_speed_m_s=sound_speed,
        density_kg_m3=density,
        set_names=set_names,
        frequencies_hz=frequencies,
        formal_frequencies_hz=formal_frequencies,
    )


def _solve_rigid_frequencies(
    case: RigidReferenceCase,
    frequencies_hz: Sequence[float],
) -> NDArray[np.complex128]:
    """以 config 正式網格與固定種子組裝一次，整批求解指定頻點。"""
    mesh = generate_shoebox_mesh(
        case.room,
        max_frequency_hz=FEM_FMAX_CAP_HZ,
        elements_per_wavelength=FEM_ELEMENTS_PER_WAVELENGTH,
        sound_speed_m_s=case.sound_speed_m_s,
        random_seed=FEM_MESH_RANDOM_SEED,
    )
    rigid_walls = {wall: None for wall in Wall.all()}
    return solve_fem_helmholtz(
        mesh,
        wall_impedances=rigid_walls,
        source=case.source,
        receiver=case.receiver,
        frequencies_hz=frequencies_hz,
        density_kg_m3=case.density_kg_m3,
        sound_speed_m_s=case.sound_speed_m_s,
    )


def solve_rigid_reference_case(
    case: RigidReferenceCase,
) -> NDArray[np.complex128]:
    """整批求解剛性解析契約的 A/B 九點。"""
    return _solve_rigid_frequencies(case, case.frequencies_hz)


def solve_rigid_formal_case(
    case: RigidReferenceCase,
) -> NDArray[np.complex128]:
    """整批求解正式 A 軸，供命令列逐點與 1/3 八度帶顯示。"""
    return _solve_rigid_frequencies(case, case.formal_frequencies_hz)


def judge_rigid_modal_pressures(
    case: RigidReferenceCase,
    actual_pressures: Sequence[complex] | NDArray[np.complex128],
    expected_pressures: Sequence[complex] | NDArray[np.complex128],
) -> RigidContractReport:
    """依 ``|v3-解析|/|解析|`` 逐點套正式相對契約。"""
    actual = np.asarray(actual_pressures, dtype=np.complex128)
    expected = np.asarray(expected_pressures, dtype=np.complex128)
    if actual.shape != expected.shape or actual.shape != (len(case.frequencies_hz),):
        raise ValueError("v3、解析答案與考點的形狀不同")
    relative = np.abs(actual - expected) / np.maximum(
        np.abs(expected), np.finfo(np.float64).tiny
    )
    points = tuple(
        RigidPointJudgment(
            set_name=set_name,
            frequency_hz=frequency,
            actual_pressure=complex(actual[index]),
            expected_pressure=complex(expected[index]),
            relative_error=float(relative[index]),
            within_contract=bool(relative[index] <= RIGID_MODAL_CONTRACT_REL),
        )
        for index, (set_name, frequency) in enumerate(
            zip(case.set_names, case.frequencies_hz, strict=True)
        )
    )
    return RigidContractReport(points)


def solve_rigid_modal_contract(
    case: RigidReferenceCase,
    expected_pressures: Sequence[complex] | NDArray[np.complex128],
) -> RigidContractReport:
    """跑一次正式剛性求解，再把同一批壓力交給契約裁判。"""
    actual = solve_rigid_reference_case(case)
    return judge_rigid_modal_pressures(case, actual, expected_pressures)


def third_octave_band_energies(
    frequencies_hz: Sequence[float],
    pressures: Sequence[complex] | NDArray[np.complex128],
) -> tuple[ThirdOctaveBandEnergy, ...]:
    """依固定中心與 ``fc·2^(±1/6)`` 分帶，平均帶內 ``|p|²``。"""
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    values = np.asarray(pressures, dtype=np.complex128)
    if frequencies.shape != values.shape or frequencies.ndim != 1:
        raise ValueError("頻率與壓力必須是同長度的一維序列")
    bands: list[ThirdOctaveBandEnergy] = []
    for center in THIRD_OCTAVE_CENTERS_HZ:
        lower = center * 2.0 ** (-1.0 / 6.0)
        upper = center * 2.0 ** (1.0 / 6.0)
        indices = np.flatnonzero((frequencies >= lower) & (frequencies <= upper))
        if indices.size == 0:
            bands.append(ThirdOctaveBandEnergy(center, (), None, None))
            continue
        mean_energy = float(np.mean(np.abs(values[indices]) ** 2))
        energy_db = -math.inf if mean_energy == 0.0 else 10.0 * math.log10(mean_energy)
        bands.append(
            ThirdOctaveBandEnergy(
                center_hz=center,
                frequencies_hz=tuple(float(frequencies[index]) for index in indices),
                mean_energy=mean_energy,
                energy_db=energy_db,
            )
        )
    return tuple(bands)


def rigid_result_table(
    case: RigidReferenceCase,
    pressures: Sequence[complex] | NDArray[np.complex128],
) -> str:
    """回傳正式 A 軸逐點壓力與依固定邊界聚合的 1/3 八度帶能量。"""
    values = np.asarray(pressures, dtype=np.complex128)
    if values.shape != (len(case.formal_frequencies_hz),):
        raise ValueError("求解壓力與正式 A 軸的形狀不同")
    lines = ["frequency_hz  |p|  phase_deg"]
    for index, frequency in enumerate(case.formal_frequencies_hz):
        pressure = complex(values[index])
        phase = math.degrees(math.atan2(pressure.imag, pressure.real))
        lines.append(
            f"{frequency:.12g}  {abs(pressure):.12g}  {phase:.9f}"
        )
    lines.extend(("", "third_octave_center_hz  frequencies_hz  mean_energy  energy_db_re_1"))
    for band in third_octave_band_energies(case.formal_frequencies_hz, values):
        if band.mean_energy is None or band.energy_db is None:
            lines.append(f"{band.center_hz:g}  none  none  none")
            continue
        members = ",".join(f"{frequency:.12g}" for frequency in band.frequencies_hz)
        lines.append(
            f"{band.center_hz:g}  {members}  {band.mean_energy:.12g}  "
            f"{band.energy_db:.9f}"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    """讀參考題目、跑正式剛性路徑並印表；錯誤回 2。"""
    parser = argparse.ArgumentParser(description="正式 P2 剛性參考房頻率響應")
    parser.add_argument("input", type=Path, help="帶 parameters 與 points 的參考 JSON")
    args = parser.parse_args(argv)
    try:
        case = load_rigid_reference_case(args.input)
        pressures = solve_rigid_formal_case(case)
        print(rigid_result_table(case, pressures), end="")
        return 0
    except Exception as exc:
        print(f"剛性參考房算不出來：{exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
