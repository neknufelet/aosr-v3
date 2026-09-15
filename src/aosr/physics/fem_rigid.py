"""剛性參考房的正式 FEM 契約、題目載入與人看命令列。

``src/aosr`` 不載入 ``blueprint``：這支只從呼叫端指定的 JSON 讀房間、音源、
接收點與解析契約考點；命令列有限元素路軸由 config 的 v3 定義供應。解析模態值不在
現有答案檔 schema，因此本段刻意沒有假裝提供 ``--rigid-compare``；解析值由考卷
在 blueprint 側獨立算好後餵給本模組的裁判。

契約界線由呼叫端從精度契約登記簿取得，再傳給本模組的裁判；獨立 oracle 不持有副本。
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
from aosr.config.capabilities import (
    capability_for,
    load_capabilities,
)
from aosr.config.precision_contracts import load_precision_contracts
from aosr.config.frequency_axis import (
    FEM_GEOMETRIC_CROSSOVER_CAP_HZ,
    FEM_LANE_FREQUENCIES_HZ,
)
from aosr.geometry.shoebox import Point, Room, Wall
from aosr.geometry.shoebox_mesh import ShoeboxMesh, generate_shoebox_mesh
from aosr.physics import capability_report
from aosr.physics.fem_helmholtz import solve_fem_helmholtz


# 這一節在能力表上的名字，以及這條路兩種材料形式：剛性邊界（六面 gamma=0），
# 與 FEniCS 凍結題目那條實數阻抗牆。
_CAPABILITY_ENTRY: Final[str] = "fem_rigid"
_CAPABILITY_ROOM: Final[str] = "shoebox"
_CAPABILITY_RIGID_MATERIALS: Final[str] = "rigid_walls"
_CAPABILITY_IMPEDANCE_MATERIALS: Final[str] = "real_frequency_independent_impedance"


def capability_line(path: Path, materials: str) -> str:
    """查這條組合本人，回傳一行給人看的 capability 節。

    ``materials`` 由呼叫端說這一跑真正用到哪種邊界；不寫死在這裡，不然
    ``--compare`` 那條實數阻抗牆會被印成 rigid_walls。印出來的那一行帶著
    表上那一條宣告的頻率範圍與輸出欄：這一跑的解在整條 300 Hz 軸上，但
    剛性那一條驗過的只有到 20 Hz 的逐點壓力，只印 status 看不出這個差別。
    """
    table = load_capabilities(path)
    record = capability_for(
        table, _CAPABILITY_ENTRY, room=_CAPABILITY_ROOM, materials=materials
    )
    return capability_report.capability_line_from_record(
        _CAPABILITY_ENTRY, _CAPABILITY_ROOM, materials, record
    )


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
    """足以重建正式剛性題目的資料；有限元素路軸來自 config。"""

    room: Room
    source: Point
    receiver: Point
    sound_speed_m_s: float
    density_kg_m3: float
    set_names: tuple[str, ...]
    frequencies_hz: tuple[float, ...]
    fem_lane_frequencies_hz: tuple[float, ...]


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
    tolerance_rel: float

    @property
    def max_relative_error(self) -> float:
        """整批考點最大的複數壓力相對差。"""
        return max(point.relative_error for point in self.points)

    @property
    def max_contract_fraction(self) -> float:
        """最大相對差用掉契約界線的比例。"""
        return self.max_relative_error / self.tolerance_rel

    @property
    def within_contract(self) -> bool:
        """只有逐點全數在界線內才為真。"""
        return all(point.within_contract for point in self.points)


@dataclass(frozen=True)
class FenicsProblem:
    """凍結的正式網格、物理條件與兩個吸音案例。"""

    mesh: ShoeboxMesh
    room: Room
    source: Point
    receiver: Point
    density_kg_m3: float
    sound_speed_m_s: float
    frequencies_hz: tuple[float, ...]
    impedance_ratios: dict[str, float]


@dataclass(frozen=True)
class FenicsAnswers:
    """FEniCS 外部答案與它指回的題目檔路徑。"""

    problem_file: str
    frequencies_hz: dict[str, tuple[float, ...]]
    pressures: dict[str, NDArray[np.complex128]]


@dataclass(frozen=True)
class FenicsPointJudgment:
    """一個案例、一個頻點的 v3／FEniCS 壓力與契約判決。"""

    case_name: str
    frequency_hz: float
    actual_pressure: complex
    expected_pressure: complex
    relative_error: float
    within_contract: bool


@dataclass(frozen=True)
class FenicsContractReport:
    """兩案例全部正式頻點的逐點判決。"""

    points: tuple[FenicsPointJudgment, ...]
    tolerance_rel: float

    @property
    def max_relative_error(self) -> float:
        """整批考點最大的複數壓力相對差。"""
        return max(point.relative_error for point in self.points)

    @property
    def max_contract_fraction(self) -> float:
        """最大相對差用掉契約界線的比例。"""
        return self.max_relative_error / self.tolerance_rel

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


def _hex_number(value: object, where: str) -> float:
    """只收 ``float.hex()`` 字串並還原成有限 float64。"""
    if not isinstance(value, str):
        raise ValueError(f"{where} 不是 float.hex() 字串")
    try:
        result = float.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{where} 不是合法 float.hex() 字串") from exc
    if not math.isfinite(result):
        raise ValueError(f"{where} 不是有限值")
    return result


def _hex_vector(value: object, width: int, where: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != width:
        raise ValueError(f"{where} 不是長度 {width} 的 float.hex() 序列")
    return tuple(_hex_number(item, f"{where}[{index}]") for index, item in enumerate(value))


def _integer_matrix(value: object, width: int, where: str) -> NDArray[np.int64]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{where} 不是非空整數矩陣")
    if any(
        not isinstance(row, list)
        or len(row) != width
        or any(isinstance(item, bool) or not isinstance(item, int) for item in row)
        for row in value
    ):
        raise ValueError(f"{where} 不是每列長度 {width} 的整數矩陣")
    return np.asarray(value, dtype=np.int64)


def _hex_matrix(value: object, width: int, where: str) -> NDArray[np.float64]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{where} 不是非空 float.hex() 矩陣")
    rows = [_hex_vector(row, width, f"{where}[{index}]") for index, row in enumerate(value)]
    return np.asarray(rows, dtype=np.float64)


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
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    if not isinstance(value, list):
        raise ValueError("points 不是一串考點")
    chosen: list[tuple[str, float]] = []
    for index, value_point in enumerate(value):
        point = _mapping(value_point, f"points[{index}]")
        set_name = point.get("set")
        frequency = _number(point.get("frequency_hz"), f"points[{index}].frequency_hz")
        if isinstance(set_name, str) and set_name in RIGID_POINT_SETS:
            if frequency <= RIGID_MODAL_FMAX_HZ:
                chosen.append((set_name, frequency))
    if not chosen:
        raise ValueError("答案檔沒有剛性解析契約考點")
    return (
        tuple(set_name for set_name, _frequency in chosen),
        tuple(frequency for _set_name, frequency in chosen),
    )


def _validated_parameters(root: dict[str, object]) -> tuple[Room, Point, Point, float, float]:
    params = _mapping(root.get("parameters"), "parameters")
    material = _mapping(params.get("material"), "parameters.material")
    boundary = material.get("boundary")
    if boundary != "rigid":
        # 這一版只吃六面 gamma=0 的剛性邊界；收到別的就明確報錯，不安靜地當剛性算。
        raise ValueError(
            f"剛性路這一版只吃 material.boundary=rigid，收到 {boundary!r}；"
            "非剛性邊界要接出去是票 #309 的事"
        )
    room_values = _mapping(params.get("room_m"), "parameters.room_m")
    room = Room(
        _number(room_values.get("Lx"), "room_m.Lx"),
        _number(room_values.get("Ly"), "room_m.Ly"),
        _number(room_values.get("Lz"), "room_m.Lz"),
    )
    source = Point(*_xyz(params.get("source_xyz_m"), "source_xyz_m"))
    receiver = Point(*_xyz(params.get("receiver_xyz_m"), "receiver_xyz_m"))
    sound_speed = _number(params.get("c_m_s"), "parameters.c_m_s")
    rho_c = _number(params.get("rho_c_pa_s_per_m"), "parameters.rho_c_pa_s_per_m")
    mesh = _mapping(params.get("mesh"), "parameters.mesh")
    expected_header = (
        FEM_FMAX_CAP_HZ,
        float(FEM_ELEMENTS_PER_WAVELENGTH),
    )
    actual_header = (
        _number(mesh.get("f_max_cap_hz"), "parameters.mesh.f_max_cap_hz"),
        _number(
            mesh.get("elements_per_wavelength"),
            "parameters.mesh.elements_per_wavelength",
        ),
    )
    if actual_header != expected_header:
        raise ValueError("答案檔的正式網格設定跟 config 不同")
    return room, source, receiver, sound_speed, rho_c / sound_speed


def _load_json(path: Path, what: str) -> dict[str, object]:
    try:
        with path.open(encoding="utf-8") as handle:
            loaded: object = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{what}讀不開：{exc}") from exc
    return _mapping(loaded, what)


def _fenics_mesh(root: dict[str, object]) -> ShoeboxMesh:
    raw = _mapping(root.get("mesh"), "mesh")
    nodes = _hex_matrix(raw.get("nodes"), 3, "mesh.nodes")
    tetrahedra = _integer_matrix(raw.get("tetrahedra"), 4, "mesh.tetrahedra")
    triangles = _integer_matrix(
        raw.get("boundary_triangles"), 3, "mesh.boundary_triangles"
    )
    wall_indices_raw = raw.get("boundary_wall_indices")
    if not isinstance(wall_indices_raw, list) or any(
        isinstance(item, bool) or not isinstance(item, int)
        for item in wall_indices_raw
    ):
        raise ValueError("mesh.boundary_wall_indices 不是整數序列")
    wall_indices = np.asarray(wall_indices_raw, dtype=np.int64)
    if wall_indices.shape != (triangles.shape[0],):
        raise ValueError("邊界牆編號沒有逐列對齊邊界三角形")
    if np.any(tetrahedra < 0) or np.any(tetrahedra >= nodes.shape[0]):
        raise ValueError("四面體引用不存在的節點")
    if np.any(triangles < 0) or np.any(triangles >= nodes.shape[0]):
        raise ValueError("邊界三角形引用不存在的節點")
    if set(int(item) for item in wall_indices) != set(range(len(Wall.all()))):
        raise ValueError("邊界牆編號沒有恰好涵蓋六面牆")
    return ShoeboxMesh(nodes, tetrahedra, triangles, wall_indices)


def _fenics_problem_header(root: dict[str, object]) -> tuple[Room, Point, Point, float, float]:
    room_values = _mapping(root.get("room_m"), "room_m")
    room = Room(*(_hex_number(room_values.get(key), f"room_m.{key}") for key in ("Lx", "Ly", "Lz")))
    source = Point(*_hex_vector(root.get("source_xyz_m"), 3, "source_xyz_m"))
    receiver = Point(*_hex_vector(root.get("receiver_xyz_m"), 3, "receiver_xyz_m"))
    density = _hex_number(root.get("density_kg_m3"), "density_kg_m3")
    sound_speed = _hex_number(root.get("sound_speed_m_s"), "sound_speed_m_s")
    return room, source, receiver, density, sound_speed


def _fenics_conditions(root: dict[str, object]) -> tuple[tuple[float, ...], dict[str, float]]:
    physics = _mapping(root.get("physics"), "physics")
    actual_physics = tuple(
        physics.get(key)
        for key in ("element", "source_strength", "time_convention", "wall_velocity")
    )
    expected_physics = ("Lagrange P2", "4*pi", "exp(+j*omega*t)", "v_n = p/Z")
    if actual_physics != expected_physics:
        raise ValueError("題目的元素、音源或時間／牆速慣例不是正式物理條件")
    values = root.get("frequencies_hz")
    if not isinstance(values, list):
        raise ValueError("frequencies_hz 不是序列")
    frequencies = tuple(
        _hex_number(value, f"frequencies_hz[{index}]")
        for index, value in enumerate(values)
    )
    if not frequencies or any(left >= right for left, right in zip(frequencies, frequencies[1:])):
        raise ValueError("frequencies_hz 必須是非空嚴格遞增序列")
    cases = _mapping(root.get("cases"), "cases")
    if set(cases) != {"flat", "lowabs"}:
        raise ValueError("題目必須恰好包含 flat 與 lowabs")
    ratios = {
        name: _hex_number(
            _mapping(cases[name], f"cases.{name}").get("wall_impedance_over_rho_c"),
            f"cases.{name}.wall_impedance_over_rho_c",
        )
        for name in ("flat", "lowabs")
    }
    if any(value <= 0.0 for value in ratios.values()):
        raise ValueError("牆阻抗比必須是正有限值")
    return frequencies, ratios


def load_fenics_problem(path: Path) -> FenicsProblem:
    """把全是 ``float.hex()`` 的凍結題目還原成正式網格與物理條件。"""
    root = _load_json(path, "FEniCS 題目檔")
    if root.get("schema") != "fem-fenics-problem/1":
        raise ValueError("FEniCS 題目檔 schema 不認得")
    room, source, receiver, density, sound_speed = _fenics_problem_header(root)
    frequencies, ratios = _fenics_conditions(root)
    return FenicsProblem(
        mesh=_fenics_mesh(root),
        room=room,
        source=source,
        receiver=receiver,
        density_kg_m3=density,
        sound_speed_m_s=sound_speed,
        frequencies_hz=frequencies,
        impedance_ratios=ratios,
    )


def load_fenics_answers(path: Path) -> FenicsAnswers:
    """讀兩案例的逐頻 FEniCS 複數壓力，不自行產生期望值。"""
    root = _load_json(path, "FEniCS 答案檔")
    if root.get("schema") != "fem-fenics-answers/1":
        raise ValueError("FEniCS 答案檔 schema 不認得")
    provenance = _mapping(root.get("provenance"), "provenance")
    problem_file = provenance.get("problem_file")
    if not isinstance(problem_file, str) or not problem_file.strip():
        raise ValueError("provenance.problem_file 不是非空路徑")
    cases = _mapping(root.get("cases"), "cases")
    if set(cases) != {"flat", "lowabs"}:
        raise ValueError("答案檔必須恰好包含 flat 與 lowabs")
    frequencies: dict[str, tuple[float, ...]] = {}
    pressures: dict[str, NDArray[np.complex128]] = {}
    for name in ("flat", "lowabs"):
        rows = cases[name]
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"cases.{name} 不是非空答案序列")
        decoded = [_mapping(row, f"cases.{name}[{index}]") for index, row in enumerate(rows)]
        frequencies[name] = tuple(
            _hex_number(row.get("frequency_hz"), f"cases.{name}.frequency_hz")
            for row in decoded
        )
        pressures[name] = np.asarray(
            [
                complex(
                    _hex_number(_mapping(row.get("pressure"), "pressure").get("real"), "pressure.real"),
                    _hex_number(_mapping(row.get("pressure"), "pressure").get("imag"), "pressure.imag"),
                )
                for row in decoded
            ],
            dtype=np.complex128,
        )
    if frequencies["flat"] != frequencies["lowabs"]:
        raise ValueError("兩案例答案的頻點沒有對齊")
    return FenicsAnswers(problem_file.strip(), frequencies, pressures)


def load_rigid_reference_case(path: Path) -> RigidReferenceCase:
    """只讀答案檔的題目欄，選出 A/B 集合中不高於物理上限的考點。"""
    with path.open(encoding="utf-8") as handle:
        loaded: object = json.load(handle)
    root = _mapping(loaded, "答案檔")
    room, source, receiver, sound_speed, density = _validated_parameters(root)
    set_names, frequencies = _reference_points(root.get("points"))
    return RigidReferenceCase(
        room=room,
        source=source,
        receiver=receiver,
        sound_speed_m_s=sound_speed,
        density_kg_m3=density,
        set_names=set_names,
        frequencies_hz=frequencies,
        fem_lane_frequencies_hz=FEM_LANE_FREQUENCIES_HZ,
    )


def _solve_rigid_frequencies(
    case: RigidReferenceCase,
    frequencies_hz: Sequence[float],
) -> NDArray[np.complex128]:
    """以有限元素路網格與固定種子組裝一次，整批求解指定頻點。"""
    mesh = generate_shoebox_mesh(
        case.room,
        max_frequency_hz=FEM_GEOMETRIC_CROSSOVER_CAP_HZ,
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


def solve_rigid_fem_lane_case(
    case: RigidReferenceCase,
) -> NDArray[np.complex128]:
    """整批求解有限元素路軸，供命令列逐點與 1/3 八度帶顯示。"""
    return _solve_rigid_frequencies(case, case.fem_lane_frequencies_hz)


def judge_rigid_modal_pressures(
    case: RigidReferenceCase,
    actual_pressures: Sequence[complex] | NDArray[np.complex128],
    expected_pressures: Sequence[complex] | NDArray[np.complex128],
    tolerance_rel: float,
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
            within_contract=bool(relative[index] <= tolerance_rel),
        )
        for index, (set_name, frequency) in enumerate(
            zip(case.set_names, case.frequencies_hz, strict=True)
        )
    )
    return RigidContractReport(points, tolerance_rel)


def solve_rigid_modal_contract(
    case: RigidReferenceCase,
    expected_pressures: Sequence[complex] | NDArray[np.complex128],
    tolerance_rel: float,
) -> RigidContractReport:
    """跑一次正式剛性求解，再把同一批壓力交給契約裁判。"""
    actual = solve_rigid_reference_case(case)
    return judge_rigid_modal_pressures(
        case, actual, expected_pressures, tolerance_rel
    )


def solve_fenics_pressures(
    problem: FenicsProblem,
) -> dict[str, NDArray[np.complex128]]:
    """在凍結正式網格上求解 flat 與 lowabs 全部頻點。"""
    pressures: dict[str, NDArray[np.complex128]] = {}
    rho_c = problem.density_kg_m3 * problem.sound_speed_m_s
    for case_name in ("flat", "lowabs"):
        impedance = problem.impedance_ratios[case_name] * rho_c
        pressures[case_name] = solve_fem_helmholtz(
            problem.mesh,
            wall_impedances={wall: impedance for wall in Wall.all()},
            source=problem.source,
            receiver=problem.receiver,
            frequencies_hz=problem.frequencies_hz,
            density_kg_m3=problem.density_kg_m3,
            sound_speed_m_s=problem.sound_speed_m_s,
        )
    return pressures


def judge_fenics_pressures(
    problem: FenicsProblem,
    actual_pressures: dict[str, NDArray[np.complex128]],
    expected_pressures: dict[str, NDArray[np.complex128]],
    tolerance_rel: float,
) -> FenicsContractReport:
    """依 ``|v3-FEniCS|/|FEniCS|`` 對兩案例逐點套正式契約。"""
    expected_cases = {"flat", "lowabs"}
    if set(actual_pressures) != expected_cases or set(expected_pressures) != expected_cases:
        raise ValueError("v3 與 FEniCS 壓力必須恰好包含 flat 與 lowabs")
    points: list[FenicsPointJudgment] = []
    expected_shape = (len(problem.frequencies_hz),)
    for case_name in ("flat", "lowabs"):
        actual = np.asarray(actual_pressures[case_name], dtype=np.complex128)
        expected = np.asarray(expected_pressures[case_name], dtype=np.complex128)
        if actual.shape != expected_shape or expected.shape != expected_shape:
            raise ValueError(f"{case_name} 的 v3、FEniCS 答案與考點形狀不同")
        relative = np.abs(actual - expected) / np.maximum(
            np.abs(expected), np.finfo(np.float64).tiny
        )
        points.extend(
            FenicsPointJudgment(
                case_name=case_name,
                frequency_hz=frequency,
                actual_pressure=complex(actual[index]),
                expected_pressure=complex(expected[index]),
                relative_error=float(relative[index]),
                within_contract=bool(relative[index] <= tolerance_rel),
            )
            for index, frequency in enumerate(problem.frequencies_hz)
        )
    return FenicsContractReport(tuple(points), tolerance_rel)


def solve_fenics_contract(
    problem: FenicsProblem,
    expected_pressures: dict[str, NDArray[np.complex128]],
    tolerance_rel: float,
) -> FenicsContractReport:
    """跑兩案例正式求解，再把同批壓力交給 FEniCS 契約裁判。"""
    return judge_fenics_pressures(
        problem,
        solve_fenics_pressures(problem),
        expected_pressures,
        tolerance_rel,
    )


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
    """回傳有限元素路軸逐點壓力與完整 1/3 八度帶能量。"""
    values = np.asarray(pressures, dtype=np.complex128)
    if values.shape != (len(case.fem_lane_frequencies_hz),):
        raise ValueError("求解壓力與有限元素路軸的形狀不同")
    lines = ["frequency_hz  |p|  phase_deg"]
    for index, frequency in enumerate(case.fem_lane_frequencies_hz):
        pressure = complex(values[index])
        phase = math.degrees(math.atan2(pressure.imag, pressure.real))
        lines.append(
            f"{frequency:.12g}  {abs(pressure):.12g}  {phase:.9f}"
        )
    lines.extend(("", "third_octave_center_hz  frequencies_hz  mean_energy  energy_db_re_1"))
    complete_bands = tuple(
        band
        for band in third_octave_band_energies(case.fem_lane_frequencies_hz, values)
        if band.center_hz * 2.0 ** (-1.0 / 6.0) >= case.fem_lane_frequencies_hz[0]
        and band.center_hz * 2.0 ** (1.0 / 6.0) <= case.fem_lane_frequencies_hz[-1]
    )
    for band in complete_bands:
        if band.mean_energy is None or band.energy_db is None:
            lines.append(f"{band.center_hz:g}  none  none  none")
            continue
        members = ",".join(f"{frequency:.12g}" for frequency in band.frequencies_hz)
        lines.append(
            f"{band.center_hz:g}  {members}  {band.mean_energy:.12g}  "
            f"{band.energy_db:.9f}"
        )
    assigned = {
        frequency
        for band in complete_bands
        for frequency in band.frequencies_hz
    }
    unassigned = ",".join(
        f"{frequency:.12g}"
        for frequency in case.fem_lane_frequencies_hz
        if frequency not in assigned
    )
    lines.append(f"unassigned_frequencies_hz  {unassigned}")
    return "\n".join(lines) + "\n"


def fenics_compare_table(report: FenicsContractReport) -> str:
    """回傳兩案例逐點 v3／FEniCS 壓力、相對差與判決。"""
    lines = [
        "case frequency_hz v3_real v3_imag answer_real answer_imag relative_error verdict"
    ]
    for point in report.points:
        verdict = "PASS" if point.within_contract else "FAIL"
        lines.append(
            f"{point.case_name} {point.frequency_hz:.12g} "
            f"{point.actual_pressure.real:.17g} {point.actual_pressure.imag:.17g} "
            f"{point.expected_pressure.real:.17g} {point.expected_pressure.imag:.17g} "
            f"{point.relative_error:.17g} {verdict}"
        )
    final = "PASS" if report.within_contract else "FAIL"
    lines.append(
        f"FINAL {final} max_relative_error={report.max_relative_error:.17g} "
        f"limit={report.tolerance_rel:.17g}"
    )
    return "\n".join(lines) + "\n"


def _run_fenics_compare(
    answer_path: Path,
    tolerance_rel: float,
) -> FenicsContractReport:
    """依答案檔指回的 repo 相對路徑讀題、核對頻點並求解。"""
    answers = load_fenics_answers(answer_path)
    problem_path = Path.cwd() / answers.problem_file
    problem = load_fenics_problem(problem_path)
    for case_name in ("flat", "lowabs"):
        if answers.frequencies_hz[case_name] != problem.frequencies_hz:
            raise ValueError(f"{case_name} 的答案頻點跟題目頻點不同")
    return solve_fenics_contract(problem, answers.pressures, tolerance_rel)


def main(argv: list[str]) -> int:
    """跑剛性表或 FEniCS 凍結答案逐點比對；錯誤回 2。"""
    parser = argparse.ArgumentParser(description="正式 P2 剛性參考房頻率響應")
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        help="帶 parameters 與 points 的剛性參考 JSON",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        help=(
            "比對 FEniCS 答案 JSON；題目取 provenance.problem_file，"
            "該路徑相對執行時的目前工作目錄"
        ),
    )
    parser.add_argument("--contracts", type=Path, help="精度契約 TOML 登記簿")
    parser.add_argument(
        "--capabilities",
        type=Path,
        required=True,
        help="能力與驗證範圍表 TOML；必給，物理層不設隱含預設",
    )
    args = parser.parse_args(argv)
    try:
        if args.compare is not None:
            if args.input is not None:
                raise ValueError("--compare 模式不收剛性 input")
            if args.contracts is None:
                raise ValueError("--compare 模式必須給 --contracts")
            line = capability_line(args.capabilities, _CAPABILITY_IMPEDANCE_MATERIALS)
            tolerance_rel = load_precision_contracts(args.contracts)[
                "fem_vs_fenics_frozen"
            ].value
            report = _run_fenics_compare(args.compare, tolerance_rel)
            print(line)
            print(fenics_compare_table(report), end="")
            return 0 if report.within_contract else 1
        if args.input is None:
            raise ValueError("要給剛性 input，或改用 --compare FEniCS答案檔")
        line = capability_line(args.capabilities, _CAPABILITY_RIGID_MATERIALS)
        case = load_rigid_reference_case(args.input)
        pressures = solve_rigid_fem_lane_case(case)
        print(line)
        print(rigid_result_table(case, pressures), end="")
        return 0
    except Exception as exc:
        print(f"剛性參考房算不出來：{exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
