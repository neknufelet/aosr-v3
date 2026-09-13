"""產生票 #213 的低頻 FEM 參考答案；必須在唯讀的 v3-donor 環境執行。

``flat`` 走上一代正式入口 ``real3d_shoebox_pressure``，六面牆皆使用
``4 * rho_c`` 的常數表面阻抗。``rigid`` 走底層
``solve_fem_3d_pressure``，邊界導納 ``gamma`` 全為零；它同時收錄正式頻率軸 A
與 VAL-1a maximin 離共振軸 B。產物記錄 donor、環境、實際命令、dtype 與可由
``float.fromhex`` 精確還原的複數壓力。每點的最近本徵頻依 VAL-1a 同一把尺，對每個
解析本徵頻 ``f_n`` 算 ``|f-f_n|/f_n`` 後取最小者，不先用 Hz 絕對距離挑模態。
這支刻意不印任何東西；成功與否由離開碼表示。
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
import os
import platform as _platform
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Final

import jax
import jax.numpy as jnp
import numpy as np


_fem_lane = importlib.import_module("lib.config.fem_lane")
_freq_axis = importlib.import_module("lib.config.freq_axis")
_materials_registry = importlib.import_module("lib.materials.registry")
_materials_response = importlib.import_module("lib.materials.response")
_fem_face_admittance = importlib.import_module("lib.physics.fem_face_admittance")
_fem_kernel = importlib.import_module("lib.physics.fem_kernel_3d")
_m4_pipeline = importlib.import_module("lib.physics.m4_pipeline")
_proxy_solver = importlib.import_module("lib.physics.proxy_solver")

FEM_FMAX_CAP_HZ = _fem_lane.FEM_FMAX_CAP_HZ
FREQ_AXIS = _freq_axis.FREQ_AXIS
constant_impedance = _materials_registry.constant_impedance
FrequencyAxis = _materials_response.FrequencyAxis
concat_boundary_faces = _fem_face_admittance.concat_boundary_faces
boundary_faces_by_id = _fem_kernel.boundary_faces_by_id
build_topology_3d = _fem_kernel.build_topology_3d
node_coords_extruded_quad = _fem_kernel.node_coords_extruded_quad
solve_fem_3d_pressure = _fem_kernel.solve_fem_3d_pressure
_real3d_mesh_dims = _m4_pipeline._real3d_mesh_dims
real3d_shoebox_pressure = _m4_pipeline.real3d_shoebox_pressure
CANONICAL_WALL_ORDER = _proxy_solver.CANONICAL_WALL_ORDER
ShoeboxGeometry = _proxy_solver.ShoeboxGeometry
ShoeboxWallSet = _proxy_solver.ShoeboxWallSet


def _donor_elements_per_wavelength() -> int:
    """讀 donor ``fem3d_mesh_resolution`` 的預設網格解析度。"""
    default = inspect.signature(_fem_kernel.fem3d_mesh_resolution).parameters[
        "elements_per_wavelength"
    ].default
    if isinstance(default, bool) or not isinstance(default, int):
        raise RuntimeError("donor elements_per_wavelength default is not an integer")
    return default


DONOR_TAG: Final[str] = "v3-donor"
ANSWER_SCHEMA: Final[int] = 1
CASES: Final[tuple[str, str]] = ("flat", "rigid")
ROOM: Final[tuple[float, float, float]] = (6.0, 4.0, 3.0)
SOURCE: Final[tuple[float, float, float]] = (1.5, 1.0, 1.2)
RECEIVER: Final[tuple[float, float, float]] = (4.0, 3.0, 1.5)
SOUND_SPEED: Final[float] = 343.0
RHO_C: Final[float] = 411.6
# 出處：donor lib.physics.fem_kernel_3d.fem3d_mesh_resolution 的預設參數。
ELEMENTS_PER_WAVELENGTH: Final[int] = _donor_elements_per_wavelength()
FLAT_IMPEDANCE: Final[complex] = complex(4.0 * RHO_C, 0.0)
RECORDED_PACKAGES: Final[tuple[str, ...]] = ("jax", "jaxlib", "numpy", "flax")
RECORDED_ENV: Final[tuple[str, ...]] = (
    "PYTHONPATH",
    "JAX_PLATFORMS",
    "PYTHONDONTWRITEBYTECODE",
)

Mesh = tuple[jax.Array, np.ndarray, np.ndarray, np.ndarray, int]


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed with exit {completed.returncode}:\n{completed.stderr}"
        )
    return completed.stdout.strip()


def _donor_root() -> Path:
    for entry in sys.path:
        if entry:
            candidate = Path(entry)
            if (candidate / "lib" / "physics" / "fem_kernel_3d.py").is_file():
                return candidate.resolve()
    raise RuntimeError("frozen donor root not found on PYTHONPATH")


def _donor_provenance(root: Path) -> dict[str, object]:
    commit = _git(root, "rev-parse", f"{DONOR_TAG}^{{commit}}")
    head = _git(root, "rev-parse", "HEAD")
    status = _git(root, "status", "--porcelain", "--ignored")
    if head != commit:
        raise RuntimeError(f"donor HEAD {head} is not {DONOR_TAG} commit {commit}")
    if status:
        raise RuntimeError(f"donor is not clean, including ignored files:\n{status}")
    return {"tag": DONOR_TAG, "commit": commit, "clean": True}


def _module_version(name: str) -> str:
    module: ModuleType = importlib.import_module(name)
    return str(getattr(module, "__version__", "unknown"))


def _command_record() -> dict[str, object]:
    return {
        "interpreter": sys.executable,
        "argv": list(sys.argv),
        "env": {name: os.environ.get(name) for name in RECORDED_ENV},
    }


def _environment_record(pressure_dtype: str) -> dict[str, object]:
    geometry = ShoeboxGeometry.of(*ROOM)
    return {
        "python": sys.version,
        "versions": {name: _module_version(name) for name in RECORDED_PACKAGES},
        "backend": str(jax.default_backend()),
        "x64": bool(getattr(jax.config, "jax_enable_x64")),
        "dtypes": {
            "frequency_axis": str(FREQ_AXIS.freqs_hz.dtype),
            "geometry": str(geometry.Lx.dtype),
            "rigid_gamma": str(jnp.zeros((1, len(CANONICAL_WALL_ORDER)), dtype=jnp.complex64).dtype),
            "pressure": pressure_dtype,
        },
        "platform": {
            "system": _platform.system(),
            "release": _platform.release(),
            "machine": _platform.machine(),
            "python_implementation": _platform.python_implementation(),
        },
    }


def _real_record(value: float) -> dict[str, str | None]:
    scalar = float(value)
    if not math.isfinite(scalar):
        return {"dec": None, "hex": None}
    return {"dec": repr(scalar), "hex": scalar.hex()}


def _complex_record(value: complex) -> dict[str, dict[str, str | None]]:
    number = complex(value)
    return {
        "real": _real_record(number.real),
        "imag": _real_record(number.imag),
        "abs": _real_record(abs(number)),
    }


def _formal_frequencies() -> list[float]:
    return [
        float(value)
        for value in np.asarray(FREQ_AXIS.freqs_hz)
        if float(value) <= float(FEM_FMAX_CAP_HZ)
    ]


def _val1a_eigenfrequencies() -> np.ndarray:
    values: list[float] = []
    for nx in range(10):
        for ny in range(10):
            for nz in range(10):
                if nx == ny == nz == 0:
                    continue
                values.append(
                    (SOUND_SPEED / 2.0)
                    * math.sqrt(
                        (nx / ROOM[0]) ** 2
                        + (ny / ROOM[1]) ** 2
                        + (nz / ROOM[2]) ** 2
                    )
                )
    return np.sort(np.asarray(values, dtype=float))


def _offresonance_frequencies() -> list[float]:
    modes = _val1a_eigenfrequencies()
    # 離共振點的 B 集合用原件的字面上下界，改成推導會讓整組位移、逐位相同破功。
    edges = np.linspace(10.0, 226.27, len(_formal_frequencies()) + 1)
    picks: list[float] = []
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        grid = np.linspace(lower, upper, 300)
        distances = np.min(
            np.abs(grid[:, None] - modes[None, :]) / modes[None, :], axis=1
        )
        picks.append(float(grid[int(np.argmax(distances))]))
    return picks


def _mesh() -> Mesh:
    geometry = ShoeboxGeometry.of(*ROOM)
    nx, ny, nz = _real3d_mesh_dims(
        geometry, None, None, None, sound_speed=SOUND_SPEED
    )
    coords = node_coords_extruded_quad(
        *(jnp.asarray(value, dtype=jnp.float32) for value in ROOM),
        jnp.zeros((4,), dtype=jnp.float32),
        nx,
        ny,
        nz,
    )
    tets = build_topology_3d(nx, ny, nz)
    tri_nodes, tri_face_idx = concat_boundary_faces(boundary_faces_by_id(nx, ny, nz))
    return coords, tets, tri_nodes, tri_face_idx, (nx + 1) * (ny + 1) * (nz + 1)


def _rigid_pressure(mesh: Mesh, frequency_hz: float) -> tuple[complex, str]:
    coords, tets, tri_nodes, tri_face_idx, n_nodes = mesh
    pressure = solve_fem_3d_pressure(
        coords,
        tets,
        tri_nodes,
        tri_face_idx,
        jnp.zeros((1, len(CANONICAL_WALL_ORDER)), dtype=jnp.complex64),
        jnp.asarray([2.0 * math.pi * frequency_hz], dtype=coords.dtype),
        jnp.asarray([SOURCE], dtype=coords.dtype),
        jnp.asarray([RECEIVER], dtype=coords.dtype),
        n_nodes,
        sound_speed=SOUND_SPEED,
    )
    pressure.block_until_ready()
    return complex(np.asarray(pressure)[0, 0]), str(pressure.dtype)


def _flat_pressure(frequency_hz: float) -> tuple[complex, str]:
    axis = FrequencyAxis.from_hz(
        jnp.asarray([frequency_hz], dtype=FREQ_AXIS.freqs_hz.dtype), resolution="custom"
    )
    material = constant_impedance(
        axis, FLAT_IMPEDANCE, material_id="ref-absorber-4rhoc"
    )
    walls = ShoeboxWallSet({wall: material for wall in CANONICAL_WALL_ORDER})
    pressure, mask = real3d_shoebox_pressure(
        walls,
        ShoeboxGeometry.of(*ROOM),
        jnp.asarray([SOURCE], dtype=jnp.float32),
        jnp.asarray([RECEIVER], dtype=jnp.float32),
        axis,
        jnp.zeros((4,), dtype=jnp.float32),
        None,
        None,
        None,
        sound_speed=SOUND_SPEED,
        rho_c=RHO_C,
    )
    pressure.block_until_ready()
    if np.asarray(mask).tolist() != [True]:
        raise RuntimeError(f"formal flat entry did not select {frequency_hz} Hz: {mask}")
    return complex(np.asarray(pressure)[0, 0]), str(pressure.dtype)


def _nearest_mode(frequency_hz: float) -> tuple[float, float]:
    """以 ``min(|f-f_n|/f_n)`` 挑模態；分母是各候選本徵頻 ``f_n``。"""
    modes = _val1a_eigenfrequencies()
    relative_distances = np.abs(modes - frequency_hz) / modes
    nearest_index = int(np.argmin(relative_distances))
    return float(modes[nearest_index]), float(relative_distances[nearest_index])


def _point(frequency_hz: float, set_name: str, pressure: complex) -> dict[str, object]:
    nearest, relative_distance = _nearest_mode(frequency_hz)
    return {
        "frequency_hz": frequency_hz,
        "set": set_name,
        "pressure": _complex_record(pressure),
        "nearest_eigenfrequency_hz": nearest,
        "nearest_eigenfrequency_relative_distance": relative_distance,
    }


def _mesh_record(mesh: Mesh) -> dict[str, object]:
    _coords, _tets, _tri_nodes, _tri_face_idx, n_nodes = mesh
    nx, ny, nz = _real3d_mesh_dims(
        ShoeboxGeometry.of(*ROOM), None, None, None, sound_speed=SOUND_SPEED
    )
    return {
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "n_nodes": n_nodes,
        "elements_per_wavelength": ELEMENTS_PER_WAVELENGTH,
        "f_max_cap_hz": float(FEM_FMAX_CAP_HZ),
    }


def _parameters(case: str, mesh: Mesh) -> dict[str, object]:
    material: object
    if case == "flat":
        material = {
            "id": "ref-absorber-4rhoc",
            "constructor": "constant_impedance",
            "all_six_faces_surface_impedance_pa_s_per_m": {
                "real": FLAT_IMPEDANCE.real,
                "imag": FLAT_IMPEDANCE.imag,
            },
        }
    else:
        material = {"boundary": "rigid", "gamma": 0.0, "faces": "all six"}
    return {
        "room_m": {"Lx": ROOM[0], "Ly": ROOM[1], "Lz": ROOM[2]},
        "source_xyz_m": list(SOURCE),
        "receiver_xyz_m": list(RECEIVER),
        "c_m_s": SOUND_SPEED,
        "rho_c_pa_s_per_m": RHO_C,
        "mesh": _mesh_record(mesh),
        "material": material,
    }


def _frequency_axis(case: str) -> dict[str, object]:
    formal = _formal_frequencies()
    sets: dict[str, list[float]] = {"A": formal}
    sources = [
        "lib/config/freq_axis.py:FREQ_AXIS",
        "lib/config/fem_lane.py:FEM_FMAX_CAP_HZ",
    ]
    if case == "rigid":
        sets["B"] = _offresonance_frequencies()
        sources.append("tests/test_val1a_analytical_fem_parity.py:VAL-1a maximin")
    return {
        "mode": "formal" if case == "flat" else "formal-plus-offresonance",
        "source": sources,
        "sets": sets,
    }


def _solve(case: str, mesh: Mesh) -> tuple[list[dict[str, object]], str]:
    axes = _frequency_axis(case)["sets"]
    if not isinstance(axes, dict):
        raise RuntimeError("frequency-axis sets are not a mapping")
    points: list[dict[str, object]] = []
    dtype = "unknown"
    for set_name, frequencies in axes.items():
        if not isinstance(set_name, str) or not isinstance(frequencies, list):
            raise RuntimeError("frequency-axis set has an invalid shape")
        for raw_frequency in frequencies:
            frequency_hz = float(raw_frequency)
            if case == "flat":
                pressure, dtype = _flat_pressure(frequency_hz)
            else:
                pressure, dtype = _rigid_pressure(mesh, frequency_hz)
            points.append(_point(frequency_hz, set_name, pressure))
    return points, dtype


def _payload(case: str) -> dict[str, object]:
    donor_root = _donor_root()
    mesh = _mesh()
    points, pressure_dtype = _solve(case, mesh)
    return {
        "schema": ANSWER_SCHEMA,
        "note": (
            "Generated by blueprint/generate_fem_answers.py from the read-only "
            "v3-donor; rerun the generator instead of editing measured values by hand."
        ),
        "donor": _donor_provenance(donor_root),
        "env": _environment_record(pressure_dtype),
        "command": _command_record(),
        "parameters": _parameters(case, mesh),
        "frequency_axis": _frequency_axis(case),
        "points": points,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = _payload(str(args.case))
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
