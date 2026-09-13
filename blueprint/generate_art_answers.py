"""把凍結 donor 的 ART 晚期混響能量跑成三份標準答案檔。

``--case flat|varied|lowabs`` 選材料，``--out`` 指定產物。這支只在上一代環境執行；它從
``PYTHONPATH`` 找唯一含 ``lib/physics/art_kernel.py`` 的 donor，量 ``v3-donor`` 的 commit、
HEAD、exact tag 與含 ignored files 的乾淨度，任何一格不符就拒絕產出。數值沿用票 #215
第 1 段原稿的同一條 donor 路徑、同一個 n_per_wall=6 與 float32 dtype；答案中的每個實數
同時存 ``dec`` 與可逐位還原的 ``hex``。
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import platform as _platform
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Final, Protocol

DONOR_TAG: Final[str] = "v3-donor"
ANSWER_SCHEMA: Final[int] = 1
CASES: Final[tuple[str, ...]] = ("flat", "varied", "lowabs")
ROOM: Final[tuple[float, float, float]] = (6.0, 4.0, 3.0)
SOUND_SPEED: Final[float] = 343.0
RHO_C: Final[float] = 411.6
FREQUENCIES_HZ: Final[tuple[float, ...]] = (125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0)
WALL_ORDER: Final[tuple[str, ...]] = ("floor", "ceiling", "x0", "xL", "y0", "yL")
RECORDED_PACKAGES: Final[tuple[str, ...]] = ("jax", "jaxlib", "numpy", "flax")
RECORDED_ENV_VARS: Final[tuple[str, ...]] = (
    "PYTHONPATH",
    "JAX_PLATFORMS",
    "PYTHONDONTWRITEBYTECODE",
)
N_PER_WALL_SOURCE: Final[str] = "lib/config/art_lane.py:5 ART_N_PER_WALL_DEFAULT"
NEUMANN_EPS_SOURCE: Final[str] = "lib/config/art_lane.py:8 ART_NEUMANN_EPS_TAIL"
POWERITER_K_SOURCE: Final[str] = "lib/config/art_lane.py:9 ART_POWERITER_K"
NEUMANN_K_SOURCE: Final[str] = "lib/config/art_lane.py:24 ART_NEUMANN_K_MAX"
DOMAIN_MAX_SOURCE: Final[str] = "lib/physics/m4_pipeline.py:1104 ART_RADIOSITY_ALPHA_BAR_MAX"
MATERIAL_ID_FLAT: Final[str] = "ref-absorber-4rhoc"
MATERIAL_ID_VARIED: Final[str] = "ref-absorber-y0-varied"
MATERIAL_ID_LOWABS: Final[str] = "ref-absorber-10rhoc-lowabs"


class _Array(Protocol):
    """產生器從 donor 陣列用到的最小形狀。"""

    dtype: object

    def __getitem__(self, key: object) -> float: ...

    def __iter__(self) -> Iterator[float]: ...


def _real(value: float) -> dict[str, str]:
    """有限實數的可讀十進位與逐位十六進位表示。"""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"答案不接受非有限值：{number!r}")
    return {"dec": repr(number), "hex": number.hex()}


def _integer(value: int | float) -> dict[str, str]:
    """整數的十進位與十六進位表示。"""
    number = int(value)
    return {"dec": str(number), "hex": hex(number)}


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.rstrip("\n")


def donor_root_from_env() -> Path:
    """從呼叫端的 module path 找唯一 donor；不在程式內改 import path。"""
    candidates = []
    for item in sys.path:
        if not item:
            continue
        candidate = Path(item)
        if (candidate / "lib" / "physics" / "art_kernel.py").is_file():
            candidates.append(candidate.resolve())
    if len(candidates) != 1:
        raise SystemExit(f"預期恰有一棵 ART donor，實際找到 {candidates!r}")
    return candidates[0]


def donor_provenance(root: Path) -> dict[str, object]:
    """量 donor 的 tag、commit 與含 ignored files 的乾淨度。"""
    commit = _git(root, "rev-parse", f"{DONOR_TAG}^{{commit}}")
    head = _git(root, "rev-parse", "HEAD")
    exact_tag = _git(root, "describe", "--tags", "--exact-match", "HEAD")
    dirty = _git(root, "status", "--porcelain", "--ignored")
    if head != commit:
        raise SystemExit(f"donor HEAD {head} 不是 {DONOR_TAG} commit {commit}")
    if exact_tag != DONOR_TAG:
        raise SystemExit(f"donor exact tag 是 {exact_tag!r}，不是 {DONOR_TAG!r}")
    if dirty:
        raise SystemExit(f"donor 含 dirty 或 ignored files，拒絕產出：\n{dirty}")
    return {"tag": exact_tag, "commit": commit, "clean": True}


def _module_version(module: ModuleType) -> str:
    return str(getattr(module, "__version__", "unknown"))


def environment() -> dict[str, object]:
    """記錄會影響答案位元的執行環境；dtypes 在現場運算後補入。"""
    jax = importlib.import_module("jax")
    versions = {
        name: _module_version(importlib.import_module(name)) for name in RECORDED_PACKAGES
    }
    return {
        "python": sys.version,
        "versions": versions,
        "backend": str(jax.default_backend()),
        "x64": bool(jax.config.jax_enable_x64),
        "platform": {
            "system": _platform.system(),
            "release": _platform.release(),
            "machine": _platform.machine(),
            "python_implementation": _platform.python_implementation(),
        },
    }


def command_record() -> dict[str, object]:
    """記錄直譯器、argv 與重跑需要的三個環境變數。"""
    return {
        "interpreter": sys.executable,
        "argv": list(sys.argv),
        "env": {name: os.environ.get(name) for name in RECORDED_ENV_VARS},
    }


def _modules() -> dict[str, ModuleType]:
    """動態載入 donor 與其數值後端；新家不 import 這支產生器。"""
    return {
        name: importlib.import_module(path)
        for name, path in (
            ("jax", "jax"),
            ("jnp", "jax.numpy"),
            ("np", "numpy"),
            ("lane", "lib.config.art_lane"),
            ("response", "lib.materials.response"),
            ("art", "lib.physics.art_kernel"),
            ("connected", "lib.physics.connected_forward"),
            ("m4", "lib.physics.m4_pipeline"),
            ("proxy", "lib.physics.proxy_solver"),
        )
    }


def _impedance_values(case: str) -> dict[str, list[complex]]:
    flat = [complex(4.0 * RHO_C, 0.0)] * len(FREQUENCIES_HZ)
    values = {wall: list(flat) for wall in WALL_ORDER}
    if case == "lowabs":
        lowabs = [complex(10.0 * RHO_C, 0.0)] * len(FREQUENCIES_HZ)
        return {wall: list(lowabs) for wall in WALL_ORDER}
    if case == "varied":
        values["y0"] = [
            complex((2.0 + 0.5 * index) * RHO_C, 0.3 * RHO_C * ((-1) ** index))
            for index in range(len(FREQUENCIES_HZ))
        ]
    return values


def _material_from_z(
    response: ModuleType,
    np: ModuleType,
    freq_axis: object,
    values: list[complex],
    material_id: str,
) -> object:
    z_array = np.asarray(values, dtype=np.complex64)
    return response.MaterialResponse.from_impedance(
        z_array,
        freq_axis,
        material_id=material_id,
        boundary_model="local_impedance",
        is_locally_reacting=True,
    )


def _wall_set(
    case: str,
    response: ModuleType,
    proxy: ModuleType,
    np: ModuleType,
    freq_axis: object,
) -> object:
    impedance = _impedance_values(case)
    if case == "lowabs":
        material = _material_from_z(
            response, np, freq_axis, impedance["floor"], MATERIAL_ID_LOWABS
        )
        return proxy.ShoeboxWallSet(
            {wall: material for wall in proxy.CANONICAL_WALL_ORDER},
            context=f"ticket-215-{case}",
        )
    flat = _material_from_z(response, np, freq_axis, impedance["floor"], MATERIAL_ID_FLAT)
    by_wall = {wall: flat for wall in proxy.CANONICAL_WALL_ORDER}
    if case == "varied":
        by_wall[proxy.WallId.Y0] = _material_from_z(
            response, np, freq_axis, impedance["y0"], MATERIAL_ID_VARIED
        )
    return proxy.ShoeboxWallSet(by_wall, context=f"ticket-215-{case}")


def _material_record(case: str, np: ModuleType) -> dict[str, object]:
    """把實際送進 MaterialResponse 的 complex64 阻抗逐格記進答案。"""
    by_wall = {}
    for wall, values in _impedance_values(case).items():
        narrowed = np.asarray(values, dtype=np.complex64)
        by_wall[wall] = [
            {"real": _real(complex(value).real), "imag": _real(complex(value).imag)}
            for value in narrowed
        ]
    descriptions = {
        "flat": "all walls Z=4*rho_c+0j",
        "varied": (
            "five walls Z=4*rho_c+0j; y0 Z_k=(2+0.5*k)*rho_c"
            "+1j*(0.3*rho_c)*((-1)**k)"
        ),
        "lowabs": "all walls Z=10*rho_c+0j",
    }
    material_ids = {
        wall: (
            MATERIAL_ID_LOWABS
            if case == "lowabs"
            else MATERIAL_ID_VARIED
            if case == "varied" and wall == "y0"
            else MATERIAL_ID_FLAT
        )
        for wall in WALL_ORDER
    }
    return {
        "case": case,
        "description": descriptions[case],
        "boundary_model": "local_impedance",
        "is_locally_reacting": True,
        "impedance_unit": "Pa*s/m",
        "material_ids_by_wall": material_ids,
        "impedance_by_wall": by_wall,
    }


def _parameters(case: str, modules: dict[str, ModuleType], n_per_wall: int) -> dict[str, object]:
    lane = modules["lane"]
    m4 = modules["m4"]
    return {
        "room_m": {"Lx": _real(ROOM[0]), "Ly": _real(ROOM[1]), "Lz": _real(ROOM[2])},
        "sound_speed_m_s": _real(SOUND_SPEED),
        "rho_c_pa_s_per_m": _real(RHO_C),
        "frequencies_hz": [_real(value) for value in FREQUENCIES_HZ],
        "wall_order": list(WALL_ORDER),
        "n_per_wall": {**_integer(n_per_wall), "source": N_PER_WALL_SOURCE},
        "ART_POWERITER_K": {
            **_integer(lane.ART_POWERITER_K),
            "source": POWERITER_K_SOURCE,
        },
        "ART_NEUMANN_K_MAX": {
            **_integer(lane.ART_NEUMANN_K_MAX),
            "source": NEUMANN_K_SOURCE,
        },
        "ART_NEUMANN_EPS_TAIL": {
            **_real(lane.ART_NEUMANN_EPS_TAIL),
            "source": NEUMANN_EPS_SOURCE,
        },
        "ART_RADIOSITY_ALPHA_BAR_MAX": {
            **_real(m4.ART_RADIOSITY_ALPHA_BAR_MAX),
            "source": DOMAIN_MAX_SOURCE,
        },
        "material": _material_record(case, modules["np"]),
    }


def _format_bands(
    raw: _Array,
    late: _Array,
    late_cell_alpha: _Array,
    spectral_radius: _Array,
    tail_steps: _Array,
    area_values: _Array,
    domain_max: float,
) -> list[dict[str, object]]:
    """把 donor 陣列依頻帶收成答案形狀；所有實數在這裡轉成 dec/hex。"""
    areas = [float(value) for value in area_values]
    total_area = sum(areas)
    bands = []
    for band_index, frequency in enumerate(FREQUENCIES_HZ):
        alpha = {
            wall: _real(late_cell_alpha[wall_index, band_index])
            for wall_index, wall in enumerate(WALL_ORDER)
        }
        alpha_bar = sum(
            area * float(late_cell_alpha[wall_index, band_index])
            for wall_index, area in enumerate(areas)
        ) / total_area
        raw_value = float(raw[band_index])
        late_value = float(late[band_index])
        ratio = late_value / raw_value if raw_value != 0.0 else 0.0
        bands.append(
            {
                "frequency_hz": _real(frequency),
                "alpha_by_wall": alpha,
                "alpha_bar": _real(alpha_bar),
                "in_domain": alpha_bar <= domain_max,
                "raw_art_rev_E": _real(raw_value),
                "late_rev_E": _real(late_value),
                "eyring_ratio": _real(ratio),
                "spectral_radius": _real(spectral_radius[band_index]),
                "neumann_orders_used": _integer(tail_steps[band_index]),
            }
        )
    return bands


def _neumann_diagnostics(
    modules: dict[str, ModuleType],
    transfer: _Array,
) -> tuple[_Array, _Array]:
    """依 donor 的 Perron root 與截尾規則重建每頻帶階數。"""
    jax = modules["jax"]
    jnp = modules["jnp"]
    lane = modules["lane"]
    art = modules["art"]
    # Perron root 與截尾階數逐式錨在 donor lib/physics/art_kernel.py:353-358。
    spectral_radius, _eigenvector = art.art_perron_eig(transfer, K=lane.ART_POWERITER_K)
    valid_rho = (spectral_radius > 0.0) & (spectral_radius < 1.0)
    safe_rho = jnp.where(valid_rho, spectral_radius, jnp.ones_like(spectral_radius) * 0.5)
    raw_tail = jnp.log(jnp.asarray(lane.ART_NEUMANN_EPS_TAIL, dtype=transfer.dtype)) / jnp.log(
        safe_rho
    )
    tail_steps = jnp.ceil(jnp.where(valid_rho, raw_tail, jnp.zeros_like(raw_tail)))
    tail_steps = jnp.clip(tail_steps, 1.0, float(lane.ART_NEUMANN_K_MAX)).astype(jnp.int32)
    jax.block_until_ready(spectral_radius)
    jax.block_until_ready(tail_steps)
    return spectral_radius, tail_steps


def _build_case(case: str, modules: dict[str, ModuleType]) -> tuple[list[dict[str, object]], dict[str, str], int]:
    """跑 donor 的 n=6 ART 與 Eyring 修正，回 bands、現場 dtypes、n_per_wall。"""
    jax = modules["jax"]
    np = modules["np"]
    lane = modules["lane"]
    response = modules["response"]
    art = modules["art"]
    connected = modules["connected"]
    m4 = modules["m4"]
    proxy = modules["proxy"]

    n_per_wall = int(lane.ART_N_PER_WALL_DEFAULT)
    freq_axis = response.FrequencyAxis.from_hz(
        np.asarray(FREQUENCIES_HZ, dtype=float),
        resolution="custom",
    )
    geom = proxy.ShoeboxGeometry.of(*ROOM)
    wall_set = _wall_set(case, response, proxy, np, freq_axis)
    token = m4.ValidatedMaterials.validate(wall_set, freq_axis, rho_c=RHO_C)
    raw = art.solve_art_reverberant_energy(
        geom,
        token.wall_set,
        freq_axis,
        sound_speed=SOUND_SPEED,
        rho_c=RHO_C,
        n_per_wall=n_per_wall,
        validate_materials=False,
    )
    late_cell_alpha = proxy.material_alpha_matrix(
        token.materials,
        freq_axis,
        RHO_C,
        validate=False,
    )
    late = connected.art_to_eyring_room_constant_correction(
        raw,
        geom.wall_areas,
        late_cell_alpha,
    )

    patches = art.art_patch_geometry(geom, n_per_wall)
    form_factors = art.art_form_factors(patches)
    alpha_patch = late_cell_alpha[patches.wall_index, :]
    transfer = art.art_transfer_operator(form_factors, alpha_patch)
    spectral_radius, tail_steps = _neumann_diagnostics(modules, transfer)
    jax.block_until_ready(late)

    domain_max = float(m4.ART_RADIOSITY_ALPHA_BAR_MAX)
    bands = _format_bands(
        raw,
        late,
        late_cell_alpha,
        spectral_radius,
        tail_steps,
        geom.wall_areas,
        domain_max,
    )
    dtypes = {
        "raw_art_rev_E": str(raw.dtype),
        "late_rev_E": str(late.dtype),
        "alpha": str(late_cell_alpha.dtype),
        "form_factors": str(form_factors.dtype),
        "transfer_operator": str(transfer.dtype),
        "spectral_radius": str(spectral_radius.dtype),
        "neumann_orders": str(tail_steps.dtype),
    }
    return bands, dtypes, n_per_wall


def build_payload(root: Path, case: str) -> dict[str, object]:
    """組成一份答案：出身、環境、重跑命令、參數與逐頻帶值。"""
    modules = _modules()
    bands, dtypes, n_per_wall = _build_case(case, modules)
    env = environment()
    env["dtypes"] = dtypes
    return {
        "schema": ANSWER_SCHEMA,
        "note": (
            "由 blueprint/generate_art_answers.py 在唯讀的凍結 donor 上產生，不准手改；"
            "數值逐格取自票 #215 第 1 段同一條 n_per_wall=6 float32 計算路徑。"
        ),
        "donor": donor_provenance(root),
        "env": env,
        "command": command_record(),
        "parameters": _parameters(case, modules, n_per_wall),
        "bands": bands,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="產生凍結 donor 的 ART 標準答案")
    parser.add_argument("--case", required=True, choices=list(CASES))
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    payload = build_payload(donor_root_from_env(), args.case)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
