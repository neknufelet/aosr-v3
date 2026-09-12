"""把 donor 的多接收點與分格材料振幅跑成票 #219 的標準答案檔。

介面是 ``--case multi_flat|multi_varied|patch_varied --out PATH``。這支只寫 ``--out``，不印
任何東西；答案的浮點格沿用 :mod:`blueprint.generate_amplitude_answers` 的 dec/hex 形狀。凍結
原件 ``run/seg1_ism_multi_patch.py`` 的 ``controls`` 記錄
``all_varied_cells_paths_match_reference_varied`` 與
``all_varied_cells_totals_match_reference_varied`` 都是 True；考卷保留這個四格全 varied 控制組，
但不另存一份答案檔。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Final, Protocol

from blueprint import generate_amplitude_answers as base

SCHEMA: Final[int] = base.ANSWER_SCHEMA
CASES: Final[tuple[str, ...]] = ("multi_flat", "multi_varied", "patch_varied")
ROOM: Final[tuple[float, float, float]] = (
    base.ROOM_LX,
    base.ROOM_LY,
    base.ROOM_LZ,
)
RECEIVERS: Final[dict[str, tuple[float, float, float]]] = {
    "R0": base.RECEIVER_XYZ,
    "R1": (2.2, 2.1, 1.15),
    "R2": (5.1, 1.3, 2.05),
}
WHOLE_WALL_GRIDS: Final[tuple[tuple[int, int], ...]] = base.GRID_SHAPES
PATCH_GRIDS: Final[tuple[tuple[int, int], ...]] = (
    (1, 1),
    (1, 1),
    (1, 1),
    (1, 1),
    (2, 2),
    (1, 1),
)
PATCH_Z_REAL: Final[float] = 10.0 * base.RHO_C


class _FrequencyAxis(Protocol):
    """本產生器只讀 donor frequency axis 的頻率陣列。"""

    freqs_hz: object


class _FloatArray(Protocol):
    """donor 回傳、以多維 index 取浮點值的陣列。"""

    shape: tuple[int, ...]

    def __getitem__(self, key: object) -> float: ...


class _ComplexArray(Protocol):
    """donor 回傳、以多維 index 取複數值的陣列。"""

    shape: tuple[int, ...]

    def __getitem__(self, key: object) -> complex: ...


class _Paths(Protocol):
    dist: _FloatArray
    delay: _FloatArray
    cell_gid: _FloatArray


class _Result(Protocol):
    reflection_product: _ComplexArray
    pressure: _ComplexArray
    ism_direct_E: _FloatArray
    ism_rev_E: _FloatArray


class _ImagePath(Protocol):
    order: int
    identity: tuple[int, ...]


def donor_provenance(root: Path) -> dict[str, object]:
    """沿用基礎產生器的出身檢查，再把 ignored 檔也納入乾淨判準。"""
    provenance = base.donor_provenance(root)
    dirty = base._git(root, "status", "--porcelain", "--ignored")
    if dirty:
        raise SystemExit(f"{root} 有 {len(dirty.splitlines())} 筆狀態，拒絕產出")
    return provenance


def _frequency_axis(modules: dict[str, ModuleType]) -> _FrequencyAxis:
    np = base._np_module()
    frequency_axis: _FrequencyAxis = modules["response"].FrequencyAxis.from_hz(
        np.asarray(base.FREQS_HZ, dtype=float), resolution="custom"
    )
    return frequency_axis


def _materials(
    modules: dict[str, ModuleType], frequency_axis: _FrequencyAxis
) -> tuple[object, object, object]:
    np = base._np_module()
    flat = modules["registry"].constant_impedance(
        frequency_axis,
        complex(base.MATERIAL_Z_FLAT_REAL, 0.0),
        material_id=base.MATERIAL_ID_FLAT,
    )
    varied_values = [
        complex(real, imag)
        for real, imag in zip(
            base.VARIED_Y0_REAL_KK, base.VARIED_Y0_IMAG_KK, strict=True
        )
    ]
    varied = modules["response"].MaterialResponse.from_impedance(
        np.asarray(varied_values, dtype=complex),
        frequency_axis,
        material_id=base.MATERIAL_ID_VARIED,
    )
    replacement = modules["registry"].constant_impedance(
        frequency_axis,
        complex(PATCH_Z_REAL, 0.0),
        material_id="ref-absorber-y0-cell-00-10rhoc",
    )
    return flat, varied, replacement


def _whole_wall_materials(
    modules: dict[str, ModuleType], frequency_axis: _FrequencyAxis, case: str
) -> object:
    flat, varied, _replacement = _materials(modules, frequency_axis)
    by_wall = {wall: flat for wall in base.CANONICAL_WALL_ORDER}
    if case == "varied":
        by_wall["y0"] = varied
    grids = tuple(
        modules["patch"].WallPatchGrid(wall, 1, 1, (by_wall[wall],))
        for wall in base.CANONICAL_WALL_ORDER
    )
    return modules["patch"].WallPatchMaterials(grids, frequency_axis)


def _patch_materials(
    modules: dict[str, ModuleType], frequency_axis: _FrequencyAxis
) -> object:
    flat, varied, replacement = _materials(modules, frequency_axis)
    y0_cells = (replacement, varied, varied, varied)
    grids = []
    for wall, (rows, cols) in zip(
        base.CANONICAL_WALL_ORDER, PATCH_GRIDS, strict=True
    ):
        cells = y0_cells if wall == "y0" else (flat,)
        grids.append(modules["patch"].WallPatchGrid(wall, rows, cols, cells))
    return modules["patch"].WallPatchMaterials(tuple(grids), frequency_axis)


def _cell_triplet(gid: int) -> dict[str, object]:
    offset = 0
    for wall, (rows, cols) in zip(
        base.CANONICAL_WALL_ORDER, PATCH_GRIDS, strict=True
    ):
        count = rows * cols
        if offset <= gid < offset + count:
            local = gid - offset
            return {"wall": wall, "row": local // cols, "col": local % cols}
        offset += count
    raise ValueError(f"cell_gid {gid} 超出 {offset} 格")


def _format_receiver(
    paths: _Paths,
    result: _Result,
    path_pressure: _ComplexArray,
    reproduced: _ComplexArray,
    image_paths: tuple[_ImagePath, ...],
    receiver_index: int,
    *,
    include_cells: bool,
) -> dict[str, object]:
    frequency_count = int(path_pressure.shape[2])
    path_count = len(image_paths)
    records = base._format_paths(
        [float(paths.dist[receiver_index, index]) for index in range(path_count)],
        [float(paths.delay[receiver_index, index]) for index in range(path_count)],
        [int(path.order) for path in image_paths],
        [[int(value) for value in path.identity] for path in image_paths],
        [
            [
                complex(result.reflection_product[receiver_index, index, band])
                for band in range(frequency_count)
            ]
            for index in range(path_count)
        ],
        [
            [
                complex(path_pressure[receiver_index, index, band])
                for band in range(frequency_count)
            ]
            for index in range(path_count)
        ],
    )
    if include_cells:
        for index, record in enumerate(records):
            record["bounce_cells"] = [
                _cell_triplet(int(paths.cell_gid[receiver_index, index, bounce]))
                for bounce in range(image_paths[index].order)
            ]
    totals = base._format_totals(
        [
            complex(result.pressure[receiver_index, band])
            for band in range(frequency_count)
        ],
        [
            complex(reproduced[receiver_index, band])
            for band in range(frequency_count)
        ],
        [
            float(result.ism_direct_E[receiver_index, band])
            for band in range(frequency_count)
        ],
        [
            float(result.ism_rev_E[receiver_index, band])
            for band in range(frequency_count)
        ],
    )
    return {"paths": records, "totals": totals}


def _run_case(
    modules: dict[str, ModuleType],
    frequency_axis: _FrequencyAxis,
    wall_materials: object,
    receivers: dict[str, tuple[float, float, float]],
    grid_shapes: tuple[tuple[int, int], ...],
    *,
    include_cells: bool,
) -> tuple[dict[str, dict[str, object]], dict[str, str]]:
    np = base._np_module()
    jnp = base._jnp_module()
    modules["ism"].validate_patch_materials(wall_materials, base.RHO_C)
    z_cells, _offsets = modules["ism"].flatten_cell_Z(wall_materials)
    geometry = modules["proxy"].ShoeboxGeometry.of(*ROOM)
    source = np.array(base.SOURCE_XYZ, dtype=float)
    receiver_array = np.array(list(receivers.values()), dtype=float)
    paths = modules["ism"].precompute_shoebox_patch_paths(
        geometry,
        source,
        receiver_array,
        grid_shapes,
        max_order=base.MAX_ORDER,
        sound_speed=base.SOUND_SPEED,
        wall_edges=None,
    )
    result = modules["ism"].solve_ism_shoebox_patched(
        paths, z_cells, frequency_axis, rho_c=base.RHO_C, directivity=None
    )
    image_paths = modules["ism"]._enumerate_image_paths(
        source, np.array(ROOM, dtype=float), base.MAX_ORDER
    )
    gid = jnp.asarray(paths.cell_gid)
    cosine = jnp.asarray(paths.cos)[..., None]
    valid = jnp.asarray(paths.valid_mask)[None, :, :, None]
    bounce_r = modules["adapter"].z_to_r_from_cos(z_cells[gid], cosine, base.RHO_C)
    reflection = jnp.prod(jnp.where(valid, bounce_r, 1.0 + 0.0j), axis=2)
    distance = jnp.maximum(jnp.asarray(paths.dist), 1e-12)
    omega = 2.0 * jnp.pi * jnp.asarray(frequency_axis.freqs_hz)
    phase = jnp.exp(-1j * omega[None, None, :] * jnp.asarray(paths.delay)[:, :, None])
    path_pressure = (1.0 / distance)[:, :, None] * reflection * phase
    reproduced = jnp.sum(path_pressure, axis=1)
    records = {
        receiver_id: _format_receiver(
            paths,
            result,
            path_pressure,
            reproduced,
            image_paths,
            receiver_index,
            include_cells=include_cells,
        )
        for receiver_index, receiver_id in enumerate(receivers)
    }
    return records, base._live_dtypes(z_cells, result, path_pressure)


def _impedance(real: list[float], imag: list[float]) -> dict[str, object]:
    return {
        "per_frequency_real": real,
        "per_frequency_imag": imag,
        "unit": "Pa·s/m",
    }


def _flat_impedance() -> dict[str, object]:
    count = len(base.FREQS_HZ)
    return _impedance([base.MATERIAL_Z_FLAT_REAL] * count, [0.0] * count)


def _varied_impedance() -> dict[str, object]:
    return _impedance(
        list(base.VARIED_Y0_REAL_KK), list(base.VARIED_Y0_IMAG_KK)
    )


def _replacement_impedance() -> dict[str, object]:
    count = len(base.FREQS_HZ)
    return _impedance([PATCH_Z_REAL] * count, [0.0] * count)


def _wall_cells() -> dict[str, list[dict[str, object]]]:
    output: dict[str, list[dict[str, object]]] = {}
    y0_rows, y0_cols = PATCH_GRIDS[base.CANONICAL_WALL_ORDER.index("y0")]
    for wall in base.CANONICAL_WALL_ORDER:
        if wall != "y0":
            output[wall] = [{"row": 0, "col": 0, "impedance": _flat_impedance()}]
            continue
        output[wall] = [
            {
                "row": row,
                "col": col,
                "impedance": _replacement_impedance()
                if (row, col) == (0, 0)
                else _varied_impedance(),
            }
            for row in range(y0_rows)
            for col in range(y0_cols)
        ]
    return output


def _common_parameters() -> dict[str, object]:
    return {
        "room": {"Lx_m": ROOM[0], "Ly_m": ROOM[1], "Lz_m": ROOM[2]},
        "source_xyz_m": {
            "x": base.SOURCE_XYZ[0],
            "y": base.SOURCE_XYZ[1],
            "z": base.SOURCE_XYZ[2],
        },
        "sound_speed_m_s": base.SOUND_SPEED,
        "rho_c_pa_s_per_m": base.RHO_C,
        "max_order": base.MAX_ORDER,
        "wall_edges": None,
        "directivity": None,
        "frequencies_hz": list(base.FREQS_HZ),
        "frequency_resolution": "custom",
        "convention": (
            "origin at room corner; x in [0,Lx], y in [0,Ly], z in [0,Lz]; "
            "z=0 is floor, z=Lz ceiling. Wall ids in canonical order: "
            + ", ".join(base.CANONICAL_WALL_ORDER)
            + "."
        ),
        "units": (
            "metres (m), seconds (s), metres/second (m/s), hertz (Hz), "
            "Pa·s/m (impedance, rho_c)"
        ),
    }


def _multi_parameters() -> dict[str, object]:
    params = _common_parameters()
    params["receivers_xyz_m"] = [
        {"id": receiver_id, "x": xyz[0], "y": xyz[1], "z": xyz[2]}
        for receiver_id, xyz in RECEIVERS.items()
    ]
    params["grid_shapes"] = [list(shape) for shape in WHOLE_WALL_GRIDS]
    params["cases"] = {
        "flat": {
            "all_six_walls": _flat_impedance(),
            "material_id": base.MATERIAL_ID_FLAT,
        },
        "varied": {
            "five_walls": _flat_impedance(),
            "five_walls_applied_to": ["floor", "ceiling", "x0", "xL", "yL"],
            "y0_wall": _varied_impedance(),
            "material_id": base.MATERIAL_ID_VARIED,
        },
    }
    return params


def _patch_parameters() -> dict[str, object]:
    params = _common_parameters()
    xyz = RECEIVERS["R0"]
    params["receiver_xyz_m"] = {"id": "R0", "x": xyz[0], "y": xyz[1], "z": xyz[2]}
    params["grid_shapes"] = [list(shape) for shape in PATCH_GRIDS]
    params["cases"] = {
        "patched_y0_cell_0_0_10rhoc": {"wall_cells": _wall_cells()}
    }
    return params


def _environment(dtypes: dict[str, str]) -> dict[str, object]:
    env = base.environment()
    env["dtypes"] = dtypes
    return env


def build_payload(root: Path, case: str) -> dict[str, object]:
    """跑指定 case，組成答案檔頂層。"""
    modules = base._donor_modules()
    frequency_axis = _frequency_axis(modules)
    if case.startswith("multi_"):
        material_case = case.removeprefix("multi_")
        records, dtypes = _run_case(
            modules,
            frequency_axis,
            _whole_wall_materials(modules, frequency_axis, material_case),
            RECEIVERS,
            WHOLE_WALL_GRIDS,
            include_cells=False,
        )
        body = {
            "parameters": _multi_parameters(),
            "receivers": records,
            "totals_by_receiver": {
                receiver_id: record["totals"]
                for receiver_id, record in records.items()
            },
        }
    else:
        records, dtypes = _run_case(
            modules,
            frequency_axis,
            _patch_materials(modules, frequency_axis),
            {"R0": RECEIVERS["R0"]},
            PATCH_GRIDS,
            include_cells=True,
        )
        body = {
            "parameters": _patch_parameters(),
            "paths": records["R0"]["paths"],
            "totals": records["R0"]["totals"],
        }
    return {
        "schema": SCHEMA,
        "note": (
            "由 blueprint/generate_multi_receiver_answers.py 在唯讀 donor 上產生，"
            "不准手改；要改就重跑產生器。"
        ),
        "donor": donor_provenance(root),
        "env": _environment(dtypes),
        "command": base.command_record(),
        **body,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=list(CASES))
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    payload = build_payload(base.donor_root_from_env(), args.case)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
