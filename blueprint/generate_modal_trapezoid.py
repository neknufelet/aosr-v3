"""一次產生梯形模態題目；凍結後考卷只讀 JSON，不再切網格。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gmsh
import numpy as np
from numpy.typing import NDArray

from aosr.geometry.shoebox import Wall
from aosr.geometry.shoebox_mesh import (
    ShoeboxMesh, _element_nodes, _orient_tetrahedra_positive, _sorted_nodes_and_tags,
)


def _walls(nodes: NDArray[np.float64], triangles: NDArray[np.int64]) -> NDArray[np.int64]:
    v = nodes[triangles]
    planes = (v[:, :, 2], v[:, :, 2] - 3, v[:, :, 0], v[:, :, 0] - 6,
              v[:, :, 1], v[:, :, 1] - 4 + v[:, :, 0] * (0.8 / 6))
    tolerance = 64 * np.finfo(float).eps * max(1.0, float(np.max(abs(nodes))))
    hits = np.asarray([np.all(abs(p) <= tolerance, axis=1) for p in planes])
    if not np.all(hits.sum(axis=0) == 1):
        raise ValueError("梯形邊界三角形必須恰屬一面牆")
    return np.asarray(np.argmax(hits, axis=0), dtype=np.int64)


def generate_mesh(c: float, cap: float, epw: int, seed: int) -> ShoeboxMesh:
    """與產品相同的單緒、固定種子 OCC 幾何與 gmsh P1 四面體流程。"""
    size = c / cap / epw
    gmsh.initialize(readConfigFiles=False)
    try:
        for name, value in (("General.Terminal", 0), ("General.NumThreads", 1),
                            ("Mesh.MaxNumThreads3D", 1), ("Mesh.RandomSeed", seed),
                            ("Mesh.MeshSizeMin", size), ("Mesh.MeshSizeMax", size)):
            gmsh.option.setNumber(name, float(value))
        gmsh.model.add("modal_trapezoid")
        points = [gmsh.model.occ.addPoint(x, y, 0) for x, y in ((0, 0), (6, 0), (6, 3.2), (0, 4))]
        edges = [gmsh.model.occ.addLine(a, b) for a, b in zip(points, (*points[1:], points[0]), strict=True)]
        face = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop(edges)])
        gmsh.model.occ.extrude([(2, face)], 0, 0, 3)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(3)
        nodes, tags = _sorted_nodes_and_tags()
        cells = _element_nodes(dimension=3, gmsh_element_type=4, nodes_per_element=4, sorted_node_tags=tags)
        triangles = _element_nodes(dimension=2, gmsh_element_type=2, nodes_per_element=3, sorted_node_tags=tags)
        return ShoeboxMesh(nodes, _orient_tetrahedra_positive(nodes, cells), triangles, _walls(nodes, triangles))
    finally:
        gmsh.finalize()


def build_problem(*, cap: float = 120.0, epw: int = 6) -> dict[str, object]:
    """案例物理量沿用既有凍結題目自己的條件；不讀產品預設。"""
    base = json.loads(Path(__file__).with_name("fem_fenics_problem.json").read_text(encoding="utf-8"))
    c, rho = float.fromhex(base["sound_speed_m_s"]), float.fromhex(base["density_kg_m3"])
    seed = 20261005
    mesh = generate_mesh(c, cap, epw, seed)
    factors: dict[str, list[float | None]] = {"rigid": [None] * len(Wall.all()), "uniform": [4.0] * len(Wall.all()),
               "unequal": [4.0, 5.0, 6.0, 7.0, 8.0, 9.0]}
    return {
        "schema": "fem-fenics-problem/modal-1",
        "geometry": {"floor_xy_m": [[0, 0], [6, 0], [6, 3.2], [0, 4]], "height_m": 3.0,
                     "wall_order": [wall.wall_name() for wall in Wall.all()]},
        "sound_speed_m_s": c.hex(), "density_kg_m3": rho.hex(), "frequency_max_hz": cap.hex(),
        "physics": {"element": "Lagrange P2", "quadrature_degree_volume": 4,
                    "quadrature_degree_boundary": 4, "time_convention": "exp(+j*omega*t)",
                    "eigenvalue": "k=omega/c", "polynomial": "K+j*k*Ct-k^2*M", "Ct": "rho*c*sum(B_w/Z_w)"},
        "v3_solver": {"shifts_hz": [30.0, 90.0, 130.0], "modes_per_shift": [48, 64, 80],
                      "count_band_edges_hz": [0.0, 40.0, 80.0, cap]},
        "cases": {name: {"wall_impedances": [None if f is None else float(f * rho * c).hex() for f in fs]}
                  for name, fs in factors.items()},
        "mesh": {"generator": "blueprint.generate_modal_trapezoid.generate_mesh", "gmsh_version": gmsh.__version__,
                 "max_frequency_hz": cap.hex(), "elements_per_wavelength": epw, "random_seed": seed,
                 "nodes": [[float(x).hex() for x in row] for row in mesh.nodes],
                 "tetrahedra": mesh.tetrahedra.tolist(), "boundary_triangles": mesh.boundary_triangles.tolist(),
                 "boundary_wall_indices": mesh.boundary_wall_indices.tolist()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cap", type=float, default=120.0)
    parser.add_argument("--epw", type=int, default=6)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("題目只能產生一次；不覆寫凍結網格")
    payload = build_problem(cap=args.cap, epw=args.epw)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
