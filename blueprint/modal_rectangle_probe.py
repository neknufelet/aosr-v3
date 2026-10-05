"""第 1 步參考房延拓證據：只讀題目、現算結果，不接搜尋或排名。

重跑：uv run python -m blueprint.modal_rectangle_probe --seed-frequency-max-hz 330
加 --max-step 0.025 可核對步長，加 --seed-frequency-max-hz 360 擴大起點範圍。
這是半解析證據，不重跑第 0 步、不作有限元素精度判定，不儲存狀態報表。
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from aosr.physics.modal_convention import ModalKind
from aosr.physics.modal_rectangle_truth import (
    ContinuationOutcome,
    RectangleModalProblem,
    trace_modes,
)
from governance.exit_codes import note


def reference_evidence(
    problem_path: Path, cap: float, seed_cap: float, step: float, *, details: bool = False,
) -> dict[str, object]:
    """案例幾何與空氣條件只讀題目檔；回傳可追查的逐指標跨界證據。"""
    data = json.loads(problem_path.read_text(encoding="utf-8"))
    room = data["room_m"]
    lengths = tuple(float.fromhex(room[axis]) for axis in ("Lx", "Ly", "Lz"))
    speed = float.fromhex(data["sound_speed_m_s"])
    density = float.fromhex(data["density_kg_m3"])
    cases: dict[str, object] = {}
    for name, case in data["cases"].items():
        beta = 1 / float.fromhex(case["wall_impedance_over_rho_c"])
        problem = RectangleModalProblem((lengths[0], lengths[1], lengths[2]), speed, density, (beta, beta, beta))
        truth = trace_modes(problem, frequency_max_hz=cap, seed_frequency_max_hz=seed_cap, max_step=step)
        case_evidence: dict[str, object] = {
            "oscillating_below_cap": len(truth.modal_table),
            "seed_branches": len(truth.modes),
            "static_wave_number": [truth.static_wave_number.real, truth.static_wave_number.imag],
            "nonoscillating": [{"index": mode.index, "omega_im_rad_s": mode.omega.imag,
                                "first_nonzero_parameter": mode.points[1].parameter,
                                "first_nonzero_omega_im_rad_s": speed * mode.points[1].wave_number.imag}
                               for mode in truth.modes if mode.reached_target
                               and mode.quantities.kind is ModalKind.NONOSCILLATING_DECAY],
            "pushed_above_cap": [{"index": mode.index,
                                  "rigid_hz": speed * mode.points[0].wave_number.real / (2 * math.pi),
                                  "damped_hz": mode.quantities.frequency_hz}
                                 for mode in truth.modes if mode.outcome is ContinuationOutcome.ABOVE_FREQUENCY_LIMIT
                                 and speed * mode.points[0].wave_number.real / (2 * math.pi) <= cap],
            "entered_from_above_cap": [{"index": mode.index, "damped_hz": mode.quantities.frequency_hz}
                                       for mode in truth.modal_table
                                       if speed * mode.points[0].wave_number.real / (2 * math.pi) > cap],
            "relation_counts": dict(Counter(relation.kind for relation in truth.relations)),
            "unresolved": [{"index": mode.index, "parameter": mode.points[-1].parameter,
                            "reason": mode.failure} for mode in truth.modes if not mode.reached_target],
            "max_accepted_residual": max(point.residual for mode in truth.modes for point in mode.points),
            "accepted_step_range": [min(b.parameter - a.parameter for mode in truth.modes
                                        for a, b in zip(mode.points, mode.points[1:])),
                                    max(b.parameter - a.parameter for mode in truth.modes
                                        for a, b in zip(mode.points, mode.points[1:]))],
        }
        if details:
            case_evidence["relations"] = [{"indices": relation.indices, "kind": relation.kind}
                                        for relation in truth.relations]
            case_evidence["traces"] = [{"index": mode.index, "outcome": mode.outcome.value,
                                      "points": [{"parameter": p.parameter,
                                                  "k_rad_m": [p.wave_number.real, p.wave_number.imag],
                                                  "axis_k_squared": [[q.real, q.imag] for q in p.axis_wave_numbers_squared],
                                                  "residual": p.residual} for p in mode.points]}
                                     for mode in truth.modes]
        cases[name] = case_evidence
    return {"problem": str(problem_path), "lengths_m": lengths, "sound_speed_m_s": speed,
            "density_kg_m3": density, "frequency_max_hz": cap, "seed_frequency_max_hz": seed_cap,
            "max_step": step, "cases": cases}


def main() -> None:
    parser = argparse.ArgumentParser(description="現算長方形房阻尼模態延拓證據（不作有限元素裁判）")
    parser.add_argument("--problem", type=Path, default=Path("blueprint/fem_fenics_problem.json"))
    parser.add_argument("--frequency-max-hz", type=float, default=300.0)
    parser.add_argument("--seed-frequency-max-hz", type=float, default=330.0)
    parser.add_argument("--max-step", type=float, default=0.05)
    parser.add_argument("--details", action="store_true", help="包含每條延拓曲線與排序交叉的指標")
    args = parser.parse_args()
    note(json.dumps(reference_evidence(args.problem, args.frequency_max_hz, args.seed_frequency_max_hz,
                                      args.max_step, details=args.details), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
