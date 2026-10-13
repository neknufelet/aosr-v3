"""C：真材料、晚期與鏡像法，有限元素用替身避免第三份全套。"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

from aosr import runtime
runtime.configure_calculation_jax()

from aosr.config.paths import config_path
from aosr.physics import three_lane_report, three_lane_report_cli


def fem(**kwargs: object) -> tuple[float, ...]:
    axis = kwargs["frequencies_hz"]
    assert isinstance(axis, tuple)
    return tuple(1. for _frequency in axis)


def main() -> None:
    path = Path(sys.argv[1])
    with patch.object(three_lane_report, "_solve_fem_energy", fem):
        code = three_lane_report_cli.main([str(path), "--format", sys.argv[2],
                                          "--capabilities", str(config_path("capabilities.toml"))])
    assert code == 0


if __name__ == "__main__":
    main()
