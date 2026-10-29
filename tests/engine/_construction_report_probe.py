"""C：真材料、晚期與鏡像法，有限元素用替身避免第三份全套。

同一個子行程依序呼叫命令列 ``main`` 一次一種格式（#972：匯入與編譯只做一次），各格式的標準輸出寫到
``<輸入檔名>.<格式>.out``；每一次都是完整的一趟（讀檔、求解、輸出），不共用上一次的結果。
"""
from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
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
    for format_name in sys.argv[2:]:
        buffer = io.StringIO()
        with patch.object(three_lane_report, "_solve_fem_energy", fem), redirect_stdout(buffer):
            code = three_lane_report_cli.main([str(path), "--format", format_name,
                                              "--capabilities", str(config_path("capabilities.toml"))])
        assert code == 0, buffer.getvalue()
        path.with_name(f"{path.stem}.{format_name}.out").write_text(buffer.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()
