"""固定面的雙精度合成整場探針；父行程用 runtime 的子行程設定開啟雙精度。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from aosr.gradient.fixed_surfaces import SCHEME_MATERIAL
from tests.engine.test_material_gradient_fixed_surface import _check_fixed, _run, _snapshot


def main() -> None:
    directory, part = Path(sys.argv[1]), sys.argv[2]
    with pytest.MonkeyPatch.context() as patch:
        if part == "fixed":
            _check_fixed(directory, patch)
        elif part == "scheme":
            _check_fixed(directory, patch, {"floor": SCHEME_MATERIAL})
        else:
            folder = directory / "control"
            _run(folder, patch)
            (directory / "snapshot.json").write_text(json.dumps(_snapshot(folder)), encoding="utf-8")


if __name__ == "__main__":
    main()
