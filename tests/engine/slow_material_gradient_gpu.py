"""本機合併前驗收的搜尋與終點重考卷；考卷本體由原檔原樣搬移。"""
from __future__ import annotations
import subprocess
from pathlib import Path
from typing import Literal
import pytest
from aosr.gradient import cli, run
from tests.engine.test_material_gradient_mount import CONTROL_PROJECT, _digest, _project
from tests.engine.test_six_category_gpu import measured as six_measured

from tests.engine.test_material_gradient_gpu import (
    _assert_preflight_start as _assert_preflight_start,
    numeric as numeric,
)


def test_project_loads_gpu_six_search(six_measured: dict[str, object]) -> None:
    assert six_measured['gpu_wrapper'] is True and six_measured['candidate_gpu_wrapper'] is True
