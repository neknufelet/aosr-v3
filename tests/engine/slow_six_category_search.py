"""本機合併前驗收的搜尋與終點重考卷；考卷本體由原檔原樣搬移。"""
from __future__ import annotations
import json
from pathlib import Path
from typing import cast
import pytest
from pydantic import ValidationError
from aosr.gradient import endpoint_run, material_gradient_ledger as ledger, run
from tests.engine.test_six_category_gpu import measured as six_measured

from tests.engine.test_six_category_search import (
    _header as _header,
    _row as _row,
    _scores as _scores,
    _snap as _snap,
    measured_nan as measured_nan,
)


def test_gpu_search_writes_six_category_scores(six_measured: dict[str, object]) -> None:
    assert six_measured['identities_checked'] is True and six_measured['result_written'] is True
