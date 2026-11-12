"""考卷共用（#1082 段 2-3c）：替身樁旁邊放一份「訓練過這一場家族」的訓練說明。

只給用替身樁（假的三件檔）考別的事的考卷用：說明裡的 ``coverage`` 照產品的 :func:`coverage_for` 由這一場的材料生成，
不手寫範圍。閘門本身（有沒有拒收、拒收講什麼）由 ``test_surrogate_coverage.py`` 與真建替身的考卷考。
:func:`fixed_checked_on` 再放一份「這一組起跑配置的固定材料都核過」的核對紀錄（#1082 段 2-8；網頁預覽與開跑要它），
配置照產品的 :func:`aosr.gui.gradient_jobs.plan_configuration` 由計畫生成。
"""
from __future__ import annotations

import json
from pathlib import Path

from aosr.gradient.surrogate_coverage import COVERAGE_VERSION, FixedTarget, coverage_for
from aosr.gradient.surrogate_files import TRAINING_FILE
from aosr.gradient.surrogate_fixed_checks import (
    FIXED_CHECKS_FILE, FixedCheckRow, FixedChecksRecord, configuration_sha256, manifest_digest, write_record,
)
from aosr.materials.search_library import SearchMaterials
from aosr.physics.fem_reduced_store import manifest_path


def trained_on(surrogate: Path, materials: SearchMaterials, fixed: tuple[FixedTarget, ...] = ()) -> Path:
    """在 ``surrogate`` 同資料夾寫 ``training.json``，記下訓練過 ``materials`` 的每個家族與 ``fixed`` 的固定材料候選；回那份檔的路徑。"""
    coverage = coverage_for(materials, fixed)
    assert coverage is not None, "只有產品設定檔家族、沒有固定材料候選的那一場不用涵蓋紀錄（舊替身照舊放行）"
    path = surrogate.parent / TRAINING_FILE
    path.write_text(json.dumps({"version": COVERAGE_VERSION, "coverage": coverage.model_dump(mode="json")},
                               ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def fixed_checked_on(surrogate: Path, configuration: dict[str, object], fixed: tuple[FixedTarget, ...]) -> Path:
    """在 ``surrogate`` 同資料夾寫核對紀錄：``configuration``（起跑配置正規寫法）下 ``fixed`` 每個候選都過了；回那份檔的路徑。"""
    path = surrogate.parent / FIXED_CHECKS_FILE
    path.unlink(missing_ok=True)
    write_record(path, FixedChecksRecord(
        configuration_sha256=configuration_sha256(configuration), configuration=configuration,
        manifest_sha256=manifest_digest(manifest_path(surrogate)),
        checks=tuple(FixedCheckRow(target=item.target, material_sha256=item.coverage().material_sha256, frequency_db=0.01,
                                   band_db=0.001, frequency_ruler=0.224, band_ruler=0.011, passed=True) for item in fixed)))
    return path
