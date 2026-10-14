"""真有限元素命令列三份方案；逐頻隔離表內與表外，不用帶平均掩蓋。

表內含施羅德兩個中頻帶，故這題的底構造變更不應經交接權重影響表內。
固定與構造場景改動前後逐位不變，是合併請求裡同機一次對照的證據，不放進考卷：
考卷不准自己叫版控工具取舊原碼（tests-isolated-from-real-env），而真有限元素與構造計算的
本機答案依 verification-split-stand-in-wiring-and-cross-cpu-tolerance.md 不能當跨處理器門檻。
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import cast
import subprocess
import time

import pytest

from aosr.physics.report_io import ReportOutput
from governance.exit_codes import note
from tests.engine.test_construction_cli_end_to_end import command, documents, environment
from tests.engine.test_measured_impedance import measured


def variants() -> tuple[dict[str, object], ...]:
    _, scene = documents()
    zones = scene['surface_zones_by_wall']
    assert isinstance(zones, dict)
    material = measured()
    material['points'] = [{'frequency_hz': 75., 'real_pa_s_m': 400., 'imag_pa_s_m': -200.},
        {'frequency_hz': 300., 'real_pa_s_m': 600., 'imag_pa_s_m': -50.},
        {'frequency_hz': 1500., 'real_pa_s_m': 800., 'imag_pa_s_m': 100.}]
    zones['x0'][0]['material'] = material
    changed_table, changed_base = deepcopy(scene), deepcopy(scene)
    table_zones = cast(dict[str, list[dict[str, object]]], changed_table['surface_zones_by_wall'])
    table_material = cast(dict[str, object], table_zones['x0'][0]['material'])
    cast(list[dict[str, object]], table_material['points'])[1]['real_pa_s_m'] = 1100.
    base_zones = cast(dict[str, list[dict[str, object]]], changed_base['surface_zones_by_wall'])
    base_material = cast(dict[str, object], base_zones['x0'][0]['material'])
    base_assembly = cast(dict[str, object], base_material['base_assembly'])
    cast(list[dict[str, object]], base_assembly['layers'])[1]['thickness_m'] = .04
    return scene, changed_table, changed_base


def run_variants(tmp_path: Path) -> list[ReportOutput]:
    children = []
    outputs = []
    start = time.perf_counter()
    try:
        for name, document in zip(('baseline', 'table', 'base'), variants(), strict=True):
            path = tmp_path / f'{name}.json'
            path.write_text(json.dumps(document))
            children.append(subprocess.Popen(command(path), cwd=tmp_path, env=environment(tmp_path),
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
        for child in children:
            stdout, stderr = child.communicate(timeout=max(.1, 120 - (time.perf_counter() - start)))
            assert child.returncode == 0, stderr
            outputs.append(ReportOutput.model_validate_json(stdout))
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait()
    note(f'實測表三份真有限元素命令列牆鐘：{time.perf_counter() - start:.3f} 秒')
    return outputs


def test_measured_three_real_cli_runs_isolate_each_frequency(tmp_path: Path) -> None:
    original, table, base = run_variants(tmp_path)
    assert original.points is not None and table.points is not None and base.points is not None
    inside_fem = []
    outside_fem = []
    inside = []
    outside = []
    for a, b, c in zip(original.points, table.points, base.points, strict=True):
        assert a.frequency_hz == b.frequency_hz == c.frequency_hz
        in_table = 75. <= a.frequency_hz <= 1500.
        if in_table:
            inside.append(a.frequency_hz)
            for field in ('reflected_energy', 'interference_energy', 'late_energy', 'scattering'):
                assert getattr(a, field) == getattr(c, field), (a.frequency_hz, field)
            if 75. < a.frequency_hz < 1500.:
                for field in ('reflected_energy', 'late_energy'):
                    assert getattr(a, field) != getattr(b, field), (a.frequency_hz, field)
        else:
            outside.append(a.frequency_hz)
            for field in ('reflected_energy', 'late_energy'):
                assert getattr(a, field) != getattr(c, field), (a.frequency_hz, field)
                assert getattr(a, field) == getattr(b, field), (a.frequency_hz, field)
        if a.fem_energy is not None and b.fem_energy is not None and c.fem_energy is not None:
            if in_table:
                inside_fem.append(a.frequency_hz)
                assert a.fem_energy == c.fem_energy
                if 75. < a.frequency_hz < 1500.:
                    assert a.fem_energy != b.fem_energy
            else:
                outside_fem.append(a.frequency_hz)
                assert a.fem_energy != c.fem_energy
                assert a.fem_energy == b.fem_energy
    assert inside and outside and inside_fem and outside_fem
    assert original.top.f_s_hz == base.top.f_s_hz
    assert original.measured_seams_by_wall is not None and base.measured_seams_by_wall is not None
    for old_seam, new_seam in zip(original.measured_seams_by_wall['x0'], base.measured_seams_by_wall['x0'], strict=True):
        assert old_seam.measured == new_seam.measured
        assert old_seam.base_assembly != new_seam.base_assembly
    assert original.construction_notes_by_wall is not None
    fill = next(n for n in original.construction_notes_by_wall['x0'] if n.code == 'measured_table_model_fill')
    assert fill.frequencies_hz and all(f < 75. or f > 1500. for f in fill.frequencies_hz)
    note('逐頻表內隔離通過；施羅德兩帶均位於表內，此題交接權重無跨頻變動。')


def test_measured_real_cli_text_contains_plain_header_and_seams(tmp_path: Path) -> None:
    path = tmp_path / 'text.json'
    path.write_text(json.dumps(variants()[0]))
    args = command(path)
    args[args.index('json')] = 'text'
    child = subprocess.run(args, cwd=tmp_path, env=environment(tmp_path), capture_output=True, text=True, timeout=120)
    assert child.returncode == 0, child.stderr
    assert '實測表涵蓋 75–1500 Hz' in child.stdout
    assert '75 Hz 以下與 1500 Hz 以上是模型補值（底構造）' in child.stdout
    assert '實測段假設與角度無關' in child.stdout
    assert '實測表只代表樣品在管子裡量到的結果，不代表現場安裝' in child.stdout
    assert '接縫報告（差＝底構造減實測）' in child.stdout
    assert '垂直入射吸音率' in child.stdout and '無規入射吸音率' in child.stdout
    assert '模型補值頻點' in child.stdout
    assert 'measured_table_model_fill' not in child.stdout and '第 None 層' not in child.stdout
