"""真有限元素命令列三份方案；逐頻隔離表內與表外，不用帶平均掩蓋。

前三份的表蓋住施羅德兩個中頻帶，底構造變更不經交接權重影響表內。另一題刻意讓表不蓋
500／1000 Hz：換底構造只准移動有限元素與幾何路的交接下端（拍板第 7 點 10-14 講準的那句），
表內各路能量逐位不變，只有交接段內的權重與合成總能量跟著動。
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

from aosr.config.frequency_axis import FEM_GEOMETRIC_CROSSOVER_CAP_HZ
from aosr.config.three_lane_crossover import CROSSOVER_LOWER_FLOOR_HZ
from aosr.physics.crossover import crossover_weights
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
    return run_documents(tmp_path, dict(zip(('baseline', 'table', 'base'), variants(), strict=True)), '實測表三份')


def run_documents(tmp_path: Path, documents_by_name: dict[str, dict[str, object]], label: str) -> list[ReportOutput]:
    children = []
    outputs = []
    start = time.perf_counter()
    try:
        for name, document in documents_by_name.items():
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
    note(f'{label}真有限元素命令列牆鐘：{time.perf_counter() - start:.3f} 秒')
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


def coupling_variants() -> tuple[dict[str, object], dict[str, object]]:
    """表只蓋 100–400 Hz，500／1000 Hz 由底構造補；兩份只差底構造（板＋空腔 → 纖維多孔）。"""
    _, scene = documents()
    zones = cast(dict[str, list[dict[str, object]]], scene['surface_zones_by_wall'])
    material = measured()
    material['points'] = [{'frequency_hz': 100., 'real_pa_s_m': 600., 'imag_pa_s_m': -900.},
        {'frequency_hz': 200., 'real_pa_s_m': 700., 'imag_pa_s_m': -400.},
        {'frequency_hz': 400., 'real_pa_s_m': 800., 'imag_pa_s_m': -100.}]
    zones['x0'][0]['material'] = material
    porous = deepcopy(scene)
    porous_zones = cast(dict[str, list[dict[str, object]]], porous['surface_zones_by_wall'])
    cast(dict[str, object], porous_zones['x0'][0]['material'])['base_assembly'] = {'kind': 'assembly', 'layers': [
        {'kind': 'porous', 'thickness_m': .05, 'flow_resistivity_pa_s_m2': 19100., 'material_class': 'fibrous'}],
        'backing': 'rigid'}
    return scene, porous


def test_base_change_moves_only_the_crossover_band_when_table_misses_schroeder_bands(tmp_path: Path) -> None:
    """拍板第 7 點（10-14 講準）：表內材料與各路能量不變；只有交接段內的房間合成結果可以動。"""
    board, porous = run_documents(tmp_path, dict(zip(('board', 'porous'), coupling_variants(), strict=True)), '交接段連動兩份')
    assert board.points is not None and porous.points is not None
    # 前提：500／1000 Hz 在表外，換底構造真的讓施羅德頻率與交接下端移動，而且兩份都沒被上限硬切。
    assert board.top.f_s_hz != porous.top.f_s_hz
    for report in (board, porous):
        # 界線不拿報告自己說的當裁判：下端照決策紙是 max(施羅德頻率, 下限)，上端固定為有限元素上限。
        assert not report.top.capped_by_upper_limit
        assert report.top.crossover_lower_hz == max(report.top.f_s_hz, CROSSOVER_LOWER_FLOOR_HZ)
        assert report.top.crossover_upper_hz == FEM_GEOMETRIC_CROSSOVER_CAP_HZ
    lower = min(board.top.crossover_lower_hz, porous.top.crossover_lower_hz)
    upper = FEM_GEOMETRIC_CROSSOVER_CAP_HZ
    assert board.top.crossover_lower_hz != porous.top.crossover_lower_hz and lower < upper
    window, steady = [], []
    for a, b in zip(board.points, porous.points, strict=True):
        assert a.frequency_hz == b.frequency_hz
        if not 100. <= a.frequency_hz <= 400.:
            continue
        for report, point in ((board, a), (porous, b)):
            # 權重照當份施羅德頻率算、合成總能量照權重加（沒有有限元素值時那一路貢獻為零）。
            expected = crossover_weights((point.frequency_hz,), report.top.f_s_hz)
            assert (point.w_fem, point.w_geo) == (expected.w_fem[0], expected.w_geo[0]), point.frequency_hz
            fem_part = 0.0 if point.fem_energy is None else point.w_fem * point.fem_energy
            assert point.total_energy == fem_part + point.w_geo * point.geometric_energy, point.frequency_hz
        for field in ('fem_energy', 'direct_energy', 'reflected_energy', 'interference_energy', 'late_energy',
                      'scattering', 'geometric_energy'):
            assert getattr(a, field) == getattr(b, field), (a.frequency_hz, field)
        if lower < a.frequency_hz < upper:
            window.append(a.frequency_hz)
            assert a.w_geo != b.w_geo, a.frequency_hz
            if a.fem_energy != a.geometric_energy:
                assert a.total_energy != b.total_energy, a.frequency_hz
        else:
            steady.append(a.frequency_hz)
            assert (a.w_fem, a.w_geo, a.total_energy) == (b.w_fem, b.w_geo, b.total_energy), a.frequency_hz
    assert window and any(f <= lower for f in steady) and any(f >= upper for f in steady), (window, steady)
    # 至少一個段內頻點兩路能量可辨別，證明合成總能量真的受交接權重牽動。
    assert any(a.total_energy != b.total_energy for a, b in zip(board.points, porous.points, strict=True)
               if a.frequency_hz in window), window
    assert board.measured_seams_by_wall is not None and porous.measured_seams_by_wall is not None
    for old_seam, new_seam in zip(board.measured_seams_by_wall['x0'], porous.measured_seams_by_wall['x0'], strict=True):
        assert old_seam.measured == new_seam.measured
    note(f'交接下端 {board.top.crossover_lower_hz:.1f}→{porous.top.crossover_lower_hz:.1f} Hz；'
         f'表內交接段 {min(window):.1f}–{max(window):.1f} Hz 的權重與合成總能量變，各路能量不變。')


def test_measured_real_cli_text_contains_plain_header_and_seams(tmp_path: Path) -> None:
    """底構造換成單層纖維多孔（流阻 60000，約 82 Hz 以下才落在小松模型量測範圍外）：
    表下方的模型補值頻點同時是材料模型外推，拍板補記要求報告開頭那一行也寫出來。"""
    document = variants()[0]
    zones = cast(dict[str, list[dict[str, object]]], document['surface_zones_by_wall'])
    cast(dict[str, object], zones['x0'][0]['material'])['base_assembly'] = {'kind': 'assembly', 'layers': [
        {'kind': 'porous', 'thickness_m': .05, 'flow_resistivity_pa_s_m2': 60000., 'material_class': 'fibrous'}],
        'backing': 'rigid'}
    path = tmp_path / 'text.json'
    path.write_text(json.dumps(document))
    args = command(path)
    args[args.index('json')] = 'text'
    child = subprocess.run(args, cwd=tmp_path, env=environment(tmp_path), capture_output=True, text=True, timeout=120)
    assert child.returncode == 0, child.stderr
    assert '實測表涵蓋 75–1500 Hz' in child.stdout
    assert '75 Hz 以下與 1500 Hz 以上是模型補值（底構造）' in child.stdout
    assert '實測段假設與角度無關' in child.stdout
    assert '實測表只代表樣品在管子裡量到的結果，不代表現場安裝' in child.stdout
    header = [line for line in child.stdout.splitlines() if line.startswith('x0：實測表涵蓋 75–1500 Hz')]
    assert header and all('底構造第 1 層另有材料模型外推' in line and '；其中 ' in line for line in header), header
    assert '接縫報告（差＝底構造減實測）' in child.stdout
    assert '垂直入射吸音率' in child.stdout and '無規入射吸音率' in child.stdout
    assert '模型補值頻點' in child.stdout
    assert 'measured_table_model_fill' not in child.stdout and '第 None 層' not in child.stdout
