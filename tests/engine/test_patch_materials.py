"""票 #219 第 3 段：分格材料的輸入、幾何、逐跳振幅與命令列考卷。"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import re

import pytest

from aosr.geometry import shoebox
from aosr.geometry.shoebox import wall_count_signature
from aosr.physics import totals
from aosr.physics import amplitude as amp
from aosr.physics import compare as compare_api
from aosr.physics.compare import PathComparison, compare_paths as _compare_paths
from aosr.physics.room_paths import RoomPath, image_source_paths, load_room_input, main
from tests.engine._precision_contracts import contract_value
from tests.engine.test_amplitude import _input_case


_ROOT = Path(__file__).resolve().parents[2]
_ANSWER = _ROOT / "blueprint/reference_amplitude_patch_varied.json"
_WALLS = amp.CANONICAL_WALLS
_REFLECTION_TOLERANCE_ULP = contract_value("reflection_product_ulp")
_DIRECT_TOLERANCE_REL = contract_value("direct_energy_vs_legacy")
_REFLECTED_TOLERANCE_REL_FLOOR = contract_value("reflected_energy_floor")


def compare_paths(
    paths: list[RoomPath],
    answers: list[dict[str, object]],
    frequencies: tuple[float, ...] | None = None,
) -> list[PathComparison]:
    return _compare_paths(paths, answers, _REFLECTION_TOLERANCE_ULP, frequencies)


def _mapping(node: object, where: str) -> dict[str, object]:
    assert isinstance(node, dict), f"{where} 不是一層表"
    return {str(key): value for key, value in node.items()}


def _patch_input() -> dict[str, object]:
    with _ANSWER.open(encoding="utf-8") as handle:
        answer = _mapping(json.load(handle), "answer")
    params = _mapping(answer["parameters"], "parameters")
    case = _mapping(_mapping(params["cases"], "cases")["patched_y0_cell_0_0_10rhoc"], "case")
    source_cells = _mapping(case["wall_cells"], "wall_cells")
    shapes = params["grid_shapes"]
    assert isinstance(shapes, list)
    materials: dict[str, object] = {
        "rho_c": params["rho_c_pa_s_per_m"],
        "frequencies_hz": params["frequencies_hz"],
    }
    for wall, shape in zip(_WALLS, shapes, strict=True):
        cells = source_cells[wall]
        assert isinstance(cells, list)
        flattened: list[list[dict[str, float]]] = []
        for cell in cells:
            impedance = _mapping(_mapping(cell, "cell")["impedance"], "impedance")
            reals = impedance["per_frequency_real"]
            imags = impedance["per_frequency_imag"]
            assert isinstance(reals, list) and isinstance(imags, list)
            flattened.append(
                [
                    {"real": float(real), "imag": float(imag)}
                    for real, imag in zip(reals, imags, strict=True)
                ]
            )
        materials[wall] = {"grid": shape, "cells": flattened}
    receiver = _mapping(params["receiver_xyz_m"], "receiver_xyz_m")
    return {
        "room": params["room"],
        "source_xyz_m": params["source_xyz_m"],
        "receiver_xyz_m": {axis: receiver[axis] for axis in "xyz"},
        "sound_speed_m_s": params["sound_speed_m_s"],
        "max_order": params["max_order"],
        "materials": materials,
    }


def _answer_root() -> dict[str, object]:
    with _ANSWER.open(encoding="utf-8") as handle:
        return _mapping(json.load(handle), "answer")


def _answer_paths() -> list[dict[str, object]]:
    raw = _answer_root()["paths"]
    assert isinstance(raw, list)
    return [_mapping(item, "path") for item in raw]


def _write_input(tmp_path: Path, value: dict[str, object]) -> Path:
    path = tmp_path / "patch.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _paths(tmp_path: Path) -> list[RoomPath]:
    inputs = load_room_input(_write_input(tmp_path, _patch_input()))
    assert inputs.materials is not None
    return image_source_paths(
        inputs.room,
        inputs.source,
        inputs.receiver,
        inputs.sound_speed,
        inputs.max_order,
        inputs.materials,
    )


def test_loads_row_major_patch_materials(tmp_path: Path) -> None:
    """載入器若仍只收整牆清單，y0 的 2×2 分格會在這題失敗。"""
    inputs = load_room_input(_write_input(tmp_path, _patch_input()))
    assert inputs.materials is not None
    assert inputs.materials.grid("y0")[:2] == (2, 2)
    parameters = _mapping(_answer_root()["parameters"], "parameters")
    rho_c_node = parameters["rho_c_pa_s_per_m"]
    assert isinstance(rho_c_node, (int, float)) and not isinstance(rho_c_node, bool)
    rho_c = float(rho_c_node)
    assert inputs.materials.impedance_at("y0", 0, 0, 0) == complex(rho_c * 10.0, 0.0)


def test_whole_wall_impedance_rejects_patched_wall(tmp_path: Path) -> None:
    """舊 API 若偷拿分格牆 (0,0)，呼叫端會把局部材料誤當整面牆。"""
    inputs = load_room_input(_write_input(tmp_path, _patch_input()))
    assert inputs.materials is not None

    with pytest.raises(
        ValueError,
        match=re.escape("這面牆有分格，用 impedance_at"),
    ):
        inputs.materials.impedance("y0", 0)

    assert "y0" not in inputs.materials.walls


def test_geometry_exposes_wall_grid_cell() -> None:
    """y0 的 x 是 col、z 是 row，內部格線沿用前代落到較大 index。"""
    room = shoebox.Room(4.0, 5.0, 3.0)
    assert shoebox.wall_grid_cell(room, "y0", (1.0, 0.0, 0.5), 2, 2) == (0, 0)
    assert shoebox.wall_grid_cell(room, "y0", (2.0, 0.0, 1.5), 2, 2) == (1, 1)


def test_paths_carry_one_cell_for_each_bounce(tmp_path: Path) -> None:
    """RoomPath 若沒有逐跳格號，振幅與 --compare 都無從驗證材料查格。"""
    for path in _paths(tmp_path):
        assert len(path.bounce_cells) == len(path.bounces)


def test_compare_names_the_path_and_hop_for_wrong_bounce_cell(tmp_path: Path) -> None:
    """裁判若沒比 bounce_cells，改壞答案裡一跳仍會假綠。"""
    answers = copy.deepcopy(_answer_paths())
    target = next(item for item in answers if item["bounce_cells"])
    cells = target["bounce_cells"]
    assert isinstance(cells, list)
    first = _mapping(cells[0], "bounce_cells[0]")
    old_col = first["col"]
    assert isinstance(old_col, int) and not isinstance(old_col, bool)
    first["col"] = old_col + 1
    cells[0] = first

    inputs = load_room_input(_write_input(tmp_path, _patch_input()))
    assert inputs.materials is not None
    result = compare_paths(_paths(tmp_path), answers, inputs.materials.frequencies_hz)
    wrong = [diff for row in result for diff in row.diffs if "bounce_cells" in diff]
    assert wrong and "第 1 跳" in wrong[0]


def test_patch_reference_paths_and_totals_are_inside_contract(tmp_path: Path) -> None:
    """逐條 bounce_cells、反射、壓力及總量都必須通過凍結答案的獨立裁判。"""
    inputs = load_room_input(_write_input(tmp_path, _patch_input()))
    assert inputs.materials is not None
    paths = _paths(tmp_path)
    rows = compare_paths(paths, _answer_paths(), inputs.materials.frequencies_hz)
    path_fraction = max(
        (max(row.max_refl_frac, row.max_pp_frac) for row in rows), default=0.0
    )
    assert not [row.diffs for row in rows if row.diffs], f"路徑用到界線 {path_fraction:.2%}"

    answer = _answer_root()
    total_result = totals.compare_totals(
        paths,
        _mapping(answer["totals"], "totals"),
        inputs.materials.frequencies_hz,
        _REFLECTION_TOLERANCE_ULP,
        _DIRECT_TOLERANCE_REL,
        _REFLECTED_TOLERANCE_REL_FLOOR,
    )
    total_fraction = max(
        total_result.max_pressure_frac,
        total_result.max_direct_frac,
        total_result.max_reflected_frac,
    )
    assert total_result.diffs == [], f"總量用到界線 {total_fraction:.2%}"


def test_patch_cli_json_and_table_show_cells(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """分格輸入的人看表與 JSON 都要帶逐跳格號。"""
    input_path = _write_input(tmp_path, _patch_input())
    assert not main([str(input_path), "--json"])
    payload = _mapping(json.loads(capsys.readouterr().out), "payload")
    records = payload["paths"]
    assert isinstance(records, list)
    assert [item["bounce_cells"] for item in records] == [
        item["bounce_cells"] for item in _answer_paths()
    ]

    assert not main([str(input_path)])
    assert "格=(" in capsys.readouterr().out


def test_all_varied_y0_cells_match_whole_wall_varied_within_contract(
    tmp_path: Path,
) -> None:
    """四格都放同一份 varied 材料時，每條／每頻帶退化回整牆 varied 答案。"""
    value = _patch_input()
    varied = _input_case("varied")
    varied_materials = _mapping(varied["materials"], "varied materials")
    y0 = varied_materials["y0"]
    materials = _mapping(value["materials"], "materials")
    y0_grid = _mapping(materials["y0"], "materials.y0")
    cells = y0_grid["cells"]
    assert isinstance(cells, list)
    y0_grid["cells"] = [copy.deepcopy(y0) for _ in cells]
    materials["y0"] = y0_grid
    value["materials"] = materials

    inputs = load_room_input(_write_input(tmp_path, value))
    assert inputs.materials is not None
    paths = image_source_paths(
        inputs.room,
        inputs.source,
        inputs.receiver,
        inputs.sound_speed,
        inputs.max_order,
        inputs.materials,
    )
    varied_answer = _ROOT / "blueprint/reference_amplitude_varied.json"
    with varied_answer.open(encoding="utf-8") as handle:
        reference = _mapping(json.load(handle), "varied answer")
    reference_paths = reference["paths"]
    assert isinstance(reference_paths, list)
    rows = compare_paths(
        paths,
        [_mapping(item, "path") for item in reference_paths],
        inputs.materials.frequencies_hz,
    )
    assert not [row.diffs for row in rows if row.diffs]
    whole_wall_inputs = load_room_input(_write_input(tmp_path, varied))
    assert whole_wall_inputs.materials is not None
    for path in paths:
        whole_wall = amp.reflection_product(
            whole_wall_inputs.materials,
            path.dist_m,
            inputs.receiver.as_tuple(),
            path.image,
            wall_count_signature(path.identity),
        )
        # 實測最大差 2.78e-17；這條守的是退化度，不是振幅契約。
        assert all(
            abs(actual - expected) <= 1e-15
            for actual, expected in zip(
                path.reflection_product, whole_wall, strict=True
            )
        )


def test_one_by_one_new_and_old_reflection_products_stay_inside_contract(
    tmp_path: Path,
) -> None:
    """全牆 1×1 時，逐跳連乘與既有 r**count 只准有雙精度運算次序尾差。"""
    value = _input_case("varied")
    input_path = _write_input(tmp_path, value)
    inputs = load_room_input(input_path)
    assert inputs.materials is not None
    for path in image_source_paths(
        inputs.room,
        inputs.source,
        inputs.receiver,
        inputs.sound_speed,
        inputs.max_order,
        inputs.materials,
    ):
        old = amp.reflection_product(
            inputs.materials,
            path.dist_m,
            inputs.receiver.as_tuple(),
            path.image,
            wall_count_signature(path.identity),
        )
        new = amp.reflection_product_by_bounce(
            inputs.materials,
            path.dist_m,
            inputs.receiver.as_tuple(),
            path.image,
            path.bounce_cells,
        )
        assert all(
            abs(a - b) <= _REFLECTION_TOLERANCE_ULP
            for a, b in zip(old, new, strict=True)
        )


@pytest.mark.parametrize("broken", ("grid", "cell_count", "band_count"))
def test_rejects_malformed_patch_materials(tmp_path: Path, broken: str) -> None:
    """grid 形狀、格數、格內頻帶數任一錯都要 ValueError 並指名 y0。"""
    value = _patch_input()
    materials = _mapping(value["materials"], "materials")
    y0 = _mapping(materials["y0"], "materials.y0")
    cells = y0["cells"]
    assert isinstance(cells, list)
    if broken == "grid":
        y0["grid"] = [2]
    elif broken == "cell_count":
        y0["cells"] = cells[:-1]
    else:
        band = cells[0]
        assert isinstance(band, list)
        cells[0] = band[:-1]
    materials["y0"] = y0
    value["materials"] = materials

    with pytest.raises(ValueError, match=re.escape("materials.y0")):
        load_room_input(_write_input(tmp_path, value))


@pytest.mark.parametrize("rows", (0, -1))
def test_rejects_non_positive_patch_grid(tmp_path: Path, rows: int) -> None:
    """grid 的列數為零或負數時不可產生空格網或負 index。"""
    value = _patch_input()
    materials = _mapping(value["materials"], "materials")
    y0 = _mapping(materials["y0"], "materials.y0")
    y0["grid"] = [rows, 2]
    materials["y0"] = y0
    value["materials"] = materials

    with pytest.raises(
        ValueError,
        match=re.escape("materials.y0.grid 必須是正整數"),
    ):
        load_room_input(_write_input(tmp_path, value))


@pytest.mark.parametrize("rows", (1.5, True))
def test_rejects_non_integer_patch_grid(tmp_path: Path, rows: object) -> None:
    """小數與 bool 都不是 grid 的整數列數。"""
    value = _patch_input()
    materials = _mapping(value["materials"], "materials")
    y0 = _mapping(materials["y0"], "materials.y0")
    y0["grid"] = [rows, 2]
    materials["y0"] = y0
    value["materials"] = materials

    with pytest.raises(
        ValueError,
        match=re.escape("materials.y0.grid[0] 不是一個整數"),
    ):
        load_room_input(_write_input(tmp_path, value))


def test_rejects_extra_patch_cell_field(tmp_path: Path) -> None:
    """cells 裡的複數格多打欄位時不可靜默忽略。"""
    value = _patch_input()
    materials = _mapping(value["materials"], "materials")
    y0 = _mapping(materials["y0"], "materials.y0")
    cells = y0["cells"]
    assert isinstance(cells, list)
    first_band = cells[0]
    assert isinstance(first_band, list)
    first_cell = first_band[0]
    assert isinstance(first_cell, dict)
    first_cell["extra"] = "typo"

    with pytest.raises(
        ValueError,
        match=re.escape("materials.y0.cells[0][0] 有多餘欄位：extra"),
    ):
        load_room_input(_write_input(tmp_path, value))


def test_accepts_whole_wall_lists_mixed_with_patch_grids(tmp_path: Path) -> None:
    """六面牆可各自選整牆清單或分格物件，不要求全用同一形狀。"""
    value = _patch_input()
    materials = _mapping(value["materials"], "materials")
    varied_materials = _mapping(_input_case("varied")["materials"], "varied materials")
    materials["floor"] = varied_materials["floor"]
    value["materials"] = materials

    inputs = load_room_input(_write_input(tmp_path, value))

    assert inputs.materials is not None
    assert inputs.materials.grid("floor")[:2] == (1, 1)
    assert inputs.materials.grid("y0")[:2] == (2, 2)


def test_swapping_y0_cells_breaks_the_amplitude_contract(tmp_path: Path) -> None:
    """互換 y0 (0,0)/(0,1) 阻抗後，至少一條振幅必須超界。"""
    value = _patch_input()
    materials = _mapping(value["materials"], "materials")
    y0 = _mapping(materials["y0"], "materials.y0")
    cells = y0["cells"]
    assert isinstance(cells, list)
    cells[0], cells[1] = cells[1], cells[0]
    paths_input = load_room_input(_write_input(tmp_path, value))
    assert paths_input.materials is not None
    paths = image_source_paths(
        paths_input.room,
        paths_input.source,
        paths_input.receiver,
        paths_input.sound_speed,
        paths_input.max_order,
        paths_input.materials,
    )
    result = compare_paths(paths, _answer_paths(), paths_input.materials.frequencies_hz)
    assert any(row.diffs for row in result)


def test_zero_reflection_boundary_turns_reference_comparison_red(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """把反射界線換成 0，浮點參考差異不可仍被裁判放成綠。"""
    monkeypatch.setattr(
        compare_api,
        "reflection_tolerance",
        lambda _f, _t, _r, _base: 0.0,
    )
    inputs = load_room_input(_write_input(tmp_path, _patch_input()))
    assert inputs.materials is not None
    result = compare_paths(_paths(tmp_path), _answer_paths(), inputs.materials.frequencies_hz)
    assert any("reflection_product" in diff for row in result for diff in row.diffs)


def test_swapping_geometry_row_and_col_breaks_bounce_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """落格函式若把 row/col 顛倒，答案檔的逐跳格號必須咬住。"""
    original = shoebox.wall_grid_cell

    def swapped(
        room: shoebox.Room,
        wall: str,
        point: tuple[float, float, float],
        rows: int,
        cols: int,
    ) -> tuple[int, int]:
        row, col = original(room, wall, point, rows, cols)
        return col, row

    monkeypatch.setattr("aosr.physics.room_paths.wall_grid_cell", swapped)
    inputs = load_room_input(_write_input(tmp_path, _patch_input()))
    assert inputs.materials is not None
    result = compare_paths(_paths(tmp_path), _answer_paths(), inputs.materials.frequencies_hz)
    assert any("bounce_cells" in diff for row in result for diff in row.diffs)
