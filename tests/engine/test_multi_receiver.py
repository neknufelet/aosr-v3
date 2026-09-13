"""票 #219 第 2d 段：多接收點輸入、計算、命令列與新形答案檔。"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from aosr.physics import receivers as receiver_api
from aosr.physics import totals
from aosr.geometry.shoebox import Point, Room
from aosr.physics.amplitude import Materials
from aosr.physics.compare import compare_paths
from aosr.physics.room_paths import RoomPath, image_source_paths, load_room_input, main
from tests.conftest import GitSandbox
from tests.engine.test_amplitude import _as_float_list, _input_case


def _multi_answer(case: str) -> dict[str, object]:
    path = Path(__file__).resolve().parents[2] / "blueprint" / f"reference_amplitude_multi_{case}.json"
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    assert isinstance(value, dict)
    return value


def _multi_input(case: str) -> dict[str, object]:
    answer = _multi_answer(case)
    parameters = answer["parameters"]
    assert isinstance(parameters, dict)
    receivers = parameters["receivers_xyz_m"]
    assert isinstance(receivers, list)
    value = _input_case(case)
    del value["receiver_xyz_m"]
    value["receivers"] = receivers
    return value


def _write_multi_input(tmp_path: Path, case: str) -> Path:
    path = tmp_path / f"multi-{case}.json"
    path.write_text(json.dumps(_multi_input(case), sort_keys=True), encoding="utf-8")
    return path


def _answer_records(case: str) -> dict[str, dict[str, object]]:
    answer = _multi_answer(case)
    records = answer["receivers"]
    assert isinstance(records, dict)
    result: dict[str, dict[str, object]] = {}
    for receiver_id, record in records.items():
        assert isinstance(record, dict)
        result[str(receiver_id)] = record
    return result


def _write_multi_answer(tmp_path: Path, data: dict[str, object]) -> Path:
    path = tmp_path / "multi-answer.json"
    path.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
    return path


def _origin_main_room_paths(tmp_path: Path, git_sandbox: GitSandbox) -> Path:
    """用隔離 git 一次封存 origin/main 全樹，再解到 tmp_path。"""
    repo = Path(__file__).resolve().parents[2]
    root = tmp_path / "origin-main"
    root.mkdir()
    archive = tmp_path / "origin-main.tar"
    git_sandbox.git(
        "archive",
        f"--remote={repo}",
        f"--output={archive}",
        "origin/main",
    )
    proc = subprocess.run(
        ["tar", "-xf", str(archive), "-C", str(root)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return root / "src"


def _run_origin_main(
    package_root: Path, input_path: Path, args: tuple[str, ...]
) -> tuple[int, str, str]:
    """在 tmp_path 跑舊 CLI，產品 import 指向這棵候選樹的相容底層模組。"""
    env = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": os.pathsep.join((str(package_root),)),
    }
    proc = subprocess.run(
        [sys.executable, "-m", "aosr.physics.room_paths", str(input_path), *args],
        cwd=package_root.parent,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _order3_geometry_input() -> dict[str, object]:
    """從第三段凍結答案只摘 CLI 的五個無材料輸入欄位。"""
    path = Path(__file__).resolve().parents[2] / "blueprint/reference_room_answers_order3.json"
    with path.open(encoding="utf-8") as handle:
        answer = json.load(handle)
    parameters = answer["parameters"]
    assert isinstance(parameters, dict)
    keys = ("room", "source_xyz_m", "receiver_xyz_m", "sound_speed_m_s", "max_order")
    return {key: parameters[key] for key in keys}


def test_loads_receivers_in_input_order(tmp_path: Path) -> None:
    """若載入器仍只收 receiver_xyz_m，這題會因 receivers 是多餘欄位而失敗。"""
    inputs = load_room_input(_write_multi_input(tmp_path, "flat"))
    raw = _multi_input("flat")["receivers"]
    assert isinstance(raw, list)

    assert [receiver.id for receiver in inputs.receivers] == [entry["id"] for entry in raw]
    assert [receiver.point.as_tuple() for receiver in inputs.receivers] == [
        tuple(float(entry[axis]) for axis in "xyz") for entry in raw
    ]


def test_rejects_duplicate_receiver_id(tmp_path: Path) -> None:
    """若只用 dict 收結果卻沒先擋重複 id，前一筆會被靜默覆蓋。"""
    value = _multi_input("flat")
    receivers = value["receivers"]
    assert isinstance(receivers, list)
    duplicate = dict(receivers[0])
    receivers.append(duplicate)
    path = tmp_path / "duplicate.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match=str(duplicate["id"])):
        load_room_input(path)


def test_rejects_single_and_multiple_receiver_inputs(tmp_path: Path) -> None:
    """兩種接收點欄位同時存在時，不可偷偷選其中一種。"""
    value = _multi_input("flat")
    value["receiver_xyz_m"] = {"x": 1.0, "y": 1.0, "z": 1.0}
    path = tmp_path / "both.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="receiver_xyz_m.*receivers"):
        load_room_input(path)


@pytest.mark.parametrize("bad_id", ["", "R 1", "R\n1", "接收點", "R" * 65])
def test_rejects_receiver_id_outside_short_ascii_contract(
    tmp_path: Path, bad_id: str
) -> None:
    """空白、非 ASCII、含空格或過長 id 都不能進結果鍵與命令列標題。"""
    value = _multi_input("flat")
    receivers = value["receivers"]
    assert isinstance(receivers, list)
    receiver = receivers[0]
    assert isinstance(receiver, dict)
    receiver["id"] = bad_id
    path = tmp_path / "bad-id.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match=r"receivers\[0\]\.id"):
        load_room_input(path)


def test_rejects_receiver_missing_x(tmp_path: Path) -> None:
    """多點中任一筆缺 x，要指名那筆與缺的軸。"""
    value = _multi_input("flat")
    receivers = value["receivers"]
    assert isinstance(receivers, list)
    receiver = receivers[1]
    assert isinstance(receiver, dict)
    del receiver["x"]

    with pytest.raises(ValueError) as caught:
        load_room_input(_write_multi_answer(tmp_path, value))

    assert str(caught.value) == "receivers[1] 缺欄位：x"


def test_missing_single_receiver_precedes_later_required_field(tmp_path: Path) -> None:
    """舊形同時缺 receiver 與後面的必填欄位時，仍先報 receiver。"""
    value = _input_case("flat")
    del value["receiver_xyz_m"]
    del value["sound_speed_m_s"]

    with pytest.raises(ValueError) as caught:
        load_room_input(_write_multi_answer(tmp_path, value))

    assert str(caught.value) == "輸入檔缺欄位：receiver_xyz_m"


@pytest.mark.parametrize("case", ["flat", "varied"])
def test_every_receiver_matches_paths_and_totals_contract(
    tmp_path: Path, case: str
) -> None:
    """少算任一接收點，或路徑／總量任一格超界，這題都會列出 id 與界線使用率。"""
    inputs = load_room_input(_write_multi_input(tmp_path, case))
    assert inputs.materials is not None
    solve = getattr(receiver_api, "solve_receivers", None)
    assert callable(solve), "多接收點求解入口尚未實作"
    results = solve(inputs)
    answers = _answer_records(case)
    answer = _multi_answer(case)
    parameters = answer["parameters"]
    assert isinstance(parameters, dict)
    frequencies = tuple(_as_float_list(parameters["frequencies_hz"], "freqs"))
    violations: list[str] = []
    usage: list[str] = []

    for receiver_id, result in results.items():
        answer = answers[receiver_id]
        answer_paths = answer["paths"]
        answer_totals = answer["totals"]
        assert isinstance(answer_paths, list)
        assert isinstance(answer_totals, dict)
        path_rows = compare_paths(result.paths, answer_paths, frequencies)
        total_result = totals.compare_totals(result.paths, answer_totals, frequencies)
        path_bad = [row for row in path_rows if row.diffs]
        if path_bad or total_result.diffs:
            violations.append(
                f"{receiver_id}: path_bad={path_bad[:2]!r}; totals={total_result.diffs[:2]!r}"
            )
        max_refl = max(row.max_refl_frac for row in path_rows)
        max_pressure = max(row.max_pp_frac for row in path_rows)
        usage.append(
            f"{case}/{receiver_id}: path refl={max_refl:.3%}, path pressure={max_pressure:.3%}, "
            f"totals pressure={total_result.max_pressure_frac:.3%}, "
            f"direct={total_result.max_direct_frac:.3%}, reflected={total_result.max_reflected_frac:.3%}"
        )

    assert list(results) == list(answers)
    assert violations == [], "; ".join([*usage, *violations])


@pytest.mark.parametrize("case", ["flat", "varied"])
def test_cli_compare_accepts_multi_answer(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], case: str
) -> None:
    """新形答案必須逐 id 比 paths 與 totals，不能拿第一點冒充整包。"""
    input_path = _write_multi_input(tmp_path, case)
    answer_path = Path(__file__).resolve().parents[2] / "blueprint" / (
        f"reference_amplitude_multi_{case}.json"
    )

    exit_code = main([str(input_path), "--compare", str(answer_path)])
    output = capsys.readouterr().out

    assert exit_code == 0
    expected_ids = list(_answer_records(case))
    for receiver_id in expected_ids:
        assert f"接收點 {receiver_id}：" in output
    receiver_lines = [line for line in output.splitlines() if line.startswith("接收點 ")]
    assert [line.removeprefix("接收點 ").partition("：")[0] for line in receiver_lines] == expected_ids
    assert "多接收點總判決：全部在契約內" in output


def test_cli_json_multi_shape_round_trips_to_compare(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """多點 JSON 必須帶 id/xyz/paths/totals，且輸出本身可直接當 compare 答案。"""
    value = _multi_input("flat")
    raw_receivers = value["receivers"]
    assert isinstance(raw_receivers, list)
    raw_receivers.reverse()
    input_path = tmp_path / "reordered-input.json"
    input_path.write_text(json.dumps(value), encoding="utf-8")
    json_exit = main([str(input_path), "--json"])
    assert json_exit == 0
    output = capsys.readouterr().out
    payload = json.loads(output)
    receivers = payload["receivers"]
    assert isinstance(receivers, dict)
    expected_ids = [str(receiver["id"]) for receiver in raw_receivers]
    assert list(receivers) == expected_ids
    assert all(set(receiver) == {"xyz", "paths", "totals"} for receiver in receivers.values())
    assert payload["totals_by_receiver"] == {
        receiver_id: receiver["totals"] for receiver_id, receiver in receivers.items()
    }
    answer_path = tmp_path / "round-trip.json"
    answer_path.write_text(output, encoding="utf-8")

    compare_exit = main([str(input_path), "--compare", str(answer_path)])
    assert compare_exit == 0


def test_cli_compare_tampered_receiver_xyz_returns_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """答案檔 id 對得到但座標被改壞時，必須列出兩邊座標並判不同。"""
    data = copy.deepcopy(_multi_answer("flat"))
    parameters = data["parameters"]
    assert isinstance(parameters, dict)
    coordinates = parameters["receivers_xyz_m"]
    assert isinstance(coordinates, list)
    target = coordinates[1]
    assert isinstance(target, dict)
    target["x"] = 2.45

    exit_code = main(
        [
            str(_write_multi_input(tmp_path, "flat")),
            "--compare",
            str(_write_multi_answer(tmp_path, data)),
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "接收點 R1：座標對不上" in output
    assert "輸入=(2.2, 2.1, 1.15)" in output
    assert "答案檔=(2.45, 2.1, 1.15)" in output


def test_multi_r0_is_bit_identical_to_single_receiver(tmp_path: Path) -> None:
    """若多點流程重排或重算 R0，RoomPath 與 Totals 的逐位 dataclass 相等會失敗。"""
    multi_inputs = load_room_input(_write_multi_input(tmp_path, "flat"))
    multi = receiver_api.solve_receivers(multi_inputs)["R0"]
    single_path = tmp_path / "single.json"
    single_path.write_text(json.dumps(_input_case("flat")), encoding="utf-8")
    single_inputs = load_room_input(single_path)
    assert single_inputs.materials is not None
    single_paths = image_source_paths(
        single_inputs.room,
        single_inputs.source,
        single_inputs.receiver,
        single_inputs.sound_speed,
        single_inputs.max_order,
        single_inputs.materials,
    )

    assert multi.paths == single_paths
    assert multi.totals == totals.totals_from_paths(single_paths)


def test_solver_calls_single_receiver_solver_once_per_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """漏點或把多點塞進一次幾何呼叫時，實際收到的座標順序會不同。"""
    from aosr.physics import room_paths

    inputs = load_room_input(_write_multi_input(tmp_path, "flat"))
    calls: list[tuple[float, float, float]] = []
    original = room_paths.image_source_paths

    def counted(
        room: Room,
        source: Point,
        receiver: Point,
        c: float,
        max_order: int = 1,
        materials: Materials | None = None,
    ) -> list[RoomPath]:
        calls.append(receiver.as_tuple())
        return original(room, source, receiver, c, max_order, materials)

    monkeypatch.setattr(room_paths, "image_source_paths", counted)
    receiver_api.solve_receivers(inputs)

    assert calls == [receiver.point.as_tuple() for receiver in inputs.receivers]


def test_receiver_error_names_image_source_receiver(tmp_path: Path) -> None:
    """多點中某一點落在鏡像聲源上時，錯誤必須能定位到該 id。"""
    value = _multi_input("flat")
    receivers = value["receivers"]
    assert isinstance(receivers, list)
    target = receivers[0]
    assert isinstance(target, dict)
    source = value["source_xyz_m"]
    assert isinstance(source, dict)
    target.update(source)
    path = tmp_path / "on-image.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    inputs = load_room_input(path)

    with pytest.raises(ValueError, match=rf"{target['id']}.*鏡像"):
        receiver_api.solve_receivers(inputs)


def test_receiver_error_names_degenerate_receiver(tmp_path: Path) -> None:
    """反彈點打在牆邊的退化組態，也必須能定位到接收點 id。"""
    value = {
        "room": {"Lx_m": 2.0, "Ly_m": 2.0, "Lz_m": 2.0},
        "source_xyz_m": {"x": 0.5, "y": 0.5, "z": 1.0},
        "receivers": [{"id": "edge", "x": 1.5, "y": 1.5, "z": 1.0}],
        "sound_speed_m_s": 343.0,
        "max_order": 2,
    }
    path = tmp_path / "edge.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="edge.*退化組態"):
        receiver_api.solve_receivers(load_room_input(path))


@pytest.mark.parametrize("case", ["flat", "varied"])
def test_cli_compare_tampered_receiver_cell_returns_one(
    tmp_path: Path, case: str
) -> None:
    """改壞某個接收點的一格 path pressure，總判決與離開碼都必須紅。"""
    data = copy.deepcopy(_multi_answer(case))
    records = data["receivers"]
    assert isinstance(records, dict)
    receiver_id = next(receiver_id for receiver_id in records if receiver_id != "R0")
    record = records[receiver_id]
    assert isinstance(record, dict)
    paths = record["paths"]
    assert isinstance(paths, list)
    pressure = paths[0]["path_pressure"][0]["real"]
    pressure["dec"] = repr(float(pressure["dec"]) + 1.0)
    answer_path = _write_multi_answer(tmp_path, data)

    exit_code = main(
        [str(_write_multi_input(tmp_path, case)), "--compare", str(answer_path)]
    )
    assert exit_code == 1


def test_cli_compare_missing_receiver_returns_one_and_names_id(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """答案少一個 id 不能 zip 截斷；輸出要明說少的是哪一點。"""
    data = copy.deepcopy(_multi_answer("flat"))
    records = data["receivers"]
    assert isinstance(records, dict)
    missing_id = next(reversed(records))
    del records[missing_id]
    answer_path = _write_multi_answer(tmp_path, data)

    exit_code = main(
        [str(_write_multi_input(tmp_path, "flat")), "--compare", str(answer_path)]
    )
    output = capsys.readouterr().out

    assert exit_code == 1
    assert missing_id in output
    assert "答案檔少了接收點" in output


def test_cli_compare_extra_receiver_returns_one_and_names_id(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """答案多一個 id 也不能忽略；輸出要明說 v3 少的是哪一點。"""
    data = copy.deepcopy(_multi_answer("flat"))
    records = data["receivers"]
    assert isinstance(records, dict)
    template = copy.deepcopy(next(iter(records.values())))
    extra_id = "answer-only"
    records[extra_id] = template
    parameters = data["parameters"]
    assert isinstance(parameters, dict)
    coordinates = parameters["receivers_xyz_m"]
    assert isinstance(coordinates, list)
    coordinates.append({"id": extra_id, "x": 1.0, "y": 1.0, "z": 1.0})
    answer_path = _write_multi_answer(tmp_path, data)

    exit_code = main(
        [str(_write_multi_input(tmp_path, "flat")), "--compare", str(answer_path)]
    )
    output = capsys.readouterr().out

    assert exit_code == 1
    assert extra_id in output
    assert "v3 少了接收點" in output


def test_multi_compare_uses_live_total_pressure_tolerance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """多點 compare 必須走 totals 的活界線；換成零後不能漂綠。"""
    monkeypatch.setattr(totals, "total_pressure_tolerance", lambda paths, i, freqs: 0.0)
    answer_path = Path(__file__).resolve().parents[2] / "blueprint/reference_amplitude_multi_flat.json"

    exit_code = main(
        [str(_write_multi_input(tmp_path, "flat")), "--compare", str(answer_path)]
    )
    assert exit_code == 1


def test_multi_human_table_has_receiver_and_totals_sections(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """人看表每點各有具名座標標題，總量也不得只印第一點。"""
    inputs = load_room_input(_write_multi_input(tmp_path, "varied"))
    exit_code = main([str(_write_multi_input(tmp_path, "varied"))])
    assert exit_code == 0
    output = capsys.readouterr().out

    for receiver in inputs.receivers:
        assert f"接收點 {receiver.id}（x={receiver.point.x!r}" in output
    assert output.splitlines().count("總量") == len(inputs.receivers)


def test_single_receiver_cli_modes_are_byte_identical_to_origin_main(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    git_sandbox: GitSandbox,
) -> None:
    """舊形單點的表格、JSON、compare 標準輸出與離開碼逐位等於 origin/main。"""
    source = _origin_main_room_paths(tmp_path, git_sandbox)
    repo = Path(__file__).resolve().parents[2]
    cases = (
        ("amplitude-flat", _input_case("flat"), repo / "blueprint/reference_amplitude_flat.json"),
        (
            "amplitude-varied",
            _input_case("varied"),
            repo / "blueprint/reference_amplitude_varied.json",
        ),
        (
            "geometry-order3",
            _order3_geometry_input(),
            repo / "blueprint/reference_room_answers_order3.json",
        ),
    )
    differences: list[str] = []

    for case, input_value, answer_path in cases:
        input_path = tmp_path / f"{case}.json"
        input_path.write_text(json.dumps(input_value, sort_keys=True), encoding="utf-8")
        modes = ((), ("--json",), ("--compare", str(answer_path)))
        for args in modes:
            expected = _run_origin_main(source, input_path, args)
            actual_code = main([str(input_path), *args])
            captured = capsys.readouterr()
            actual = (actual_code, captured.out, captured.err)
            if actual != expected:
                differences.append(
                    f"{case}/{args or ('table',)}: code {actual_code}/{expected[0]}, "
                    f"stdout {len(captured.out.encode())}/{len(expected[1].encode())}, "
                    f"stderr {len(captured.err.encode())}/{len(expected[2].encode())}"
                )

    assert differences == [], differences
