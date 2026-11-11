"""#562 審查修補：物理命令列、實測接縫、公開分界入口與第二區構造註記。"""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from aosr.config.paths import config_path
from aosr.geometry.shoebox import Wall
from aosr.physics import report_io, report_output, three_lane_report_cli
from aosr.runtime import child_process_env
from tests.engine._directivity import DIRECTIVITY
from tests.engine._furniture_cases import CAPABILITIES, pair
from tests.engine._zoned_report_cases import small_scheme
from tests.engine.test_measured_impedance import measured


def test_physics_cli_directs_rectangles_to_scheme_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """合法分區輸入應告知可用入口，不可落成六面材料缺漏。"""
    source = tmp_path / "input.json"
    source.write_text(report_io.load_input_document(pair(small_scheme(zoned=True)),
        CAPABILITIES, DIRECTIVITY).model_dump_json())
    assert three_lane_report_cli.main([str(source), "--capabilities", str(config_path("capabilities.toml"))]) == 2
    assert "分區方案請走 aosr.reporting.scheme_cli run" in capsys.readouterr().out


@pytest.mark.parametrize("wall,index", (("x0", 0), ("x0", 1), ("floor", 0)))
def test_zoned_measured_material_is_calculable(wall: str, index: int) -> None:
    """#1084：首區、第二區或其他整面貼實測表都查得到能力列、不再拒收；接縫各報一處由雙精度探針
    （test_zoned_measured_seams 的 routing 段）守。"""
    inputs = report_io.load_input_document(pair(small_scheme(zoned=True)), CAPABILITIES, DIRECTIVITY)
    document = inputs.model_dump(mode="json")
    zones = inputs.surface_zones_by_wall
    assert zones is not None
    updated = {name: [zone.model_dump(mode="json") for zone in cells] for name, cells in zones.items()}
    updated[wall][index]["material"] = measured()
    inputs = report_io.load_input_document(document | {"surface_zones_by_wall": updated}, CAPABILITIES, DIRECTIVITY)
    assert report_output.report_capability(CAPABILITIES, {}, inputs.surface_zones_by_wall).materials == "measured_impedance_by_wall"


def test_public_partition_check_rejects_gap_and_accepts_full_coverage() -> None:
    """純資料消費端可公開驗分界，且不得漏掉覆蓋洞。"""
    from aosr.geometry.shoebox_mesh import checked_partitions

    room = small_scheme().scene.room_m
    checked_partitions(room, {Wall.X0: (("a", (0.0, 1.0), (0.0, 1.5)),
                                      ("b", (1.0, 2.0), (0.0, 1.5)))})
    with pytest.raises(ValueError, match="蓋滿整面"):
        checked_partitions(room, {Wall.X0: (("a", (0.0, 1.0), (0.0, 1.5)),)})


def test_second_zone_normal_notes_stay_on_own_wall(tmp_path: Path) -> None:
    """真雙精度子行程：第二區註記應跟無規入射一致，不能沾到其餘五面。"""
    probe = Path(__file__).with_name("_zoned_report_notes_probe.py")
    source = tmp_path / "scheme.json"
    source.write_text(small_scheme(zoned=True).model_dump_json())
    result = subprocess.run([sys.executable, str(probe), str(source)], cwd=tmp_path,
        env=child_process_env(threads=1), capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout+result.stderr
