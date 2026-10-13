"""#559 路徑表控制組：主線 958caaa6465b4a60f8b6e47511dc2e9adff7f1c0，2026-10-08 乾淨樹實跑。

參考房 left→main，完整細軸與預設指向性；只求真幾何早期路，不需要有限元素替身。
錄製：UV_CACHE_DIR=<本次暫存>/uv-cache uv run --no-sync python <本次暫存>/control.py。
重驗：uv run --no-sync pytest -n 4 tests/engine/test_furniture_path_control.py。
舊 SHA-256 與九格 hex 答案保留。第五步在 HEAD 5339182ca3fd1831a684a19ab5c03462eb5b7e1f
用 bash <本次暫存>/record.sh 重錄，舊雜湊同跑核過；新答案住 _furniture_path_answers.py。
report_path_table._path_row 的 relative_direct_energy 經 amplitude.path_pressure 的 cmath.exp、
apply_pressure_factor 的 numpy.power／expm1 及次方，逐格①級、絕對 0；沒有晚期能量或解方程組。
report_path_table._direction 經 atan2／hypot；source_directivity.off_axis_degrees 經 asin，角度逐格①級、絕對 0。
frequencies_hz 是紙指定的軸鍵例外；六面同散射時 geometric_lane._scattering_at 直接回輸入，留骨架。
delay_s、distance_m 來自 shoebox.distance 的四則／sqrt；direction_vector 只除以距離，留骨架逐字。
骨架取產品原文，只換機器相關字面，保留順序、格式、字串、整數與其餘浮點。不再證明跨處理器逐位。
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from aosr.config.capabilities import load_capabilities
from aosr.config.directivity_defaults import load_directivity_defaults
from aosr.config.frequency_axis import GEOMETRIC_LANE_FREQUENCIES_HZ
from aosr.config.paths import config_path
from aosr.physics.geometric_lane import solve_geometric_early_lane
from aosr.physics.report_io import PathTableSection, solver_inputs
from aosr.physics.report_path_table import build_path_table_section
from aosr.reporting.scheme import Scheme
from aosr.reporting.validation import checked_inputs
from tests.engine._cross_cpu_tolerance import GEOMETRIC_REL, assert_within, cell
from tests.engine._furniture_path_answers import ANSWERS
from tests.engine._json_control import assert_json_control


MAIN_JSON_SHA256 = "c43660ce956025d417a6f0a4aedaf6df0d5515809bb74609553a41c2092c2941"
MAIN_ENERGY_HEX = {
    ("x0",): ("0x1.6efcfe2f08523p-4", "0x1.f929e12b5dd02p-5", "0x1.bf1cf5b1cd6e1p-8"),
    ("floor",): ("0x1.9c6a39d25ec90p-4", "0x1.78f99a714fd4cp-4", "0x1.da4c018784b83p-5"),
    ("ceiling",): ("0x1.34a51708324d1p-4", "0x1.0f289ecae02ddp-4", "0x1.10b625f391ac8p-5"),
}


def reference_section() -> PathTableSection:
    table = load_capabilities(config_path("capabilities.toml"))
    directivity = load_directivity_defaults(config_path("directivity_defaults.toml"))
    path = Path(__file__).resolve().parents[2] / "blueprint" / "scheme_reference_room.json"
    scheme = Scheme.model_validate_json(path.read_text())
    _, documents = checked_inputs(scheme, capabilities=table, directivity=directivity)
    solved = solver_inputs(documents[("left", "main")][1])
    assert solved.impedance_by_wall is not None
    lane = solve_geometric_early_lane(
        source_model=solved.source_model, room=solved.room, source=solved.source, receiver=solved.receiver,
        sound_speed_m_s=solved.sound_speed_m_s, rho_c_pa_s_per_m=solved.density_kg_m3 * solved.sound_speed_m_s,
        frequencies_hz=GEOMETRIC_LANE_FREQUENCIES_HZ,
        impedance_by_wall={wall.wall_name(): z for wall, z in solved.impedance_by_wall.items()},
        scattering_by_wall={wall.wall_name(): s for wall, s in (solved.scattering_by_wall or {}).items()},
        reflection_order_k=solved.reflection_order_k, furniture=None,
    )
    return build_path_table_section(SimpleNamespace(geometric_lane=lane), solved)


def test_unfurnished_reference_json_and_wall_energy_are_bitwise_main() -> None:
    section = reference_section()
    assert_json_control(section.model_dump_json(), "path", ANSWERS["path"])
    for sequence, expected in MAIN_ENERGY_HEX.items():
        row = next(row for row in section.rows if row.wall_sequence == sequence)
        assert_within(cell(f"牆序列{sequence}.relative_direct_energy[{i}]", row.relative_direct_energy[i], answer,
                           rel=GEOMETRIC_REL, abs_=0.0)
                      for i, answer in zip((20, 80, 140), expected, strict=True))

