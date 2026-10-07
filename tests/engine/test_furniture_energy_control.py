"""沒家具逐位控制組：2026-10-07 主線 d71d2f983887e22826d465c4bbf1b751da7b291a 實跑。

錄製命令：UV_CACHE_DIR=<本次新建暫存目錄> uv run --no-sync python - <<'PY'
幾何呼叫與下面 test_no_furniture_geometric_control_two_models 的引數相同，但尚無 furniture 引數，
逐欄 print(tuple(value.hex() for value in getattr(result, field)))；頻點取正式細軸 20、80、140。
三路呼叫同下面 reference_report_control，但錄製時不換任何求解器，真算參考房左喇叭→主位，
print({i: (report.points[i].frequency_hz.hex(), report.points[i].total_energy.hex()) for i in (0, 80, 100, 219)})。
額外取 140 點的值自同次保存的真報表。這裡只比 FEM 權重為零的總量；考卷換掉有限元素與
衰減的慢求解（兩者不影響所比的總量），幾何、晚期逐階能量、交接與總量均真跑。
答案是施工前控制組的字串，沒有用被測接線現算答案。
"""
from __future__ import annotations

import pytest

from aosr.geometry.shoebox import Point
from aosr.physics.geometric_lane import solve_geometric_lane
from aosr.physics.report_source import SourceModelKind, SourceModelSpec
from aosr.physics import three_lane_report as report
from tests.engine import _furniture_energy_cases as case, _source_model_control as stand_ins
from tests.engine._directivity import DIRECTIVITY

FIELDS = ("geometric_energy", "reflected_energy", "interference_energy", "late_energy")
GEOMETRIC_HEX = (
    (("0x1.d39f8690e7c68p-2", "0x1.1e1f75167d48dp-3", "0x1.6a72d4149be1fp-5"),
     ("0x1.b97a3c3445789p-2", "0x1.d918fb18d3d31p-7", "0x1.1b23c92ab73d7p-7"),
     ("-0x1.da905e602e5c8p-5", "0x1.6f1fa6257580ap-5", "-0x1.2eeec29f7d2c3p-5"),
     ("0x1.e9249967661f1p-7", "0x1.88bc9100732bap-7", "0x1.2225b74bd38cdp-8")),
    (("0x1.d83d48a1d4f49p-2", "0x1.33d602a1fff71p-3", "0x1.60038763f5ae1p-5"),
     ("0x1.befb78d58f5fcp-2", "0x1.09a96cf76f2a5p-6", "0x1.2b543819a228dp-7"),
     ("-0x1.e3f3a1c612880p-5", "0x1.a0ebe43fab37fp-5", "-0x1.433c984c4fa5ap-5"),
     ("0x1.f21cbfc76df6bp-7", "0x1.e0c74436dc5d1p-7", "0x1.556e770e178d9p-8")),
)
REPORT_HEX = {100: "0x1.03ade9dda6ee7p-1", 140: "0x1.40fef991dfcb2p-4", 219: "0x1.09fdb728a88dfp-4"}


@pytest.mark.parametrize("index", range(len(case.CONTROL_CURVES)))
def test_no_furniture_geometric_control_two_models(index: int) -> None:
    actual = solve_geometric_lane(source_model=SourceModelSpec(
        SourceModelKind.ANALYTIC_AXISYMMETRIC_TWO_PARAMETER_V1,
        case.CONTROL_CURVES[index], Point(4.8, 2.6, 1.2)),
        room=case.ROOM, source=Point(1.2, 1.3, 1.1), receiver=Point(4.7, 2.8, 1.4),
        sound_speed_m_s=case.SPEED, rho_c_pa_s_per_m=case.RHO_C,
        frequencies_hz=case.CONTROL_FREQUENCIES,
        impedance_by_wall={wall.wall_name(): value for wall, value in case.WALLS.items()},
        scattering_by_wall={wall.wall_name(): 0.2 for wall in case.WALLS}, reflection_order_k=3, furniture=None)
    assert tuple(tuple(v.hex() for v in getattr(actual, field)) for field in FIELDS) == GEOMETRIC_HEX[index]
    assert actual.furniture is None


def test_no_furniture_reference_report_control(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(report, "_solve_fem_energy", stand_ins.fake_fem_energy)
    monkeypatch.setattr(report, "_solve_report_late_decay", stand_ins.fast_late_decay)
    actual = report.solve_three_lane_report(source_model=SourceModelSpec(
        SourceModelKind.ANALYTIC_AXISYMMETRIC_TWO_PARAMETER_V1, DIRECTIVITY.two_parameter, Point(3.2, 1.9, 1.2)),
        room=case.ROOM, source=Point(1.0, 1.3, 1.2), receiver=Point(3.2, 1.9, 1.2),
        sound_speed_m_s=case.SPEED, density_kg_m3=case.DENSITY, impedance_by_wall=case.WALLS, furniture=None)
    for i, expected in REPORT_HEX.items():
        assert actual.points[i].w_fem == 0.0
        assert actual.points[i].total_energy.hex() == expected
    assert actual.furniture is None
