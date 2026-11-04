"""分區加權的凸組合界、四區交集與極細分區。"""
from dataclasses import replace

import numpy as np
import pytest

from aosr.geometry.shoebox import Room, Wall
from aosr.physics import late_energy, wall_surfaces
from aosr.physics.report_surfaces import RectangleZone
from tests.engine._precision_contracts import contract_value
from tests.engine.test_late_energy_zones import CONSTANTS, MATERIAL, synthetic


def rounding_problem(alphas: tuple[float, ...]) -> late_energy.LateEnergyInputs:
    """0.1 m、0.3 m 切線讓區面積和與整牆乘法的最後一位不同。"""
    u, v = (0.0, 0.1, 6.7), (0.0, 0.3, 4.1)
    cells = tuple(wall_surfaces.RectangleAbsorption(RectangleZone(area="rectangle", id=f"q{i}{j}",
        u_m=(u[i], u[i+1]), v_m=(v[j], v[j+1]), material=MATERIAL), (alphas[2*i+j],))
        for i in range(2) for j in range(2))
    return late_energy.LateEnergyInputs(Room(6.7, 4.1, 2.9), CONSTANTS.rho_c, (500.0,),
        dict.fromkeys(Wall.wall_names(), (0.25,)), 2, 0.9, absorption_by_zone={"floor": cells})


@pytest.mark.parametrize("alpha", (1.0, 0.7))
def test_uniform_zone_roundoff_keeps_exact_alpha_and_product_accepts(alpha: float) -> None:
    """牆面積分母且不夾界會令全吸地板超過 1，產品入口拒收合法分區。"""
    problem = rounding_problem((alpha,) * 4)
    result = late_energy.solve_late_energy(problem)
    assert result.bands[0].alpha_by_wall["floor"] == alpha
    assert problem.absorption_by_zone is not None
    effective = wall_surfaces.equivalent_absorption_rows(problem.room, problem.absorption_by_wall,
                                                        problem.absorption_by_zone)
    assert effective["floor"] == (alpha,)
    np.testing.assert_array_equal(late_energy._patch_absorption(problem)[:4], alpha)


def test_mixed_zone_average_stays_within_participating_alpha_bounds() -> None:
    """平均不得跑出各頻帶參與區域的界；只夾全牆 [0,1] 不夠。"""
    problem = rounding_problem((0.2, 0.4, 0.6, 0.8))
    assert problem.absorption_by_zone is not None
    effective = wall_surfaces.equivalent_absorption_rows(problem.room, problem.absorption_by_wall,
                                                        problem.absorption_by_zone)
    assert 0.2 <= effective["floor"][0] <= 0.8
    # 後三塊只碰上排、右列或右上區；界必須排除零交集的區。
    patches = late_energy._patch_absorption(problem)[:4, 0]
    assert np.all(patches >= (0.2, 0.4, 0.6, 0.8))
    assert np.all(patches <= 0.8)


def test_one_patch_overlaps_all_four_zones_with_hand_areas() -> None:
    """左下塊 3×2，切在 u=3/4、v=1/2：交集為 3/8、9/8、9/8、27/8。"""
    problem = synthetic(Wall.FLOOR, 2, 0.125, 0.125)
    assert problem.absorption_by_zone is not None
    weights = wall_surfaces.rectangle_patch_weights(problem.room, "floor", problem.absorption_by_zone["floor"], 2)
    np.testing.assert_array_equal(weights[0], (1/16, 3/16, 3/16, 9/16))
    # 手算四區材料 1/8、1/4、1/2、3/4，不能只保留前兩個非零交集。
    assert late_energy._patch_absorption(problem)[0, 0] == 73/128


def test_thin_zone_contributes_its_full_intersection_weight() -> None:
    """牆寬的 1e-6 仍有正交集；清掉小面積會把本題吸音率整個抹掉。"""
    problem = synthetic(Wall.FLOOR, 2, 1e-6, 0.5)
    assert problem.absorption_by_zone is not None
    cells = tuple(replace(cell, absorption=(1.0, 1.0) if i < 2 else (0.0, 0.0))
                  for i, cell in enumerate(problem.absorption_by_zone["floor"]))
    problem = replace(problem, absorption_by_zone={"floor": cells})
    weights = wall_surfaces.rectangle_patch_weights(problem.room, "floor", cells, 2)
    # 左下塊面積 6，細條交集 6e-6×2=12e-6，比例 2e-6。
    np.testing.assert_allclose(weights[0], (2e-6, 0.0, 1.0-2e-6, 0.0),
        rtol=contract_value("late_energy_physical_property"), atol=0.0)
    assert late_energy._patch_absorption(problem)[0, 0] == pytest.approx(2e-6,
        rel=contract_value("late_energy_physical_property"), abs=0.0)
