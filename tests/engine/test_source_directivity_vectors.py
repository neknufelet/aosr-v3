"""方向倍率共用介面與施工前主線牆面逐位控制組。

控制組出身：2026-10-07，乾淨主線 e60f71dc94e5d7803165f645940cdde03162f16b，Python 3.12.3。
動手前以 uv run --no-sync python 探針，按本檔房間／介質／軸／路徑條件呼叫
主線 apply_pressure_factor；D 用主線 _pressure_function(x)，x 用主線 departure_direction、
speaker_axis、one_minus_cos。各點 float.hex() 標準輸出抄入下方，測試不重算答案。
"""
import math
from dataclasses import replace

import numpy as np
import pytest

from aosr.config.directivity_defaults import load_directivity_defaults
from aosr.config.paths import config_path
from aosr.geometry.shoebox import Point, Room
from aosr.physics.amplitude import CANONICAL_WALLS, Materials
from aosr.physics.room_paths import image_source_paths
from aosr.physics import source_directivity as core


def test_downward_table_departure_matches_hand_angle() -> None:
    """決策紙第 27 條：往下方向的指向性尚未獨立驗證；此題只驗公式接線。"""
    # 桌頂 z=.6，S(-.6,0,1.05)→H(0,0,.6)：(.6,0,-.45)÷.75=(.8,0,-.6)。
    # 主位 E(.6,0,1.05)，軸線 (1.2,0,0)÷1.2=(1,0,0)，cos=.8、x=.2。
    x = 1.0 - 0.8
    angle = math.degrees(math.acos(0.8))
    assert math.isclose(angle, math.degrees(math.atan2(0.6, 0.8)), rel_tol=1e-14)
    params = load_directivity_defaults(config_path("directivity_defaults.toml"))
    frequencies = (125.0, 1000.0, 8000.0)
    expected = core.two_parameter_pressure_factor(x, frequencies, params)
    got = core.pressure_factor_for_direction(
        (0.8, 0.0, -0.6), frequencies, core.SourceModel.TWO_PARAMETER,
        source=Point(-0.6, 0.0, 1.05), aim=Point(0.6, 0.0, 1.05), params=params)
    np.testing.assert_allclose(got, expected, rtol=1e-14, atol=0.0)


_DIRECT_CONTROL = (
    ("0x1.0000000000000p+0", "0x1.0000000000000p+0", "0x1.0000000000000p+0"),
    (("0x1.5ab8216383cebp-3", "-0x1.0f2c6bce7c0b2p-2"),
     ("-0x1.9f0560fc5b2e8p-5", "-0x1.3da4b81435c87p-2"),
     ("0x1.5e6b60c9a3740p-4", "-0x1.35b32c9460ff8p-2")),
)
_WALL_CONTROLS = {
    core.SourceModel.TWO_PARAMETER: (
        _DIRECT_CONTROL,
        (("0x1.ce710d323f953p-1", "0x1.4d112475e085ep-2", "0x1.2302a7e6b86c5p-5"),
         (("0x1.4ff7bf480a275p-7", "0x1.d8ce69251fae6p-5"),
          ("0x1.a42859896dd1dp-8", "-0x1.3cc446a2be86bp-5"),
          ("0x1.55377a9d51e83p-10", "0x1.456e555bd1216p-8"))),
        (("0x1.cb4e788aed651p-1", "0x1.3afa042a5a273p-2", "0x1.c0eead8f64eaap-6"),
         (("0x1.257d783c20076p-11", "-0x1.eaf8d3a40c787p-9"),
          ("0x1.135bcfa1e051dp-8", "0x1.54b5c4a0ef0acp-7"),
          ("-0x1.ba83e09dcf679p-10", "-0x1.ebb839d14aa51p-14"))),
    ),
    core.SourceModel.V2_COMPAT: (
        _DIRECT_CONTROL,
        (("0x1.e8e65e96e8222p-1", "0x1.5615e0915ab09p-2", "0x1.f507717da5b58p-8"),
         (("0x1.63309ecc6b8f9p-7", "0x1.f3db89ebd39d2p-5"),
          ("0x1.af88c64e2fa31p-8", "-0x1.45580580954a7p-5"),
          ("0x1.25bbff582dcdfp-12", "0x1.182533967f0dep-10"))),
        (("0x1.e72776dff9633p-1", "0x1.324a4aafde959p-2", "0x1.29af9f92c4cbep-6"),
         (("0x1.3748caaa63c74p-11", "-0x1.045ea7374ee43p-8"),
          ("0x1.0bc3c9c9377adp-8", "0x1.4b505dd751837p-7"),
          ("-0x1.256e49342af16p-10", "-0x1.460ee57e0fb84p-14"))),
    ),
}


@pytest.mark.parametrize("model", tuple(_WALL_CONTROLS))
def test_wall_control_hex_pins_direction_factor_and_pressure(model: core.SourceModel) -> None:
    frequencies = (125.0, 1000.0, 8000.0)
    source, receiver = Point(0.9, 1.3, 1.1), Point(3.8, 2.6, 1.25)
    params = load_directivity_defaults(config_path("directivity_defaults.toml"))
    materials = Materials(415.03, frequencies,
        {wall: (830.06 + 0j, 1660.12 + 0j, 2490.18 + 0j) for wall in CANONICAL_WALLS})
    identities = ((0, 1, 0, 1, 0, 1), (0, -1, 0, 1, 0, 1), (0, -1, 0, -1, 0, 1))
    all_paths = {path.identity: path for path in image_source_paths(
        Room(5.3, 3.7, 2.9), source, receiver, 343.0, max_order=2, materials=materials)}
    paths = [all_paths[key] for key in identities]
    assert tuple(path.order for path in paths) == (0, 1, 2)
    adjusted = core.apply_pressure_factor(
        paths, receiver, frequencies, model, source=source, aim=receiver, params=params,
        baffle_width_m=0.21, piston_radius_m=0.08, sound_speed_m_s=343.0)
    for path, after, (expected_d, expected_p) in zip(paths, adjusted, _WALL_CONTROLS[model], strict=True):
        factors = core.pressure_factor_for_direction(
            core.departure_direction(path, receiver), frequencies, model,
            source=source, aim=receiver, params=params, baffle_width_m=0.21,
            piston_radius_m=0.08, sound_speed_m_s=343.0)
        assert tuple(float(d).hex() for d in factors) == expected_d
        assert tuple((p.real.hex(), p.imag.hex()) for p in after.path_pressure) == expected_p


def test_omnidirectional_factor_needs_no_axis_geometry_or_parameters() -> None:
    got = core.pressure_factor_for_direction((0.8, 0.0, -0.6), (125.0, 1000.0),
                                             core.SourceModel.OMNIDIRECTIONAL)
    assert tuple(float(value).hex() for value in got) == ("0x1.0000000000000p+0", "0x1.0000000000000p+0")


def test_room_paths_use_the_public_direction_factor(monkeypatch: pytest.MonkeyPatch) -> None:
    """改成另一份牆面判法就紅；倍率替身只用來證明這個共用接點被呼叫。"""
    source, receiver = Point(1.0, 1.0, 1.0), Point(3.0, 2.0, 1.0)
    direct = next(path for path in image_source_paths(Room(4.0, 4.0, 3.0), source, receiver, 343.0,
                                                     max_order=1) if path.order == 0)
    path = replace(direct, path_pressure=(1.0 + 2.0j, -2.0 + 3.0j, 4.0 - 1.0j))
    def factor(*args: object, **kwargs: object) -> np.ndarray:
        return np.asarray((0.25, 0.5, 0.75))
    monkeypatch.setattr(core, "pressure_factor_for_direction", factor)
    params = load_directivity_defaults(config_path("directivity_defaults.toml"))
    got = core.apply_pressure_factor([path], receiver, (125.0, 1000.0, 8000.0), core.SourceModel.TWO_PARAMETER,
                                    source=source, aim=receiver, params=params)
    assert got[0].path_pressure == (0.25 + 0.5j, -1.0 + 1.5j, 3.0 - 0.75j)
