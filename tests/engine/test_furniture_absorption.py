"""家具整件量進晚期一族：真求解、總量守恆與轉向逐位契約。"""
from __future__ import annotations

import random
import tomllib
from pathlib import Path
from typing import NoReturn

import pytest

from aosr.config.art_lane import ART_N_PER_WALL_DEFAULT
from aosr.config.frequency_axis import LATE_DECAY_FREQUENCIES_HZ
from aosr.config.paths import config_path
from aosr.config.three_lane_crossover import SCHROEDER_T60_BANDS_HZ
from aosr.geometry.furniture import FurnitureKind
from aosr.geometry.shoebox import Wall
from aosr.physics import furniture_scene, late_energy, three_lane_report as report, three_lane_report_batch as batch
from aosr.physics.report_furniture import AbsoluteFurniture
from aosr.physics.crossover import _room_measures, eyring_t60_by_band
from aosr.physics.furniture_scene import furniture_absorption_area_m2, furniture_adjusted_absorption
from tests.engine import _furniture_energy_cases as case
from tests.engine._furniture_absorption_cases import ITEMS, late_kwargs
from tests.engine._precision_contracts import contract_value
from tests.engine._report_cache import _key_value


@pytest.mark.parametrize("item", ITEMS, ids=lambda item: item.furniture_id)
def test_default_object_area_times_raw_catalog_absorption(item: AbsoluteFurniture) -> None:
    """抓貼地少扣底面、懸空少算底面，或未用材料檔的原始值。"""
    with config_path("furniture_materials.toml").open("rb") as stream:
        curve = tomllib.load(stream)[item.material]["default"]
    alpha = curve["absorption"][curve["band_center_hz"].index(1000.0)]
    # 考卷自行寫三件預設的封閉式，不從盒子或受驗函式反求面積。
    areas = {"sofa": 1.80 * 0.75 + 2.0 * 0.65 * (1.80 + 0.75),
             "table": 2.0 * 1.20 * 0.60 + 2.0 * 0.03 * (1.20 + 0.60),
             "cloud": 2.0 * 1.35 * 1.80 + 2.0 * 0.05 * (1.35 + 1.80)}
    assert furniture_absorption_area_m2((item,), (1000.0,), case.RHO_C) == (areas[item.furniture_id] * alpha,)


def _catalog_alpha(material: str, frequency_hz: float) -> float:
    with config_path("furniture_materials.toml").open("rb") as stream:
        curve = tomllib.load(stream)[material]["default"]
    return float(curve["absorption"][curve["band_center_hz"].index(frequency_hz)])


@pytest.mark.parametrize("kind", tuple(FurnitureKind), ids=str)
def test_every_kind_uses_its_own_closed_form(kind: FurnitureKind) -> None:
    """抓把「貼地」寫成只認沙發：椅子會被當懸空、多算底面；書桌少算一面也抓。"""
    material = {"sofa": "fabric", "chair": "fabric", "ceiling_cloud": "absorptive_cloud"}.get(kind, "wood")
    item = ITEMS[0].model_copy(update={"kind": kind, "material": material, "width_m": 1.1, "depth_m": 0.7, "height_m": 0.4})
    # 考卷自己寫哪兩種貼地，不讀產品的 grounded。
    w, d, h = 1.1, 0.7, 0.4
    area = w * d + 2.0 * h * (w + d) if kind in ("sofa", "chair") else 2.0 * w * d + 2.0 * h * (w + d)
    assert furniture_absorption_area_m2((item,), (1000.0,), case.RHO_C) == (area * _catalog_alpha(material, 1000.0),)


def test_several_items_sum_by_furniture_id_from_left() -> None:
    """抓多件只留最後一件、或每件都拿第一件的吸音率；預期值由封閉式與材料檔自算，不呼叫受驗函式。"""
    frequencies = (250.0, 1000.0, 4000.0)
    areas = {"sofa": 1.80 * 0.75 + 2.0 * 0.65 * (1.80 + 0.75),
             "table": 2.0 * 1.20 * 0.60 + 2.0 * 0.03 * (1.20 + 0.60),
             "cloud": 2.0 * 1.35 * 1.80 + 2.0 * 0.05 * (1.35 + 1.80)}
    expected = []
    for frequency in frequencies:
        total = 0.0
        for item in sorted(ITEMS, key=lambda item: item.furniture_id):
            total += areas[item.furniture_id] * _catalog_alpha(item.material, frequency)
        expected.append(total)
    assert furniture_absorption_area_m2(ITEMS, frequencies, case.RHO_C) == tuple(expected)


def test_absorption_is_bitwise_invariant_under_yaw_position_and_declaration_order() -> None:
    """抓從旋轉後 extents 或面序累加而改變末位，也抓依宣告順序加總。"""
    rng = random.Random(703)
    for _ in range(100):
        items = tuple(item.model_copy(update={name: rng.randint(1, 300) / 100.0
            for name in ("width_m", "depth_m", "height_m")}) for item in ITEMS)
        expected = repr(furniture_absorption_area_m2(items, case.FREQUENCIES, case.RHO_C))
        for yaw in (0.0, 90.0, 180.0, 270.0):
            turned = tuple(item.model_copy(update={"yaw_deg": yaw, "bottom_center_m": (1.0, 1.0, 1.0)})
                           for item in reversed(items))
            assert repr(furniture_absorption_area_m2(turned, case.FREQUENCIES, case.RHO_C)) == expected


def test_raw_curve_is_interpolated_and_extended_without_paris_clamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """把真材料檔天雲預設設成原有上界，抓 A_f 誤用夾值；仍真跑材料與求解來源。"""
    import math
    import re
    source = config_path("furniture_materials.toml").read_text()
    cloud = tomllib.loads(source)["absorptive_cloud"]
    bound = cloud["bounds"][0]["upper"]
    upper = {"band_center_hz": cloud["default"]["band_center_hz"],
             "absorption": [bound["absorption"][bound["band_center_hz"].index(f)]
                            for f in cloud["default"]["band_center_hz"]]}
    custom = tmp_path / "furniture_materials.toml"
    head, marker, tail = source.partition("[absorptive_cloud.default]")
    tail = re.sub(r"(?m)^band_center_hz = .*", f"band_center_hz = {upper['band_center_hz']}", tail, count=1)
    tail = re.sub(r"(?m)^absorption = .*", f"absorption = {upper['absorption']}", tail, count=1)
    custom.write_text(head + marker + tail)
    monkeypatch.setattr(furniture_scene, "config_path", lambda _name: custom)
    axis = (20.0, 750.0, 1000.0, 16000.0)
    item = ITEMS[-1]
    area = 2.0 * 1.35 * 1.80 + 2.0 * 0.05 * (1.35 + 1.80)
    frequencies, alpha = upper["band_center_hz"], upper["absorption"]
    lo, hi = frequencies.index(500.0), frequencies.index(1000.0)
    middle = alpha[lo] + (alpha[hi] - alpha[lo]) * math.log10(750.0 / 500.0) / math.log10(1000.0 / 500.0)
    expected = tuple(area * a for a in (alpha[0], middle, alpha[hi], alpha[-1]))
    assert furniture_absorption_area_m2((item,), axis, case.RHO_C) == pytest.approx(
        expected, rel=contract_value("late_energy_physical_property"))
    impedance = furniture_scene.furniture_impedances((item,), axis, case.RHO_C)[item.furniture_id]
    assert impedance.catalog_absorption[axis.index(1000.0)] > impedance.absorption[axis.index(1000.0)]


def test_late_family_eyring_and_art_share_total_absorption() -> None:
    """抓只改 Eyring、ART 或衰減其中一路、重複加家具、分配總量錯或回傳了調過的牆。"""
    surfaces, t60, _fs, late, decay = batch._prepare_late_family(**late_kwargs(), furniture=ITEMS)
    original = batch._prepare_late_family(**late_kwargs())[0]
    assert surfaces == original
    _, areas = _room_measures(case.ROOM)
    total = sum(areas.values())
    af = furniture_absorption_area_m2(ITEMS, surfaces.frequencies_hz, case.RHO_C)
    averages = {f: (sum(areas[w] * surfaces.absorption_by_wall[w][i] for w in Wall.all()) + af[i]) / total
                for i, f in enumerate(surfaces.frequencies_hz)}
    uniform = {w: {f: averages[f] for f in SCHROEDER_T60_BANDS_HZ} for w in Wall.all()}
    assert t60 == pytest.approx(eyring_t60_by_band(case.ROOM, uniform), rel=contract_value("late_energy_physical_property"))
    af_on_report_axis = furniture_absorption_area_m2(ITEMS, case.FREQUENCIES, case.RHO_C)
    adjusted = furniture_adjusted_absorption(case.ROOM, case.FREQUENCIES,
        {w: tuple(surfaces.absorption_by_wall[w][surfaces.frequencies_hz.index(f)] for f in case.FREQUENCIES)
         for w in Wall.all()}, af_on_report_axis)
    energy = late_energy.solve_late_energy(late_energy.LateEnergyInputs(room=case.ROOM,
        rho_c_pa_s_per_m=case.RHO_C, frequencies_hz=case.FREQUENCIES,
        absorption_by_wall={w.wall_name(): row for w, row in adjusted.items()}, n_per_wall=ART_N_PER_WALL_DEFAULT, domain_alpha_bar_max=float("inf")))
    for band, by_order in zip(energy.bands, late.bands, strict=True):
        assert band.late_reverberant_energy == by_order.late_reverberant_energy
        assert band.alpha_bar == pytest.approx(averages[band.frequency_hz], rel=contract_value("late_energy_physical_property"))
    # 六面同阻抗，分完仍是同值；衰減拿平均吸音率的均勻列重算一次，不經分配函式。
    af_on_decay_axis = furniture_absorption_area_m2(ITEMS, LATE_DECAY_FREQUENCIES_HZ, case.RHO_C)
    rows = tuple((sum(areas[w] * surfaces.absorption_by_wall[w][surfaces.frequencies_hz.index(f)] for w in Wall.all())
                  + af_on_decay_axis[i]) / total for i, f in enumerate(LATE_DECAY_FREQUENCIES_HZ))
    reference = report._solve_report_late_decay(room=case.ROOM, absorption_by_wall={w.wall_name(): rows for w in Wall.all()},
        rho_c_pa_s_per_m=case.RHO_C, sound_speed_m_s=case.SPEED)
    assert decay.unavailable_by_center_hz.keys() == reference.unavailable_by_center_hz.keys()
    for got, want in zip(decay.result.bands, reference.result.bands, strict=True):
        assert got.t20_s == pytest.approx(want.t20_s, rel=contract_value("late_energy_physical_property"))
        assert (got.t30_s is None) == (want.t30_s is None)
        if want.t30_s is not None:
            assert got.t30_s == pytest.approx(want.t30_s, rel=contract_value("late_energy_physical_property"))


@pytest.mark.parametrize("af, alpha", ((1000.0, 0.2), (0.0, 1.0), (-1.0, 0.2), (float("nan"), 0.2), (float("inf"), 0.2)))
def test_invalid_remaining_reflectivity_is_rejected_with_diagnostics(af: float, alpha: float) -> None:
    with pytest.raises(ValueError, match="125.*A_f.*denom"):
        furniture_adjusted_absorption(case.ROOM, (125.0,), {w: (alpha,) for w in Wall.all()}, (af,))


def test_zero_furniture_absorption_is_accepted_and_leaves_walls_unchanged() -> None:
    """抓把 ε＝0 也拒收（下界寫成嚴格大於）。"""
    rows = {w: (0.2,) for w in Wall.all()}
    assert furniture_adjusted_absorption(case.ROOM, (125.0,), rows, (0.0,)) == rows


def test_no_furniture_keeps_all_five_outputs_bitwise_equal(monkeypatch: pytest.MonkeyPatch) -> None:
    """抓沒家具仍走分配那條路（例如條件寫成 is not None，空清單也進去算 ε＝0）。"""
    expected = repr(_key_value(batch._prepare_late_family(**late_kwargs())))

    def forbidden(*_args: object, **_kwargs: object) -> NoReturn:
        raise AssertionError("沒家具不准算家具吸音")
    monkeypatch.setattr(batch, "furniture_absorption_area_m2", forbidden)
    monkeypatch.setattr(batch, "furniture_adjusted_absorption", forbidden)
    assert repr(_key_value(batch._prepare_late_family(**late_kwargs(), furniture=None))) == expected
    assert repr(_key_value(batch._prepare_late_family(**late_kwargs(), furniture=()))) == expected


def test_late_family_rejects_furniture_that_exhausts_reflectivity() -> None:
    giant = ITEMS[0].model_copy(update={"width_m": 100.0, "depth_m": 100.0})
    with pytest.raises(ValueError, match="A_f.*denom"):
        batch._prepare_late_family(**late_kwargs(), furniture=(giant,))


def test_nonuniform_walls_keep_total_area_and_remaining_reflectivity_ratios() -> None:
    """抓把家具量平均加到牆上、或 denominator（分母）沒用牆自己的吸音率。"""
    rows = {wall: (i / 10.0, i / 12.0) for i, wall in enumerate(Wall.all())}
    frequencies, af = (125.0, 1000.0), (2.0, 3.0)
    adjusted = furniture_adjusted_absorption(case.ROOM, frequencies, rows, af)
    _, areas = _room_measures(case.ROOM)
    for i, extra in enumerate(af):
        before = sum(areas[w] * rows[w][i] for w in Wall.all())
        after = sum(areas[w] * adjusted[w][i] for w in Wall.all())
        assert after == pytest.approx(before + extra, rel=contract_value("late_energy_physical_property"))
        factors = tuple((1.0 - adjusted[w][i]) / (1.0 - rows[w][i]) for w in Wall.all())
        assert factors == pytest.approx((factors[0],) * len(factors), rel=contract_value("late_energy_physical_property"))
    remaining = sum(areas[w] * (1.0 - rows[w][0]) for w in Wall.all())
    with pytest.raises(ValueError, match="125.*A_f.*denom"):
        furniture_adjusted_absorption(case.ROOM, (125.0,), {w: (rows[w][0],) for w in Wall.all()}, (remaining,))
