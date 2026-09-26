"""交接階數 K 是呼叫端與輸入檔可調的設定（票 #341）。

票 #337 把幾何路與晚期混響改成逐階分工，但 K 寫死在產品設定
``config.three_lane_crossover.REFLECTION_ORDER_K``；這一支守的是它**露出來**之後的四件事：

1. 給不同的 K 真的算出不同的東西（不是收下參數然後照舊用常數）；
2. 不給 K 時逐位等於產品設定那一個（預設不變）；
3. K 出界要報錯，兩端的界線一律從 ``room_paths.SUPPORTED_MIN_ORDER``／
   ``SUPPORTED_MAX_ORDER`` 現算——寫死 9 的話，上限哪天再放寬，這一題就從「守出界」
   變成「守一個合法值」而照樣綠；
4. 輸入檔那一格 ``reflection_order_k`` 要一路走到報表印出來的那一格。

**這一支只驗性質，不對任何凍結答案，也不新增門檻數字**：需要容差的地方拿既有精度契約
登記簿的相對界線縮放本題量級，跟 ``test_geometric_order_split.py`` 同一條路。

**交線反射也要跑到合法上限**：票 #305 採所有相交牆面一起算後，參考房的整齊座標與挪開
幾公分的座標都應跑到上限；本題只守可用性與階數帳，不替其他 K 補數值答案。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from pydantic import BaseModel

from aosr.config.three_lane_crossover import REFLECTION_ORDER_K
from aosr.geometry.shoebox import Point, Room, Wall
from aosr.physics.report_source import SourceModelKind, SourceModelSpec
from aosr.physics.geometric_lane import (
    GeometricEarlyResult,
    GeometricLaneResult,
    average_geometric_lane_to_bands_with_dense_early,
    solve_geometric_early_lane,
    solve_geometric_lane,
)
from aosr.physics.report_io import (
    ReportInput,
    ReportOutput,
    TopFields,
    load_input_document,
    solver_inputs,
)
from aosr.physics.room_paths import (
    SUPPORTED_MAX_ORDER,
    SUPPORTED_MIN_ORDER,
    image_source_paths,
)
from tests.engine._precision_contracts import contract_value
from tests.engine.test_report_io_contract import (
    _TABLE_PATH,
    _fake_fem_energy,
    _input_document,
    _table,
)


_ROOM = Room(Lx=6.0, Ly=4.0, Lz=3.0)
_SOURCE = Point(x=1.2, y=1.3, z=1.1)
_RECEIVER = Point(x=4.7, y=2.8, z=1.4)
# 同一間房、把座標挪幾公分，用來跟會精確命中交線的整齊座標並列。
_SHIFTED_SOURCE = Point(x=1.17, y=1.31, z=1.13)
_SHIFTED_RECEIVER = Point(x=4.73, y=2.79, z=1.37)
_SOUND_SPEED_M_S = 343.0
_RHO_C_PA_S_PER_M = 411.6
_FREQUENCIES_HZ = (63.0, 125.0, 500.0, 2000.0)
_TOLERANCE_REL = contract_value("direct_energy_vs_legacy")


def _walls(impedance_multiple: float) -> dict[str, complex]:
    return {
        wall: complex(impedance_multiple * _RHO_C_PA_S_PER_M, 0.0)
        for wall in Wall.wall_names()
    }


def _scattering(value: float = 0.2) -> dict[str, float]:
    return {wall: value for wall in Wall.wall_names()}


def _lane(reflection_order_k: int) -> GeometricLaneResult:
    """細軸幾何路，明著給 K。"""
    return solve_geometric_lane(
        source_model=SourceModelSpec(kind=SourceModelKind.OMNIDIRECTIONAL),
        room=_ROOM,
        source=_SOURCE,
        receiver=_RECEIVER,
        sound_speed_m_s=_SOUND_SPEED_M_S,
        rho_c_pa_s_per_m=_RHO_C_PA_S_PER_M,
        frequencies_hz=_FREQUENCIES_HZ,
        impedance_by_wall=_walls(4.0),
        scattering_by_wall=_scattering(),
        reflection_order_k=reflection_order_k,
    )


def _lane_without_k() -> GeometricLaneResult:
    """同一組輸入，但**整格不給** K——除了那一格，跟 :func:`_lane` 一字不差。"""
    return solve_geometric_lane(
        source_model=SourceModelSpec(kind=SourceModelKind.OMNIDIRECTIONAL),
        room=_ROOM,
        source=_SOURCE,
        receiver=_RECEIVER,
        sound_speed_m_s=_SOUND_SPEED_M_S,
        rho_c_pa_s_per_m=_RHO_C_PA_S_PER_M,
        frequencies_hz=_FREQUENCIES_HZ,
        impedance_by_wall=_walls(4.0),
        scattering_by_wall=_scattering(),
    )


# ── ① 不同的 K 真的算出不同的東西 ─────────────────────────────────────────────
def test_a_lower_k_changes_the_reflected_column() -> None:
    """K=1 與 K=3 的反射欄必須不同：參數被收下卻照舊用常數的話這一題紅。

    一階鏡面是六面牆各一次、三階多出上百條路徑，兩者的同調和不可能逐位相同。
    """
    one = _lane(SUPPORTED_MIN_ORDER)
    three = _lane(REFLECTION_ORDER_K)

    assert one.reflection_order_k == SUPPORTED_MIN_ORDER
    assert three.reflection_order_k == REFLECTION_ORDER_K
    assert one.reflected_energy != three.reflected_energy
    assert one.late_energy != three.late_energy
    # 直達那一欄跟階數無關：它是同一條直達路徑，換 K 不准動到它。
    assert one.direct_energy == three.direct_energy


def test_the_early_lane_also_follows_the_given_k() -> None:
    """早期那一支（密軸走的那條）也要跟著 K 走，不是只有細軸那一支。"""
    early_one = solve_geometric_early_lane(
        source_model=SourceModelSpec(kind=SourceModelKind.OMNIDIRECTIONAL),
        room=_ROOM,
        source=_SOURCE,
        receiver=_RECEIVER,
        sound_speed_m_s=_SOUND_SPEED_M_S,
        rho_c_pa_s_per_m=_RHO_C_PA_S_PER_M,
        frequencies_hz=_FREQUENCIES_HZ,
        impedance_by_wall=_walls(4.0),
        reflection_order_k=SUPPORTED_MIN_ORDER,
    )
    early_default = solve_geometric_early_lane(
        source_model=SourceModelSpec(kind=SourceModelKind.OMNIDIRECTIONAL),
        room=_ROOM,
        source=_SOURCE,
        receiver=_RECEIVER,
        sound_speed_m_s=_SOUND_SPEED_M_S,
        rho_c_pa_s_per_m=_RHO_C_PA_S_PER_M,
        frequencies_hz=_FREQUENCIES_HZ,
        impedance_by_wall=_walls(4.0),
    )

    assert early_one.reflection_order_k == SUPPORTED_MIN_ORDER
    assert early_default.reflection_order_k == REFLECTION_ORDER_K
    assert early_one.reflected_energy != early_default.reflected_energy


# ── ② 不給 K 就是產品設定那一個（預設不變）────────────────────────────────────
def test_leaving_k_out_is_bit_for_bit_the_product_default() -> None:
    """沒給 K 的那一跑要逐位等於明著給 ``REFLECTION_ORDER_K``——預設不准悄悄改。"""
    assert _lane_without_k() == _lane(REFLECTION_ORDER_K)


# ── ③ 出界要報錯（界線現算，不寫死）───────────────────────────────────────────
@pytest.mark.parametrize(
    "bad_k",
    (SUPPORTED_MIN_ORDER - 1, SUPPORTED_MAX_ORDER + 1),
    ids=("below", "above"),
)
def test_k_outside_the_supported_range_is_refused_by_the_lane(bad_k: int) -> None:
    """幾何路收到出界的 K 要丟 ValueError（界線住 room_paths，這一層不抄第二份）。"""
    with pytest.raises(ValueError, match="max_order"):
        _lane(bad_k)


@pytest.mark.parametrize(
    "bad_k",
    (SUPPORTED_MIN_ORDER - 1, SUPPORTED_MAX_ORDER + 1),
    ids=("below", "above"),
)
def test_k_outside_the_supported_range_is_refused_by_the_input_model(bad_k: int) -> None:
    """輸入檔那一格也要擋在同一條界線上，而且訊息指得出是哪一欄。"""
    with pytest.raises(ValueError, match="reflection_order_k"):
        load_input_document(_input_document(reflection_order_k=bad_k), _table())


def test_the_two_axes_and_the_report_must_agree_on_k() -> None:
    """頻帶平均收到兩份 K 不同的結果要當場報錯，不准平均出一份誰的 K 都不是的報表。"""
    fine = _lane(REFLECTION_ORDER_K)
    mismatched = GeometricEarlyResult(
        source_model=SourceModelSpec(kind=SourceModelKind.OMNIDIRECTIONAL),
        frequencies_hz=fine.frequencies_hz,
        direct_energy=fine.direct_energy,
        reflected_energy=fine.reflected_energy,
        interference_energy=fine.interference_energy,
        scattering=fine.scattering,
        reflection_order_k=REFLECTION_ORDER_K + 1,
    )

    with pytest.raises(ValueError, match="同一個 K"):
        average_geometric_lane_to_bands_with_dense_early(
            fine,
            mismatched,
            reflection_order_k=REFLECTION_ORDER_K,
        )


# ── ④ 輸入檔那一格一路走到報表 ────────────────────────────────────────────────
def test_the_input_field_defaults_to_the_product_setting() -> None:
    """輸入檔不給那一格時，求解層收到的就是產品設定那一個。"""
    inputs = load_input_document(_input_document(), _table())

    assert inputs.reflection_order_k == REFLECTION_ORDER_K
    assert solver_inputs(inputs).reflection_order_k == REFLECTION_ORDER_K


@pytest.mark.parametrize("given_k", (1, 2), ids=("k1", "k2"))
def test_the_report_prints_the_k_the_input_file_asked_for(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    given_k: int,
) -> None:
    """輸入檔給了 K，報表頂層印出來的就要是那一個（有限元素照既有考卷換成假的）。

    走的是真的命令列：輸入模型、求解層、輸出契約三段都在裡面，只有昂貴的有限元素那一半
    是假的。K 挑 1 與 2 是因為它們都不等於預設的 3——印出 3 就代表那一格被丟掉了。
    """
    from aosr.physics import three_lane_report, three_lane_report_cli

    input_path = tmp_path / "room.json"
    input_path.write_text(
        json.dumps(_input_document(reflection_order_k=given_k)), encoding="utf-8"
    )
    monkeypatch.setattr(three_lane_report, "_solve_fem_energy", _fake_fem_energy)

    exit_code = three_lane_report_cli.main(
        [str(input_path), "--format", "json", "--capabilities", str(_TABLE_PATH)]
    )
    output = capsys.readouterr().out
    assert exit_code == 0, output

    parsed = ReportOutput.model_validate_json(output)
    assert parsed.top.reflection_order_k == given_k
    # ⑤ 報表在 K≠3 時照樣守得住四欄相加與總能量非負（這兩條不是只有預設 K 成立）。
    for band in parsed.bands:
        four_columns = (
            band.direct_energy
            + band.reflected_energy
            + band.interference_energy
            + band.late_energy
        )
        assert math.isclose(
            band.geometric_energy,
            four_columns,
            rel_tol=_TOLERANCE_REL,
            abs_tol=_TOLERANCE_REL * abs(four_columns),
        )
        assert band.total_energy >= 0.0


# ── ⑤ 幾何路本人在 K≠3 時的四欄與非負 ─────────────────────────────────────────
@pytest.mark.parametrize("given_k", (1, 2, 5), ids=("k1", "k2", "k5"))
def test_four_columns_add_up_and_stay_nonnegative_at_other_k(given_k: int) -> None:
    """四欄相加等於幾何能量、總能量非負這兩條在 K≠3 時也要成立。

    第一項是模平方、其餘兩項非負，所以總能量恆不為負；這條性質跟 K 無關，
    但「跟 K 無關」要真的換 K 跑過才算數。
    """
    actual = _lane(given_k)

    assert actual.reflection_order_k == given_k
    for index in range(len(_FREQUENCIES_HZ)):
        assert actual.geometric_energy[index] == (
            actual.direct_energy[index]
            + actual.reflected_energy[index]
            + actual.interference_energy[index]
            + actual.late_energy[index]
        )
        assert actual.geometric_energy[index] >= 0.0
        assert actual.late_energy[index] >= 0.0
        assert actual.reflected_energy[index] >= 0.0


# ── ⑥ 交線反射不再限制合法階數（票 #305）──────────────────────────────────────
@pytest.mark.parametrize(
    ("source", "receiver"),
    ((_SOURCE, _RECEIVER), (_SHIFTED_SOURCE, _SHIFTED_RECEIVER)),
)
def test_high_orders_compute_for_symmetric_and_shifted_positions(
    source: Point, receiver: Point
) -> None:
    """同一間房、同一個合法上限：整齊座標與挪開座標都跑得完。"""
    paths = image_source_paths(
        _ROOM,
        source,
        receiver,
        _SOUND_SPEED_M_S,
        max_order=SUPPORTED_MAX_ORDER,
    )
    assert paths
    assert all(
        sum(len(bounce.walls) for bounce in path.bounces) == path.order
        for path in paths
    )

    assert max(path.order for path in paths) == SUPPORTED_MAX_ORDER
    assert len(paths) > len(
        image_source_paths(
            _ROOM,
            _SHIFTED_SOURCE,
            _SHIFTED_RECEIVER,
            _SOUND_SPEED_M_S,
            max_order=REFLECTION_ORDER_K,
        )
    )


def _declared_order_bounds(model: type[BaseModel]) -> tuple[int, int]:
    """模型那一格現在宣告的（下界, 上界），直接問欄位、不看散文。"""
    metadata = model.model_fields["reflection_order_k"].metadata
    lower = next((item.ge for item in metadata if hasattr(item, "ge")), None)
    upper = next((item.le for item in metadata if hasattr(item, "le")), None)
    assert lower is not None, f"{model.__name__} 的 K 沒有宣告下界"
    assert upper is not None, f"{model.__name__} 的 K 沒有宣告上界"
    return int(lower), int(upper)


def test_both_report_models_take_the_order_bounds_from_one_place() -> None:
    """輸入契約與輸出契約的 K 界線必須是同一份，不准各抄一份字面量。

    抄成字面量的話，``room_paths`` 那個上限哪天動了，輸入會跟、輸出不會——輸出契約就會
    收下一個輸入契約已經拒收的 K，而兩邊都還是綠的。
    """
    for model in (ReportInput, TopFields):
        assert _declared_order_bounds(model) == (
            SUPPORTED_MIN_ORDER,
            SUPPORTED_MAX_ORDER,
        )
