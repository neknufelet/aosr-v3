"""家具能力範圍取自決策紙第 2、10、11、16、20、22、27 條，不宣稱實物精度已驗證。"""
from importlib.util import find_spec

from aosr.config.capabilities import load_capabilities
from aosr.config.paths import config_path
from aosr.physics.report_io import quantity_table
from aosr.physics.three_lane_report_cli import FURNITURE_CLI_UNSUPPORTED


def test_furniture_entry_declares_experimental_boxes_and_unsupported_angles() -> None:
    entry = load_capabilities(config_path("capabilities.toml")).for_entry("furniture_reflections")
    assert entry.module == "aosr.physics.furniture_paths" and find_spec(entry.module) is not None
    rows = {row.materials: row for row in entry.capability}
    assert set(rows) == {"axis_aligned_boxes_estimated_absorption", "tilted_or_non_right_angle_boxes"}
    assert all(row.room == "shoebox" and row.frequency_hz == (20.0, 11166.0) for row in rows.values())
    assert all(not row.evidence for row in rows.values())
    boxes = rows["axis_aligned_boxes_estimated_absorption"]
    assert boxes.status == "experimental"
    assert rows["tilted_or_non_right_angle_boxes"].status == "unsupported"
    assert rows["tilted_or_non_right_angle_boxes"].outputs == ("furniture.yaw_deg",)
    # 第 2 條原句；說反（例如「任意角度照算」）就紅。
    assert rows["tilted_or_non_right_angle_boxes"].note == "不支援的方向（傾斜、非直角的轉角）明確拒收，不忽略角度。"
    # 宣告本身就是答案：手寫整組，少宣告或多宣告一個都紅；每一個都要是報表真的有的鍵。
    assert set(boxes.outputs) == {
        "path_table.furniture_ids", "path_table.furniture_model", "path_table.blocked_wall_paths",
        "path_table.furniture_materials", "path_table.furniture_materials.furniture_id",
        "path_table.furniture_materials.material", "path_table.furniture_materials.unknown_bands_hz",
        "path_table.rows.furniture_id", "path_table.rows.furniture_face", "path_table.rows.reflection_point_m"}
    assert set(boxes.outputs) <= quantity_table().keys()
    assert {"path_table.furniture_model", "path_table.rows.furniture_id", "path_table.rows.furniture_face",
            "path_table.rows.reflection_point_m", "path_table.blocked_wall_paths"} <= set(boxes.outputs)
    assert all("energy" not in output for output in boxes.outputs)
    assert not entry.not_modeled and not entry.manual_checks


def test_furniture_notes_keep_decision_scope_and_limits() -> None:
    entry = load_capabilities(config_path("capabilities.toml")).for_entry("furniture_reflections")
    for phrase in (
        "這一節借 materials 欄記家具形狀與材質來源，不是牆面材料", "正放的盒子", "水平天雲",
        "不把桌下或板下空間填實", "家具只算一次反射", "不佔反射階數 K", "混合反射未納入",
        "透射不算", "有限尺寸修正", "尺寸遠小於距離", "沒有相位與邊緣繞射路徑", "只收矩形面",
        "估計，非本件實測", "五面或六面同值是資料限制", "63 與 8000 Hz 都未知", "玻璃的 125 Hz",
        "未知（計算時用相鄰頻帶延伸代算）", "暫緩完整繞射路徑", "遮擋邊界上的反射會突然出現或消失",
        "家具可以不對稱", "顫動警戒第一版只看三對牆", "家具形成的平行面", "未評估",
    ):
        assert phrase in entry.note
    for phrase in ("家具路徑不乘整房合成散射的 (1−s) 那一項", "天雲材質第一版開兩種", "吸音天雲",
                   "數值借用木質桌面那組估計值，懸空的板直接借用是近似"):
        assert phrase in entry.note
    boxes = next(row for row in entry.capability if row.status == "experimental")
    assert "能力表新條目在有獨立真值前最多標試驗中" in boxes.note
    assert "考卷只證明公式實作一致，不寫成實際家具精度已驗證" in boxes.note
    # 不准說過頭：除了那一句「不寫成……已驗證」本身，哪裡都不准出現「已驗證」「實測驗證」。
    for text in (entry.note, boxes.note.replace("不寫成實際家具精度已驗證", "")):
        assert "已驗證" not in text and "實測驗證" not in text


def test_related_notes_explain_downward_directivity_modal_scope_and_cli_entry() -> None:
    table = load_capabilities(config_path("capabilities.toml"))
    analytic = next(row for row in table.for_entry("source_directivity").capability
                    if row.materials == "analytic_axisymmetric_two_parameter_v1")
    assert "上下方向沿用同一條曲線、尚未獨立驗證，桌面反射強度靠這個假設" in analytic.note
    assert "桌面、沙發、天雲都不進有限元素網格，低頻模態未含家具" in table.for_entry("modal_diagnosis").note
    assert FURNITURE_CLI_UNSUPPORTED in table.for_entry("three_lane_report").note
