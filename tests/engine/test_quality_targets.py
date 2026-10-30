"""品質登記簿的載入、拒收與設定指紋考卷（票 #356）。"""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from aosr.config.frequency_axis import FEM_GEOMETRIC_CROSSOVER_CAP_HZ
from aosr.config.paths import config_path
from aosr.config.quality_targets import (
    QualityTargets, QualityPurpose,
    SettingEntry,
    TargetEntry,
    WeightTable,
    WeightItem,
    load_quality_targets,
)


_REGISTRY = config_path("quality_targets.toml")
_SOURCE = "測試基線，未查證；正式值等 #358"
_RIPPLE_SOURCE = (
    f"{_SOURCE}；上限 8000 由票 #345/#347 拍板；下限 20 Hz：票 #668 老闆 2026-10-20「音色改 傾斜不改」，"
    "對齊 #347 老闆 2026-09-18 拍的音色評估範圍 20～8,000 Hz"
)
_LISTENING_AREA_SOURCE = (
    f"{_SOURCE}；in_range_best 的代價 1 落在容許帶外再加上 "
    "worse_reference，不是落在 worse_reference"
)
_TIMBRE_ALERT_SOURCE = (
    "暫定複核警戒、不是淘汰線（票 #444 老闆 2026-09-22 拍 C）；峰：Toole & Olive 1988 "
    "Q=10 剛好聽得出約 ±3 dB、Sullivan 等 1986 單一峰的峰谷比偵測門檻約 6 dB（量的是峰谷比"
    "不是對擬合線的峰高，只當量級參考）；谷：Bücklein 1962 谷比峰難聽出、Peacock 等 1995 "
    "小房間尖銳的谷到 20 dB 耳朵能編輯掉、Olive 等 1997 脈衝＋高 Q 時谷跟峰一樣聽得出；"
    "房間量級 Toole 1982 低頻高低點差 20–30 dB、Kyriakakis 等 1998 位置間 ±15 dB 都是位置差、"
    "推不出單谷門檻"
)
_REFLECTION_BASELINE_SOURCE = "票 #351 老闆 2026-09-24 拍板；目前作為產品基線"
_REFLECTION_COST_SOURCE = "票 #351 老闆看過佔位表（2026-09-24）、這一格沒改；查無文獻，是產品基線"
_REFLECTION_WEIGHT_SOURCE = "票 #351 老闆 2026-09-24 選四區平均（A）；等權是主對話照「平均」記的基線，查無文獻"
_REFLECTION_FLUTTER_BANDS_SOURCE = (
    "票 #351 老闆 2026-09-24 拍板：顫動警戒採標稱 400 Hz–10 kHz 共 15 個 1/3 八度帶；"
    "是範圍政策、沒有外部出處，所以登記成基線"
)
_REFLECTION_FLUTTER_SOURCE = (
    "票 #351：由「跟本房同一帶殘響比」推出的一致性要求（報表 t20_s 是外推到 60 dB 的時間），"
    "不是老闆拍的數字；目前作為產品基線"
)
# 第一階段（#558）那一段的三句出處：老闆拍的是產品選擇、沒有外部出處，所以只准是基線。
_STAGE_ONE_RANGE_SOURCE = (
    "票 #558 老闆 2026-10-19 回低音分數第 2 題（重問）「1」＝選項 1：20～300 Hz，跟有限元素算的範圍一樣；是產品選擇、沒有外部出處，所以登記成基線"
)
_STAGE_ONE_REFERENCE_SOURCE = (
    "票 #558 老闆 2026-10-20 回第一階段第 2 題（權重）：「實作要核對參考尺度：兩項各增加 1 dB，對代價的增量應相同。不能權重填 1:1，卻因各自除以不同尺度而變成另一種比例。」較差參考只為無單位化，兩項取同一個值、合分程式不相等就報錯；代價數字本身沒有意義，顯示分項一律用原始 dB"
)
_STAGE_ONE_WEIGHT_SOURCE = (
    "票 #558 老闆 2026-10-20 回第一階段第 2 題（權重）：「先採原始兩項指標的有效權重 1:1，標為 baseline、product_choice。這是暫定的工程取捨，不是兩項在聽感上等價的保證。」每次搜尋固定、不隨當批候選分布自動改；第一批後檢視取捨，看過分布也不升 calibrated"
)
# 第三階段梯度目標專用三格（stage_three_gradient.*，#561 壬一）：溫度一句、目標與權重同一句；從登記簿逐字抄。
_STAGE_THREE_PROXY_TEMPERATURE_SOURCE = (
    '主對話 2026-10-30 暫定：軟最大值高估真大峰最多 T·ln N（N＝兩支喇叭的計分範圍內部點數，示範房 188 點軸約 2×186、ln 約 5.9），T＝0.25 dB 時最多約 1.5 dB；T 越小越貼近真大峰、目標越接近不平滑，第三場跑完再看要不要調'
)
_STAGE_THREE_PROXY_SOURCE = (
    '老闆 2026-10-30（對話，#562 下一塊那一題）：「選 B，是因為它是最小、能直接驗證目前問題的實驗。」「0.25 只能當初始試驗值。尤其第二場相對原牆約改善 0.494 分、最大峰指標卻惡化約 6.09 dB；如果代理剛好等於這個峰值，0.25 對這段變化的代價就是約 1.52 分，足以逆轉排名。所以第三場一定要同時列舊分數與新分數，避免只因加了權重、排名改變，就誤以為聲學真的改善。」較差參考取 1.0、跟兩個分項同一把尺，1 dB 的代理乘權重就是代價增量'
)
# 還掛著這幾句出處的條目只准是基線（沒查證過、或不是老闆拍的數字）。
_BASELINE_SOURCES = frozenset({
    '票 #558 老闆 2026-10-20 回評價規格第二題「Ａ」：對照那一間＝前 5 間以外、大峰比前 5 間裡最低的還低 1 dB 以上的候選中合分最好的一間；1 dB 是主對話暫定（10-05 舊搜尋兩軸峰高差最大 0.73 dB），第一批跑完再看',
    '票 #558 老闆 2026-10-23 回「主位參與大」門檻：「好，先採 −10 dB，登記為暫定工程的「優先檢視門檻」，不是聽感門檻或聲學合格線。」'
    '量法沿用各喇叭對主位、相對同一喇叭到同一主位最強的共振換算 dB、左右取較大；拖尾超線且參與 ≥−10 dB 列優先組，其餘超線完整保留在後面；文獻沒有直接支持哪個數字，第一批跑完看名單再調',
    _SOURCE,
    _RIPPLE_SOURCE,
    _LISTENING_AREA_SOURCE,
    _TIMBRE_ALERT_SOURCE,
    _REFLECTION_BASELINE_SOURCE,
    _REFLECTION_COST_SOURCE,
    _REFLECTION_WEIGHT_SOURCE,
    _REFLECTION_FLUTTER_SOURCE,
    _REFLECTION_FLUTTER_BANDS_SOURCE,
    _STAGE_ONE_RANGE_SOURCE,
    _STAGE_ONE_REFERENCE_SOURCE,
    _STAGE_ONE_WEIGHT_SOURCE,
    _STAGE_THREE_PROXY_TEMPERATURE_SOURCE,
    _STAGE_THREE_PROXY_SOURCE,
})
# 准是正式數字的名冊：一條一個鍵（權重列寫成「表鍵.列名」）。老闆拍一題、帶一張決策紙，
# 才准往這裡加一行——機器分不出「合法升等」與「偷偷蓋章」，這份名冊就是那道摩擦。
_CALIBRATED_KEYS = frozenset(
    {
        "timbre_balance.target_tilt_db_per_octave",
        "reflections_and_echo.window_upper_ms",
        "reflections_and_echo.frequency_range_hz",
        "reflections_and_echo.zone_threshold_db.front",
        "reflections_and_echo.zone_threshold_db.lateral",
        "reflections_and_echo.zone_threshold_db.rear",
        "reflections_and_echo.zone_threshold_db.vertical",
        "reverberation.target_t20_nominal_s_by_band",
        "reverberation.target_t20_tolerance_s_by_band",
        "low_frequency_decay.reference_anchor_frequencies_hz",
        "low_frequency_decay.reference_anchor_t60_s",
        "low_frequency_decay.reference_extension_range_hz",
        "low_frequency_decay.reference_extension_t60_s",
        "low_frequency_decay.strict_reference_frequencies_hz",
        "low_frequency_decay.strict_reference_t60_s",
    }
)


def _source(status: str = "baseline", source_kind: str = "product_choice") -> str:
    return (
        f'status = "{status}"\n'
        f'source_kind = "{source_kind}"\n'
        f'source = "{_SOURCE}"\n'
    )


def _setting(key: str, value: str = "[20.0, 8000.0]") -> str:
    return (
        "[[purpose.setting]]\n"
        f'key = "{key}"\n'
        f"value = {value}\n"
        'unit = "Hz"\n'
        + _source()
    )


def _target() -> str:
    return (
        "[[purpose.target]]\n"
        'key = "timbre_balance.target_tilt_db_per_octave"\n'
        "value = 0.0\n"
        'unit = "dB/oct"\n'
        'cost_shape = "in_range_best"\n'
        "tolerance = 0.5\n"
        "worse_reference = 2.0\n"
        + _source()
    )


def _weight() -> str:
    return (
        "[[purpose.weight]]\n"
        'key = "ranking.category_weights"\n'
        "[[purpose.weight.item]]\n"
        'name = "timbre_balance"\n'
        "value = 1.0\n"
        'note = "等權是試驗基線、不是中立"\n'
        + _source()
    )


def _qualification() -> str:
    return (
        "[[purpose.qualification]]\n"
        'key = "ranking.mandatory_categories"\n'
        'value = ["timbre_balance"]\n'
        + _source()
    )


def _structured_source() -> str:
    return (
        'source_id = "example-standard:2026"\n'
        'source_version = "2026"\n'
        'locator = "table 1"\n'
        'conditions = "dedicated listening room"\n'
        "frequency_range_hz = [20.0, 8000.0]\n"
        'context = "two-channel reproduction"\n'
        f'verification_digest = "sha256:{"a" * 64}"\n'
    )


def _document(*, reverse_settings: bool = False) -> str:
    settings: tuple[str, ...] = (
        _setting("timbre_balance.coverage_range_hz"),
        _setting("timbre_balance.tilt_fit_range_hz", "[80.0, 4000.0]"),
    )
    if reverse_settings:
        settings = tuple(reversed(settings))
    return (
        "schema_version = 1\n"
        "[[purpose]]\n"
        'name = "dedicated_two_channel_listening_room"\n'
        + "".join(settings)
        + _target()
        + _weight()
        + _qualification()
    )


def _load(path: Path) -> QualityTargets:
    return load_quality_targets(path)


def _write(path: Path, document: str) -> Path:
    path.write_text(document, encoding="utf-8")
    return path


def _assert_formal_sources(purpose: QualityPurpose) -> None:
    """逐列核出處；保留原檢查原文，新增工程基線不冒充文獻校準。"""
    structured_source_fields = (
        "source_id",
        "source_version",
        "locator",
        "conditions",
        "frequency_range_hz",
        "context",
        "verification_digest",
    )
    for entry in purpose.records:
        if isinstance(entry, WeightItem) and entry.name == "low_frequency_response":
            # #558：新增類權重是暫定工程選擇，沒有冒充文獻校準；只放行這一列並守住出處。
            assert entry.status == "baseline" and entry.source_kind == "engineering_recommendation"
            assert entry.note == "暫定工程選擇，不宣稱聽感校準"
            assert "2026-10-25" in entry.source and "#558" in entry.source
            continue
        if isinstance(entry, SettingEntry) and entry.key in {"timbre_balance.ripple_range_hz", "timbre_balance.feature_range_hz", "channel_matching.ripple_range_hz"}:
            # #558：產品選擇沒有外部文獻欄；新考卷核對登記出處，這裡仍擋偷偷升正式。
            assert entry.status == "baseline" and entry.source_kind == "product_choice"
            assert "2026-10-25" in entry.source and "#558" in entry.source
            continue
        if entry.source in _BASELINE_SOURCES:
            # 還掛著佔位那一句的條目不准被蓋成正式數字：沒查證過的數字蓋了章就查不回來。
            assert entry.status == "baseline", entry.source
            continue
        for field in structured_source_fields:
            value = getattr(entry, field)
            assert value is not None, field
            if isinstance(value, str):
                assert value.strip(), field


def test_formal_registry_loads_with_baseline_provenance() -> None:
    """佔位那一句不准被蓋成正式數字，拍完的那一條必須帶齊出處。"""
    registry = _load(_REGISTRY)
    purpose = registry.purpose("dedicated_two_channel_listening_room")
    keys = {entry.key for entry in purpose.entries}
    required = {
        "timbre_balance.coverage_range_hz",
        "timbre_balance.tilt_fit_range_hz",
        "timbre_balance.ripple_range_hz",
        "timbre_balance.feature_range_hz",
        "channel_matching.ripple_range_hz",
        "timbre_balance.smoothing_width_octave_tilt",
        "timbre_balance.smoothing_width_octave_ripple",
        "timbre_balance.feature_min_width_octave",
        "timbre_balance.min_points",
        "timbre_balance.target_tilt_db_per_octave",
        "timbre_balance.target_deviation_rms_db",
        "timbre_balance.residual_rms_db",
        "timbre_balance.peak_depth_db",
        "timbre_balance.dip_depth_db",
        "timbre_balance.within_category_weights",
        "ranking.category_weights",
        "ranking.mandatory_categories",
        "ranking.optional_categories",
        "ranking.eligibility.max_unavailable_bands",
        "ranking.eligibility.critical_bands",
        "ranking.eligibility.min_valid_bands",
    }

    assert required <= keys
    _assert_formal_sources(purpose)
    calibrated = {
        entry.key
        for entry in (*purpose.setting, *purpose.target, *purpose.qualification)
        if entry.status == "calibrated"
    } | {
        f"{table.key}.{item.name}"
        for table in purpose.weight
        for item in table.item
        if item.status == "calibrated"
    }
    assert calibrated == _CALIBRATED_KEYS

    coverage = purpose.entry("timbre_balance.coverage_range_hz")
    assert isinstance(coverage, SettingEntry)
    assert coverage.value == (20.0, 8000.0)


def test_stage_one_section_is_baseline_with_rostered_sources() -> None:
    """第一階段那一段不在用途裡，上一題的名冊走不到它：這裡逐列守同一把尺，而且整段沒有一格是正式數字。"""
    registry = _load(_REGISTRY)
    section = registry.stage_one("dedicated_two_channel_listening_room")
    assert {entry.key for entry in section.entries} >= {
        "stage_one_low_frequency.scored_range_hz",
        "stage_one_low_frequency.own_line_ripple_rms_db",
        "stage_one_low_frequency.seat_robustness_rms_db",
        "stage_one_low_frequency.component_weights",
    }
    for entry in section.records:
        assert entry.source in _BASELINE_SOURCES, entry.source
        assert entry.status == "baseline", entry.source
    assert {
        entry.key for entry in (*section.setting, *section.target, *section.qualification)
        if entry.status == "calibrated"
    } | {
        f"{table.key}.{item.name}" for table in section.weight for item in table.item
        if item.status == "calibrated"
    } <= _CALIBRATED_KEYS


def test_stage_one_section_must_name_a_known_purpose(tmp_path: Path) -> None:
    """第一階段那一段掛的用途不在登記簿就拒收：用途仍是房間的使用情境，不拿來裝階段。"""
    text = _REGISTRY.read_text(encoding="utf-8")
    head, marker, tail = text.partition("\n[[stage_one]]\nname = ")
    assert marker
    changed = head + marker + tail.replace('"dedicated_two_channel_listening_room"', '"stage_one_room_screening"', 1)

    with pytest.raises(ValidationError, match="第一階段那一段的用途不在登記簿"):
        _load(_write(tmp_path / "quality_targets.toml", changed))


def test_ripple_scoring_reaches_the_registered_diagnostic_ceiling() -> None:
    """起伏上緣若仍停在舊值，20～8000 Hz 的診斷範圍會有一段永遠不參與計分。"""
    purpose = _load(_REGISTRY).purpose("dedicated_two_channel_listening_room")
    coverage = purpose.entry("timbre_balance.coverage_range_hz")
    ripple = purpose.entry("timbre_balance.ripple_range_hz")

    assert isinstance(coverage, SettingEntry)
    assert isinstance(ripple, SettingEntry)
    coverage_value = coverage.value
    ripple_value = ripple.value
    assert isinstance(coverage_value, tuple)
    assert isinstance(ripple_value, tuple)
    assert ripple_value[1] == coverage_value[1]


@pytest.mark.parametrize("known_unit", ["Hz", "dB/oct"])
def test_unknown_unit_is_rejected_while_loading(tmp_path: Path, known_unit: str) -> None:
    """若 unit 仍是任意字串，拼成 decibel 的登記簿會被當成有效契約。

    ``Hz`` 第一次出現在量法設定、``dB/oct`` 第一次出現在品質目標，兩種條目各守一次。
    """
    original = f'unit = "{known_unit}"'
    assert original in _document()
    changed = _document().replace(original, 'unit = "decibel"', 1)

    with pytest.raises(ValidationError, match="decibel"):
        _load(_write(tmp_path / "quality_targets.toml", changed))


def test_angle_unit_loads_from_registry(tmp_path: Path) -> None:
    """方向分區的角度設定可以由登記簿載入。"""
    document = (
        "schema_version = 1\n"
        "[[purpose]]\n"
        'name = "dedicated_two_channel_listening_room"\n'
        "target = []\n"
        "weight = []\n"
        "qualification = []\n"
        "[[purpose.setting]]\n"
        'key = "reflection.direction_boundary"\n'
        "value = 30.0\n"
        'unit = "deg"\n'
        + _source()
    )
    loaded = _load(_write(tmp_path / "quality_targets.toml", document))
    entry = loaded.purpose("dedicated_two_channel_listening_room").entry(
        "reflection.direction_boundary"
    )
    assert isinstance(entry, SettingEntry)
    assert entry.unit == "deg"


def test_all_listening_area_stability_targets_and_weights_are_provisional() -> None:
    """聆聽區數字尚未校準：新增或改動任一條都只能留在 baseline 佔位狀態。"""
    purpose = _load(_REGISTRY).purpose("dedicated_two_channel_listening_room")
    prefix = "listening_area_stability."
    entries = tuple(entry for entry in purpose.entries if entry.key.startswith(prefix))
    target_keys = {entry.key for entry in entries if isinstance(entry, TargetEntry)}
    assert {
        "listening_area_stability.tilt_weighted_mean_deviation",
        "listening_area_stability.ripple_rms_weighted_mean_deviation",
        "listening_area_stability.overall_level_weighted_mean_deviation",
        "listening_area_stability.tilt_worst_deviation",
        "listening_area_stability.ripple_rms_worst_deviation",
        "listening_area_stability.overall_level_worst_deviation",
    } <= target_keys
    weights = purpose.entry("listening_area_stability.within_category_weights")
    assert isinstance(weights, WeightTable)
    records = tuple(
        record
        for entry in entries
        for record in (entry.item if isinstance(entry, WeightTable) else (entry,))
    )

    assert {record.status for record in records} == {"baseline"}
    assert {record.source_kind for record in records} == {"engineering_recommendation"}
    targets = tuple(record for record in records if isinstance(record, TargetEntry))
    assert {record.source for record in targets} == {_LISTENING_AREA_SOURCE}


def test_decided_tilt_entry_keeps_the_three_numbers_that_were_ruled_on() -> None:
    """傾斜那一條的三個數字是拍板過的（#368 目標值、#376 容許帶與較差參考）。

    改其中任何一個都要帶一張新的決策紙，所以這裡逐格釘住，順手改一個數會在這裡紅。
    這一條三個數字出處不同種，登記簿一條只有一格出處種類，照票上的做法整條標產品選擇、
    出處字串逐項寫明，所以這裡驗的種類是產品選擇，不是標準原文。
    """
    registry = _load(_REGISTRY)
    purpose = registry.purpose("dedicated_two_channel_listening_room")
    target_tilt = purpose.entry("timbre_balance.target_tilt_db_per_octave")

    assert isinstance(target_tilt, TargetEntry)
    assert target_tilt.value == 0.0
    assert target_tilt.tolerance == 1.0
    assert target_tilt.worse_reference == 2.0
    assert target_tilt.status == "calibrated"
    assert target_tilt.source_kind == "product_choice"
    assert target_tilt.verification_digest is not None
    assert target_tilt.verification_digest.startswith("sha256:")


def test_unknown_purpose_is_rejected(tmp_path: Path) -> None:
    """用途拼錯不准偷偷退回唯一一節，否則換用途會拿到錯的品質尺。"""
    registry = _load(_write(tmp_path / "targets.toml", _document()))

    with pytest.raises(KeyError, match="沒有這個用途"):
        registry.purpose("unknown_room")


def test_missing_field_is_rejected(tmp_path: Path) -> None:
    """拿掉單位時載入必須紅；沒有單位的數字不能交給第二層判讀。"""
    document = _document().replace('unit = "Hz"\n', "", 1)

    with pytest.raises(ValidationError, match="unit"):
        _load(_write(tmp_path / "targets.toml", document))


@pytest.mark.parametrize(
    ("old", "new", "message"),
    (
        ('source_kind = "product_choice"', 'source_kind = "guess"', "source_kind"),
        ('cost_shape = "in_range_best"', 'cost_shape = "nearest"', "cost_shape"),
        ('status = "baseline"', 'status = "draft"', "status"),
    ),
)
def test_unknown_controlled_value_is_rejected(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    """受控值放寬成任意字串時，拼錯的來源、代價型或狀態會溜進正式表。"""
    document = _document().replace(old, new, 1)

    with pytest.raises(ValidationError, match=message):
        _load(_write(tmp_path / "targets.toml", document))


def test_calibrated_entry_requires_structured_source(tmp_path: Path) -> None:
    """只把 baseline 改字成 calibrated、卻沒有穩定來源與查證雜湊時必須紅。"""
    document = _document().replace('status = "baseline"', 'status = "calibrated"', 1)

    with pytest.raises(ValidationError, match="calibrated.*source_id"):
        _load(_write(tmp_path / "targets.toml", document))


@pytest.mark.parametrize(
    "missing",
    (
        "source_id",
        "source_version",
        "locator",
        "conditions",
        "frequency_range_hz",
        "context",
        "source_kind",
        "verification_digest",
    ),
)
def test_calibrated_entry_rejects_each_missing_source_field(
    tmp_path: Path, missing: str
) -> None:
    """已校準出處的每一格都是收據的一部分，拿掉任一格都不能載入。"""
    structured = _structured_source()
    calibrated = _source(status="calibrated") + structured
    line = next(row for row in calibrated.splitlines(keepends=True) if row.startswith(missing))
    document = _document().replace(_source(), calibrated.replace(line, "", 1), 1)

    with pytest.raises(ValidationError, match=missing):
        _load(_write(tmp_path / "targets.toml", document))


def test_complete_calibrated_source_is_accepted(tmp_path: Path) -> None:
    """結構化出處齊全時要能升 calibrated，否則 #358 沒有合法出口。"""
    calibrated = _source(status="calibrated") + _structured_source()
    document = _document().replace(_source(), calibrated, 1)

    registry = _load(_write(tmp_path / "targets.toml", document))
    coverage = registry.purpose("dedicated_two_channel_listening_room").entry(
        "timbre_balance.coverage_range_hz"
    )

    assert isinstance(coverage, SettingEntry)
    assert coverage.status == "calibrated"


def test_calibrated_verification_digest_must_be_a_hash(tmp_path: Path) -> None:
    """任意非空字串不是查證表雜湊，不能讓條目冒充 calibrated。"""
    structured = _structured_source().replace("sha256:" + "a" * 64, "not-a-hash")
    calibrated = _source(status="calibrated") + structured
    document = _document().replace(_source(), calibrated, 1)

    with pytest.raises(ValidationError, match="verification_digest"):
        _load(_write(tmp_path / "targets.toml", document))


def test_duplicate_key_within_purpose_is_rejected(tmp_path: Path) -> None:
    """同用途同鍵若出現兩次，呼叫端就無法知道該拿哪一個值。"""
    document = _document() + _setting("timbre_balance.coverage_range_hz")

    with pytest.raises(ValidationError, match="重複.*coverage_range_hz"):
        _load(_write(tmp_path / "targets.toml", document))


@pytest.mark.parametrize(
    ("old", "new", "message"),
    (
        (f'source = "{_SOURCE}"', 'source = "   "', "空白"),
        ("value = 0.0\n", "value = true\n", "value"),
        ("worse_reference = 2.0", "worse_reference = true", "worse_reference"),
        ("worse_reference = 2.0", "worse_reference = -2.0", "worse_reference"),
        ("tolerance = 0.5", "tolerance = -0.5", "tolerance"),
    ),
)
def test_blank_source_bool_value_and_negative_scale_are_rejected(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    """空白出處、被當成 1 的布林值、負的較差參考或容許帶，都不是一把能用的品質尺。"""
    document = _document().replace(old, new, 1)
    assert document != _document()

    with pytest.raises(ValidationError, match=message):
        _load(_write(tmp_path / "targets.toml", document))


def test_nan_is_rejected(tmp_path: Path) -> None:
    """TOML 能剖開 nan 不代表品質尺能收；非有限數會污染全部代價。"""
    document = _document().replace("value = 0.0", "value = nan", 1)

    with pytest.raises(ValidationError, match="finite_number"):
        _load(_write(tmp_path / "targets.toml", document))


def test_fingerprint_ignores_entry_and_field_order(tmp_path: Path) -> None:
    """同一內容只換 TOML 排列，不能被排名層誤判成不同設定。"""
    first = _load(_write(tmp_path / "first.toml", _document()))
    reordered = _document(reverse_settings=True).replace(
        'key = "timbre_balance.target_tilt_db_per_octave"\nvalue = 0.0',
        'value = 0.0\nkey = "timbre_balance.target_tilt_db_per_octave"',
    )
    second = _load(_write(tmp_path / "second.toml", reordered))

    assert first.fingerprint == second.fingerprint


def test_fingerprint_changes_with_a_value(tmp_path: Path) -> None:
    """任一登記值改動卻沿用舊指紋，會讓不同考卷的結果被放進同一張表。"""
    first = _load(_write(tmp_path / "first.toml", _document()))
    changed = _document().replace("value = 0.0", "value = 0.25", 1)
    second = _load(_write(tmp_path / "second.toml", changed))

    assert first.fingerprint != second.fingerprint


def test_fingerprint_is_lowercase_hexadecimal(tmp_path: Path) -> None:
    """指紋不是十六進位時，跨工具轉送會需要另一套編碼約定。"""
    registry = _load(_write(tmp_path / "targets.toml", _document()))

    assert registry.fingerprint == registry.fingerprint.lower()
    assert set(registry.fingerprint) <= set("0123456789abcdef")


def test_loaded_registry_is_frozen(tmp_path: Path) -> None:
    """載入後若能就地改值，物件帶的指紋就不再描述它自己。"""
    registry = _load(_write(tmp_path / "targets.toml", _document()))

    with pytest.raises(ValidationError, match="frozen"):
        setattr(registry, "schema_version", 2)


def test_reflection_registry_values_have_units_and_provenance() -> None:
    purpose = _load(_REGISTRY).purpose("dedicated_two_channel_listening_room")
    expected = {
        "reflections_and_echo.window_upper_ms": (15.0, "ms", "calibrated"),
        "reflections_and_echo.frequency_range_hz": ((300.0, 8000.0), "Hz", "calibrated"),
        "direction_zones.vertical_min_abs_elevation_deg": (30.0, "deg", "baseline"),
        "direction_zones.front_max_abs_azimuth_deg": (40.0, "deg", "baseline"),
        "direction_zones.rear_min_abs_azimuth_deg": (135.0, "deg", "baseline"),
        "reflections_and_echo.flutter_decay_db": (60.0, "dB", "baseline"),
        "reflections_and_echo.flutter_alert_band_centers_hz": (
            (400.0, 500.0, 630.0, 800.0, 1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0,
             4000.0, 5000.0, 6300.0, 8000.0, 10000.0), "Hz", "baseline",
        ),
    }
    expected.update({
        f"reflections_and_echo.zone_threshold_db.{zone}": (-10.0, "dB", "calibrated")
        for zone in ("front", "lateral", "rear", "vertical")
    })
    cost_target = purpose.entry("reflections_and_echo.zone_excess_db")
    assert isinstance(cost_target, TargetEntry)
    assert (cost_target.value, cost_target.unit, cost_target.cost_shape,
            cost_target.worse_reference, cost_target.status) == (
                0.0, "dB", "less_is_better", 10.0, "baseline")
    weights = purpose.entry("reflections_and_echo.within_category_weights")
    assert isinstance(weights, WeightTable)
    assert {item.name: (item.value, item.status) for item in weights.item} == {
        zone: (1.0, "baseline") for zone in ("front", "lateral", "rear", "vertical")}
    for key, (value, unit, status) in expected.items():
        entry = purpose.entry(key)
        assert isinstance(entry, SettingEntry)
        assert (entry.value, entry.unit, entry.status) == (value, unit, status)
    # 正式那幾條：出處照殘響那條的寫法（產品選擇、引用標準），查證摘要指到同一份查證表
    reverberation = purpose.entry("reverberation.target_t20_nominal_s_by_band")
    assert isinstance(reverberation, SettingEntry)
    for key, (_value, _unit, status) in expected.items():
        entry = purpose.entry(key)
        assert isinstance(entry, SettingEntry)
        if status != "calibrated":
            continue
        assert entry.source_kind == "product_choice", key
        assert (entry.source_id, entry.source_version) == ("ITU-R BS.1116-3", "02/2015"), key
        assert entry.locator is not None and entry.locator.startswith("§8.3.3.1"), key
        assert entry.frequency_range_hz == (1000.0, 8000.0), key
        assert entry.verification_digest == reverberation.verification_digest, key


def test_reflection_range_starts_where_the_finite_element_lane_ends() -> None:
    """老闆 2026-09-24：早期反射從 300 Hz 起，參考有限元素只算到 300 Hz；交接點改了這裡沒跟著改就紅。

    交接點是目前範圍政策的參考，不是幾何模型在 300 Hz 以上有效的證明（老闆同日原話）。
    """
    purpose = _load(_REGISTRY).purpose("dedicated_two_channel_listening_room")
    entry = purpose.entry("reflections_and_echo.frequency_range_hz")
    assert isinstance(entry, SettingEntry)
    assert isinstance(entry.value, tuple)
    assert entry.value[0] == FEM_GEOMETRIC_CROSSOVER_CAP_HZ
