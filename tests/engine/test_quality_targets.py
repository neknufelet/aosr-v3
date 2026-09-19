"""品質登記簿的載入、拒收與設定指紋考卷（票 #356）。"""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from aosr.config.paths import config_path
from aosr.config.quality_targets import (
    QualityTargets,
    SettingEntry,
    TargetEntry,
    load_quality_targets,
)


_REGISTRY = config_path("quality_targets.toml")
_SOURCE = "測試基線，未查證；正式值等 #358"


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


def test_formal_registry_loads_with_baseline_provenance() -> None:
    """正式表守住基線狀態，非佔位出處必須完整且平坦傾斜有標準來源。"""
    registry = _load(_REGISTRY)
    purpose = registry.purpose("dedicated_two_channel_listening_room")
    keys = {entry.key for entry in purpose.entries}
    required = {
        "timbre_balance.coverage_range_hz",
        "timbre_balance.tilt_fit_range_hz",
        "timbre_balance.ripple_range_hz",
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
    assert all(entry.status == "baseline" for entry in purpose.records)
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
        if entry.source == _SOURCE:
            continue
        for field in structured_source_fields:
            value = getattr(entry, field)
            assert value is not None, field
            if isinstance(value, str):
                assert value.strip(), field
    coverage = purpose.entry("timbre_balance.coverage_range_hz")
    target_tilt = purpose.entry("timbre_balance.target_tilt_db_per_octave")
    assert isinstance(coverage, SettingEntry)
    assert isinstance(target_tilt, TargetEntry)
    assert coverage.value == (20.0, 8000.0)
    assert target_tilt.value == 0.0
    assert target_tilt.source_kind == "standard"
    assert target_tilt.status == "baseline"
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
