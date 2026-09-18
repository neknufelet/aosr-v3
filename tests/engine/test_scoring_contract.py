"""評估器交給排名層的凍結共用契約考卷（票 #356）。"""
from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from aosr.scoring import contract as _CONTRACT
from aosr.scoring.contract import CategoryEvaluation


def _provenance() -> dict[str, str]:
    return {
        "report_id": "reference-room-flat",
        "engine_commit": "unknown",
        "speaker_id": "left",
        "receiver_id": "main-seat",
    }


def _timbre_payload() -> dict[str, object]:
    return {
        "category": "timbre_balance",
        "tilt_db_per_octave": -0.25,
        "tilt_fit_range_hz": [80.0, 4000.0],
        "target_tilt_db_per_octave": 0.0,
        "target_deviation_rms_db": 1.5,
        "deviation_curve": [[100.0, -1.0], [200.0, 1.0]],
        "residual_rms_db": 1.0,
        "ripple_range_hz": [40.0, 4000.0],
        "features": [
            {
                "kind": "dip",
                "center_frequency_hz": 100.0,
                "depth_db": 3.0,
                "width_octave": None,
                "flags": ["feature_boundary_incomplete"],
            }
        ],
        "strongest_peak_index": None,
        "deepest_dip_index": 0,
        "data_range_hz": [20.0, 8000.0],
        "coverage_range_hz": [20.0, 8000.0],
    }


def _evaluation(*, state: str = "costed") -> dict[str, object]:
    return {
        "schema_version": _CONTRACT.CONTRACT_SCHEMA_VERSION,
        "candidate_id": "candidate-a",
        "category": "timbre_balance",
        "state": state,
        "payload": _timbre_payload(),
        "raw_quantities": [{"name": "tilt", "value": -0.25, "unit": "dB/oct"}],
        "category_cost": (
            {
                "value": 0.25,
                "components": {"tilt": 0.1, "residual_rms": 0.15},
                "cost_settings_fingerprint": "f" * 64,
            }
            if state == "costed"
            else None
        ),
        "flags": ["baseline_settings"],
        "reason_codes": [],
        "evaluator_version": "timbre-v1",
        "settings_fingerprint": "a" * 64,
        "provenance": _provenance(),
    }


def _validate(document: dict[str, object]) -> CategoryEvaluation:
    return _CONTRACT.CategoryEvaluation.model_validate(document)


def test_legal_costed_evaluation_is_frozen() -> None:
    """合法樣本應能跨層傳遞，而且建立後不能換掉候選身分。"""
    evaluation = _validate(_evaluation())

    with pytest.raises(ValidationError, match="frozen"):
        evaluation.candidate_id = "candidate-b"


@pytest.mark.parametrize("output", ("category", "candidate"))
def test_invariant_1_schema_version_is_exact(output: str) -> None:
    """版本不合仍被接收時，排名層會用舊語意解讀新資料。"""
    document = _evaluation()
    document["schema_version"] = "future-version"

    with pytest.raises(ValidationError, match="schema_version"):
        if output == "category":
            _validate(document)
        else:
            _CONTRACT.CandidateEvaluation.model_validate(
                {
                    "schema_version": "future-version",
                    "candidate_id": "candidate-a",
                    "provenance": _provenance(),
                    "evaluations": [],
                }
            )


@pytest.mark.parametrize("mutation", ("unit", "payload"))
def test_invariant_2_quantities_are_typed_and_payload_matches_category(
    mutation: str,
) -> None:
    """原始量單位或 payload 類別放錯時，外層類別就不能再解釋裡面的數字。"""
    document = _evaluation()
    if mutation == "unit":
        document["raw_quantities"] = [{"name": "tilt", "value": 0.0, "unit": "watts"}]
    else:
        document["payload"] = {"category": "reverberation"}

    with pytest.raises(ValidationError):
        _validate(document)


@pytest.mark.parametrize("state", ("measured", "costed"))
def test_invariant_3_state_and_category_cost_agree(state: str) -> None:
    """可估的兩個狀態若不約束代價有無，排名層只看 nullable 就會猜錯。"""
    document = _evaluation(state=state)
    document["category_cost"] = {"value": 0.5, "components": {}, "cost_settings_fingerprint": "f"}
    if state == "costed":
        document["category_cost"] = None

    with pytest.raises(ValidationError, match="category_cost"):
        _validate(document)


@pytest.mark.parametrize("state", ("measured", "costed"))
@pytest.mark.parametrize("leak", ("reason_codes", "raw_quantities"))
def test_invariant_3_estimable_states_do_not_leak_unavailable_shape(state: str, leak: str) -> None:
    """可估卻帶不可估原因、或「已量」卻零原始量，三個狀態就互相滲透、排名層又得猜。"""
    document = _evaluation(state=state)
    if leak == "reason_codes":
        document["reason_codes"] = ["solver_unavailable"]
    else:
        document["raw_quantities"] = []

    with pytest.raises(ValidationError, match=leak):
        _validate(document)


@pytest.mark.parametrize("violation", ("reason", "cost", "payload"))
def test_invariant_4_unavailable_has_no_cost_and_has_reason(violation: str) -> None:
    """不可估若沒有原因、還帶代價、或帶一份捏造的數值 payload，零值會被誤當成好成績。"""
    document = _evaluation(state="unavailable")
    document["payload"] = None
    if violation == "reason":
        document["reason_codes"] = []
    elif violation == "payload":
        document["reason_codes"] = ["missing_points"]
        document["payload"] = _timbre_payload()
    else:
        document["reason_codes"] = ["missing_points"]
        document["category_cost"] = {
            "value": 0.5,
            "components": {},
            "cost_settings_fingerprint": "f",
        }

    with pytest.raises(ValidationError, match="reason_codes|category_cost|payload"):
        _validate(document)


def test_invariant_5_costed_value_is_finite() -> None:
    """可排名代價若收下無限值，排序結果不再是有限品質尺上的比較（由欄位層 allow_inf_nan=False 守）。"""
    document = _evaluation()
    assert isinstance(document["category_cost"], dict)
    document["category_cost"]["value"] = float("inf")

    with pytest.raises(ValidationError, match="finite_number"):
        _validate(document)


@pytest.mark.parametrize("field", ("evaluator_version", "settings_fingerprint"))
def test_invariant_6_version_and_fingerprint_are_nonempty(field: str) -> None:
    """評估器版本或設定指紋空白時，排名層無法判定結果能不能同表。"""
    document = _evaluation()
    document[field] = "   "

    with pytest.raises(ValidationError, match=field):
        _validate(document)


@pytest.mark.parametrize("field", ("flags", "reason_codes"))
def test_invariant_7_unknown_flag_or_reason_is_rejected(field: str) -> None:
    """未知標記或原因若靜靜通過，消費端會漏掉它不認識的判決語意。"""
    document = _evaluation()
    document[field] = ["invented_code"]

    with pytest.raises(ValidationError):
        _validate(document)


@pytest.mark.parametrize("where", ("raw", "payload", "component"))
def test_invariant_8_every_numeric_position_rejects_nan(where: str) -> None:
    """NaN 躲進原始量、類別 payload 或代價分項都會讓後續比較失真。"""
    document = _evaluation()
    if where == "raw":
        document["raw_quantities"] = [{"name": "tilt", "value": float("nan"), "unit": "dB/oct"}]
    elif where == "payload":
        assert isinstance(document["payload"], dict)
        document["payload"]["residual_rms_db"] = float("nan")
    else:
        assert isinstance(document["category_cost"], dict)
        document["category_cost"]["components"] = {"tilt": float("nan")}

    with pytest.raises(ValidationError, match="finite_number"):
        _validate(document)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("tilt_fit_range_hz", [4000.0, 80.0], "遞增"),
        ("coverage_range_hz", [8000.0, 8000.0], "遞增"),
        ("strongest_peak_index", 5, "超出"),
        ("strongest_peak_index", 0, "不是 peak"),
    ),
)
def test_timbre_payload_ranges_and_summary_indices_are_consistent(
    field: str, value: object, message: str
) -> None:
    """顛倒的頻率範圍或指到不存在／種類不符的摘要索引，會讓排名表印出捏造的摘要。"""
    document = _evaluation()
    assert isinstance(document["payload"], dict)
    document["payload"][field] = value

    with pytest.raises(ValidationError, match=message):
        _validate(document)


def test_feature_width_must_be_positive_when_known() -> None:
    """負的或零的特徵寬度不是「未知」，是壞值；未知只准用 None 表達。"""
    document = _evaluation()
    assert isinstance(document["payload"], dict)
    document["payload"]["features"][0]["width_octave"] = -1.0

    with pytest.raises(ValidationError, match="width_octave"):
        _validate(document)


@pytest.mark.parametrize("mismatch", ("candidate", "provenance"))
def test_invariant_9_candidate_envelope_identity_matches(mismatch: str) -> None:
    """包內候選或報表出身不同時，排名層不得把別份評估冒充同一候選。"""
    evaluation = _evaluation()
    envelope = {
        "schema_version": _CONTRACT.CONTRACT_SCHEMA_VERSION,
        "candidate_id": "candidate-a",
        "provenance": _provenance(),
        "evaluations": [evaluation],
    }
    if mismatch == "candidate":
        evaluation["candidate_id"] = "candidate-b"
    else:
        evaluation["provenance"] = {**_provenance(), "report_id": "another-report"}

    with pytest.raises(ValidationError, match=mismatch):
        _CONTRACT.CandidateEvaluation.model_validate(envelope)


def test_candidate_has_at_most_one_evaluation_per_category() -> None:
    """同一候選同類出現兩次時，排名層無法知道該用哪一條類代價。"""
    evaluation = _evaluation()
    envelope = {
        "schema_version": _CONTRACT.CONTRACT_SCHEMA_VERSION,
        "candidate_id": "candidate-a",
        "provenance": _provenance(),
        "evaluations": [evaluation, deepcopy(evaluation)],
    }

    with pytest.raises(ValidationError, match="重複.*category"):
        _CONTRACT.CandidateEvaluation.model_validate(envelope)


def test_quality_category_and_code_vocabularies_are_complete() -> None:
    """拿掉拍板類別、標記或原因時，後續評估器會被迫另造契約外字串。"""
    assert {item.value for item in _CONTRACT.QualityCategory} == {
        "timbre_balance",
        "listening_area_stability",
        "low_frequency_decay",
        "reflections_and_echo",
        "reverberation",
        "channel_matching",
        "spatial_impression",
    }
    assert {
        "crossover_band",
        "unvalidated",
        "no_directivity",
        "data_coverage_short",
        "feature_too_narrow",
        "feature_boundary_incomplete",
        "baseline_settings",
    } <= {item.value for item in _CONTRACT.Flag}
    assert {
        "insufficient_coverage",
        "missing_points",
        "non_positive_energy",
        "solver_unavailable",
        "evaluator_not_implemented",
    } <= {item.value for item in _CONTRACT.ReasonCode}


@pytest.mark.parametrize(
    "category",
    (
        "listening_area_stability",
        "low_frequency_decay",
        "reflections_and_echo",
        "reverberation",
        "channel_matching",
        "spatial_impression",
    ),
)
def test_future_category_payloads_are_discriminated_placeholders(category: str) -> None:
    """未實作的六類只收類別辨識欄，不假裝已有量法欄位。"""
    document = _evaluation(state="measured")
    document["category"] = category
    document["payload"] = {"category": category}

    evaluation = _validate(document)

    assert evaluation.payload is not None
    assert evaluation.payload.category == category


def test_unavailable_evaluation_is_explicit_and_legal() -> None:
    """缺資料應有明確 unavailable 與原因，不必捏造一份音色數值 payload。"""
    document = _evaluation(state="unavailable")
    document["payload"] = None
    document["reason_codes"] = ["missing_points"]

    evaluation = _validate(document)

    assert evaluation.state.value == "unavailable"
    assert evaluation.category_cost is None


def test_contract_models_export_json_schema() -> None:
    """拿掉可匯出的型別資訊時，#359 就無法從凍結模型產生正式 schema。"""
    category_schema = _CONTRACT.CategoryEvaluation.model_json_schema()
    candidate_schema = _CONTRACT.CandidateEvaluation.model_json_schema()

    assert category_schema["type"] == "object"
    assert candidate_schema["type"] == "object"
