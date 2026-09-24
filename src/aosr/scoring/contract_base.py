"""評估器契約的共用底座：凍結設定、量值狀態、受控標記與原因代碼（票 #351 從 ``contract.py`` 原文搬出）。

``contract.py`` 已經貼著單檔行數上限，新的類別 payload 要住自己的模組；那些模組要用這裡的底座，
而 ``contract.py`` 又要把它們收進 ``CategoryPayload`` 聯集——底座留在 ``contract.py`` 就會繞成一圈。
這一支只准往下拿，不准回頭拿 ``contract.py``；``contract.py`` 用原名再匯出，既有呼叫端不用改匯入。
"""
from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


FROZEN = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
FrequencyRange = tuple[Annotated[float, Field(gt=0.0)], Annotated[float, Field(gt=0.0)]]


class MetricState(StrEnum):
    """單一量值狀態；unavailable 含缺值或壞值，原因碼分流，並與衍生量不可計算分開。"""

    MEASURED = "measured"
    UNAVAILABLE = "unavailable"
    NOT_COMPUTABLE = "not_computable"


class Flag(StrEnum):
    """跨評估器傳遞、但不直接等於不可估原因的受控標記。"""

    CROSSOVER_BAND = "crossover_band"
    UNVALIDATED = "unvalidated"
    NO_DIRECTIVITY = "no_directivity"
    DATA_COVERAGE_SHORT = "data_coverage_short"
    FEATURE_TOO_NARROW = "feature_too_narrow"
    FEATURE_BOUNDARY_INCOMPLETE = "feature_boundary_incomplete"
    FEATURE_NARROWER_THAN_AXIS = "feature_narrower_than_axis"
    BASELINE_SETTINGS = "baseline_settings"
    PARTIAL_FREQUENCY_OVERLAP = "partial_frequency_overlap"
    LISTENING_AREA_PEER_GROUP_MISSING = "listening_area_peer_group_missing"
    WINDOW_ONLY_DELAY_SCREEN = "window_only_delay_screen"
    GEOMETRY_MATERIAL_CONSERVATIVE_SCREEN = "geometry_material_conservative_screen"
    REFLECTION_FRONT_ABOVE_THRESHOLD = "reflection_front_above_threshold"
    REFLECTION_LATERAL_ABOVE_THRESHOLD = "reflection_lateral_above_threshold"
    REFLECTION_REAR_ABOVE_THRESHOLD = "reflection_rear_above_threshold"
    REFLECTION_VERTICAL_ABOVE_THRESHOLD = "reflection_vertical_above_threshold"


class ReasonCode(StrEnum):
    """不可估原因，以及 payload 局部沒有可量配對時的受控原因代碼。"""

    INSUFFICIENT_COVERAGE = "insufficient_coverage"
    # 音色的計分依賴範圍未完整覆蓋，或範圍內有缺段。
    TIMBRE_SCORING_RANGE_GAP = "timbre_scoring_range_gap"
    MISSING_POINTS = "missing_points"
    NON_POSITIVE_ENERGY = "non_positive_energy"
    SOLVER_UNAVAILABLE = "solver_unavailable"
    EVALUATOR_NOT_IMPLEMENTED = "evaluator_not_implemented"
    CANDIDATE_ID_MISMATCH = "candidate_id_mismatch"
    SPEAKER_ID_MISMATCH = "speaker_id_mismatch"
    RECEIVER_SET_FINGERPRINT_MISMATCH = "receiver_set_fingerprint_mismatch"
    EVALUATOR_VERSION_MISMATCH = "evaluator_version_mismatch"
    SCENE_FINGERPRINT_MISMATCH = "scene_fingerprint_mismatch"
    PLACEMENT_MISMATCH = "placement_mismatch"
    SETTINGS_FINGERPRINT_MISMATCH = "settings_fingerprint_mismatch"
    TIMBRE_SETTINGS_FINGERPRINT_MISMATCH = "timbre_settings_fingerprint_mismatch"
    LISTENING_AREA_SETTINGS_FINGERPRINT_MISMATCH = (
        "listening_area_settings_fingerprint_mismatch"
    )
    CHANNEL_GROUP_FINGERPRINT_MISMATCH = "channel_group_fingerprint_mismatch"
    CHANNEL_RESULT_UNAVAILABLE = "channel_result_unavailable"
    REQUIRED_CHANNEL_POINT_UNAVAILABLE = "required_channel_point_unavailable"
    CHANNEL_ROLE_MISMATCH = "channel_role_mismatch"
    FREQUENCY_AXIS_MISMATCH = "frequency_axis_mismatch"
    INVALID_DIRECT_DISTANCE = "invalid_direct_distance"
    RECEIVER_ID_MISMATCH = "receiver_id_mismatch"
    TIMBRE_NOT_MEASURED = "timbre_not_measured"
    ZERO_TOTAL_IMPORTANCE = "zero_total_importance"
    NO_SURROUNDING_PAIRS = "no_surrounding_pairs"
    INSUFFICIENT_DECAY_RANGE = "insufficient_decay_range"
    BAND_ROW_MISSING = "band_row_missing"
    NON_POSITIVE_VALUE = "non_positive_value"
    OTHER_ERROR = "other_error"
    PATH_TABLE_MISSING = "path_table_missing"
    REFLECTION_SCREEN_OR_WINDOW_MISSING = "reflection_screen_or_window_missing"
    REFLECTION_SCREEN_OR_WINDOW_MISMATCH = "reflection_screen_or_window_mismatch"
    REFLECTION_WINDOW_INCOMPLETE = "reflection_window_incomplete"
    LISTENING_AXIS_UNDEFINED = "listening_axis_undefined"
    NO_REFLECTION_IN_ZONE_POINT = "no_reflection_in_zone_point"
    ZERO_REFLECTION_ENERGY = "zero_reflection_energy"
    ZERO_RETENTION = "zero_retention"
    FULL_REFLECTION = "full_reflection"
    T20_BAND_UNAVAILABLE = "t20_band_unavailable"


class FrozenModel(BaseModel):
    """共用凍結、拒收多餘欄位與非有限數的模型底座。"""

    model_config = FROZEN


_FrozenModel = FrozenModel


class InputProvenance(_FrozenModel):
    """評估器吃到哪份報表與哪個聲源／接收點；排名層只轉不造。"""

    report_id: str = Field(min_length=1)
    engine_commit: str = Field(min_length=1)
    speaker_id: str = Field(min_length=1)
    receiver_id: str = Field(min_length=1)
