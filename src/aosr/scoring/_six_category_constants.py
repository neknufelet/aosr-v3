"""六類組裝器的扁平常數；所有欄都能直接 np.asarray 並逐欄雜湊。

用閉包帶進 jit（編譯）；靜態索引仍是 CPU 陣列，不把常數本身當可追蹤參數。
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from aosr.scoring.gradient_objective_jax import GradientObjectiveConstants
from aosr.scoring.stage_one_objective_jax import StageOneObjectiveConstants
from aosr.scoring.reverberation_cost_jax import ReverberationCostSettings
from aosr.scoring.timbre_jax import ResponseConstants
from aosr.scoring.timbre_cost_jax import CostConstants, PairConstants, ShapeConstants


class SixCategoryConstants(NamedTuple):
    """面片、路徑、兩組低頻尺與各類設定，沒有巢狀欄位。"""

    frequencies_hz: NDArray[np.float64]
    fem_frequencies_hz: NDArray[np.float64]
    union_frequencies_hz: NDArray[np.float64]
    report_indices: NDArray[np.int64]
    decay_indices: NDArray[np.int64]
    midband_indices: NDArray[np.int64]
    paris_angles_deg: NDArray[np.float64]
    source_ids: NDArray[np.str_]
    patch_ids: NDArray[np.str_]
    angle_offsets: NDArray[np.int64]
    angles_deg: NDArray[np.float64]
    cosines: NDArray[np.float64]
    path_offsets: NDArray[np.int64]
    order: NDArray[np.int64]
    dist_m: NDArray[np.float64]
    delay_s: NDArray[np.float64]
    bounce_wall: NDArray[np.int64]
    bounce_cos: NDArray[np.float64]
    bounce_mask: NDArray[np.bool_]
    bounce_patch: NDArray[np.int64]
    bounce_angle: NDArray[np.int64]
    inverse_distance: NDArray[np.float64]
    phase: NDArray[np.complex128]
    directivity: NDArray[np.float64]
    furniture_pressure: NDArray[np.complex128]
    patch_areas_m2: NDArray[np.float64]
    rho_c: NDArray[np.float64]
    reflection_order_k: NDArray[np.int64]
    primary_pair_indices: NDArray[np.int64]
    direct_distances_m: NDArray[np.float64]
    reflection_zone_mask: NDArray[np.bool_]
    reflection_frequency_weights: NDArray[np.float64]
    reflection_thresholds_db: NDArray[np.float64]
    reflection_zone_weights: NDArray[np.float64]
    reflection_worse_reference_db: NDArray[np.float64]
    reflection_coverage: NDArray[np.bool_]
    reflection_furniture_energy: NDArray[np.float64]
    form_factors: NDArray[np.float64]
    late_patch_areas: NDArray[np.float64]
    late_wall_index: NDArray[np.int64]
    wall_areas: NDArray[np.float64]
    wall_weights: NDArray[np.float64]
    patch_weights: NDArray[np.float64]
    collision_frequency_hz: NDArray[np.float64]
    volume_m3: NDArray[np.float64]
    air_absorption_fraction: NDArray[np.float64]
    furniture_absorption_area_m2: NDArray[np.float64]
    has_furniture: NDArray[np.bool_]
    scattering_by_patch: NDArray[np.float64]
    source_power_ratio: NDArray[np.float64]
    decay_band_mask: NDArray[np.bool_]
    category_weights: NDArray[np.float64]
    timbre_input_layout: NDArray[np.bool_]
    timbre_primary: NDArray[np.int64]
    timbre_measured: NDArray[np.int64]
    timbre_primary_pairs_indices: NDArray[np.int64]
    timbre_primary_pairs_weights: NDArray[np.float64]
    timbre_peer_pairs_indices: NDArray[np.int64]
    timbre_peer_pairs_weights: NDArray[np.float64]
    timbre_channel_pairs: NDArray[np.int64]
    timbre_importance: NDArray[np.float64]
    timbre_weights: NDArray[np.float64]
    listening_weights: NDArray[np.float64]
    matching_weights: NDArray[np.float64]
    sound_speed_m_s: NDArray[np.float64]
    timbre_shape_center: NDArray[np.float64]
    timbre_shape_tolerance: NDArray[np.float64]
    timbre_shape_worse_reference: NDArray[np.float64]
    timbre_shape_linear: NDArray[np.bool_]
    listening_shape_center: NDArray[np.float64]
    listening_shape_tolerance: NDArray[np.float64]
    listening_shape_worse_reference: NDArray[np.float64]
    listening_shape_linear: NDArray[np.bool_]
    matching_shape_center: NDArray[np.float64]
    matching_shape_tolerance: NDArray[np.float64]
    matching_shape_worse_reference: NDArray[np.float64]
    matching_shape_linear: NDArray[np.bool_]
    response_octaves: NDArray[np.float64]
    response_tilt_smoothing: NDArray[np.float64]
    response_ripple_smoothing: NDArray[np.float64]
    response_fit_weights: NDArray[np.float64]
    response_ripple_weights: NDArray[np.float64]
    response_matching_weights: NDArray[np.float64]
    response_broadband_weights: NDArray[np.float64]
    response_insufficient_coverage: NDArray[np.bool_]
    response_scoring_range_gap: NDArray[np.bool_]
    response_matching_coverage: NDArray[np.bool_]
    product_log2_frequencies: NDArray[np.float64]
    product_frequency_indices: NDArray[np.int64]
    product_frequency_weights: NDArray[np.float64]
    product_primary_index: NDArray[np.int64]
    product_surrounding_indices: NDArray[np.int64]
    product_speaker_indices: NDArray[np.int64]
    product_component_weights: NDArray[np.float64]
    product_worse_references: NDArray[np.float64]
    objective_log2_frequencies: NDArray[np.float64]
    objective_frequency_indices: NDArray[np.int64]
    objective_frequency_weights: NDArray[np.float64]
    objective_primary_index: NDArray[np.int64]
    objective_surrounding_indices: NDArray[np.int64]
    objective_speaker_indices: NDArray[np.int64]
    objective_component_weights: NDArray[np.float64]
    objective_worse_references: NDArray[np.float64]
    objective_interior_positions: NDArray[np.int64]
    objective_proxy_temperature_db: NDArray[np.float64]
    objective_proxy_weight: NDArray[np.float64]
    objective_proxy_worse_reference: NDArray[np.float64]
    reverb_lower_s: NDArray[np.float64]
    reverb_upper_s: NDArray[np.float64]
    reverb_interval_scale_s: NDArray[np.float64]
    reverb_jump_threshold: NDArray[np.float64]
    reverb_jump_scale: NDArray[np.float64]
    reverb_logarithm_base: NDArray[np.float64]
    reverb_weights: NDArray[np.float64]

    def response(self) -> ResponseConstants:
        """重建既有核心吃的常數，不重新求值。"""
        return ResponseConstants(
            self.response_octaves, self.response_tilt_smoothing, self.response_ripple_smoothing,
            self.response_fit_weights, self.response_ripple_weights, self.response_matching_weights,
            self.response_broadband_weights, self.response_insufficient_coverage, self.response_scoring_range_gap,
            self.response_matching_coverage
        )

    def product(self) -> StageOneObjectiveConstants:
        """重建既有核心吃的常數，不重新求值。"""
        return StageOneObjectiveConstants(
            self.product_log2_frequencies, self.product_frequency_indices, self.product_frequency_weights,
            self.product_primary_index, self.product_surrounding_indices, self.product_speaker_indices,
            self.product_component_weights, self.product_worse_references
        )

    def objective(self) -> GradientObjectiveConstants:
        """重建既有核心吃的常數，不重新求值。"""
        return GradientObjectiveConstants(
            self.objective_log2_frequencies, self.objective_frequency_indices, self.objective_frequency_weights,
            self.objective_primary_index, self.objective_surrounding_indices, self.objective_speaker_indices,
            self.objective_component_weights, self.objective_worse_references, self.objective_interior_positions,
            self.objective_proxy_temperature_db, self.objective_proxy_weight, self.objective_proxy_worse_reference
        )

    def reverb(self) -> ReverberationCostSettings:
        """重建既有核心吃的常數，不重新求值。"""
        return ReverberationCostSettings(
            self.reverb_lower_s, self.reverb_upper_s, float(self.reverb_interval_scale_s),
            float(self.reverb_jump_threshold), float(self.reverb_jump_scale), float(self.reverb_logarithm_base),
            self.reverb_weights
        )

    def timbre(self) -> CostConstants:
        """攤平的音色設定重組；每個葉子仍在指紋欄位內。"""
        return CostConstants(
            self.response(), self.timbre_input_layout, self.timbre_primary, self.timbre_measured,
            PairConstants(self.timbre_primary_pairs_indices, self.timbre_primary_pairs_weights),
            PairConstants(self.timbre_peer_pairs_indices, self.timbre_peer_pairs_weights),
            self.timbre_channel_pairs, self.timbre_importance,
            ShapeConstants(self.timbre_shape_center, self.timbre_shape_tolerance,
                           self.timbre_shape_worse_reference, self.timbre_shape_linear),
            ShapeConstants(self.listening_shape_center, self.listening_shape_tolerance,
                           self.listening_shape_worse_reference, self.listening_shape_linear),
            ShapeConstants(self.matching_shape_center, self.matching_shape_tolerance,
                           self.matching_shape_worse_reference, self.matching_shape_linear),
            self.timbre_weights, self.listening_weights, self.matching_weights, self.sound_speed_m_s)
