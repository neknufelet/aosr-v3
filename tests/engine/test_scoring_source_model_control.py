"""#505 第三刀控制組：全向的兩個候選，六類評估與排名跟改動前的主線相同。

評分層讀聲源模型之後，全向那條路的分數、旗標、狀態、名次與分表都不准動。答案是主對話在改動前的主線上實跑
``_scoring_source_model_control.py`` 錄下的（``_scoring_source_model_answers.py``）；這一題只拿現在的結果去比，
不拿被測函式現算的值當答案。按設計會變的格子（評估器版本、折了版本的兩個設定指紋、反射 payload 換型那一格、
新的聲源模型指紋；#577 起還有品質登記簿整份的指紋）在管線那一支逐一列出並排除。#518（聆聽區左右各算）按設計會變的幾格同樣在管線那一支列出（``REPLACED_BY_518``）；聆聽區與總代價的新值寫在逐一記（HEADLINE），由改動前主線上的左右兩支單錄手算。整條管線放在同一題：平行跑法下模組層的準備會在每個工人重做。


#785 第七步接手核對：本控制組沒有家具路徑；以下延遲是純幾何四則與 sqrt，
不是由名字推定。relative_direct_delay_s 由 reflections.py::_paths 的兩個 delay_s 相減，
上游 report_path_table.py::_path_row 轉交 room_paths.py::_one_path 的距離／聲速，
距離由 src/aosr/geometry/shoebox.py::distance 的 math.sqrt 算出；strongest_delay_s 由 reflections.py::_point 選取轉交。
左右 delay_s 沿 channel_matching_reflections.py::_side 轉交上述 strongest_delay_s。
牆對 round_trip_delay_s.value 沿 reflections.py::_wall_pairs 轉交 reflection_screen.py::_wall_pair 的 2×房長／聲速。
只核准 machine_float 內五條完整延遲路徑（另可有 evaluation 前綴）；未知位置的 delay_s 保守逐格。
泛名 raw_quantities[].value 另讀同列 name：feature_count、band_count、assessed_point_count 是整數計數轉 float；
round_trip_delay_* 是上述牆對延遲；其他量保守逐格，不能由 value 或時間單位猜測。

#785 第七步欄位分類（決策紙第 6 條；所有有關格統一③級，未知保守有關）：
下表以 HEAD 同跑結果列出每個浮點路徑樣式；非浮點全部留骨架。
deviation_curve[][] 的 [0] 為頻率鍵、[1] 為分貝；不是整欄一刀切。
raw_quantities[].value 依同列 name：feature_count（含角色前綴）、band_count、assessed_point_count
及 round_trip_delay_* 為無關，其餘量測量為有關；無關浮點的值、型別、索引與順序都在骨架。
reflection_asymmetry 的 delay_s 由 strongest_delay_s 轉交，來自 room_paths.py::_one_path 的 sqrt／四則；
channel_matching 直達時間由本控制組 _distance 的 math.dist 轉交，沒有正確捨入保證，保守有關。
listening_axis_xy 用 math.hypot，亦保守有關。帶界為頻率鍵；timbre 的 dependency_range 經次方，逐格。
peak_dip_consistency 只核准下表兩條完整路徑（計數與四則）留骨架，底下新增的浮點保守逐格；其選擇結果與狀態／旗標逐字比。
骨架只由葉子組成：空字典、空清單不產生葉子；兩者互換、或多出不改變其他葉子路徑的空容器，骨架看不到（舊分組裁判同）。
分項代價可能由評估器超越函式傳播；無法只從數值判成純算術者保守有關（含恰好為零的代價）。
此題證明骨架逐字、機器相關浮點在③級內，不再證明跨處理器逐位相等。
欄位樣式（保留根層；[] 去掉索引） | 無關格數 | 有關格數 | 來源與依據
--- | --- | --- | ---
.category_cost | 0 | 8 | src/aosr/scoring/ranking.py::_settle、src/aosr/scoring/ranking.py::_category_line（傳播六類評估代價）
.category_weight | 8 | 0 | src/aosr/scoring/ranking.py::_category_line（登記簿權重）
.components[].cost | 0 | 86 | src/aosr/scoring/ranking.py::_settle、src/aosr/scoring/ranking.py::_category_line（傳播六類評估代價）
.components[].raw_value | 0 | 16 | src/aosr/scoring/timbre.py::evaluate_timbre、src/aosr/scoring/timbre.py::_dependency_ranges、src/aosr/scoring/timbre.py::_features（軸／範圍／目標為輸入；依賴範圍經非整數次方；量測經陣列 log／擬合）
.components[].weight | 8 | 0 | src/aosr/scoring/ranking.py::_timbre_channel_lines（登記權重除聲道數）
.eligibility_rules[].value[] | 2 | 0 | src/aosr/scoring/ranking.py::_eligibility_rules（登記簿輸入）
.evaluation.category_cost.components.adjacent_t20_jump | 0 | 2 | src/aosr/scoring/ranking.py::_settle、src/aosr/scoring/ranking.py::_category_line（傳播六類評估代價）
.evaluation.category_cost.components.adjacent_t20_jump.1000-2000Hz | 0 | 2 | src/aosr/scoring/ranking.py::_settle、src/aosr/scoring/ranking.py::_category_line（傳播六類評估代價）
.evaluation.category_cost.components.adjacent_t20_jump.125-250Hz | 0 | 2 | src/aosr/scoring/ranking.py::_settle、src/aosr/scoring/ranking.py::_category_line（傳播六類評估代價）
.evaluation.category_cost.components.adjacent_t20_jump.2000-4000Hz | 0 | 2 | src/aosr/scoring/ranking.py::_settle、src/aosr/scoring/ranking.py::_category_line（傳播六類評估代價）
.evaluation.category_cost.components.adjacent_t20_jump.250-500Hz | 0 | 2 | src/aosr/scoring/ranking.py::_settle、src/aosr/scoring/ranking.py::_category_line（傳播六類評估代價）
.evaluation.category_cost.components.adjacent_t20_jump.4000-8000Hz | 0 | 2 | src/aosr/scoring/ranking.py::_settle、src/aosr/scoring/ranking.py::_category_line（傳播六類評估代價）
.evaluation.category_cost.components.adjacent_t20_jump.500-1000Hz | 0 | 2 | src/aosr/scoring/ranking.py::_settle、src/aosr/scoring/ranking.py::_category_line（傳播六類評估代價）
.evaluation.category_cost.components.broadband_level_difference.weighted_mean | 0 | 2 | src/aosr/scoring/ranking.py::_settle、src/aosr/scoring/ranking.py::_category_line（傳播六類評估代價）
.evaluation.category_cost.components.broadband_level_difference.weighted_mean.left-right | 0 | 2 | src/aosr/scoring/ranking.py::_settle、src/aosr/scoring/ranking.py::_category_line（傳播六類評估代價）
.evaluation.category_cost.components.broadband_level_difference.worst | 0 | 2 | src/aosr/scoring/ranking.py::_settle、src/aosr/scoring/ranking.py::_category_line（傳播六類評估代價）
.evaluation.category_cost.components.broadband_level_difference.worst.left-right.front | 0 | 2 | src/aosr/scoring/ranking.py::_settle、src/aosr/scoring/ranking.py::_category_line（傳播六類評估代價）
.evaluation.category_cost.components.left.front | 0 | 2 | src/aosr/scoring/ranking.py::_settle、src/aosr/scoring/ranking.py::_category_line（傳播六類評估代價）
.evaluation.category_cost.components.left.lateral | 0 | 2 | src/aosr/scoring/ranking.py::_settle、src/aosr/scoring/ranking.py::_category_line（傳播六類評估代價）
.evaluation.category_cost.components.left.peaks_dips | 0 | 2 | src/aosr/scoring/ranking.py::_settle、src/aosr/scoring/ranking.py::_category_line（傳播六類評估代價）
.evaluation.category_cost.components.left.rear | 0 | 2 | src/aosr/scoring/ranking.py::_settle、src/aosr/scoring/ranking.py::_category_line（傳播六類評估代價）
.evaluation.category_cost.components.left.residual_rms | 0 | 2 | src/aosr/scoring/ranking.py::_settle、src/aosr/scoring/ranking.py::_category_line（傳播六類評估代價）
.evaluation.category_cost.components.left.target_deviation | 0 | 2 | src/aosr/scoring/ranking.py::_settle、src/aosr/scoring/ranking.py::_category_line（傳播六類評估代價）
.evaluation.category_cost.components.left.tilt | 0 | 2 | src/aosr/scoring/ranking.py::_settle、src/aosr/scoring/ranking.py::_category_line（傳播六類評估代價）
.evaluation.category_cost.components.left.vertical | 0 | 2 | src/aosr/scoring/ranking.py::_settle、src/aosr/scoring/ranking.py::_category_line（傳播六類評估代價）
.evaluation.category_cost.components.right.front | 0 | 2 | src/aosr/scoring/ranking.py::_settle、src/aosr/scoring/ranking.py::_category_line（傳播六類評估代價）
.evaluation.category_cost.components.right.lateral | 0 | 2 | src/aosr/scoring/ranking.py::_settle、src/aosr/scoring/ranking.py::_category_line（傳播六類評估代價）
.evaluation.category_cost.components.right.peaks_dips | 0 | 2 | src/aosr/scoring/ranking.py::_settle、src/aosr/scoring/ranking.py::_category_line（傳播六類評估代價）
.evaluation.category_cost.components.right.rear | 0 | 2 | src/aosr/scoring/ranking.py::_settle、src/aosr/scoring/ranking.py::_category_line（傳播六類評估代價）
.evaluation.category_cost.components.right.residual_rms | 0 | 2 | src/aosr/scoring/ranking.py::_settle、src/aosr/scoring/ranking.py::_category_line（傳播六類評估代價）
.evaluation.category_cost.components.right.target_deviation | 0 | 2 | src/aosr/scoring/ranking.py::_settle、src/aosr/scoring/ranking.py::_category_line（傳播六類評估代價）
.evaluation.category_cost.components.right.tilt | 0 | 2 | src/aosr/scoring/ranking.py::_settle、src/aosr/scoring/ranking.py::_category_line（傳播六類評估代價）
.evaluation.category_cost.components.right.vertical | 0 | 2 | src/aosr/scoring/ranking.py::_settle、src/aosr/scoring/ranking.py::_category_line（傳播六類評估代價）
.evaluation.category_cost.components.ripple_rms_difference.weighted_mean | 0 | 2 | src/aosr/scoring/ranking.py::_settle、src/aosr/scoring/ranking.py::_category_line（傳播六類評估代價）
.evaluation.category_cost.components.ripple_rms_difference.weighted_mean.left-right | 0 | 2 | src/aosr/scoring/ranking.py::_settle、src/aosr/scoring/ranking.py::_category_line（傳播六類評估代價）
.evaluation.category_cost.components.ripple_rms_difference.worst | 0 | 2 | src/aosr/scoring/ranking.py::_settle、src/aosr/scoring/ranking.py::_category_line（傳播六類評估代價）
.evaluation.category_cost.components.ripple_rms_difference.worst.left-right.front | 0 | 2 | src/aosr/scoring/ranking.py::_settle、src/aosr/scoring/ranking.py::_category_line（傳播六類評估代價）
.evaluation.category_cost.components.t20_target_interval | 0 | 2 | src/aosr/scoring/ranking.py::_settle、src/aosr/scoring/ranking.py::_category_line（傳播六類評估代價）
.evaluation.category_cost.components.t20_target_interval.1000Hz | 0 | 2 | src/aosr/scoring/ranking.py::_settle、src/aosr/scoring/ranking.py::_category_line（傳播六類評估代價）
.evaluation.category_cost.components.t20_target_interval.125Hz | 0 | 2 | src/aosr/scoring/ranking.py::_settle、src/aosr/scoring/ranking.py::_category_line（傳播六類評估代價）
.evaluation.category_cost.components.t20_target_interval.2000Hz | 0 | 2 | src/aosr/scoring/ranking.py::_settle、src/aosr/scoring/ranking.py::_category_line（傳播六類評估代價）
.evaluation.category_cost.components.t20_target_interval.250Hz | 0 | 2 | src/aosr/scoring/ranking.py::_settle、src/aosr/scoring/ranking.py::_category_line（傳播六類評估代價）
.evaluation.category_cost.components.t20_target_interval.4000Hz | 0 | 2 | src/aosr/scoring/ranking.py::_settle、src/aosr/scoring/ranking.py::_category_line（傳播六類評估代價）
.evaluation.category_cost.components.t20_target_interval.500Hz | 0 | 2 | src/aosr/scoring/ranking.py::_settle、src/aosr/scoring/ranking.py::_category_line（傳播六類評估代價）
.evaluation.category_cost.components.t20_target_interval.8000Hz | 0 | 2 | src/aosr/scoring/ranking.py::_settle、src/aosr/scoring/ranking.py::_category_line（傳播六類評估代價）
.evaluation.category_cost.components.tilt_difference.weighted_mean | 0 | 2 | src/aosr/scoring/ranking.py::_settle、src/aosr/scoring/ranking.py::_category_line（傳播六類評估代價）
.evaluation.category_cost.components.tilt_difference.weighted_mean.left-right | 0 | 2 | src/aosr/scoring/ranking.py::_settle、src/aosr/scoring/ranking.py::_category_line（傳播六類評估代價）
.evaluation.category_cost.components.tilt_difference.worst | 0 | 2 | src/aosr/scoring/ranking.py::_settle、src/aosr/scoring/ranking.py::_category_line（傳播六類評估代價）
.evaluation.category_cost.components.tilt_difference.worst.left-right.main | 0 | 2 | src/aosr/scoring/ranking.py::_settle、src/aosr/scoring/ranking.py::_category_line（傳播六類評估代價）
.evaluation.category_cost.value | 0 | 8 | src/aosr/scoring/ranking.py::_settle、src/aosr/scoring/ranking.py::_category_line（傳播六類評估代價）
.evaluation.payload.adjacent_band_changes[].lower_center_frequency_hz | 12 | 0 | src/aosr/scoring/reverberation.py::_evaluate_band、src/aosr/scoring/reverberation.py::_band_range、src/aosr/scoring/reverberation.py::_adjacent_change（頻率鍵／sqrt 帶界；衰減來自陣列帶平均；log 比值；輸入底數）
.evaluation.payload.adjacent_band_changes[].signed_log_ratio | 0 | 12 | src/aosr/scoring/reverberation.py::_evaluate_band、src/aosr/scoring/reverberation.py::_band_range、src/aosr/scoring/reverberation.py::_adjacent_change（頻率鍵／sqrt 帶界；衰減來自陣列帶平均；log 比值；輸入底數）
.evaluation.payload.adjacent_band_changes[].upper_center_frequency_hz | 12 | 0 | src/aosr/scoring/reverberation.py::_evaluate_band、src/aosr/scoring/reverberation.py::_band_range、src/aosr/scoring/reverberation.py::_adjacent_change（頻率鍵／sqrt 帶界；衰減來自陣列帶平均；log 比值；輸入底數）
.evaluation.payload.aggregates[].broadband_level_difference.weighted_mean_absolute_difference | 0 | 2 | src/aosr/scoring/channel_matching.py::_metric_aggregate（分貝傳播；直達時間來自控制組 math.dist，保守有關）
.evaluation.payload.aggregates[].broadband_level_difference.worst_absolute_difference | 0 | 2 | src/aosr/scoring/channel_matching.py::_metric_aggregate（分貝傳播；直達時間來自控制組 math.dist，保守有關）
.evaluation.payload.aggregates[].direct_time_difference.weighted_mean_absolute_difference | 0 | 2 | src/aosr/scoring/channel_matching.py::_metric_aggregate（分貝傳播；直達時間來自控制組 math.dist，保守有關）
.evaluation.payload.aggregates[].direct_time_difference.worst_absolute_difference | 0 | 2 | src/aosr/scoring/channel_matching.py::_metric_aggregate（分貝傳播；直達時間來自控制組 math.dist，保守有關）
.evaluation.payload.aggregates[].ripple_rms_difference.weighted_mean_absolute_difference | 0 | 2 | src/aosr/scoring/channel_matching.py::_metric_aggregate（分貝傳播；直達時間來自控制組 math.dist，保守有關）
.evaluation.payload.aggregates[].ripple_rms_difference.worst_absolute_difference | 0 | 2 | src/aosr/scoring/channel_matching.py::_metric_aggregate（分貝傳播；直達時間來自控制組 math.dist，保守有關）
.evaluation.payload.aggregates[].tilt_difference.weighted_mean_absolute_difference | 0 | 2 | src/aosr/scoring/channel_matching.py::_metric_aggregate（分貝傳播；直達時間來自控制組 math.dist，保守有關）
.evaluation.payload.aggregates[].tilt_difference.worst_absolute_difference | 0 | 2 | src/aosr/scoring/channel_matching.py::_metric_aggregate（分貝傳播；直達時間來自控制組 math.dist，保守有關）
.evaluation.payload.bands[].band_range_hz[] | 28 | 0 | src/aosr/scoring/reverberation.py::_evaluate_band、src/aosr/scoring/reverberation.py::_band_range、src/aosr/scoring/reverberation.py::_adjacent_change（頻率鍵／sqrt 帶界；衰減來自陣列帶平均；log 比值；輸入底數）
.evaluation.payload.bands[].center_frequency_hz | 14 | 0 | src/aosr/scoring/reverberation.py::_evaluate_band、src/aosr/scoring/reverberation.py::_band_range、src/aosr/scoring/reverberation.py::_adjacent_change（頻率鍵／sqrt 帶界；衰減來自陣列帶平均；log 比值；輸入底數）
.evaluation.payload.bands[].fitting_difference.value | 0 | 14 | src/aosr/scoring/reverberation.py::_evaluate_band、src/aosr/scoring/reverberation.py::_band_range、src/aosr/scoring/reverberation.py::_adjacent_change（頻率鍵／sqrt 帶界；衰減來自陣列帶平均；log 比值；輸入底數）
.evaluation.payload.bands[].t20.value | 0 | 14 | src/aosr/scoring/reverberation.py::_evaluate_band、src/aosr/scoring/reverberation.py::_band_range、src/aosr/scoring/reverberation.py::_adjacent_change（頻率鍵／sqrt 帶界；衰減來自陣列帶平均；log 比值；輸入底數）
.evaluation.payload.bands[].t30.value | 0 | 14 | src/aosr/scoring/reverberation.py::_evaluate_band、src/aosr/scoring/reverberation.py::_band_range、src/aosr/scoring/reverberation.py::_adjacent_change（頻率鍵／sqrt 帶界；衰減來自陣列帶平均；log 比值；輸入底數）
.evaluation.payload.broadband_support.frequencies_hz[] | 416 | 0 | src/aosr/scoring/channel_matching.py::_broadband_support（軸轉交為鍵）
.evaluation.payload.broadband_support.highest_frequency_hz | 2 | 0 | src/aosr/scoring/channel_matching.py::_broadband_support（軸轉交為鍵）
.evaluation.payload.broadband_support.lowest_frequency_hz | 2 | 0 | src/aosr/scoring/channel_matching.py::_broadband_support（軸轉交為鍵）
.evaluation.payload.channels[].payload.coverage_range_hz[] | 8 | 0 | src/aosr/scoring/timbre.py::evaluate_timbre、src/aosr/scoring/timbre.py::_dependency_ranges、src/aosr/scoring/timbre.py::_features（軸／範圍／目標為輸入；依賴範圍經非整數次方；量測經陣列 log／擬合）
.evaluation.payload.channels[].payload.data_range_hz[] | 8 | 0 | src/aosr/scoring/timbre.py::evaluate_timbre、src/aosr/scoring/timbre.py::_dependency_ranges、src/aosr/scoring/timbre.py::_features（軸／範圍／目標為輸入；依賴範圍經非整數次方；量測經陣列 log／擬合）
.evaluation.payload.channels[].payload.deviation_curve[][] | 832 | 832 | src/aosr/scoring/timbre.py::evaluate_timbre、src/aosr/scoring/timbre.py::_dependency_ranges、src/aosr/scoring/timbre.py::_features（軸／範圍／目標為輸入；依賴範圍經非整數次方；量測經陣列 log／擬合）
.evaluation.payload.channels[].payload.features[].center_frequency_hz | 189 | 0 | src/aosr/scoring/timbre.py::evaluate_timbre、src/aosr/scoring/timbre.py::_dependency_ranges、src/aosr/scoring/timbre.py::_features（軸／範圍／目標為輸入；依賴範圍經非整數次方；量測經陣列 log／擬合）
.evaluation.payload.channels[].payload.features[].depth_db | 0 | 189 | src/aosr/scoring/timbre.py::evaluate_timbre、src/aosr/scoring/timbre.py::_dependency_ranges、src/aosr/scoring/timbre.py::_features（軸／範圍／目標為輸入；依賴範圍經非整數次方；量測經陣列 log／擬合）
.evaluation.payload.channels[].payload.features[].width_octave | 0 | 172 | src/aosr/scoring/timbre.py::evaluate_timbre、src/aosr/scoring/timbre.py::_dependency_ranges、src/aosr/scoring/timbre.py::_features（軸／範圍／目標為輸入；依賴範圍經非整數次方；量測經陣列 log／擬合）
.evaluation.payload.channels[].payload.residual_rms_db | 0 | 4 | src/aosr/scoring/timbre.py::evaluate_timbre、src/aosr/scoring/timbre.py::_dependency_ranges、src/aosr/scoring/timbre.py::_features（軸／範圍／目標為輸入；依賴範圍經非整數次方；量測經陣列 log／擬合）
.evaluation.payload.channels[].payload.ripple_dependency_range_hz[] | 0 | 8 | src/aosr/scoring/timbre.py::evaluate_timbre、src/aosr/scoring/timbre.py::_dependency_ranges、src/aosr/scoring/timbre.py::_features（軸／範圍／目標為輸入；依賴範圍經非整數次方；量測經陣列 log／擬合）
.evaluation.payload.channels[].payload.ripple_range_hz[] | 8 | 0 | src/aosr/scoring/timbre.py::evaluate_timbre、src/aosr/scoring/timbre.py::_dependency_ranges、src/aosr/scoring/timbre.py::_features（軸／範圍／目標為輸入；依賴範圍經非整數次方；量測經陣列 log／擬合）
.evaluation.payload.channels[].payload.target_deviation_rms_db | 0 | 4 | src/aosr/scoring/timbre.py::evaluate_timbre、src/aosr/scoring/timbre.py::_dependency_ranges、src/aosr/scoring/timbre.py::_features（軸／範圍／目標為輸入；依賴範圍經非整數次方；量測經陣列 log／擬合）
.evaluation.payload.channels[].payload.target_tilt_db_per_octave | 4 | 0 | src/aosr/scoring/timbre.py::evaluate_timbre、src/aosr/scoring/timbre.py::_dependency_ranges、src/aosr/scoring/timbre.py::_features（軸／範圍／目標為輸入；依賴範圍經非整數次方；量測經陣列 log／擬合）
.evaluation.payload.channels[].payload.tilt_db_per_octave | 0 | 4 | src/aosr/scoring/timbre.py::evaluate_timbre、src/aosr/scoring/timbre.py::_dependency_ranges、src/aosr/scoring/timbre.py::_features（軸／範圍／目標為輸入；依賴範圍經非整數次方；量測經陣列 log／擬合）
.evaluation.payload.channels[].payload.tilt_dependency_range_hz[] | 0 | 8 | src/aosr/scoring/timbre.py::evaluate_timbre、src/aosr/scoring/timbre.py::_dependency_ranges、src/aosr/scoring/timbre.py::_features（軸／範圍／目標為輸入；依賴範圍經非整數次方；量測經陣列 log／擬合）
.evaluation.payload.channels[].payload.tilt_fit_range_hz[] | 8 | 0 | src/aosr/scoring/timbre.py::evaluate_timbre、src/aosr/scoring/timbre.py::_dependency_ranges、src/aosr/scoring/timbre.py::_features（軸／範圍／目標為輸入；依賴範圍經非整數次方；量測經陣列 log／擬合）
.evaluation.payload.channels[].reflections[].broadband_level_db | 0 | 514 | src/aosr/scoring/reflections.py::_paths（延遲四則；角度 atan2；分貝陣列 log）
.evaluation.payload.channels[].reflections[].listening_azimuth_deg | 0 | 514 | src/aosr/scoring/reflections.py::_paths（延遲四則；角度 atan2；分貝陣列 log）
.evaluation.payload.channels[].reflections[].listening_elevation_deg | 0 | 514 | src/aosr/scoring/reflections.py::_paths（延遲四則；角度 atan2；分貝陣列 log）
.evaluation.payload.channels[].reflections[].relative_direct_delay_s | 514 | 0 | src/aosr/scoring/reflections.py::_paths（延遲四則；角度 atan2；分貝陣列 log）
.evaluation.payload.channels[].reflections[].room_azimuth_deg | 0 | 514 | src/aosr/scoring/reflections.py::_paths（延遲四則；角度 atan2；分貝陣列 log）
.evaluation.payload.channels[].reflections[].room_elevation_deg | 0 | 514 | src/aosr/scoring/reflections.py::_paths（延遲四則；角度 atan2；分貝陣列 log）
.evaluation.payload.channels[].total_window_energy_db[].value | 0 | 912 | src/aosr/scoring/reflections.py::_channel（經 _sum_level 的 log10）
.evaluation.payload.channels[].zones[].points[].frequency_hz | 3648 | 0 | src/aosr/scoring/reflections.py::_point（頻率鍵；延遲轉交；能量經 log10）
.evaluation.payload.channels[].zones[].points[].strongest_delay_s | 3192 | 0 | src/aosr/scoring/reflections.py::_point（頻率鍵；延遲轉交；能量經 log10）
.evaluation.payload.channels[].zones[].points[].strongest_level_db | 0 | 3192 | src/aosr/scoring/reflections.py::_point（頻率鍵；延遲轉交；能量經 log10）
.evaluation.payload.channels[].zones[].points[].total_energy_db.value | 0 | 3192 | src/aosr/scoring/reflections.py::_point（頻率鍵；延遲轉交；能量經 log10）
.evaluation.payload.frequency_range_hz[] | 4 | 0 | src/aosr/scoring/reflections.py::evaluate_reflections、src/aosr/scoring/channel_matching.py::_payload（登記設定或其四則轉換）
.evaluation.payload.listening_axis_xy[] | 0 | 4 | src/aosr/scoring/direction_zones.py::listening_axis（math.hypot 不保證正確捨入，保守有關）
.evaluation.payload.logarithm_base | 2 | 0 | src/aosr/scoring/reverberation.py::_evaluate_band、src/aosr/scoring/reverberation.py::_band_range、src/aosr/scoring/reverberation.py::_adjacent_change（頻率鍵／sqrt 帶界；衰減來自陣列帶平均；log 比值；輸入底數）
.evaluation.payload.point_results[].broadband_level_difference_db | 0 | 4 | src/aosr/scoring/channel_matching.py::_measured_point、src/aosr/scoring/channel_matching.py::_difference_curve（頻率鍵、輸入 importance；分貝／特徵已經陣列，時間來自 math.dist）
.evaluation.payload.point_results[].direct_time_difference_ms | 0 | 4 | src/aosr/scoring/channel_matching.py::_measured_point、src/aosr/scoring/channel_matching.py::_difference_curve（頻率鍵、輸入 importance；分貝／特徵已經陣列，時間來自 math.dist）
.evaluation.payload.point_results[].frequency_difference_curve_db[].frequency_hz | 880 | 0 | src/aosr/scoring/channel_matching.py::_measured_point、src/aosr/scoring/channel_matching.py::_difference_curve（頻率鍵、輸入 importance；分貝／特徵已經陣列，時間來自 math.dist）
.evaluation.payload.point_results[].frequency_difference_curve_db[].left_minus_right_db | 0 | 880 | src/aosr/scoring/channel_matching.py::_measured_point、src/aosr/scoring/channel_matching.py::_difference_curve（頻率鍵、輸入 importance；分貝／特徵已經陣列，時間來自 math.dist）
.evaluation.payload.point_results[].importance | 4 | 0 | src/aosr/scoring/channel_matching.py::_measured_point、src/aosr/scoring/channel_matching.py::_difference_curve（頻率鍵、輸入 importance；分貝／特徵已經陣列，時間來自 math.dist）
.evaluation.payload.point_results[].ripple_rms_difference_db | 0 | 4 | src/aosr/scoring/channel_matching.py::_measured_point、src/aosr/scoring/channel_matching.py::_difference_curve（頻率鍵、輸入 importance；分貝／特徵已經陣列，時間來自 math.dist）
.evaluation.payload.point_results[].tilt_difference_db_per_octave | 0 | 4 | src/aosr/scoring/channel_matching.py::_measured_point、src/aosr/scoring/channel_matching.py::_difference_curve（頻率鍵、輸入 importance；分貝／特徵已經陣列，時間來自 math.dist）
.evaluation.payload.point_results[].unmatched_features[].feature.center_frequency_hz | 250 | 0 | src/aosr/scoring/channel_matching.py::_measured_point、src/aosr/scoring/channel_matching.py::_difference_curve（頻率鍵、輸入 importance；分貝／特徵已經陣列，時間來自 math.dist）
.evaluation.payload.point_results[].unmatched_features[].feature.depth_db | 0 | 250 | src/aosr/scoring/channel_matching.py::_measured_point、src/aosr/scoring/channel_matching.py::_difference_curve（頻率鍵、輸入 importance；分貝／特徵已經陣列，時間來自 math.dist）
.evaluation.payload.point_results[].unmatched_features[].feature.width_octave | 0 | 224 | src/aosr/scoring/channel_matching.py::_measured_point、src/aosr/scoring/channel_matching.py::_difference_curve（頻率鍵、輸入 importance；分貝／特徵已經陣列，時間來自 math.dist）
.evaluation.payload.point_sources[].channels[].payload.coverage_range_hz[] | 16 | 0 | src/aosr/scoring/timbre.py::evaluate_timbre、src/aosr/scoring/timbre.py::_dependency_ranges、src/aosr/scoring/timbre.py::_features（軸／範圍／目標為輸入；依賴範圍經非整數次方；量測經陣列 log／擬合）
.evaluation.payload.point_sources[].channels[].payload.data_range_hz[] | 16 | 0 | src/aosr/scoring/timbre.py::evaluate_timbre、src/aosr/scoring/timbre.py::_dependency_ranges、src/aosr/scoring/timbre.py::_features（軸／範圍／目標為輸入；依賴範圍經非整數次方；量測經陣列 log／擬合）
.evaluation.payload.point_sources[].channels[].payload.deviation_curve[][] | 1664 | 1664 | src/aosr/scoring/timbre.py::evaluate_timbre、src/aosr/scoring/timbre.py::_dependency_ranges、src/aosr/scoring/timbre.py::_features（軸／範圍／目標為輸入；依賴範圍經非整數次方；量測經陣列 log／擬合）
.evaluation.payload.point_sources[].channels[].payload.features[].center_frequency_hz | 382 | 0 | src/aosr/scoring/timbre.py::evaluate_timbre、src/aosr/scoring/timbre.py::_dependency_ranges、src/aosr/scoring/timbre.py::_features（軸／範圍／目標為輸入；依賴範圍經非整數次方；量測經陣列 log／擬合）
.evaluation.payload.point_sources[].channels[].payload.features[].depth_db | 0 | 382 | src/aosr/scoring/timbre.py::evaluate_timbre、src/aosr/scoring/timbre.py::_dependency_ranges、src/aosr/scoring/timbre.py::_features（軸／範圍／目標為輸入；依賴範圍經非整數次方；量測經陣列 log／擬合）
.evaluation.payload.point_sources[].channels[].payload.features[].width_octave | 0 | 349 | src/aosr/scoring/timbre.py::evaluate_timbre、src/aosr/scoring/timbre.py::_dependency_ranges、src/aosr/scoring/timbre.py::_features（軸／範圍／目標為輸入；依賴範圍經非整數次方；量測經陣列 log／擬合）
.evaluation.payload.point_sources[].channels[].payload.residual_rms_db | 0 | 8 | src/aosr/scoring/timbre.py::evaluate_timbre、src/aosr/scoring/timbre.py::_dependency_ranges、src/aosr/scoring/timbre.py::_features（軸／範圍／目標為輸入；依賴範圍經非整數次方；量測經陣列 log／擬合）
.evaluation.payload.point_sources[].channels[].payload.ripple_dependency_range_hz[] | 0 | 16 | src/aosr/scoring/timbre.py::evaluate_timbre、src/aosr/scoring/timbre.py::_dependency_ranges、src/aosr/scoring/timbre.py::_features（軸／範圍／目標為輸入；依賴範圍經非整數次方；量測經陣列 log／擬合）
.evaluation.payload.point_sources[].channels[].payload.ripple_range_hz[] | 16 | 0 | src/aosr/scoring/timbre.py::evaluate_timbre、src/aosr/scoring/timbre.py::_dependency_ranges、src/aosr/scoring/timbre.py::_features（軸／範圍／目標為輸入；依賴範圍經非整數次方；量測經陣列 log／擬合）
.evaluation.payload.point_sources[].channels[].payload.target_deviation_rms_db | 0 | 8 | src/aosr/scoring/timbre.py::evaluate_timbre、src/aosr/scoring/timbre.py::_dependency_ranges、src/aosr/scoring/timbre.py::_features（軸／範圍／目標為輸入；依賴範圍經非整數次方；量測經陣列 log／擬合）
.evaluation.payload.point_sources[].channels[].payload.target_tilt_db_per_octave | 8 | 0 | src/aosr/scoring/timbre.py::evaluate_timbre、src/aosr/scoring/timbre.py::_dependency_ranges、src/aosr/scoring/timbre.py::_features（軸／範圍／目標為輸入；依賴範圍經非整數次方；量測經陣列 log／擬合）
.evaluation.payload.point_sources[].channels[].payload.tilt_db_per_octave | 0 | 8 | src/aosr/scoring/timbre.py::evaluate_timbre、src/aosr/scoring/timbre.py::_dependency_ranges、src/aosr/scoring/timbre.py::_features（軸／範圍／目標為輸入；依賴範圍經非整數次方；量測經陣列 log／擬合）
.evaluation.payload.point_sources[].channels[].payload.tilt_dependency_range_hz[] | 0 | 16 | src/aosr/scoring/timbre.py::evaluate_timbre、src/aosr/scoring/timbre.py::_dependency_ranges、src/aosr/scoring/timbre.py::_features（軸／範圍／目標為輸入；依賴範圍經非整數次方；量測經陣列 log／擬合）
.evaluation.payload.point_sources[].channels[].payload.tilt_fit_range_hz[] | 16 | 0 | src/aosr/scoring/timbre.py::evaluate_timbre、src/aosr/scoring/timbre.py::_dependency_ranges、src/aosr/scoring/timbre.py::_features（軸／範圍／目標為輸入；依賴範圍經非整數次方；量測經陣列 log／擬合）
.evaluation.payload.point_sources[].channels[].raw_quantities[].value | 8 | 24 | src/aosr/scoring/timbre.py::_raw_quantities; src/aosr/scoring/reverberation.py::_raw_quantities; src/aosr/scoring/reflections.py::evaluate_reflections; src/aosr/scoring/channel_matching.py::_raw_quantities; src/aosr/scoring/timbre_channels.py::_raw_quantities
.evaluation.payload.reflection_asymmetry.frequency_range_hz[] | 4 | 0 | src/aosr/scoring/channel_matching.py::_payload（由反射結果轉交；延遲四則、分貝已經 log）
.evaluation.payload.reflection_asymmetry.points[].frequency_hz | 1824 | 0 | src/aosr/scoring/channel_matching.py::_payload（由反射結果轉交；延遲四則、分貝已經 log）
.evaluation.payload.reflection_asymmetry.points[].left.delay_s | 1596 | 0 | src/aosr/scoring/channel_matching.py::_payload（由反射結果轉交；延遲四則、分貝已經 log）
.evaluation.payload.reflection_asymmetry.points[].left.level_db | 0 | 1596 | src/aosr/scoring/channel_matching.py::_payload（由反射結果轉交；延遲四則、分貝已經 log）
.evaluation.payload.reflection_asymmetry.points[].left_minus_right_db | 0 | 1596 | src/aosr/scoring/channel_matching.py::_payload（由反射結果轉交；延遲四則、分貝已經 log）
.evaluation.payload.reflection_asymmetry.points[].right.delay_s | 1596 | 0 | src/aosr/scoring/channel_matching.py::_payload（由反射結果轉交；延遲四則、分貝已經 log）
.evaluation.payload.reflection_asymmetry.points[].right.level_db | 0 | 1596 | src/aosr/scoring/channel_matching.py::_payload（由反射結果轉交；延遲四則、分貝已經 log）
.evaluation.payload.reflection_asymmetry.window_upper_ms | 2 | 0 | src/aosr/scoring/channel_matching.py::_payload（由反射結果轉交；延遲四則、分貝已經 log）
.evaluation.payload.wall_pairs[].bands[].decay_duration_s.value | 0 | 126 | src/aosr/scoring/reflections.py::_wall_pairs、src/aosr/scoring/reflections.py::_wall_band（延遲四則；衰減與分貝經 log10／陣列帶平均；頻率為鍵）
.evaluation.payload.wall_pairs[].bands[].frequency_hz | 126 | 0 | src/aosr/scoring/reflections.py::_wall_pairs、src/aosr/scoring/reflections.py::_wall_band（延遲四則；衰減與分貝經 log10／陣列帶平均；頻率為鍵）
.evaluation.payload.wall_pairs[].bands[].lower_hz | 126 | 0 | src/aosr/scoring/reflections.py::_wall_pairs、src/aosr/scoring/reflections.py::_wall_band（延遲四則；衰減與分貝經 log10／陣列帶平均；頻率為鍵）
.evaluation.payload.wall_pairs[].bands[].room_t20_s.value | 0 | 126 | src/aosr/scoring/reflections.py::_wall_pairs、src/aosr/scoring/reflections.py::_wall_band（延遲四則；衰減與分貝經 log10／陣列帶平均；頻率為鍵）
.evaluation.payload.wall_pairs[].bands[].round_trip_loss_db.value | 0 | 126 | src/aosr/scoring/reflections.py::_wall_pairs、src/aosr/scoring/reflections.py::_wall_band（延遲四則；衰減與分貝經 log10／陣列帶平均；頻率為鍵）
.evaluation.payload.wall_pairs[].bands[].upper_hz | 126 | 0 | src/aosr/scoring/reflections.py::_wall_pairs、src/aosr/scoring/reflections.py::_wall_band（延遲四則；衰減與分貝經 log10／陣列帶平均；頻率為鍵）
.evaluation.payload.wall_pairs[].round_trip_delay_s.value | 6 | 0 | src/aosr/scoring/reflections.py::_wall_pairs、src/aosr/scoring/reflections.py::_wall_band（延遲四則；衰減與分貝經 log10／陣列帶平均；頻率為鍵）
.evaluation.payload.window_upper_ms | 2 | 0 | src/aosr/scoring/reflections.py::evaluate_reflections、src/aosr/scoring/channel_matching.py::_payload（登記設定或其四則轉換）
.evaluation.payload.window_upper_s | 2 | 0 | src/aosr/scoring/reflections.py::evaluate_reflections、src/aosr/scoring/channel_matching.py::_payload（登記設定或其四則轉換）
.evaluation.payload.zone_limits.front_max_abs_azimuth_deg | 2 | 0 | src/aosr/scoring/direction_zones.py::zone_limits（登記輸入）
.evaluation.payload.zone_limits.rear_min_abs_azimuth_deg | 2 | 0 | src/aosr/scoring/direction_zones.py::zone_limits（登記輸入）
.evaluation.payload.zone_limits.vertical_min_abs_elevation_deg | 2 | 0 | src/aosr/scoring/direction_zones.py::zone_limits（登記輸入）
.evaluation.placement.receiver_positions_m[][][] | 36 | 0 | src/aosr/scoring/reflections.py::_placement（報表輸入座標；其餘評估也原樣轉交）
.evaluation.placement.speaker_positions_m[][][] | 42 | 0 | src/aosr/scoring/reflections.py::_placement（報表輸入座標；其餘評估也原樣轉交）
.evaluation.raw_quantities[].value | 14 | 82 | src/aosr/scoring/timbre.py::_raw_quantities; src/aosr/scoring/reverberation.py::_raw_quantities; src/aosr/scoring/reflections.py::evaluate_reflections; src/aosr/scoring/channel_matching.py::_raw_quantities; src/aosr/scoring/timbre_channels.py::_raw_quantities
.payload.adjacent_band_changes[].lower_center_frequency_hz | 12 | 0 | src/aosr/scoring/reverberation.py::_evaluate_band、src/aosr/scoring/reverberation.py::_band_range、src/aosr/scoring/reverberation.py::_adjacent_change（頻率鍵／sqrt 帶界；衰減來自陣列帶平均；log 比值；輸入底數）
.payload.adjacent_band_changes[].signed_log_ratio | 0 | 12 | src/aosr/scoring/reverberation.py::_evaluate_band、src/aosr/scoring/reverberation.py::_band_range、src/aosr/scoring/reverberation.py::_adjacent_change（頻率鍵／sqrt 帶界；衰減來自陣列帶平均；log 比值；輸入底數）
.payload.adjacent_band_changes[].upper_center_frequency_hz | 12 | 0 | src/aosr/scoring/reverberation.py::_evaluate_band、src/aosr/scoring/reverberation.py::_band_range、src/aosr/scoring/reverberation.py::_adjacent_change（頻率鍵／sqrt 帶界；衰減來自陣列帶平均；log 比值；輸入底數）
.payload.aggregates[].broadband_level_difference.weighted_mean_absolute_difference | 0 | 2 | src/aosr/scoring/channel_matching.py::_metric_aggregate（分貝傳播；直達時間來自控制組 math.dist，保守有關）
.payload.aggregates[].broadband_level_difference.worst_absolute_difference | 0 | 2 | src/aosr/scoring/channel_matching.py::_metric_aggregate（分貝傳播；直達時間來自控制組 math.dist，保守有關）
.payload.aggregates[].direct_time_difference.weighted_mean_absolute_difference | 0 | 2 | src/aosr/scoring/channel_matching.py::_metric_aggregate（分貝傳播；直達時間來自控制組 math.dist，保守有關）
.payload.aggregates[].direct_time_difference.worst_absolute_difference | 0 | 2 | src/aosr/scoring/channel_matching.py::_metric_aggregate（分貝傳播；直達時間來自控制組 math.dist，保守有關）
.payload.aggregates[].ripple_rms_difference.weighted_mean_absolute_difference | 0 | 2 | src/aosr/scoring/channel_matching.py::_metric_aggregate（分貝傳播；直達時間來自控制組 math.dist，保守有關）
.payload.aggregates[].ripple_rms_difference.worst_absolute_difference | 0 | 2 | src/aosr/scoring/channel_matching.py::_metric_aggregate（分貝傳播；直達時間來自控制組 math.dist，保守有關）
.payload.aggregates[].tilt_difference.weighted_mean_absolute_difference | 0 | 2 | src/aosr/scoring/channel_matching.py::_metric_aggregate（分貝傳播；直達時間來自控制組 math.dist，保守有關）
.payload.aggregates[].tilt_difference.worst_absolute_difference | 0 | 2 | src/aosr/scoring/channel_matching.py::_metric_aggregate（分貝傳播；直達時間來自控制組 math.dist，保守有關）
.payload.bands[].band_range_hz[] | 28 | 0 | src/aosr/scoring/reverberation.py::_evaluate_band、src/aosr/scoring/reverberation.py::_band_range、src/aosr/scoring/reverberation.py::_adjacent_change（頻率鍵／sqrt 帶界；衰減來自陣列帶平均；log 比值；輸入底數）
.payload.bands[].center_frequency_hz | 14 | 0 | src/aosr/scoring/reverberation.py::_evaluate_band、src/aosr/scoring/reverberation.py::_band_range、src/aosr/scoring/reverberation.py::_adjacent_change（頻率鍵／sqrt 帶界；衰減來自陣列帶平均；log 比值；輸入底數）
.payload.bands[].fitting_difference.value | 0 | 14 | src/aosr/scoring/reverberation.py::_evaluate_band、src/aosr/scoring/reverberation.py::_band_range、src/aosr/scoring/reverberation.py::_adjacent_change（頻率鍵／sqrt 帶界；衰減來自陣列帶平均；log 比值；輸入底數）
.payload.bands[].t20.value | 0 | 14 | src/aosr/scoring/reverberation.py::_evaluate_band、src/aosr/scoring/reverberation.py::_band_range、src/aosr/scoring/reverberation.py::_adjacent_change（頻率鍵／sqrt 帶界；衰減來自陣列帶平均；log 比值；輸入底數）
.payload.bands[].t30.value | 0 | 14 | src/aosr/scoring/reverberation.py::_evaluate_band、src/aosr/scoring/reverberation.py::_band_range、src/aosr/scoring/reverberation.py::_adjacent_change（頻率鍵／sqrt 帶界；衰減來自陣列帶平均；log 比值；輸入底數）
.payload.broadband_support.frequencies_hz[] | 416 | 0 | src/aosr/scoring/channel_matching.py::_broadband_support（軸轉交為鍵）
.payload.broadband_support.highest_frequency_hz | 2 | 0 | src/aosr/scoring/channel_matching.py::_broadband_support（軸轉交為鍵）
.payload.broadband_support.lowest_frequency_hz | 2 | 0 | src/aosr/scoring/channel_matching.py::_broadband_support（軸轉交為鍵）
.payload.channels[].payload.coverage_range_hz[] | 8 | 0 | src/aosr/scoring/timbre.py::evaluate_timbre、src/aosr/scoring/timbre.py::_dependency_ranges、src/aosr/scoring/timbre.py::_features（軸／範圍／目標為輸入；依賴範圍經非整數次方；量測經陣列 log／擬合）
.payload.channels[].payload.data_range_hz[] | 8 | 0 | src/aosr/scoring/timbre.py::evaluate_timbre、src/aosr/scoring/timbre.py::_dependency_ranges、src/aosr/scoring/timbre.py::_features（軸／範圍／目標為輸入；依賴範圍經非整數次方；量測經陣列 log／擬合）
.payload.channels[].payload.deviation_curve[][] | 832 | 832 | src/aosr/scoring/timbre.py::evaluate_timbre、src/aosr/scoring/timbre.py::_dependency_ranges、src/aosr/scoring/timbre.py::_features（軸／範圍／目標為輸入；依賴範圍經非整數次方；量測經陣列 log／擬合）
.payload.channels[].payload.features[].center_frequency_hz | 189 | 0 | src/aosr/scoring/timbre.py::evaluate_timbre、src/aosr/scoring/timbre.py::_dependency_ranges、src/aosr/scoring/timbre.py::_features（軸／範圍／目標為輸入；依賴範圍經非整數次方；量測經陣列 log／擬合）
.payload.channels[].payload.features[].depth_db | 0 | 189 | src/aosr/scoring/timbre.py::evaluate_timbre、src/aosr/scoring/timbre.py::_dependency_ranges、src/aosr/scoring/timbre.py::_features（軸／範圍／目標為輸入；依賴範圍經非整數次方；量測經陣列 log／擬合）
.payload.channels[].payload.features[].width_octave | 0 | 172 | src/aosr/scoring/timbre.py::evaluate_timbre、src/aosr/scoring/timbre.py::_dependency_ranges、src/aosr/scoring/timbre.py::_features（軸／範圍／目標為輸入；依賴範圍經非整數次方；量測經陣列 log／擬合）
.payload.channels[].payload.residual_rms_db | 0 | 4 | src/aosr/scoring/timbre.py::evaluate_timbre、src/aosr/scoring/timbre.py::_dependency_ranges、src/aosr/scoring/timbre.py::_features（軸／範圍／目標為輸入；依賴範圍經非整數次方；量測經陣列 log／擬合）
.payload.channels[].payload.ripple_dependency_range_hz[] | 0 | 8 | src/aosr/scoring/timbre.py::evaluate_timbre、src/aosr/scoring/timbre.py::_dependency_ranges、src/aosr/scoring/timbre.py::_features（軸／範圍／目標為輸入；依賴範圍經非整數次方；量測經陣列 log／擬合）
.payload.channels[].payload.ripple_range_hz[] | 8 | 0 | src/aosr/scoring/timbre.py::evaluate_timbre、src/aosr/scoring/timbre.py::_dependency_ranges、src/aosr/scoring/timbre.py::_features（軸／範圍／目標為輸入；依賴範圍經非整數次方；量測經陣列 log／擬合）
.payload.channels[].payload.target_deviation_rms_db | 0 | 4 | src/aosr/scoring/timbre.py::evaluate_timbre、src/aosr/scoring/timbre.py::_dependency_ranges、src/aosr/scoring/timbre.py::_features（軸／範圍／目標為輸入；依賴範圍經非整數次方；量測經陣列 log／擬合）
.payload.channels[].payload.target_tilt_db_per_octave | 4 | 0 | src/aosr/scoring/timbre.py::evaluate_timbre、src/aosr/scoring/timbre.py::_dependency_ranges、src/aosr/scoring/timbre.py::_features（軸／範圍／目標為輸入；依賴範圍經非整數次方；量測經陣列 log／擬合）
.payload.channels[].payload.tilt_db_per_octave | 0 | 4 | src/aosr/scoring/timbre.py::evaluate_timbre、src/aosr/scoring/timbre.py::_dependency_ranges、src/aosr/scoring/timbre.py::_features（軸／範圍／目標為輸入；依賴範圍經非整數次方；量測經陣列 log／擬合）
.payload.channels[].payload.tilt_dependency_range_hz[] | 0 | 8 | src/aosr/scoring/timbre.py::evaluate_timbre、src/aosr/scoring/timbre.py::_dependency_ranges、src/aosr/scoring/timbre.py::_features（軸／範圍／目標為輸入；依賴範圍經非整數次方；量測經陣列 log／擬合）
.payload.channels[].payload.tilt_fit_range_hz[] | 8 | 0 | src/aosr/scoring/timbre.py::evaluate_timbre、src/aosr/scoring/timbre.py::_dependency_ranges、src/aosr/scoring/timbre.py::_features（軸／範圍／目標為輸入；依賴範圍經非整數次方；量測經陣列 log／擬合）
.payload.channels[].reflections[].broadband_level_db | 0 | 514 | src/aosr/scoring/reflections.py::_paths（延遲四則；角度 atan2；分貝陣列 log）
.payload.channels[].reflections[].listening_azimuth_deg | 0 | 514 | src/aosr/scoring/reflections.py::_paths（延遲四則；角度 atan2；分貝陣列 log）
.payload.channels[].reflections[].listening_elevation_deg | 0 | 514 | src/aosr/scoring/reflections.py::_paths（延遲四則；角度 atan2；分貝陣列 log）
.payload.channels[].reflections[].relative_direct_delay_s | 514 | 0 | src/aosr/scoring/reflections.py::_paths（延遲四則；角度 atan2；分貝陣列 log）
.payload.channels[].reflections[].room_azimuth_deg | 0 | 514 | src/aosr/scoring/reflections.py::_paths（延遲四則；角度 atan2；分貝陣列 log）
.payload.channels[].reflections[].room_elevation_deg | 0 | 514 | src/aosr/scoring/reflections.py::_paths（延遲四則；角度 atan2；分貝陣列 log）
.payload.channels[].total_window_energy_db[].value | 0 | 912 | src/aosr/scoring/reflections.py::_channel（經 _sum_level 的 log10）
.payload.channels[].zones[].points[].frequency_hz | 3648 | 0 | src/aosr/scoring/reflections.py::_point（頻率鍵；延遲轉交；能量經 log10）
.payload.channels[].zones[].points[].strongest_delay_s | 3192 | 0 | src/aosr/scoring/reflections.py::_point（頻率鍵；延遲轉交；能量經 log10）
.payload.channels[].zones[].points[].strongest_level_db | 0 | 3192 | src/aosr/scoring/reflections.py::_point（頻率鍵；延遲轉交；能量經 log10）
.payload.channels[].zones[].points[].total_energy_db.value | 0 | 3192 | src/aosr/scoring/reflections.py::_point（頻率鍵；延遲轉交；能量經 log10）
.payload.frequency_range_hz[] | 4 | 0 | src/aosr/scoring/reflections.py::evaluate_reflections、src/aosr/scoring/channel_matching.py::_payload（登記設定或其四則轉換）
.payload.frequency_support.overall_level_frequencies_hz[] | 832 | 0 | src/aosr/scoring/listening_area.py::_payload（軸轉交為鍵）
.payload.frequency_support.points[].timbre_frequencies_hz[] | 1664 | 0 | src/aosr/scoring/listening_area.py::_payload（軸轉交為鍵）
.payload.listening_axis_xy[] | 0 | 4 | src/aosr/scoring/direction_zones.py::listening_axis（math.hypot 不保證正確捨入，保守有關）
.payload.logarithm_base | 2 | 0 | src/aosr/scoring/reverberation.py::_evaluate_band、src/aosr/scoring/reverberation.py::_band_range、src/aosr/scoring/reverberation.py::_adjacent_change（頻率鍵／sqrt 帶界；衰減來自陣列帶平均；log 比值；輸入底數）
.payload.overall_level_stability.primary_to_surrounding.weighted_mean_deviation | 0 | 4 | src/aosr/scoring/listening_area.py::_aggregate（分貝量傳播；端點 importance 為輸入）
.payload.overall_level_stability.primary_to_surrounding.worst_deviation.receiver.importance | 4 | 0 | src/aosr/scoring/listening_area.py::_aggregate（分貝量傳播；端點 importance 為輸入）
.payload.overall_level_stability.primary_to_surrounding.worst_deviation.reference.importance | 4 | 0 | src/aosr/scoring/listening_area.py::_aggregate（分貝量傳播；端點 importance 為輸入）
.payload.overall_level_stability.primary_to_surrounding.worst_deviation.value | 0 | 4 | src/aosr/scoring/listening_area.py::_aggregate（分貝量傳播；端點 importance 為輸入）
.payload.peak_dip_consistency.primary_to_surrounding.weighted_mean_deviation | 4 | 0 | src/aosr/scoring/listening_area.py::_feature_distance、src/aosr/scoring/listening_area.py::_aggregate（整數計數與四則；端點 importance 為輸入）
.payload.peak_dip_consistency.primary_to_surrounding.worst_deviation.receiver.importance | 4 | 0 | src/aosr/scoring/listening_area.py::_feature_distance、src/aosr/scoring/listening_area.py::_aggregate（整數計數與四則；端點 importance 為輸入）
.payload.peak_dip_consistency.primary_to_surrounding.worst_deviation.reference.importance | 4 | 0 | src/aosr/scoring/listening_area.py::_feature_distance、src/aosr/scoring/listening_area.py::_aggregate（整數計數與四則；端點 importance 為輸入）
.payload.peak_dip_consistency.primary_to_surrounding.worst_deviation.value | 4 | 0 | src/aosr/scoring/listening_area.py::_feature_distance、src/aosr/scoring/listening_area.py::_aggregate（整數計數與四則；端點 importance 為輸入）
.payload.peak_dip_occurrences[].primary_center_frequency_hz | 189 | 0 | src/aosr/scoring/listening_area.py::_peak_dip_occurrences（特徵中心頻率鍵）
.payload.point_results[].broadband_level_difference_db | 0 | 4 | src/aosr/scoring/channel_matching.py::_measured_point、src/aosr/scoring/channel_matching.py::_difference_curve（頻率鍵、輸入 importance；分貝／特徵已經陣列，時間來自 math.dist）
.payload.point_results[].direct_time_difference_ms | 0 | 4 | src/aosr/scoring/channel_matching.py::_measured_point、src/aosr/scoring/channel_matching.py::_difference_curve（頻率鍵、輸入 importance；分貝／特徵已經陣列，時間來自 math.dist）
.payload.point_results[].frequency_difference_curve_db[].frequency_hz | 880 | 0 | src/aosr/scoring/channel_matching.py::_measured_point、src/aosr/scoring/channel_matching.py::_difference_curve（頻率鍵、輸入 importance；分貝／特徵已經陣列，時間來自 math.dist）
.payload.point_results[].frequency_difference_curve_db[].left_minus_right_db | 0 | 880 | src/aosr/scoring/channel_matching.py::_measured_point、src/aosr/scoring/channel_matching.py::_difference_curve（頻率鍵、輸入 importance；分貝／特徵已經陣列，時間來自 math.dist）
.payload.point_results[].importance | 4 | 0 | src/aosr/scoring/channel_matching.py::_measured_point、src/aosr/scoring/channel_matching.py::_difference_curve（頻率鍵、輸入 importance；分貝／特徵已經陣列，時間來自 math.dist）
.payload.point_results[].ripple_rms_difference_db | 0 | 4 | src/aosr/scoring/channel_matching.py::_measured_point、src/aosr/scoring/channel_matching.py::_difference_curve（頻率鍵、輸入 importance；分貝／特徵已經陣列，時間來自 math.dist）
.payload.point_results[].tilt_difference_db_per_octave | 0 | 4 | src/aosr/scoring/channel_matching.py::_measured_point、src/aosr/scoring/channel_matching.py::_difference_curve（頻率鍵、輸入 importance；分貝／特徵已經陣列，時間來自 math.dist）
.payload.point_results[].unmatched_features[].feature.center_frequency_hz | 250 | 0 | src/aosr/scoring/channel_matching.py::_measured_point、src/aosr/scoring/channel_matching.py::_difference_curve（頻率鍵、輸入 importance；分貝／特徵已經陣列，時間來自 math.dist）
.payload.point_results[].unmatched_features[].feature.depth_db | 0 | 250 | src/aosr/scoring/channel_matching.py::_measured_point、src/aosr/scoring/channel_matching.py::_difference_curve（頻率鍵、輸入 importance；分貝／特徵已經陣列，時間來自 math.dist）
.payload.point_results[].unmatched_features[].feature.width_octave | 0 | 224 | src/aosr/scoring/channel_matching.py::_measured_point、src/aosr/scoring/channel_matching.py::_difference_curve（頻率鍵、輸入 importance；分貝／特徵已經陣列，時間來自 math.dist）
.payload.point_sources[].channels[].payload.coverage_range_hz[] | 16 | 0 | src/aosr/scoring/timbre.py::evaluate_timbre、src/aosr/scoring/timbre.py::_dependency_ranges、src/aosr/scoring/timbre.py::_features（軸／範圍／目標為輸入；依賴範圍經非整數次方；量測經陣列 log／擬合）
.payload.point_sources[].channels[].payload.data_range_hz[] | 16 | 0 | src/aosr/scoring/timbre.py::evaluate_timbre、src/aosr/scoring/timbre.py::_dependency_ranges、src/aosr/scoring/timbre.py::_features（軸／範圍／目標為輸入；依賴範圍經非整數次方；量測經陣列 log／擬合）
.payload.point_sources[].channels[].payload.deviation_curve[][] | 1664 | 1664 | src/aosr/scoring/timbre.py::evaluate_timbre、src/aosr/scoring/timbre.py::_dependency_ranges、src/aosr/scoring/timbre.py::_features（軸／範圍／目標為輸入；依賴範圍經非整數次方；量測經陣列 log／擬合）
.payload.point_sources[].channels[].payload.features[].center_frequency_hz | 382 | 0 | src/aosr/scoring/timbre.py::evaluate_timbre、src/aosr/scoring/timbre.py::_dependency_ranges、src/aosr/scoring/timbre.py::_features（軸／範圍／目標為輸入；依賴範圍經非整數次方；量測經陣列 log／擬合）
.payload.point_sources[].channels[].payload.features[].depth_db | 0 | 382 | src/aosr/scoring/timbre.py::evaluate_timbre、src/aosr/scoring/timbre.py::_dependency_ranges、src/aosr/scoring/timbre.py::_features（軸／範圍／目標為輸入；依賴範圍經非整數次方；量測經陣列 log／擬合）
.payload.point_sources[].channels[].payload.features[].width_octave | 0 | 349 | src/aosr/scoring/timbre.py::evaluate_timbre、src/aosr/scoring/timbre.py::_dependency_ranges、src/aosr/scoring/timbre.py::_features（軸／範圍／目標為輸入；依賴範圍經非整數次方；量測經陣列 log／擬合）
.payload.point_sources[].channels[].payload.residual_rms_db | 0 | 8 | src/aosr/scoring/timbre.py::evaluate_timbre、src/aosr/scoring/timbre.py::_dependency_ranges、src/aosr/scoring/timbre.py::_features（軸／範圍／目標為輸入；依賴範圍經非整數次方；量測經陣列 log／擬合）
.payload.point_sources[].channels[].payload.ripple_dependency_range_hz[] | 0 | 16 | src/aosr/scoring/timbre.py::evaluate_timbre、src/aosr/scoring/timbre.py::_dependency_ranges、src/aosr/scoring/timbre.py::_features（軸／範圍／目標為輸入；依賴範圍經非整數次方；量測經陣列 log／擬合）
.payload.point_sources[].channels[].payload.ripple_range_hz[] | 16 | 0 | src/aosr/scoring/timbre.py::evaluate_timbre、src/aosr/scoring/timbre.py::_dependency_ranges、src/aosr/scoring/timbre.py::_features（軸／範圍／目標為輸入；依賴範圍經非整數次方；量測經陣列 log／擬合）
.payload.point_sources[].channels[].payload.target_deviation_rms_db | 0 | 8 | src/aosr/scoring/timbre.py::evaluate_timbre、src/aosr/scoring/timbre.py::_dependency_ranges、src/aosr/scoring/timbre.py::_features（軸／範圍／目標為輸入；依賴範圍經非整數次方；量測經陣列 log／擬合）
.payload.point_sources[].channels[].payload.target_tilt_db_per_octave | 8 | 0 | src/aosr/scoring/timbre.py::evaluate_timbre、src/aosr/scoring/timbre.py::_dependency_ranges、src/aosr/scoring/timbre.py::_features（軸／範圍／目標為輸入；依賴範圍經非整數次方；量測經陣列 log／擬合）
.payload.point_sources[].channels[].payload.tilt_db_per_octave | 0 | 8 | src/aosr/scoring/timbre.py::evaluate_timbre、src/aosr/scoring/timbre.py::_dependency_ranges、src/aosr/scoring/timbre.py::_features（軸／範圍／目標為輸入；依賴範圍經非整數次方；量測經陣列 log／擬合）
.payload.point_sources[].channels[].payload.tilt_dependency_range_hz[] | 0 | 16 | src/aosr/scoring/timbre.py::evaluate_timbre、src/aosr/scoring/timbre.py::_dependency_ranges、src/aosr/scoring/timbre.py::_features（軸／範圍／目標為輸入；依賴範圍經非整數次方；量測經陣列 log／擬合）
.payload.point_sources[].channels[].payload.tilt_fit_range_hz[] | 16 | 0 | src/aosr/scoring/timbre.py::evaluate_timbre、src/aosr/scoring/timbre.py::_dependency_ranges、src/aosr/scoring/timbre.py::_features（軸／範圍／目標為輸入；依賴範圍經非整數次方；量測經陣列 log／擬合）
.payload.point_sources[].channels[].raw_quantities[].value | 8 | 24 | src/aosr/scoring/timbre.py::_raw_quantities; src/aosr/scoring/reverberation.py::_raw_quantities; src/aosr/scoring/reflections.py::evaluate_reflections; src/aosr/scoring/channel_matching.py::_raw_quantities; src/aosr/scoring/timbre_channels.py::_raw_quantities
.payload.reflection_asymmetry.frequency_range_hz[] | 4 | 0 | src/aosr/scoring/channel_matching.py::_payload（由反射結果轉交；延遲四則、分貝已經 log）
.payload.reflection_asymmetry.points[].frequency_hz | 1824 | 0 | src/aosr/scoring/channel_matching.py::_payload（由反射結果轉交；延遲四則、分貝已經 log）
.payload.reflection_asymmetry.points[].left.delay_s | 1596 | 0 | src/aosr/scoring/channel_matching.py::_payload（由反射結果轉交；延遲四則、分貝已經 log）
.payload.reflection_asymmetry.points[].left.level_db | 0 | 1596 | src/aosr/scoring/channel_matching.py::_payload（由反射結果轉交；延遲四則、分貝已經 log）
.payload.reflection_asymmetry.points[].left_minus_right_db | 0 | 1596 | src/aosr/scoring/channel_matching.py::_payload（由反射結果轉交；延遲四則、分貝已經 log）
.payload.reflection_asymmetry.points[].right.delay_s | 1596 | 0 | src/aosr/scoring/channel_matching.py::_payload（由反射結果轉交；延遲四則、分貝已經 log）
.payload.reflection_asymmetry.points[].right.level_db | 0 | 1596 | src/aosr/scoring/channel_matching.py::_payload（由反射結果轉交；延遲四則、分貝已經 log）
.payload.reflection_asymmetry.window_upper_ms | 2 | 0 | src/aosr/scoring/channel_matching.py::_payload（由反射結果轉交；延遲四則、分貝已經 log）
.payload.ripple_rms_stability.primary_to_surrounding.weighted_mean_deviation | 0 | 4 | src/aosr/scoring/listening_area.py::_aggregate（分貝量傳播；端點 importance 為輸入）
.payload.ripple_rms_stability.primary_to_surrounding.worst_deviation.receiver.importance | 4 | 0 | src/aosr/scoring/listening_area.py::_aggregate（分貝量傳播；端點 importance 為輸入）
.payload.ripple_rms_stability.primary_to_surrounding.worst_deviation.reference.importance | 4 | 0 | src/aosr/scoring/listening_area.py::_aggregate（分貝量傳播；端點 importance 為輸入）
.payload.ripple_rms_stability.primary_to_surrounding.worst_deviation.value | 0 | 4 | src/aosr/scoring/listening_area.py::_aggregate（分貝量傳播；端點 importance 為輸入）
.payload.target_deviation_position_spread_curve_db[].frequency_hz | 832 | 0 | src/aosr/scoring/listening_area.py::_diagnostic_curve（頻率鍵；分貝經陣列計算）
.payload.target_deviation_position_spread_curve_db[].spread_db | 0 | 832 | src/aosr/scoring/listening_area.py::_diagnostic_curve（頻率鍵；分貝經陣列計算）
.payload.tilt_stability.primary_to_surrounding.weighted_mean_deviation | 0 | 4 | src/aosr/scoring/listening_area.py::_aggregate（分貝量傳播；端點 importance 為輸入）
.payload.tilt_stability.primary_to_surrounding.worst_deviation.receiver.importance | 4 | 0 | src/aosr/scoring/listening_area.py::_aggregate（分貝量傳播；端點 importance 為輸入）
.payload.tilt_stability.primary_to_surrounding.worst_deviation.reference.importance | 4 | 0 | src/aosr/scoring/listening_area.py::_aggregate（分貝量傳播；端點 importance 為輸入）
.payload.tilt_stability.primary_to_surrounding.worst_deviation.value | 0 | 4 | src/aosr/scoring/listening_area.py::_aggregate（分貝量傳播；端點 importance 為輸入）
.payload.wall_pairs[].bands[].decay_duration_s.value | 0 | 126 | src/aosr/scoring/reflections.py::_wall_pairs、src/aosr/scoring/reflections.py::_wall_band（延遲四則；衰減與分貝經 log10／陣列帶平均；頻率為鍵）
.payload.wall_pairs[].bands[].frequency_hz | 126 | 0 | src/aosr/scoring/reflections.py::_wall_pairs、src/aosr/scoring/reflections.py::_wall_band（延遲四則；衰減與分貝經 log10／陣列帶平均；頻率為鍵）
.payload.wall_pairs[].bands[].lower_hz | 126 | 0 | src/aosr/scoring/reflections.py::_wall_pairs、src/aosr/scoring/reflections.py::_wall_band（延遲四則；衰減與分貝經 log10／陣列帶平均；頻率為鍵）
.payload.wall_pairs[].bands[].room_t20_s.value | 0 | 126 | src/aosr/scoring/reflections.py::_wall_pairs、src/aosr/scoring/reflections.py::_wall_band（延遲四則；衰減與分貝經 log10／陣列帶平均；頻率為鍵）
.payload.wall_pairs[].bands[].round_trip_loss_db.value | 0 | 126 | src/aosr/scoring/reflections.py::_wall_pairs、src/aosr/scoring/reflections.py::_wall_band（延遲四則；衰減與分貝經 log10／陣列帶平均；頻率為鍵）
.payload.wall_pairs[].bands[].upper_hz | 126 | 0 | src/aosr/scoring/reflections.py::_wall_pairs、src/aosr/scoring/reflections.py::_wall_band（延遲四則；衰減與分貝經 log10／陣列帶平均；頻率為鍵）
.payload.wall_pairs[].round_trip_delay_s.value | 6 | 0 | src/aosr/scoring/reflections.py::_wall_pairs、src/aosr/scoring/reflections.py::_wall_band（延遲四則；衰減與分貝經 log10／陣列帶平均；頻率為鍵）
.payload.window_upper_ms | 2 | 0 | src/aosr/scoring/reflections.py::evaluate_reflections、src/aosr/scoring/channel_matching.py::_payload（登記設定或其四則轉換）
.payload.window_upper_s | 2 | 0 | src/aosr/scoring/reflections.py::evaluate_reflections、src/aosr/scoring/channel_matching.py::_payload（登記設定或其四則轉換）
.payload.zone_limits.front_max_abs_azimuth_deg | 2 | 0 | src/aosr/scoring/direction_zones.py::zone_limits（登記輸入）
.payload.zone_limits.rear_min_abs_azimuth_deg | 2 | 0 | src/aosr/scoring/direction_zones.py::zone_limits（登記輸入）
.payload.zone_limits.vertical_min_abs_elevation_deg | 2 | 0 | src/aosr/scoring/direction_zones.py::zone_limits（登記輸入）
.placement.receiver_positions_m[][][] | 36 | 0 | src/aosr/scoring/reflections.py::_placement（報表輸入座標；其餘評估也原樣轉交）
.placement.speaker_positions_m[][][] | 42 | 0 | src/aosr/scoring/reflections.py::_placement（報表輸入座標；其餘評估也原樣轉交）
.raw_quantities[].value | 14 | 82 | src/aosr/scoring/timbre.py::_raw_quantities; src/aosr/scoring/reverberation.py::_raw_quantities; src/aosr/scoring/reflections.py::evaluate_reflections; src/aosr/scoring/channel_matching.py::_raw_quantities; src/aosr/scoring/timbre_channels.py::_raw_quantities
.review_alerts[].center_frequency_hz | 60 | 0 | src/aosr/scoring/timbre_cost.py::_payload_alerts、src/aosr/scoring/reflections_cost.py::_flutter_alert（量測值／頻率鍵／登記限值）
.review_alerts[].decay_db | 2 | 0 | src/aosr/scoring/timbre_cost.py::_payload_alerts、src/aosr/scoring/reflections_cost.py::_flutter_alert（量測值／頻率鍵／登記限值）
.review_alerts[].decay_duration_s | 0 | 2 | src/aosr/scoring/timbre_cost.py::_payload_alerts、src/aosr/scoring/reflections_cost.py::_flutter_alert（量測值／頻率鍵／登記限值）
.review_alerts[].depth_db | 0 | 58 | src/aosr/scoring/timbre_cost.py::_payload_alerts、src/aosr/scoring/reflections_cost.py::_flutter_alert（量測值／頻率鍵／登記限值）
.review_alerts[].limit_db | 58 | 0 | src/aosr/scoring/timbre_cost.py::_payload_alerts、src/aosr/scoring/reflections_cost.py::_flutter_alert（量測值／頻率鍵／登記限值）
.review_alerts[].lower_hz | 2 | 0 | src/aosr/scoring/timbre_cost.py::_payload_alerts、src/aosr/scoring/reflections_cost.py::_flutter_alert（量測值／頻率鍵／登記限值）
.review_alerts[].room_t20_s | 0 | 2 | src/aosr/scoring/timbre_cost.py::_payload_alerts、src/aosr/scoring/reflections_cost.py::_flutter_alert（量測值／頻率鍵／登記限值）
.review_alerts[].upper_hz | 2 | 0 | src/aosr/scoring/timbre_cost.py::_payload_alerts、src/aosr/scoring/reflections_cost.py::_flutter_alert（量測值／頻率鍵／登記限值）
.review_alerts[].width_octave | 0 | 53 | src/aosr/scoring/timbre_cost.py::_payload_alerts、src/aosr/scoring/reflections_cost.py::_flutter_alert（量測值／頻率鍵／登記限值）
.weighted_cost | 0 | 8 | src/aosr/scoring/ranking.py::_settle、src/aosr/scoring/ranking.py::_category_line（傳播六類評估代價）
"""
from __future__ import annotations

import pytest

from tests.engine import _scoring_source_model_answers as answers
from tests.engine import _scoring_source_model_control as control
from tests.engine._report_cache import control_pair_cache
from tests.engine._cross_cpu_tolerance import SCORING_REL, SCORING_ABS, cell, assert_within



def _hex_leaves(node: object, path: str = "") -> list[tuple[str, str]]:
    if isinstance(node, dict):
        return [leaf for key in sorted(node) for leaf in _hex_leaves(node[key], f"{path}.{key}")]
    assert isinstance(node, str), path
    return [(path, node)]


def _assert_headline(found: dict[str, object]) -> None:
    got, expected = _hex_leaves(found), _hex_leaves(answers.HEADLINE)
    assert [path for path, _ in got] == [path for path, _ in expected]
    assert_within(cell(f"HEADLINE{path}", value, answer, rel=SCORING_REL, abs_=SCORING_ABS)
                  for (path, value), (_, answer) in zip(got, expected, strict=True))


def _assert_section(name: str, found: dict[str, object], expected: dict[str, object]) -> None:
    assert found["exact"] == expected["exact"], f"{name}：骨架（路徑、順序、型別、鍵、輸入與判定）變了"
    values, answer_values = found["floats"], expected["floats"]
    assert isinstance(values, list) and isinstance(answer_values, list)
    assert len(values) == len(answer_values), name
    if values:
        assert_within(cell(f"{name}{path}", value, answer, rel=SCORING_REL, abs_=SCORING_ABS)
                      for (path, value), answer in zip(values, answer_values, strict=True))


def test_omnidirectional_scoring_and_ranking_match_main(monkeypatch: pytest.MonkeyPatch) -> None:
    """全向兩候選：骨架逐字相同；機器相關浮點依葉子順序逐格在評分層③級內。
    每一格各用答案的尺，不受其他格大小、分組權重或抵銷影響；鍵、型別、位置、順序與判定仍逐字。
    不證明跨處理器逐位相同；SUMMARY 逐字，HEADLINE 逐格比③級（原為手算答案，#668 起改錄產品在新登記簿下算出的值，出身見答案檔）。
    """
    for module, name, stand_in in control.STAND_INS:
        monkeypatch.setattr(module, name, stand_in)
    candidates, ranking = control.run()

    assert control.summary(candidates, ranking) == answers.SUMMARY
    _assert_headline(control.headline(ranking))
    found = control.sections(candidates, ranking)
    assert found.keys() == answers.SECTIONS.keys()
    for name, section in found.items():
        expected = answers.SECTIONS[name]
        assert isinstance(expected, dict)
        _assert_section(name, section, expected)


@pytest.mark.parametrize("document", [
    {"payload": {"deviation_curve": [[20.0, 2.0]]}},
    {"payload": {"channels": [{"zones": [{"points": [{"strongest_delay_s": 0.02}]}]}]}},
    {"raw_quantities": [{"name": "band_count", "value": 7.0}]},
])
def test_keys_and_arithmetic_leaves_stay_in_skeleton(document: dict[str, object]) -> None:
    """防止頻率列、幾何延遲與計數誤進容差；其餘曲線分貝仍要有逐格答案。"""
    found = control.flatten(document, {})
    values = found["floats"]
    assert isinstance(values, list)
    if "raw_quantities" in document:
        assert values == []
    else:
        payload = document["payload"]
        assert isinstance(payload, dict)
        if "deviation_curve" in payload:
            assert values == [(".payload.deviation_curve[0][1]", 2.0.hex())]
        else:
            assert values == []


def test_unknown_float_keeps_its_position_and_type() -> None:
    """未分類浮點保守逐格；變整數或少一葉子必須改骨架。"""
    found = control.flatten({"new_metric": [1.25]}, {})
    assert found["floats"] == [(".new_metric[0]", 1.25.hex())]
    assert found["exact"] != control.flatten({"new_metric": [1]}, {})["exact"]
    assert found["exact"] != control.flatten({"new_metric": []}, {})["exact"]
    assert control.flatten({"new_metric": {"delay_s": 1.25}}, {})["floats"] == [
        (".new_metric.delay_s", 1.25.hex()),
    ]
    assert control.flatten({"payload": {"peak_dip_consistency": {"depth_db": 1.25}}}, {})["floats"] == [
        (".payload.peak_dip_consistency.depth_db", 1.25.hex()),
    ]


def test_section_judge_compares_skeleton_and_every_cell() -> None:
    """裁判本身要真的比：骨架雜湊不同、或任一格超出③，逐格其餘全等也紅；③之內綠。"""
    found = control.flatten({"payload": {"level_db": [-3.0, 0.5]}}, {})
    cells = found["floats"]
    assert isinstance(cells, list)
    answer = {"exact": found["exact"], "floats": [value for _, value in cells]}
    _assert_section("自製段", found, answer)
    _assert_section("自製段", found, {**answer, "floats": [(-3.0 + 1e-12).hex(), 0.5.hex()]})
    with pytest.raises(AssertionError, match="超出跨處理器容差"):
        _assert_section("自製段", found, {**answer, "floats": [(-3.0 + 1e-10).hex(), 0.5.hex()]})
    with pytest.raises(AssertionError, match="骨架"):
        _assert_section("自製段", found, {**answer, "exact": "0" * 64})
