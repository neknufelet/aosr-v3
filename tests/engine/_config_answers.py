"""讀標準答案檔、把答案檔裡的記號還原成 Python 值的共用零件。

這一支不是考卷（檔名不以 ``test_`` 開頭），是考卷共用的兩件事：

1. **答案檔在哪。** ``tests/engine/answers/config_cut1.json``，由
   ``blueprint/generate_config_cut1_answers.py`` 在唯讀的 v2 工作樹上跑出來（票 #127 第 1 刀
   的那 11 支模組，v2 的考卷一支都帶不走——每一支都還 import 了別的層或搬走的模組，
   所以裁判改由這一份標準答案接手）。**人不碰裡面的數字**：要改就重跑產生器。
   答案檔存的是值本身，不是「怎麼寫的」——tuple 與 list 在檔案裡是同一種 list 記號
   （這一刀的合約是數值一致、結構隨便改）。真的需要分辨那兩種寫法的地方只有一處：
   pydantic 的訊息會把 ``input_value=(nan,)`` 與 ``input_value=[nan]`` 講成兩句，
   所以 ``validate_message`` 只比它認定的問題種類，不比輸入值那一格。

2. **記號怎麼還原。** 產生器把值編成帶型別的小記號（浮點存 ``float.hex()``——那是精確的、
   可以逐位元還原的寫法，十進位字串會漂）。這一支把那些記號還原回來。

還原時**逐位元比對**（``float.fromhex`` 之後用 ``==``）：這一刀的合約是「行為與數值跟上一代
一致」，不是「差不多」。``4*math.pi`` 這種「算出來的」值，答案檔另外記了運算式
（``expr``），這一支當場算一次並確認那個記號算出來真的等於它——不然答案檔寫著
``4*math.pi`` 而值是別的東西，比對會比心酸的。

**A/B/C 表（逐符號）。** 判定的單位是**公開符號**（公開函式／類別／全大寫常數），11 支加起來 88 個；`SplOutputConfig` 的兩個屬性（`l_ref_db`／`is_relative`）是額外列出來的，因為它們各有自己的判定與裁判。證據那一欄寫的是**上一代（v2）**的考卷檔名——**行號沒有照抄**：那些考卷這一刀都帶不走，抄行號只會過期。

每一列的「處置」欄是這一刀真的做的事；「覆蓋方式」欄寫那一筆是用什麼蓋的，
「殘餘風險」欄具名寫出那一筆蓋不到什麼（只靠標準答案的符號一律要寫）。

### art_lane

| 符號 | 判定 | 證據（v2 考卷：行） | 處置 | 覆蓋方式 | 殘餘風險 |
|---|---|---|---|---|---|
| ART_N_PER_WALL_DEFAULT | B | 8 支考卷 import 這一支，但這一個常數只被當輸入（`test_art_polygon`／`test_art_kernel`／`test_p4_4_blend` 等） | donor 標準答案 | donor 標準答案（1 個凍結值） | 只比凍結值，沒驗它在 ART dispatch 裡怎麼被用（那些考卷綁 physics／materials，帶不走） |
| ART_FACE_GRID_N | B | 同上，只當輸入 | donor 標準答案 | donor 標準答案（1 個凍結值） | 同上；另外 v2 的 `test_art_polygon_coarse` 驗的是粗格 ART 的收斂行為，那一支沒帶過來 |
| ART_NEUMANN_EPS_TAIL | B | 只當輸入（Neumann 尾段） | donor 標準答案 | donor 標準答案（1 個凍結值） | 沒驗它與 `ART_NEUMANN_K_MAX` 的搭配是否仍收斂（那要跑 kernel） |
| ART_POWERITER_K | B | 只當輸入（冪迭代次數） | donor 標準答案 | donor 標準答案（1 個凍結值） | 沒驗收斂品質；v2 也沒有考卷直接驗它 |
| ART_FORMFACTOR_EPS | B | 只當輸入 | donor 標準答案 | donor 標準答案（1 個凍結值） | 沒驗它對形狀因子數值的影響 |
| ART_RECIPROCITY_TOL | B | 只當輸入（互易性容忍） | donor 標準答案 | donor 標準答案（1 個凍結值） | 沒驗互易性檢查真的用它（那在 kernel 裡） |
| ART_P_CAP | A | `test_art_polygon` 寫死 `ART_P_CAP=8192`（訊息裡） | donor 標準答案 ＋ 補的新考卷 | 兩者（v2 考卷那一條斷言沒有帶走，改用補的邊界考卷 ＋ donor 標準答案） | v2 那一條是在真 ART dispatch 上驗的；補的考卷只驗 6·n² 與上限的交界（36／37），沒驗記憶體估算那一段訊息 |
| ART_NEUMANN_K_MAX | B | 只當輸入 | donor 標準答案 | donor 標準答案（1 個凍結值） | 沒驗它真的限制了 `lax.scan` 的長度 |
| ART_WLS_T20_HI_DB | B | 只當輸入（T20 視窗） | donor 標準答案 | donor 標準答案（1 個凍結值） | 沒驗 WLS 擬合出來的值；v2 的 VAL-F6 考卷綁 scoring／physics |
| ART_WLS_T20_LO_DB | B | 同上 | donor 標準答案 | donor 標準答案（1 個凍結值） | 同上 |
| ART_WLS_WINDOW_SOFTNESS_DB | B | 同上 | donor 標準答案 | donor 標準答案（1 個凍結值） | 同上 |
| guard_art_patch_count | C | v2 零考卷呼叫（只有生產碼用；姊妹函式有考卷） | 補的新考卷 | 補的新考卷（14 個輸入點：負值／1／6／36／37／116，含 `context` 覆寫） | 補的考卷驗的是這一支自己的行為；v2 那一條 `test_art_polygon` 驗的是它在陣列路徑上的效果（`match="P=8193 exceeds ART_P_CAP=8192"`），那一條綁 physics，沒帶走 |
| guard_polygon_art_patch_count | A | `test_art_polygon` 寫死 `match="P=8193 exceeds ART_P_CAP=8192"` | donor 標準答案 ＋ 補的新考卷 | 兩者（補的新考卷 6 個輸入點 ＋ donor 標準答案） | v2 那一條的 `P=8193` 情境是在真 dispatch 上；補的考卷直接餵 `n_tris`，沒驗那個訊息在真實陣列路徑上仍一樣 |

### art_rt_guard

| 符號 | 判定 | 證據（v2 考卷：行） | 處置 | 覆蓋方式 | 殘餘風險 |
|---|---|---|---|---|---|
| ART_RT_T_CAP_S | A | `test_scoring_phase0_art_rt_production_switch` 寫死 `== 8.0` | donor 標準答案 | donor 標準答案（1 個凍結值） | v2 那一條考卷同時驗 log-compressor 的效果（`t_cap + s·log1p(...)`）；那一段沒帶走，只驗了常數 |
| ART_RT_GUARD_SMOOTHNESS_S | A | 同上，`== 1.0` | donor 標準答案 | donor 標準答案（1 個凍結值） | 同上：軟化係數的效果沒驗 |
| RENDER_DEFAULT_SAMPLE_RATE_HZ | A | 同上，`== 48_000`（另外用它算 cutoff 驗 band 遮罩） | donor 標準答案 | donor 標準答案（1 個凍結值） | band 遮罩那一段（哪些 art_hf 頻帶被排除）沒帶走；那要頻率軸與 scoring 的層 |

### authoring_defaults

| 符號 | 判定 | 證據（v2 考卷：行） | 處置 | 覆蓋方式 | 殘餘風險 |
|---|---|---|---|---|---|
| APP_WALL_PREFIX | B | 17 支考卷用到這一支，只當輸入 | donor 標準答案 ＋ 補的新考卷（接線） | 兩者（補的接線考卷 ＋ donor 標準答案） | 接線只驗「助手用的是這個常數」；v2 沒有考卷驗這個前綴本身 |
| APP_PATCH_PREFIX | B | 同上 | donor 標準答案 ＋ 補的新考卷（接線） | 兩者 | 同上 |
| RHINO_PREFIX | B | 同上 | donor 標準答案 ＋ 補的新考卷（接線） | 兩者 | 同上 |
| GRID_PRESETS | A | `test_m13_authoring` 寫死 `GRID_PRESETS["coarse"]==(2,3)` | donor 標準答案 | donor 標準答案（整張表逐鍵比） | `test_m13_authoring` 還驗它與 materials／geometry 的互動，沒帶走；只驗了表本身 |
| TOTAL_THICKNESS_MAX_M | A | 同上，`total_thickness_max_m==0.20` | donor 標準答案 | donor 標準答案（1 個凍結值） | 沒驗那個上限被誰用到（materials 那一層） |
| BARE_MATERIAL_ID | B | 只當輸入 | donor 標準答案 | donor 標準答案（1 個凍結值） | 沒驗它與材料載入器的對應 |
| BARE_TEMPLATE_ID | B | 只當輸入 | donor 標準答案 | donor 標準答案（1 個凍結值） | 同上 |
| BROADBAND_POROUS_TEMPLATE_ID | B | 只當輸入 | donor 標準答案 | donor 標準答案（1 個凍結值） | 同上 |
| RESONANT_PANEL_TEMPLATE_ID | B | 只當輸入 | donor 標準答案 | donor 標準答案（1 個凍結值） | 同上 |
| CUSTOM_STACK_TEMPLATE_ID | B | 只當輸入 | donor 標準答案 | donor 標準答案（1 個凍結值） | 同上 |
| FIXED_ALPHA_TEMPLATE_ID | B | 只當輸入（`test_p6_fixed_alpha` 用它） | donor 標準答案 | donor 標準答案（1 個凍結值） | `test_p6_fixed_alpha` 驗的是固定 α 材料真的不走優化器，綁 materials／physics |
| FLOW_RESISTIVITY_MIN | B | 只當輸入（穿孔板的參數範圍） | donor 標準答案 | donor 標準答案（1 個凍結值） | 沒驗驗證器真的用它（那在 materials 的 schema 裡） |
| FLOW_RESISTIVITY_MAX | B | 同上 | donor 標準答案 | donor 標準答案（1 個凍結值） | 同上 |
| SPACING_MIN_M | B | 同上 | donor 標準答案 | donor 標準答案（1 個凍結值） | 同上 |
| SPACING_MAX_M | B | 同上 | donor 標準答案 | donor 標準答案（1 個凍結值） | 同上 |
| HOLE_RATIO_MIN | B | 同上 | donor 標準答案 | donor 標準答案（1 個凍結值） | 同上 |
| HOLE_RATIO_MAX | B | 同上 | donor 標準答案 | donor 標準答案（1 個凍結值） | 同上 |
| PERFORATED_FIXED_THICKNESS_M | B | 只當輸入 | donor 標準答案 | donor 標準答案（1 個凍結值） | 沒驗它與 `RESONANT_PANEL_LAYER_STACK_SPEC` 裡那 9.0mm 的一致性 |
| FABRIC_RS | B | 只當輸入（布料的流阻） | donor 標準答案 | donor 標準答案（1 個凍結值） | 同上 |
| FABRIC_MS | B | 只當輸入（布料的面密度） | donor 標準答案 | donor 標準答案（1 個凍結值） | 同上 |
| BARE_LAYER_STACK_SPEC | B | 只當輸入（空堆疊） | donor 標準答案 | donor 標準答案（1 個凍結值） | 沒驗它被組進材料程式時的行為 |
| BROADBAND_POROUS_LAYER_STACK_SPEC | B | 只當輸入 | donor 標準答案 | donor 標準答案（逐鍵逐值） | 同上；另外沒驗那份 TOML 的 `flow_resistivity` 與 `FABRIC_RS` 是不是同一個來源 |
| RESONANT_PANEL_LAYER_STACK_SPEC | B | 只當輸入 | donor 標準答案 | donor 標準答案（逐鍵逐值） | 同上 |
| app_wall_id | B | `test_m13_authoring` 等 17 支只把它當輸入（拿字串再塞別處） | 補的新考卷 ＋ donor 標準答案 | 兩者（3 個輸入點 ＋ donor 標準答案） | 補的考卷驗的是「同一組輸入 → 同一個字串」；v2 考卷驗的是那些 id 進到 materials 之後的對應（帶不走） |
| app_patch_id | B | 同上 | 補的新考卷 ＋ donor 標準答案 | 兩者（3 個輸入點 ＋ donor 標準答案） | 同上 |
| rhino_boundary_id | B | 同上 | 補的新考卷 ＋ donor 標準答案 | 兩者（3 個輸入點 ＋ donor 標準答案） | 同上 |

### crossover_axis

| 符號 | 判定 | 證據（v2 考卷：行） | 處置 | 覆蓋方式 | 殘餘風險 |
|---|---|---|---|---|---|
| DEFAULT_T_C_CENTER_S | A | `test_engine2_crossover_axis` 寫死 `t_c_center=0.08` | donor 標準答案 ＋ 補的新考卷（接線） | 兩者 | v2 那一條是驗它流進 render 參數之後的值；那一段綁 dsp／physics，沒帶走 |
| DEFAULT_T_C_FADE_S | A | 同上，`t_c_fade=0.02` | donor 標準答案 ＋ 補的新考卷（接線） | 兩者 | 同上 |
| CanonicalCrossover | B | 考卷拿它的實例再往下傳（`test_engine2_late_field` 等） | donor 標準答案 | donor 標準答案（4 個輸入點，逐欄比對） | 它是 frozen dataclass：補的考卷沒驗 `frozen`（賦值要丟例外）；v2 也沒有那一條 |
| build_canonical_crossover | A | `test_engine2_crossover_axis` 寫死 `seam_f_s≈321`、center／fade 一致 | 補的新考卷 ＋ donor 標準答案 | 兩者（4 個輸入點 ＋ donor 標準答案） | v2 那一條驗的是它在真 auralize／bridge 路徑上的接縫；那一段沒帶走，只驗了建構出來的三個欄位 |

### default_geometry

| 符號 | 判定 | 證據（v2 考卷：行） | 處置 | 覆蓋方式 | 殘餘風險 |
|---|---|---|---|---|---|
| DEFAULT_DIMS_M | B | `test_m12_crossover_2way_e2e` 當輸入（註解寫 `(5.5, 4.2, 2.8)`，沒有 assert） | donor 標準答案 | donor 標準答案（1 個凍結值） | v2 那一支考卷綁 `freq_axis`／materials／physics／scoring；它驗的是那組幾何跑完 e2e 的結果，沒帶走 |
| DEFAULT_SOURCE_XYZ | B | 同上，只當種子座標 | donor 標準答案 | donor 標準答案（1 個凍結值） | 沒驗它真的餵進哪個求解器 |
| DEFAULT_RECEIVER_XYZ | B | 同上 | donor 標準答案 | donor 標準答案（1 個凍結值） | 同上 |
| DEFAULT_EAR_HEIGHT_M | B | 只當輸入 | donor 標準答案 | donor 標準答案（1 個凍結值） | 它與 `receiver_grid.EAR_HEIGHT_M` 同值（1.2）但沒有考卷驗那個一致；v2 也沒有 |

### ism_lane

| 符號 | 判定 | 證據（v2 考卷：行） | 處置 | 覆蓋方式 | 殘餘風險 |
|---|---|---|---|---|---|
| ISM_FACE_GRID_N | B | `test_pmo7_di3_snapped_parity` 間接使用 | donor 標準答案 | donor 標準答案（1 個凍結值） | 那一支驗的是 DI-3 對稱性收斂（綁 geometry／materials／physics）；「N=16 讓誤差 ≤ 5e-2」那個結論沒帶走 |
| ISM_FACE_GRID_SAMPLES | B | 同上 | donor 標準答案 | donor 標準答案（1 個凍結值） | 同上 |

### phase2_report_bands

| 符號 | 判定 | 證據（v2 考卷：行） | 處置 | 覆蓋方式 | 殘餘風險 |
|---|---|---|---|---|---|
| PHASE2_RFZ_THRESHOLD_DB | C | v2 零考卷 import；消費者那一支考卷把 `-6.0` 抄在測試裡 | donor 標準答案 | donor 標準答案（1 個凍結值） | **最弱的一格**：到 reporting 那一塊長出來之前，沒有任何考卷驗這 5 個常數被誰用；v2 的 `test_reporting_phase2_observables` 用的是自己抄的數字，它也不會因為這裡改壞而紅 |
| PHASE2_RFZ_BAND_HZ | C | 同上 | donor 標準答案 | donor 標準答案（1 個凍結值） | 同上 |
| PHASE2_LATE_RT_BAND_HZ | C | 同上 | donor 標準答案 | donor 標準答案（1 個凍結值） | 同上 |
| PHASE2_RFZ_WINDOW_S | C | 同上 | donor 標準答案 | donor 標準答案（1 個凍結值） | 同上 |
| PHASE2_TARGET_SLOPE_DB_OCT | C | 同上 | donor 標準答案 | donor 標準答案（1 個凍結值） | 同上 |

### receiver_grid

| 符號 | 判定 | 證據（v2 考卷：行） | 處置 | 覆蓋方式 | 殘餘風險 |
|---|---|---|---|---|---|
| GRID_N | A | `test_receiver_grid` 寫死 `GRID_N*GRID_N==16`，並與 `scripts.choras_bridge_pkg.geometry_grid` 比同一物件 | donor 標準答案 | donor 標準答案（1 個凍結值） | 「跨 facade 是同一物件」那一條綁 `scripts`；沒帶走。而且 v2 那個寫法只驗得出「別人也指向我」，驗不出「我是對的」 |
| GRID_MARGIN_M | A | 同上，`wall_offset_m == GRID_MARGIN_M` | donor 標準答案 | donor 標準答案（1 個凍結值） | 同上 |
| EAR_HEIGHT_M | A | 同上，`EAR_HEIGHT == EAR_HEIGHT_M` | donor 標準答案 | donor 標準答案（1 個凍結值） | 同上；另外它與 `default_geometry.DEFAULT_EAR_HEIGHT_M` 同值但沒有考卷驗那個一致（v2 也沒有） |

### source_reference

| 符號 | 判定 | 證據（v2 考卷：行） | 處置 | 覆蓋方式 | 殘餘風險 |
|---|---|---|---|---|---|
| CANONICAL_MONOPOLE_STRENGTH | B | 9 支考卷用它當輸入，只做跨模組一致性斷言（`test_m6_multisource` 等） | donor 標準答案 | donor 標準答案（1 個凍結值，以 `4*math.pi` 運算式記號存） | 那 9 支驗的是「各 lane 用同一個參考」——那要 physics／scoring 的層，沒帶走；只驗了值 |
| DIFFUSE_MONOPOLE_4PI | B | 同上（`test_art_kernel`／`test_m6_ray_lane` 當輸入） | donor 標準答案 | donor 標準答案（1 個凍結值） | 「它是 `CANONICAL_MONOPOLE_STRENGTH` 的別名、不是第二個家」這件事：補的考卷只比兩邊的值相等，沒驗它是同一個物件或同一個來源 |
| REFERENCE_ANCHOR_KIND | B | 只當輸入／字串標記 | donor 標準答案 | donor 標準答案（1 個凍結值） | 沒驗它被誰讀（絕對位準的命名錨） |
| REFERENCE_SENSITIVITY_DB_SPL_1M_1W | B | 只當輸入（94 dB 那個參考） | donor 標準答案 | donor 標準答案（1 個凍結值） | 沒驗它與 `HIGH_SPL_REFERENCE_*` 的關係 |
| REFERENCE_SENSITIVITY_PRESSURE_PA | B | 只當輸入 | donor 標準答案 | donor 標準答案（1 個凍結值） | 同上 |
| REFERENCE_DRIVE_POWER_W | B | 只當輸入 | donor 標準答案 | donor 標準答案（1 個凍結值） | 同上 |
| HIGH_SPL_REFERENCE_DB_SPL_1M_1W | B | 只當輸入 | donor 標準答案 | donor 標準答案（1 個凍結值） | 同上 |
| HIGH_SPL_REFERENCE_PRESSURE_PA | B | 只當輸入 | donor 標準答案 | donor 標準答案（1 個凍結值） | 同上 |

### speaker_directivity

| 符號 | 判定 | 證據（v2 考卷：行） | 處置 | 覆蓋方式 | 殘餘風險 |
|---|---|---|---|---|---|
| DIRECTIVITY_MODEL_VERSION | A | `test_scene17_default_directivity` 寫死 `MODEL_VERSION=="scene1-analytic-v1"` | donor 標準答案 | donor 標準答案（1 個凍結值） | 那一支還驗 provenance（結果物件的 `version` 是不是它）；補的考卷在解析成功那幾筆裡比了 `version` 欄位，但沒驗它與這一格的來源關係 |
| DIRECTIVITY_GOLDEN_VERSION | A | 同上，`GOLDEN_VERSION=="scene1-directivity-golden-v3"` | donor 標準答案 | donor 標準答案（1 個凍結值） | golden 檔本身在 v2 的 `tests/golden`，沒帶走；這一格只是字串 |
| DIRECTIVITY_ENABLED_KEY | B | 考卷用它組 payload（只當輸入） | donor 標準答案 | donor 標準答案（1 個凍結值） | 沒驗 bridge 真的讀這個鍵（綁 `scripts.choras_bridge`） |
| DIRECTIVITY_DEFAULT_ENABLED | A | `test_scene17_default_directivity` 寫死 `is True` | donor 標準答案 ＋ 補的新考卷 | 兩者 | v2 那一條驗的是「鍵缺席時產品預設是開」的整條路徑（bridge 那一層）；補的考卷只驗常數與解析函式的預設，沒驗 bridge 的行為 |
| SPEAKER_TYPE_KEY | B | 只當輸入／訊息的一部分 | donor 標準答案 | donor 標準答案（1 個凍結值） | 沒驗 bridge 真的讀這個鍵 |
| SPEAKER_WIDTH_OVERRIDE_KEY | B | 只當輸入（覆寫寬度的鍵） | donor 標準答案 | donor 標準答案（1 個凍結值） | 同上；錯誤訊息裡用到它的那條路徑有補到（見 `resolve_speaker_directivity`） |
| PISTON_RADIUS_OVERRIDE_KEY | B | 同上 | donor 標準答案 | donor 標準答案（1 個凍結值） | 同上 |
| DEFAULT_SPEAKER_TYPE | B | 只當輸入（`test_scene2_p1_oriented_schema` 用它挑 preset） | donor 標準答案 | donor 標準答案（1 個凍結值） | 沒驗它真的是解析函式的預設（補的考卷有餵「不給型別」那一筆，算接線到一半） |
| DIRECTIVITY_REAR_GAIN_MIN | B | 只當輸入 | donor 標準答案 | donor 標準答案（1 個凍結值） | 沒驗背向增益的下限在哪裡被用（那在解析式壓力場裡） |
| DIRECTIVITY_NORMALIZATION_GL_ORDER | B | 只當輸入（球面調和正規化的階數） | donor 標準答案 | donor 標準答案（1 個凍結值） | 沒驗正規化的數值結果；v2 也沒有考卷直接驗它 |
| DIRECTIVITY_TOE_IN_ANCHOR | B | 只當輸入（toe-in 的錨字串） | donor 標準答案 | donor 標準答案（1 個凍結值） | 沒驗誰讀這個錨 |
| SPEAKER_TYPES | B | 只當輸入（解析函式的白名單來源） | donor 標準答案 ＋ 補的新考卷（間接） | 兩者 | 補的考卷靠「不支援的型別會炸」間接蓋到它；沒有一條直接驗「這個 tuple 就是白名單」 |
| SPEAKER_PRESETS | B | `test_scene2_p1_oriented_schema` 把 `_W=_BOOKSHELF.width_m` 當輸入再用它去斷言別處 | donor 標準答案 ＋ 補的新考卷 | 兩者（逐欄比對 ＋ 18 個解析輸入點） | v2 那一支驗的是那組尺寸流進幾何之後的行為；補的考卷只驗五個欄位的值。**`inwall` 沒有 preset** 這件事也只是一條註解，沒有考卷 |
| SpeakerPreset | B | 只當資料容器（考卷讀 `.width_m` 等） | donor 標準答案 | donor 標準答案（逐欄比對，2 個 preset） | 沒驗它是 frozen（賦值要丟例外）；v2 也沒有 |
| SpeakerDirectivity | B | `test_scene163_polygon_directivity` 只 `assert resolved is not None` | 補的新考卷 ＋ donor 標準答案 | 兩者 | 沒驗它是 frozen；四個欄位比對了，但「哪種型別才會生出這個物件」只有部分蓋到（`inwall` 回 `None` 有蓋到） |
| resolve_speaker_directivity | B | 16 支考卷用到，但只驗 ON／OFF 與 provenance、或只當輸入 | 補的新考卷 ＋ donor 標準答案 | 兩者（18 個輸入點：開／關 × 型別 × 覆寫值 × 錯誤值） | **v2 考卷帶走的那條路本來會蓋到的保護**：bridge 層的鍵解析、警告的 `stacklevel`、`inwall` 在真實 payload 裡的行為。補的考卷直接呼叫函式，蓋不到那一層 |

### spl_output

| 符號 | 判定 | 證據（v2 考卷：行） | 處置 | 覆蓋方式 | 殘餘風險 |
|---|---|---|---|---|---|
| DEFAULT_SENSITIVITY_DB | A | `test_m15_p5_2_inc4_absolute_spl` 寫死 `== 0.0`（並驗 byte-identity） | 帶走的 v2 考卷 ＋ donor 標準答案 | 兩者 | v2 那一支的 byte-identity 端到端閘（整份 golden 逐鍵比）沒帶走——那要 bridge 那一層 |
| DEFAULT_PLAYBACK_LEVEL_DB | A | 同上，`== 0.0` | 帶走的 v2 考卷 ＋ donor 標準答案 | 兩者 | 同上 |
| FLAT_EQUAL_TOL_DB | A | `test_m15_p5_2_inc4_absolute_spl` 用它算「容差內／外」的邊界 | 帶走的 v2 考卷 ＋ donor 標準答案 | 兩者 | 同上 |
| SplOutputConfig（類別本身） | A | `test_m15_p5_2_inc4_absolute_spl` 建構它、驗預設與非有限值被擋 | 帶走的 v2 考卷 ＋ donor 標準答案 | 兩者（12 個建構輸入點 ＋ v2 那 4 條斷言） | 建構子本身的行為蓋到了；`model_config` 的 `frozen` 只有 v2 的 `test_config` 對 `PhysicsConstants` 驗過（那一支不是這一刀），這一刀沒有一條驗「`SplOutputConfig` 賦值會丟例外」 |
| SplOutputConfig.l_ref_db（屬性） | A | `test_m15_p5_2_inc4_absolute_spl` 寫死 `(88.0,-6.0)→82.0`、不等靈敏度丟錯、`meta["l_ref_db"]==82.0` | 帶走的 v2 考卷 ＋ donor 標準答案 | 兩者（v2 那 4 條斷言照抄 ＋ 12 個建構輸入點的答案） | `meta["l_ref_db"]` 那一條（值有沒有進到輸出中介資料）綁 bridge，沒帶走 |
| SplOutputConfig.is_relative（屬性） | A | 同上，`is True/False` | 帶走的 v2 考卷 ＋ donor 標準答案 | 兩者 | 同上 |

### 表外一筆（不屬於這一刀的符號，寫在這裡以免它消失在切線上）

上一代的 `test_receiver_grid` 有一條驗「`ReceiverGridSpec` 的預設與接收格點的 SSOT 一致」。那個類別住 `experiment_schema`——**搬去 materials 那一塊**（票 #134），所以這一刀沒有任何東西可以拿來對齊它。這一筆跟著 #134 走，不在這一刀的 88 個符號裡。

**這張表跟這一支的關係。** 這一支是那 11 支考卷共用的零件（讀答案檔、還原記號），表住在這裡而不是 PR 內文：它是「有沒有漏掉」的答案，要跟著考卷一起進版控、一起被讀。
"""

from __future__ import annotations

import json
import math
from dataclasses import fields, is_dataclass
from pathlib import Path

# 答案檔的位置：`tests/engine/answers/config_cut1.json`。用 `__file__` 推出來，跟 cwd 無關。
ANSWER_PATH: Path = Path(__file__).resolve().parent / "answers" / "config_cut1.json"

# 這一刀（票 #127 前半）的 11 支模組。順序照產生器那一份，不另外抄一份篩選邏輯。
CUT1_MODULES: tuple[str, ...] = (
    "art_lane",
    "art_rt_guard",
    "authoring_defaults",
    "crossover_axis",
    "default_geometry",
    "ism_lane",
    "phase2_report_bands",
    "receiver_grid",
    "source_reference",
    "speaker_directivity",
    "spl_output",
)

# `4*math.pi` 這種以運算式記下來的值。只有它一個，寫成一張表而不是散在程式裡。
EXPR_VALUES: dict[str, float] = {"4*math.pi": 4.0 * math.pi}


def _as_mapping(node: object, where: str) -> dict[str, object]:
    """把一個節點收窄成「一層表」。不是表就當場炸——我沒看懂就不出結論。"""
    if not isinstance(node, dict):
        raise AssertionError(f"{where} 不是一層表：{node!r}")
    return {str(key): value for key, value in node.items()}


def load_answers() -> dict[str, object]:
    """讀答案檔。檔頭少一格就當場炸——沒有 donor 記號或 sha 的答案檔不算證據。"""
    with ANSWER_PATH.open(encoding="utf-8") as handle:
        data: object = json.load(handle)
    root = _as_mapping(data, ANSWER_PATH.name)
    donor = _as_mapping(root.get("donor"), f"{ANSWER_PATH.name} 的 donor 檔頭")
    if not donor.get("tag") or not donor.get("commit"):
        raise AssertionError(f"{ANSWER_PATH.name} 的檔頭少了 donor 記號或 commit——那不是標準答案")
    return root


def module_answers(name: str) -> dict[str, object]:
    """某一支模組的那一塊（常數 ＋ 探針）。"""
    modules = _as_mapping(load_answers().get("modules"), "答案檔的 modules")
    if name not in modules:
        raise AssertionError(f"答案檔裡沒有 {name}——產生器沒跑到它，或這一刀的名單變了")
    return _as_mapping(modules[name], f"答案檔的 modules.{name}")


def decode(node: object) -> object:
    """把答案檔裡的一個記號還原成 Python 值。"""
    marker = _as_mapping(node, "答案檔的一個記號")
    kind = marker.get("kind")
    if kind == "scalar":
        return marker.get("value")
    if kind == "float":
        raw = marker.get("hex")
        if not isinstance(raw, str):
            raise AssertionError(f"答案檔的浮點記號沒有 hex：{marker!r}")
        value = float.fromhex(raw)
        expr = marker.get("expr")
        if isinstance(expr, str) and value != EXPR_VALUES[expr]:
            raise AssertionError(f"答案檔記著運算式 {expr}，但那個值不等於它——這份答案檔壞了")
        return value
    if kind == "list":
        return [decode(item) for item in _sequence(marker.get("items"), "答案檔的 list 記號")]
    if kind == "dict":
        return {key: decode(item) for key, item in _as_mapping(marker.get("items"), "答案檔的 dict 記號").items()}
    if kind == "object":
        return {key: decode(item) for key, item in _as_mapping(marker.get("fields"), "答案檔的 object 記號").items()}
    if kind == "model":
        return decode(marker.get("fields"))
    raise AssertionError(f"答案檔裡有不認識的記號種類：{kind!r}")


def _sequence(node: object, where: str) -> list[object]:
    """把一個節點收窄成一個序列。不是序列就當場炸。"""
    if not isinstance(node, list):
        raise AssertionError(f"{where} 不是一串東西：{node!r}")
    return node


def constant(module: str, name: str) -> object:
    """某一支模組某一個常數的凍結值（donor 那一邊跑出來的）。"""
    constants = _as_mapping(module_answers(module).get("constants"), f"{module} 的常數表")
    if name not in constants:
        raise AssertionError(f"答案檔裡沒有 {module}.{name}——這一格沒有裁判")
    return decode(constants[name])


def probe_block(expected: dict[str, object], key: str) -> dict[str, object]:
    """一筆探針某一格的**原始結果格**（``{"value": 記號}``／``{"raised": {...}}``／…）。

    每一格的形狀不一樣，這一支只負責「那一格在不在、是不是一格」。要值用
    :func:`probe_value`，要例外用 :func:`probe_raised`，要警告用 :func:`probe_warned`。
    """
    if key not in expected:
        raise AssertionError(f"答案檔這一筆沒有 {key}——這一格沒有裁判")
    return _as_mapping(expected[key], f"答案檔這一筆的 {key}")


def probe_value(expected: dict[str, object], key: str = "value") -> object:
    """一筆探針某一格的值（已經解好，可以直接跟新家算出來的東西比）。

    答案檔的每一格都包一層（``{"value": 記號}`` 或 ``{"raised": …}``）；屬性那兩格因為
    產生器是從真的物件上讀的，還會再多包一層。這裡一路拆到看見記號為止。
    """
    node: object = probe_block(expected, key)
    while isinstance(node, dict) and "kind" not in node and "value" in node:
        node = _as_mapping(node, f"答案檔這一筆的 {key}")["value"]
    return decode(node)


def is_approx(actual: object, expected: object) -> bool:
    """浮點逐位元比、其餘用 ``==``。``nan`` 特殊處理：兩邊都是 ``nan`` 才算一樣。

    這裡**不用** ``pytest.approx``：答案檔存的浮點是精確的十六進位寫法，而這一刀的合約
    是「數值跟上一代一致」——容差會把「算出來差一點」放過去。

    tuple 與 list 在這裡算同一種東西（逐項比）：答案檔把它們都存成 list 記號，而這一刀的
    合約是數值一致、**結構隨便改**——「上游那幾支寫 tuple 還是 list」不是這一刀要釘的東西。
    dataclass 實例（例如喇叭預設）比它的欄位表，不比記憶體位址。
    """
    if isinstance(expected, float) and isinstance(actual, float):
        if math.isnan(expected) or math.isnan(actual):
            return math.isnan(expected) and math.isnan(actual)
        return actual == expected
    left = list(actual) if isinstance(actual, tuple) and isinstance(expected, list) else actual
    right = list(expected) if isinstance(expected, tuple) and isinstance(actual, list) else expected
    if isinstance(right, list) and isinstance(left, list):
        try:
            return all(is_approx(item, want) for item, want in zip(left, right, strict=True))
        except ValueError:
            return False
    if isinstance(right, dict) and isinstance(left, dict):
        return set(left) == set(right) and all(
            is_approx(left[key], value) for key, value in right.items()
        )
    if is_dataclass(actual) and not isinstance(actual, type):
        return is_approx(
            {field.name: getattr(actual, field.name) for field in fields(actual)}, expected
        )
    return bool(actual == expected)


def probe_raised(expected: dict[str, object]) -> dict[str, object] | None:
    """一筆探針「炸掉」的那一塊（沒有炸就回 ``None``）。"""
    if "raised" not in expected:
        return None
    return _as_mapping(expected["raised"], "答案檔這一筆的 raised")


def probe_warned(expected: dict[str, object]) -> list[dict[str, object]]:
    """一筆探針記下來的警告（沒有就回空清單）。"""
    if "warned" not in expected:
        return []
    return [_as_mapping(item, "答案檔這一筆的一條警告") for item in _sequence(expected["warned"], "答案檔這一筆的 warned")]


def validate_message(actual: str, expected: str) -> bool:
    """比對 pydantic 的 ``ValidationError`` 訊息：只比它認定的問題清單（``[type=…]``）。

    為什麼不比整串：那一串裡有兩件**不是契約**的東西——錯誤說明的網址帶著庫自己的版本號，
    而輸入值那一格印的是「它收到的那個容器」（``input_value=(nan,)`` 與 ``input_value=[nan]``
    是同一種錯的兩種寫法，取決於餵進去的是 tuple 還是 list）。**問題清單**才是這一條要釘的
    東西：哪幾種錯、幾筆、順序。訊息整串逐字比對會把「換個容器寫法」判成行為改變。
    """
    if actual == expected:
        return True
    return bool(_error_types(actual)) and _error_types(actual) == _error_types(expected)


def _error_types(message: str) -> list[str]:
    """訊息裡每一筆問題的種類（``[type=finite_number, input_value=…]`` → ``finite_number``）。

    問題種類就是「這一句在講哪一種錯」那一格的名字；後面的 ``input_value`` 只是它順便
    印出來的輸入，不是種類的一部分。
    """
    marker = "[type="
    return [part.split(",")[0].split("]")[0].strip() for part in message.split(marker)[1:]]


# 探針參數的種類：答案檔只有「字串／數字／布林／null／一串東西」這幾種 JSON 形狀，
# 而每個函式要的是特定的型別（`bool`、`float`、`float | None`、`str`、`tuple[float, ...]`）。
# 這裡一張表把「這一刀的探針會出現哪些參數」寫清楚，考卷用 probe_args() 拿到的就是
# 可以直接 ** 展開的具體型別（mypy 嚴格模式下不必每個呼叫點各寫一個 ignore）。
PARAM_KINDS: dict[str, str] = {
    # art_lane 的兩支護欄
    "n_per_wall": "int",
    "n_tris": "int",
    "context": "str",
    # authoring_defaults 的三支 id 助手
    "wall_id": "str",
    "row": "int",
    "col": "int",
    "guid": "str",
    # crossover_axis
    "seam_f_s": "float",
    "t_c_center": "float",
    "t_c_fade": "float",
    # speaker_directivity
    "enabled": "bool",
    "speaker_type": "str",
    "baffle_width_m": "opt_float",
    "piston_radius_m": "opt_float",
    # spl_output
    "sensitivity_db": "float_tuple",
    "playback_level_db": "float",
}


def _kind(name: str) -> str:
    if name not in PARAM_KINDS:
        raise AssertionError(f"探針參數 {name} 沒有登記種類——答案檔多了這一格就是漏了一種")
    return PARAM_KINDS[name]


def probe_args(expected: dict[str, object]) -> dict[str, object]:
    """一筆探針的參數，轉成對應的 Python 型別（可以直接 ``**`` 展開進被測函式）。

    轉不出來就當場炸：讀回來的東西跟登記的種類不合，是這份答案檔壞了或這一刀的
    函式簽章變了，兩種都不該靜靜地跑過去。
    """
    args = _as_mapping(expected.get("args"), "答案檔這一筆的 args")
    out: dict[str, object] = {}
    for name, raw in args.items():
        kind = _kind(name)
        if kind == "str":
            if not isinstance(raw, str):
                raise AssertionError(f"{name} 應該是字串：{raw!r}")
            out[name] = raw
        elif kind == "bool":
            if not isinstance(raw, bool):
                raise AssertionError(f"{name} 應該是布林：{raw!r}")
            out[name] = raw
        elif kind == "int":
            if not isinstance(raw, int) or isinstance(raw, bool):
                raise AssertionError(f"{name} 應該是整數：{raw!r}")
            out[name] = raw
        elif kind == "float":
            out[name] = _as_float(raw, name)
        elif kind == "opt_float":
            out[name] = None if raw is None else _as_float(raw, name)
        elif kind == "float_tuple":
            out[name] = tuple(_as_float(item, name) for item in _sequence(raw, f"{name} 那一格"))
        else:
            raise AssertionError(f"{name} 登記了不認識的種類：{kind}")
    return out


def _as_float(raw: object, name: str) -> float:
    if not isinstance(raw, (int, float)) or isinstance(raw, bool):
        raise AssertionError(f"{name} 應該是數字：{raw!r}")
    return float(raw)
