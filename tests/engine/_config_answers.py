"""讀標準答案檔、把答案檔裡的記號還原成 Python 值的共用零件。

這一支不是考卷（檔名不以 ``test_`` 開頭），是考卷共用的兩件事：

1. **答案檔在哪、誰決定它有哪些 case。** ``blueprint/config_cut1_answers.json``，由
   ``blueprint/generate_config_cut1_answers.py`` 在唯讀的 v2 工作樹上跑出來（票 #127 第 1 刀
   的那 11 支模組，v2 的考卷一支都帶不走——每一支都還 import 了別的層或搬走的模組，
   所以裁判改由這一份標準答案接手）。**人不碰裡面的數字**：要改就重跑產生器。
   答案檔存的是值本身，不是「怎麼寫的」——tuple 與 list 在檔案裡是同一種 list 記號
   （這一刀的合約是數值一致、結構隨便改）。真的需要分辨那兩種寫法的地方只有一處：
   pydantic 的訊息會把 ``input_value=(nan,)`` 與 ``input_value=[nan]`` 講成兩句，
   所以 ``validate_message`` 只比它認定的問題種類，不比輸入值那一格。
   **有哪些 case 不是這一支決定的**：``blueprint/config_cut1_cases.py`` 是版控裡的
   單一來源，產生器照它跑、考卷用 :func:`check_case_ids` 斷言答案檔的 id 集合**等於**
   它宣告的集合。

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

### 這一刀**新蓋**的兩支（不是從 donor 搬的）

`paths.py` 與 `early_reflection.py` 是這一刀唯二真的新寫的程式（其餘 11 支是照抄）。
它們**沒有 donor 可以對**——裁判就是下面這幾條，理由寫在每一列。

| 符號 | 判定 | 證據 | 處置 | 覆蓋方式 | 殘餘風險 |
|---|---|---|---|---|---|
| `paths.CONFIG_DIR` | 新蓋 | 上一代 5 支各自算路徑、4 支靠呼叫端傳；新家收成一支（檔頭指名「往上數兩層」那個安靜的錯法） | 補的新考卷 | 補的新考卷（3 條形狀斷言） | 蓋不到「`data/` 底下真的有那 9 個 `.toml`」——那是第 2 刀；也蓋不到「別的模組有沒有繞過這一支自己算路徑」（第 2 刀那 9 支的考卷要看） |
| `paths.config_path` | 新蓋 | 同上 | 補的新考卷 | 補的新考卷（3 條：名字、子目錄、不檢查存在） | 沒驗呼叫端都用它（今天沒有呼叫端） |
| `early_reflection.EarlyReflectionConfig` | 新蓋（型別從 zoning 沉下來） | 上一代住 zoning 那一層，第 2 層反過來拿它 | 補的新考卷 | 補的新考卷（4 條：欄位名與種類、零預設、frozen、extra 不准多） | **沒驗值**（15／10／(200,8000) 在 `perceptual.toml`，第 2 刀）；也沒驗「perceptual 那一支真的從這裡拿」（那一支是第 2 刀） |

**為什麼「補的新考卷」在這裡就夠。** 這兩支的風險不是「值跟上一代不一樣」（它們沒有上一代
對應物），是「**路徑算錯**」與「**契約鬆掉**」——前者住 ``CONFIG_DIR`` 的形狀，後者住
``model_fields`` 與 ``extra``／``frozen``。上面那幾條各釘住其中一種，而且每一條都在
「有人把那個錯法寫回去」時會紅（不是只驗『有這個名字』）。

### case 表（答案檔的份量下限）

| 東西 | 判定 | 處置 | 覆蓋方式 | 殘餘風險 |
|---|---|---|---|---|
| `blueprint/config_cut1_cases.py` 宣告的 case 集合 | 新蓋 | 版控裡的單一來源：產生器照它跑、考卷比對集合 | 補的新考卷（`test_config_cut1_case_table` 的集合等值＋非空＋每支都有常數） | 答案檔住 `blueprint/`，**CI 上沒有機器掃它**；這一條要求的是「答案檔有的 case 就是表上那些」，它蓋不到「表上那些 case 本身夠不夠兇」（那是每一支考卷的事） |
| donor 出身（記號／commit／clean） | 新蓋 | 產生器用量的（`rev-parse v3-donor^{commit}`、`status --porcelain` 空才准產出），考卷驗檔頭有那三格 | 補的新考卷（`test_donor_provenance_is_measured_not_declared`） | **v2 不在 CI**：那個 sha 是「產生時量的 ＋ 人工可重跑」，不是 CI 驗的——考卷只能驗答案檔自己說的那三格 |

### 表外一筆（不屬於這一刀的符號，寫在這裡以免它消失在切線上）

上一代的 `test_receiver_grid` 有一條驗「`ReceiverGridSpec` 的預設與接收格點的 SSOT 一致」。那個類別住 `experiment_schema`——**搬去 materials 那一塊**（票 #134），所以這一刀沒有任何東西可以拿來對齊它。這一筆跟著 #134 走，不在這一刀的 88 個符號裡。
"""

from __future__ import annotations

import importlib
import json
import math
import tempfile
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import cast

from blueprint import config_cut1_cases as cases

# 答案檔的形狀版本，要跟 blueprint/generate_config_cut1_answers.py 的 ANSWER_SCHEMA 一樣。
# 第 2 版加了 case 的 id 與 donor 的 clean 那一格；第 3 版加了這一刀（讀檔那 9 支）的
# `op` 與載入器結果。
ANSWER_SCHEMA: int = 3

# 答案檔的位置：`blueprint/config_cut1_answers.json`（產生器與它的產物住同一層）。
# 從這一支往上一層是 tests/engine，再往上兩層是 repo 根；跟 cwd 無關。
ANSWER_PATH: Path = Path(__file__).resolve().parents[2] / "blueprint" / "config_cut1_answers.json"

# 第一刀（票 #127 前半）的 11 支模組。這一刀（後半，讀檔那 9 支）的名單在下面
# `CUT2_TABLE_MODULES`——兩份都從 case 表推出來，這裡不另外抄篩選邏輯。
CUT1_MODULES: tuple[str, ...] = tuple(
    name for name in sorted(cases.MODULES) if not name.startswith("cut2_")
)

# 這一刀那 9 支模組在 case 表裡的名字（`cut2_<模組>`）。
CUT2_TABLE_MODULES: tuple[str, ...] = tuple(f"cut2_{name}" for name in cases.CUT2_MODULES)

# `4*math.pi` 這種以運算式記下來的值。只有它一個，寫成一張表而不是散在程式裡。
EXPR_VALUES: dict[str, float] = {"4*math.pi": 4.0 * math.pi}


def _as_mapping(node: object, where: str) -> dict[str, object]:
    """把一個節點收窄成「一層表」。不是表就當場炸——我沒看懂就不出結論。"""
    if not isinstance(node, dict):
        raise AssertionError(f"{where} 不是一層表：{node!r}")
    return {str(key): value for key, value in node.items()}


def load_answers() -> dict[str, object]:
    """讀答案檔。檔頭少一格就當場炸——沒有 donor 出身或 sha 的答案檔不算證據。"""
    with ANSWER_PATH.open(encoding="utf-8") as handle:
        data: object = json.load(handle)
    root = _as_mapping(data, ANSWER_PATH.name)
    schema = root.get("schema")
    if schema != ANSWER_SCHEMA:
        raise AssertionError(
            f"{ANSWER_PATH.name} 的 schema 是 {schema!r}，這一支讀的是 {ANSWER_SCHEMA}"
            "——產生器改了形狀就要一起改讀的人"
        )
    donor = _as_mapping(root.get("donor"), f"{ANSWER_PATH.name} 的 donor 檔頭")
    for key in ("tag", "commit"):
        if not donor.get(key):
            raise AssertionError(f"{ANSWER_PATH.name} 的 donor 檔頭少了 {key}")
    if donor.get("clean") is not True:
        raise AssertionError(
            f"{ANSWER_PATH.name} 的 donor 檔頭沒有 clean=true"
            "——那代表它是在一棵被改過的樹上跑出來的，不是上一代的值"
        )
    return root


def donor_provenance() -> dict[str, object]:
    """答案檔檔頭記的 donor 出身（記號、解析出來的 commit、樹乾不乾淨）。"""
    return _as_mapping(load_answers().get("donor"), "答案檔的 donor 檔頭")


def declared_case_ids() -> set[str]:
    """case 表（`blueprint/config_cut1_cases.py`）宣告的每一個 id。

    常數那一筆的 id 是 `<module>.const.<NAME>`，跟答案檔那一邊的寫法同一套。
    """
    return cases.all_case_ids() | cases.constant_ids()


def answer_case_ids() -> set[str]:
    """答案檔裡每一個 case 的 id（常數那一筆也在內）。"""
    ids: set[str] = set()
    modules = _as_mapping(load_answers().get("modules"), "答案檔的 modules")
    for name, raw in modules.items():
        module = _as_mapping(raw, f"答案檔的 modules.{name}")
        for constant in _as_mapping(module.get("constants"), f"{name} 的常數表"):
            ids.add(f"{name}.const.{constant}")
        probes = module.get("probes")
        if not isinstance(probes, list):
            raise AssertionError(f"答案檔的 {name} 探針不是一串東西")
        for record in probes:
            case = _as_mapping(record, f"答案檔 {name} 的一筆探針")
            case_id = case.get("id")
            if not isinstance(case_id, str) or not case_id:
                raise AssertionError(f"答案檔 {name} 有一筆探針沒有 id：{record!r}")
            ids.add(case_id)
    return ids


def check_case_ids() -> None:
    """答案檔的 id 集合必須**等於** case 表宣告的集合（少一筆紅、多一筆也紅）。

    比的是具名的集合，不是筆數（`assertions-not-pinned-to-counts` 咬後者）。這一條是
    答案檔的「份量下限」：答案檔住 `blueprint/`，`identity-strings-generated` 與
    `refs-and-links-resolve` 都不掃它，所以「它還有幾筆」只有這裡在守。
    """
    declared = declared_case_ids()
    present = answer_case_ids()
    missing = sorted(declared - present)
    extra = sorted(present - declared)
    assert not missing, f"答案檔少了 case 表宣告的這幾筆：{missing}"
    assert not extra, f"答案檔有 case 表沒宣告的這幾筆：{extra}"


def module_answers(name: str) -> dict[str, object]:
    """某一支模組的那一塊（常數 ＋ 探針）。"""
    modules = _as_mapping(load_answers().get("modules"), "答案檔的 modules")
    if name not in modules:
        raise AssertionError(f"答案檔裡沒有 {name}——產生器沒跑到它，或這一刀的名單變了")
    return _as_mapping(modules[name], f"答案檔的 modules.{name}")


def module_probes(module: str) -> list[dict[str, object]]:
    """某一支模組在答案檔裡的所有探針（每一筆都有 id）。"""
    probes = module_answers(module).get("probes")
    if not isinstance(probes, list):
        raise AssertionError(f"答案檔的 {module} 探針不是一串東西")
    return [_as_mapping(record, f"答案檔 {module} 的一筆探針") for record in probes]


def probe_ids(module: str) -> set[str]:
    """某一支模組在答案檔裡的 case id 集合（用來挑「這一支有哪幾筆」）。"""
    return {str(case["id"]) for case in module_probes(module)}


def probe_by_id(case_id: str) -> dict[str, object]:
    """答案檔裡那一筆探針（找不到就當場炸——考卷要用的每一筆都必須在）。"""
    module_name = case_id.split(".", 1)[0]
    probes = module_answers(module_name).get("probes")
    if not isinstance(probes, list):
        raise AssertionError(f"答案檔的 {module_name} 探針不是一串東西")
    for record in probes:
        case = _as_mapping(record, f"答案檔 {module_name} 的一筆探針")
        if case.get("id") == case_id:
            return case
    raise AssertionError(f"答案檔裡沒有 case {case_id!r}——這一格沒有裁判")


def case_args(case: dict[str, object]) -> dict[str, object]:
    """一筆 case 的參數，整格轉成可以直接 ``**`` 展開的 Python 值。

    唯一要還原的是容器：答案檔是 JSON，上游宣告成 ``tuple[float, ...]`` 的那一格讀回來
    是一串 list，而 pydantic 的訊息會把 ``input_value=[nan]`` 與 ``input_value=(nan,)``
    講成兩句——所以那一格要還原成 tuple（跟產生器餵進去的是同一個容器）。
    """
    args = _as_mapping(case.get("args"), f"答案檔 case {case.get('id')!r} 的 args")
    resolved: dict[str, object] = dict(args)
    for name in ("sensitivity_db",):
        value = resolved.get(name)
        if isinstance(value, list):
            resolved[name] = tuple(value)
    return resolved


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
    if kind == "type":
        # 模組層的類別物件（這一刀那 9 支的 `CalibrationConfig` 那一種常數）：只記名字。
        name = marker.get("name")
        if not isinstance(name, str) or not name:
            raise AssertionError(f"答案檔的類別記號沒有名字：{marker!r}")
        return name
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


def expected_block(case: dict[str, object]) -> dict[str, object]:
    """一筆探針的 ``expected``（收窄成表；形狀不對就當場炸）。

    答案檔讀回來是動態的表，``case["expected"]`` 的靜態型別是 ``object``——考卷每一支
    都要它，所以在這裡收一次。
    """
    return _as_mapping(case.get("expected"), f"答案檔 case {case.get('id')!r} 的 expected")


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
    dataclass 實例與 pydantic 模型先走 :func:`as_plain` 變成普通容器（欄位名與值），
    再逐欄比對，不比記憶體位址。
    """
    left = as_plain(actual)
    right = as_plain(expected)
    # **bool 與數字不是同一種東西**（這裡刻意選了嚴的那一邊）：`True` 在 Python 裡
    # `== 1`，但答案檔把布林與整數編成兩種記號，而 case 表也有專門的 `true()`／
    # `false()`——一個凍結值從 `True` 變成 `1` 是值變了，不是寫法不同。所以一邊是布林、
    # 另一邊不是，就直接不近似（`True` vs `True` 走下面的 `==`）。
    # 反過來，`1` 與 `1.0` **算同一種**：那是同一個數字的兩種寫法（TOML 的整數與 Python
    # 的浮點在兩代之間換過寫法），不是值變了。
    if isinstance(left, bool) != isinstance(right, bool):
        return False
    if isinstance(right, float) and isinstance(left, float):
        if math.isnan(right) or math.isnan(left):
            return math.isnan(right) and math.isnan(left)
        return left == right
    if isinstance(right, list) and isinstance(left, list):
        try:
            return all(is_approx(item, want) for item, want in zip(left, right, strict=True))
        except ValueError:
            return False
    if isinstance(right, dict) and isinstance(left, dict):
        return set(left) == set(right) and all(
            is_approx(left[key], value) for key, value in right.items()
        )
    return bool(left == right)


def as_plain(value: object) -> object:
    """把一個值收成「只有 list／dict／純量」的形狀（比對用）。

    這一刀（讀檔那 9 支）的載入器回傳的是 dataclass 與 pydantic 模型（還有一層層巢狀
    子模型）；答案檔那一邊解出來的是同樣形狀的普通容器。兩邊都走這一支再比，才比得到
    「欄位名與值一不一致」，而不是「型別物件的記憶體位址」。

    ``bool`` 要在 ``int`` 之前判（Python 的 ``True`` 是一個 ``int``）。
    **類別物件**（模組層的常數，例如 `CalibrationConfig`）收成它的**名字**——答案檔那一邊
    記的就是名字（``{"kind": "type", "name": …}``），而且不這樣做的話它會被當成 pydantic
    的**實例**去呼叫未綁定的 `model_dump`（實測就是 `TypeError: model_dump() missing 1
    required positional argument: 'self'`）。
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, type):
        return value.__name__
    if isinstance(value, dict):
        return {str(key): as_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_plain(item) for item in value]
    if is_dataclass(value):
        return {field.name: as_plain(getattr(value, field.name)) for field in fields(value)}
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return as_plain(dump())
    enumerable = getattr(value, "__dict__", None)
    if isinstance(enumerable, dict):
        return {str(key): as_plain(item) for key, item in enumerable.items()}
    return value


def probe_raised(expected: dict[str, object]) -> dict[str, object] | None:
    """一筆探針「炸掉」的那一塊（沒有炸就回 ``None``）。

    三種存放形狀都認：

    * 第 1 刀那些純呼叫的探針——例外放在最上層（``expected["raised"]``）；
    * 這一刀那 9 支的載入器探針——例外放在結果格裡（``expected["value"]["raised"]``）；
    * 這一刀那些**成功**的載入器探針——``expected`` 本身就是一個值記號
      （``{"kind": "model", "fields": …}``），沒有包一層。
    """
    node: object = expected.get("raised")
    if node is None:
        value_node = expected.get("value")
        if isinstance(value_node, dict):
            node = value_node.get("raised")
    if node is None:
        return None
    return _as_mapping(node, "答案檔這一筆的 raised")


@contextmanager
def loader_case_run(case_id: str) -> Iterator[object]:
    """照 case 表跑一筆載入器 case，**在暫存檔還在的時候**把結果交給呼叫端。

    這是一支 context manager 而不是普通函式，理由跟產生器那一邊一樣：突變過的 TOML
    離開 `TemporaryDirectory` 就被刪掉，而呼叫端要拿載入器回傳的活物件跟答案檔比——
    比的那一刻檔案必須還在。

    產生的東西跟產生器那一邊**同一個形狀**：成功回傳載入器的回傳值（呼叫端再用
    :func:`as_plain` 收成普通容器），失敗回傳 ``{"raised": {"type": …, "message": …}}``，
    訊息裡的暫存路徑換成 ``<tmp>``。
    """
    table_name = case_id.split(".", 1)[0]
    if table_name not in cases.MODULES:
        raise AssertionError(f"case 表裡沒有 {table_name!r}——這一筆沒有裁判")
    case = next((item for item in cases.cases_for(table_name) if item["id"] == case_id), None)
    if case is None:
        raise AssertionError(f"case 表裡沒有 {case_id!r}——這一筆沒有裁判")
    op = cases.op_for(case)
    fn_name = op.get("fn")
    if not isinstance(fn_name, str):
        raise AssertionError(f"{case_id} 的 op 沒有 fn——這一筆不知道要跑哪一支載入器")
    engine_name = cases.engine_module_name(table_name)
    module = importlib.import_module(f"aosr.config.{engine_name}")
    loader = cast(Callable[..., object], getattr(module, fn_name))
    kwargs_node = op.get("kwargs")
    kwargs = {} if kwargs_node is None else _as_mapping(kwargs_node, f"{case_id} 的 op.kwargs")
    resolved_kwargs = {name: cases.resolve(value) for name, value in kwargs.items()}
    path_mode = op.get("path")
    if path_mode == "missing":
        yield _capture(lambda: loader(cases.missing_path()))
        return
    if path_mode not in ("default", None):
        raise AssertionError(f"{case_id} 的 op.path 看不懂：{path_mode!r}")
    # 先看有沒有突變：`path: "default"` 只說「不傳路徑」，兩者可以同時出現。
    mutations = op.get("mutations") or []
    if not isinstance(mutations, list):
        raise AssertionError(f"{case_id} 的 op.mutations 不是一串東西")
    if not mutations:
        # `op.path == "default"`：把真的設定檔的路徑明著餵進去（跟產生器那一邊同一件事）。
        # 不餵 `None`：那五支必填的載入器在上一代收到 `None` 是 `TypeError`，餵 `None`
        # 等於在比一個上一代沒有的行為。
        yield _capture(
            lambda: loader(config_data_dir() / f"{engine_name}.toml", **resolved_kwargs)
        )
        return
    text = (config_data_dir() / f"{engine_name}.toml").read_text(encoding="utf-8")
    for mutation in mutations:
        before = mutation["before"]
        count = text.count(before)
        if count != 1:
            raise AssertionError(
                f"{case_id} 的突變原文在 {engine_name}.toml 裡出現 {count} 次，不是 1 次：{before!r}"
            )
        text = text.replace(before, mutation["after"])
    with tempfile.TemporaryDirectory(dir=_tmp_root()) as work:
        probe = Path(work) / f"{engine_name}.toml"
        probe.write_text(text, encoding="utf-8")
        yield _capture(lambda: loader(probe, **resolved_kwargs))


def _capture(call: Callable[[], object]) -> object:
    """跑一次載入器：成功就回傳結果，失敗就回一個帶型別與訊息的表。

    訊息的處理跟產生器那一邊**逐字相同**：`cases.normalise_paths` 把絕對路徑前綴收成
    ``<path>/``。少了這一步，答案檔在 CI 上會因為「repo 不在 ``/home/florian/...``」
    而整批紅——那不是契約，是 checkout 在哪。

    先做一次「絕對路徑換成相對 repo 根」：`config_path()` 的預設吐出來的是絕對路徑，
    而產生器那一邊餵給上一代載入器的是相對路徑（同一份檔），兩邊不先對齊就會在
    「路徑寫法」上比出假的差異。
    """
    try:
        return call()
    except Exception as exc:  # noqa: BLE001  # expires=2026-12-08 reason=這一格要記錄「炸什麼」而不是讓它往外炸，例外的種類不影響判準
        return {
            "raised": {
                "type": type(exc).__name__,
                "message": cases.normalise_paths(_as_repo_relative(str(exc))),
            }
        }


def _as_repo_relative(message: str) -> str:
    """把訊息裡的 repo 根那段砍掉（絕對路徑 -> 相對 repo 根）。"""
    return message.replace(str(repo_root()) + "/", "")


def repo_root() -> Path:
    """新家這一棵樹的根（`<root>/src/aosr/config/data` 往上三層）。"""
    return config_data_dir().parents[3]


def config_data_dir() -> Path:
    """新家那 9 個真的設定檔住哪（載入器自己的預設也是指到這裡）。"""
    return Path(cases.__file__).resolve().parents[1] / "src" / "aosr" / "config" / "data"


def _tmp_root() -> Path:
    """突變過的 TOML 寫在哪（跟產生器同一個固定位置，訊息才比得起來）。"""
    root = Path(cases.tmp_root())
    root.mkdir(parents=True, exist_ok=True)
    return root


def probe_warned(expected: dict[str, object]) -> list[dict[str, object]]:
    """一筆探針記下來的警告（沒有就回空清單）。"""
    if "warned" not in expected:
        return []
    return [
        _as_mapping(item, "答案檔這一筆的一條警告")
        for item in _sequence(expected["warned"], "答案檔這一筆的 warned")
    ]


@contextmanager
def observed_warnings() -> Iterator[list[warnings.WarningMessage]]:
    """把一段程式發出的警告收下來（考卷用它驗「不該警告的時候不准警告」）。

    用 `warnings.catch_warnings` ＋ `simplefilter("always")`：donor 說沒有警告的那些
    case，新家只要多發一個警告就紅——**那一面原本沒有考卷**（找碴 NOTE 2）。
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield caught


def validate_message(actual: str, expected: str) -> bool:
    """比對 pydantic 的 ``ValidationError`` 訊息：比**欄位路徑**與**問題種類**。

    不比整串的原因有兩件不是契約的東西：錯誤說明的網址帶著庫自己的版本號；輸入值那一格
    印的是「它收到的那個容器」（``input_value=(nan,)`` 與 ``input_value=[nan]`` 是同一種
    錯的兩種寫法）。除此之外的兩格是契約——**哪一個欄位**（``sensitivity_db.0``）與
    **哪一種問題**（``finite_number``、``too_short``、``missing``），這裡逐筆比。
    """
    if actual == expected:
        return True
    return bool(_problems(actual)) and _problems(actual) == _problems(expected)


def _problems(message: str) -> list[tuple[str, str]]:
    """一份 pydantic 訊息裡的每一筆問題：``(欄位路徑, 問題種類)``。

    欄位路徑是 ``[type=`` 之前那一行（去縮排、去掉尾端說明）；問題種類是 ``[type=…]``
    那一格的第一段。兩者都不是「怎麼印」而是「哪一格壞了、壞在哪」。
    """
    problems: list[tuple[str, str]] = []
    for chunk in message.split("[type=")[1:]:
        kind = chunk.split(",")[0].split("]")[0].strip()
        head = chunk.split("[type=")[0]
        field = head.strip().splitlines()[-1].strip() if head.strip() else ""
        problems.append((field, kind))
    return problems
