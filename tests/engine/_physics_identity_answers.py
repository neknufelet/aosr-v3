"""由目前 physics_identity_parts 程式產生的物理變更偵測答案。

這不是答案正確性的考卷，是「物理變了要看得見」的偵測器。
閉包、資料清單或程式＋資料摘要變了，代表老闆存下的結果要重跑物理，
每份約 6 分鐘。改這三個值的合併請求內文必須寫明「物理變更」與原因。
環境摘要不釘在這裡，套件版本或 Python 版本另由物理身分反映。
"""

from aosr.config.paths import config_path

CLOSURE = (
    'aosr',
    'aosr.config',
    'aosr.config._furniture_records',
    'aosr.config.art_lane',
    'aosr.config.capabilities',
    'aosr.config.construction_environment',
    'aosr.config.directivity_defaults',
    'aosr.config.fem_lane',
    'aosr.config.frequency_axis',
    'aosr.config.furniture_materials',
    'aosr.config.paths',
    'aosr.config.physics_constants',
    'aosr.config.porous_model_ranges',
    'aosr.config.precision_contracts',
    'aosr.config.source_reference',
    'aosr.config.speaker_directivity',
    'aosr.config.three_lane_crossover',
    'aosr.geometry',
    'aosr.geometry.furniture',
    'aosr.geometry.shoebox',
    'aosr.geometry.shoebox_mesh',
    'aosr.materials',
    'aosr.materials.catalog_absorption',
    'aosr.materials.construction',
    'aosr.materials.construction_models',
    'aosr.materials.furniture_materials',
    'aosr.materials.scattering_defaults',
    'aosr.materials.transfer_matrix',
    'aosr.physics',
    'aosr.physics.amplitude',
    'aosr.physics.compare',
    'aosr.physics.crossover',
    'aosr.physics.fem_batch',
    'aosr.physics.fem_helmholtz',
    'aosr.physics.finite_reflector',
    'aosr.physics.furniture_paths',
    'aosr.physics.furniture_scene',
    'aosr.physics.geometric_lane',
    'aosr.physics.late_decay',
    'aosr.physics.late_energy',
    'aosr.physics.receivers',
    'aosr.physics.reflection_screen',
    'aosr.physics.report_facts',
    'aosr.physics.report_furniture',
    'aosr.physics.report_io',
    'aosr.physics.report_measured',
    'aosr.physics.report_output',
    'aosr.physics.report_path_output',
    'aosr.physics.report_path_table',
    'aosr.physics.report_source',
    'aosr.physics.report_surfaces',
    'aosr.physics.report_validation',
    'aosr.physics.room_path_output',
    'aosr.physics.room_paths',
    'aosr.physics.source_directivity',
    'aosr.physics.third_octave_decay',
    'aosr.physics.three_lane_report',
    'aosr.physics.three_lane_report_batch',
    'aosr.physics.three_lane_report_geometry',
    'aosr.physics.three_lane_report_materials',
    'aosr.physics.totals',
    'aosr.physics.wall_reflections',
    'aosr.physics.wall_surfaces',
    'aosr.reporting',
    'aosr.reporting.fem_slices',
    'aosr.reporting.furniture_layout',
    'aosr.reporting.physics_stage',
    'aosr.reporting.scheme',
    'aosr.reporting.validation',
    'aosr.runtime',
    'aosr.scoring',
    'aosr.scoring.channel_group',
    'aosr.scoring.receiver_set',
)

DATA_FILES = (
    config_path('fem_lane.toml').name,
    config_path('furniture_materials.toml').name,
    config_path('physics_constants.toml').name,
    config_path('porous_model_ranges.toml').name,
)

# 程式摘要雜湊的是 ast.dump 的文字，格式跟著 Python 小版本走：直譯器小版本不同時要重錄，那不是物理變更。
PYTHON_MINOR = (3, 12)

# 物理身分變更：#309 之五丙第二批；三路當次構造註記合併存報表、固定輸出省略此欄，材料模型描述補中文辨識名，外推頻點分低高兩段寫；審查修補：固定阻抗描述改成可直接上畫面的中文、外推頻點綱要不重複。
# 出處：之五丙第二批，以正式 physics_identity_parts 實跑重釘；閉包、資料清單與模態摘要不變。
# #309 實測阻抗表：格式、表內外計算、註記與接縫輸出。
# 程式閉包加入接縫報告，能力表新增實測材料列；物理結果需重算。
# #435 物理變更：新增正式擺位搜尋 48 點低頻軸與軸說明；保留 24 點預設與 1 Hz 驗證，舊結果照現有機制標重算。
# #703 物理變更：家具整件原始吸音量加進 Eyring、交接頻率、ART、T20／T30；模態閉包不變。
# #309 施工一：四種面層、背面貼布、流阻來源欄、Komatsu 4,000 放行；正式入口實跑重錄，閉包與檔名清單不變。
# #990 物理變更：新增材料種類「貼在原牆上」（新增層背後接原牆阻抗）；正式入口實跑重錄，閉包與檔名清單不變，既有構造逐位不變。
CODE_DIGEST = "4a1f9bd59c0c8c2aa339f62721c4b66ec4a4f26fc1949127629cf734599d7496"
