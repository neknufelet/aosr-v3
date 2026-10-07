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
    'aosr.config.directivity_defaults',
    'aosr.config.fem_lane',
    'aosr.config.frequency_axis',
    'aosr.config.furniture_materials',
    'aosr.config.paths',
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
    'aosr.materials.furniture_materials',
    'aosr.materials.scattering_defaults',
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
    'aosr.physics.report_output',
    'aosr.physics.report_path_output',
    'aosr.physics.report_path_table',
    'aosr.physics.report_source',
    'aosr.physics.room_path_output',
    'aosr.physics.room_paths',
    'aosr.physics.source_directivity',
    'aosr.physics.third_octave_decay',
    'aosr.physics.three_lane_report',
    'aosr.physics.three_lane_report_batch',
    'aosr.physics.three_lane_report_materials',
    'aosr.physics.totals',
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
)

# 程式摘要雜湊的是 ast.dump 的文字，格式跟著 Python 小版本走：直譯器小版本不同時要重錄，那不是物理變更。
PYTHON_MINOR = (3, 12)

# 物理變更：家具接入兩軸幾何能量與三路批次，材質檔及接觸界線納入身分；空房控制組逐位不變。
CODE_DIGEST = 'becc7c2db20d9b3558238de1c367c137f92a6870e5fc718a63e78fc3d1232730'
