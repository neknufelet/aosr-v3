"""測試共用的正式指向性登記簿；範圍和值都從同一份檔載入。"""

from aosr.config.directivity_defaults import load_directivity_defaults
from aosr.config.paths import config_path


DIRECTIVITY = load_directivity_defaults(config_path("directivity_defaults.toml"))
