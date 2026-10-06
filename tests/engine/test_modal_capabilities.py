"""模態診斷能力另開考卷，不擴充已滿的能力表考卷檔。"""
from aosr.config.capabilities import load_capabilities
from aosr.config.paths import config_path
from aosr.reporting.display import LOW_FREQUENCY_DECAY_NOTE


def test_modal_diagnosis_is_experimental_with_module_and_notes() -> None:
    table = load_capabilities(config_path("capabilities.toml"))
    entry = table.for_entry("modal_diagnosis")
    assert entry.module == "aosr.reporting.modal_lookup"
    assert entry.note and entry.capability
    assert all(row.status == "experimental" and row.note for row in entry.capability)
    assert config_path("capabilities.toml").parents[2].joinpath(*entry.module.split(".")[1:]).with_suffix(".py").is_file()
    pipeline = table.for_entry("scheme_pipeline")
    assert LOW_FREQUENCY_DECAY_NOTE in pipeline.not_modeled
    assert LOW_FREQUENCY_DECAY_NOTE in pipeline.note
    assert all(LOW_FREQUENCY_DECAY_NOTE in row.note for row in pipeline.capability)
