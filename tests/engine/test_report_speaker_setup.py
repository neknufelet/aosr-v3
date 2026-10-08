"""結果檔的方案快照沿用型別序列化；物理零件不收喇叭箱體描述。"""
import json

import pytest

from aosr.reporting.result import SchemeResult
from aosr.reporting.scheme import Scheme
from tests.engine._speaker_setup_cases import setup
from tests.engine.test_scheme_pipeline import shared_control_result


def test_result_scheme_snapshot_omits_null_and_preserves_setup(
    tmp_path_factory: pytest.TempPathFactory, worker_id: str,
) -> None:
    original = shared_control_result(tmp_path_factory, worker_id, "wall-1")
    null_scheme = Scheme.model_validate(original.scheme.model_dump() | {"speaker_setup": None})
    for scheme in (original.scheme, null_scheme):
        saved = json.loads(original.model_copy(update={"scheme": scheme}).model_dump_json())
        assert saved["scheme"] == json.loads(original.scheme.model_dump_json())
        assert "speaker_setup" not in saved["scheme"]
    described = Scheme.model_validate(original.scheme.model_dump() | {"speaker_setup": setup()})
    saved_result = original.model_copy(update={"scheme": described})
    saved = json.loads(saved_result.model_dump_json())
    assert saved["scheme"]["speaker_setup"] == setup()
    reloaded = SchemeResult.model_validate_json(saved_result.model_dump_json())
    assert reloaded.scheme.speaker_setup == described.speaker_setup
    for pair in reloaded.pairs:
        assert "speaker_setup" not in pair.input_document
