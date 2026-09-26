"""評分考卷共用的全向聲源模型與產品算法算出的身分。"""

from aosr.physics.report_source import SourceModelKind, SourceModelSection, SourceModelSpec
from aosr.scoring.source_model_identity import source_model_fingerprint


OMNI_SOURCE_MODEL = SourceModelSection.from_spec(
    SourceModelSpec(kind=SourceModelKind.OMNIDIRECTIONAL), None
)
OMNI_SOURCE_MODEL_FINGERPRINT = source_model_fingerprint(OMNI_SOURCE_MODEL)
