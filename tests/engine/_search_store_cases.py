"""搜尋存放考卷共用的合成輸入；讀取參考方案前先複製到暫存目錄。"""

import hashlib
import json
import shutil
from pathlib import Path

from aosr.reporting.result import PurposeSettings
from aosr.reporting.scheme import Scheme, load_scheme
from aosr.search.layout_settings import Cabinet, LayoutSettings, Span


def settings_document() -> dict[str, object]:
    layout = LayoutSettings(
        front_wall="x0", speaker_height_m=1.25, ear_height_m=1.25,
        front_distance_m=Span(low=0.25, high=2.0), spacing_m=Span(low=0.5, high=2.0),
        listening_distance_m=Span(low=0.5, high=3.0),
        cabinet=Cabinet(width_m=0.25, depth_m=0.25, height_m=0.5),
    )
    return {"purpose": "listening", "layout": layout.model_dump(mode="json"), "seed": 0,
            "n_startup_trials": 2, "constant_liar": True, "batch_size": 3,
            "max_workers": 1, "budget": 11, "convergence_run": 5}


def reference_project(tmp_path: Path) -> Scheme:
    source = next((Path(__file__).resolve().parents[2] / "blueprint").glob("scheme_reference_room.*"))
    target = tmp_path / "reference"
    shutil.copyfile(source, target)
    return load_scheme(target)


def purpose_settings(purpose: str) -> PurposeSettings:
    content: dict[str, object] = {"purpose": purpose, "label": "當次評分", "weights": {"example": 1.0}}
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return PurposeSettings(purpose=purpose, fingerprint=hashlib.sha256(canonical.encode()).hexdigest(), content=content)
