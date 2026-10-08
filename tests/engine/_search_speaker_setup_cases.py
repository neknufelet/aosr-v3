"""第五步手寫幾何：3–4–5 朝向與主線端到端房間；不從被測限制取答案。"""
from pathlib import Path
from typing import cast

from aosr.reporting.scheme import Scheme
from aosr.search.layout_settings import LayoutSettings
from aosr.search.settings import SearchSettings
from aosr.search.store import SearchIdentity, SearchStore
from tests.engine._furniture_cases import reference_document, relative_item
from tests.engine._search_run_cases import ENGINE, registry_copy
from tests.engine._search_store_cases import settings_document
from tests.engine._speaker_setup_cases import setup
from aosr.reporting.evaluation import purpose_settings


def geometric(mount: str, *, forward: float = 0.7, bottom: float = 0.7,
              thickness: float = 0.04, center: float = 0.2) -> tuple[Scheme, LayoutSettings]:
    doc = reference_document()
    height = {"stand": 1.2, "desk": (bottom + thickness) + center, "floor": center}[mount]
    doc["speaker_setup"] = setup("floorstanding" if mount == "floor" else "bookshelf", mount)
    cabinet = {"width_m": 0.2, "depth_m": 0.25,
        "height_m": 1.1 if mount == "floor" else 0.35,
        "acoustic_center_behind_front_m": 0.0, "acoustic_center_above_bottom_m": center}
    cast(dict[str, object], doc["speaker_setup"])["cabinet"] = cabinet
    doc["speakers"] = {"left": {"x": 1.0, "y": 1.25, "z": height},
                       "right": {"x": 1.0, "y": 2.75, "z": height}}
    doc["receiver_set"] = {"points": [{"receiver_id": "main", "role": "primary",
        "position_m": [2.0, 2.0, 1.2], "importance": 1.0},
        {"receiver_id": "side", "role": "surrounding", "position_m": [2.1, 2.0, 1.2],
         "importance": 1.0, "direction_relative_to_primary": "left"}]}
    doc["furniture"] = [relative_item(furniture_id="table", kind="desk", material="wood",
        width_m=2.0, depth_m=0.8, height_m=thickness,
        placement={"forward_m": forward, "left_m": 0.0, "bottom_height_m": bottom, "yaw_deg": 0})]
    settings = cast(dict[str, object], settings_document()["layout"]) | {"speaker_height_m": height, "cabinet": cabinet}
    return Scheme.model_validate(doc), LayoutSettings.model_validate(settings)


def flow_project(mount: str) -> Scheme:
    doc = reference_document()
    height = {"stand": 1.2, "desk": 0.935, "floor": 0.8}[mount]
    doc["speaker_setup"] = setup("floorstanding" if mount == "floor" else "bookshelf", mount)
    doc["speakers"] = {"left": {"x": 1.0, "y": 1.4, "z": height},
                       "right": {"x": 1.0, "y": 2.6, "z": height}}
    doc["receiver_set"] = {"points": [{"receiver_id": "main", "role": "primary",
        "position_m": [1.85 if mount == "desk" else 3.2, 2.0, 1.2], "importance": 1.0},
        {"receiver_id": "side", "role": "surrounding",
         "position_m": [1.95 if mount == "desk" else 3.3, 2.0, 1.2],
         "importance": 1.0, "direction_relative_to_primary": "left"}]}
    doc["furniture"] = [relative_item(furniture_id="table", kind="coffee_table" if mount == "floor" else "desk",
        material="wood", width_m=2.0, depth_m=0.6, height_m=0.03,
        placement={"forward_m": 1.0, "left_m": 0.0,
                   "bottom_height_m": 0.4 if mount == "floor" else 0.7, "yaw_deg": 0})]
    return Scheme.model_validate(doc)


def store_for(tmp_path: Path, project: Scheme, *, layout_changes: dict[str, object] | None = None,
              budget: int = 3) -> tuple[SearchStore, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    registry = registry_copy(tmp_path)
    doc = settings_document() | {"purpose": project.purpose, "budget": budget, "batch_size": budget,
                                "convergence_run": 100}
    assert project.speaker_setup is not None
    doc["layout"] = cast(dict[str, object], doc["layout"]) | {"speaker_height_m": project.speakers[next(
        channel.speaker_id for channel in project.channel_group.channels if channel.role == "left")].z,
        "cabinet": project.speaker_setup.cabinet.model_dump()} | (layout_changes or {})
    identity = SearchIdentity("phys-v1:" + "b" * 64, ENGINE, purpose_settings(registry, project.purpose))
    return SearchStore.create(tmp_path / "searches", project=project, settings=SearchSettings.model_validate(doc),
        identity=identity, versions={"python": "test", "optuna": "test", "numpy": "test"}), registry
