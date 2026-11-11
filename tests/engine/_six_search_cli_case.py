"""六類可估的命令列小房；原小房的反射窗到第八階仍證明不了完整覆蓋。

只放大房殼，喇叭與座位不變；另建與驗證替身，絕不把舊網格冒充新房。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from aosr.geometry.shoebox import Room
from aosr.physics.fem_reduced import FullOperators, ReducedModel, export_room_operators
from aosr.physics.fem_reduced_build import build_reduced_model
from aosr.physics.fem_reduced_store import load_reduced, save_reduced
from tests.engine._scheme_cache import shared_json
from tests.engine.test_fem_reduced import AXIS, DENSITY, SEATS, SOUND_SPEED, SOURCES, TRAINING, _settings

ROOM = Room(2.4, 1.8, 1.5)
# 終點正式 numpy 路徑需避開反射的浮點邊界；搜尋控制保留原房殼與原判準。
ENDPOINT_ROOM = Room(2.5, 2., 1.5)
ENDPOINT_MODULES = frozenset(('test_material_gradient_endpoint_cli', 'test_gradient_endpoint_range_migration',
    'test_material_gradient_fixed_surface_cli', 'test_material_gradient_mount', 'test_material_gradient_templates_cli',
    'test_six_search_review_fixes', 'slow_material_gradient_endpoint_cli', 'slow_gradient_endpoint_range_migration',
    'slow_material_gradient_fixed_surface_cli', 'slow_material_gradient_mount',
    'slow_material_gradient_templates_cli', 'slow_six_search_review_fixes', 'slow_gui_gradient_jobs'))


@pytest.fixture(scope='module')
def operators(request: pytest.FixtureRequest) -> FullOperators:
    room = ENDPOINT_ROOM if request.module.__name__.rsplit(".", 1)[-1] in ENDPOINT_MODULES else ROOM
    return export_room_operators(room, SOURCES, SEATS, density_kg_m3=DENSITY, sound_speed_m_s=SOUND_SPEED,
        physics_identity='phys-v1:'+'0'*64, max_frequency_hz=300.)


@pytest.fixture(scope='module')
def model(operators: FullOperators, tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> ReducedModel:
    def produce() -> str:
        directory = tmp_path_factory.mktemp('six-search-cli-model')
        path = directory / 'rom.npz'
        save_reduced(build_reduced_model(operators, AXIS, TRAINING, settings=_settings()), path)
        return str(path)
    return load_reduced(Path(shared_json(tmp_path_factory, worker_id, f'six-search-cli-model-{operators.mesh_hash}', produce)))
