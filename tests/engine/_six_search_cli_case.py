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


@pytest.fixture(scope='module')
def operators() -> FullOperators:
    return export_room_operators(ROOM, SOURCES, SEATS, density_kg_m3=DENSITY, sound_speed_m_s=SOUND_SPEED,
        physics_identity='phys-v1:'+'0'*64, max_frequency_hz=300.)


@pytest.fixture(scope='module')
def model(operators: FullOperators, tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> ReducedModel:
    def produce() -> str:
        directory = tmp_path_factory.mktemp('six-search-cli-model')
        path = directory / 'rom.npz'
        save_reduced(build_reduced_model(operators, AXIS, TRAINING, settings=_settings()), path)
        return str(path)
    return load_reduced(Path(shared_json(tmp_path_factory, worker_id, 'six-search-cli-model', produce)))
