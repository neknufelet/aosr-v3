"""六類常數兩題要在雙精度行程裡建常數的斷言；由 ``test_six_category_jax`` 在冷子行程呼叫。"""
from __future__ import annotations

import hashlib
import sys
from unittest.mock import patch

import pytest

import numpy as np

from aosr.config.capabilities import load_capabilities
from aosr.config.paths import config_path
from aosr.config.quality_targets import QualityPurpose, load_quality_targets
from aosr.config.precision_contracts import default_precision_contracts_path, furniture_contact_rel
from aosr.physics.furniture_scene import furniture_lane_inputs
from aosr.physics.furniture_paths import single_bounce_furniture_paths
from aosr.scoring.reflections_jax import reflection_setting
from aosr.reporting.validation import checked_inputs
from aosr.scoring import _six_category_build as builder
from aosr.scoring.reverberation_cost import target_intervals
from tests.engine._directivity import DIRECTIVITY
from aosr.gradient.boundary import scheme_patch_keys, surface_ids
from tests.engine import _six_category_jax_cases as cases


def _fingerprint(constants: tuple[object, ...], names: tuple[str, ...]) -> str:
    """帳本 objective_fingerprint 同一個逐欄算法；不收容器的 repr。"""
    digest = hashlib.sha256()
    for name, value in zip(names, constants, strict=True):
        array = np.asarray(value)
        for part in (name.encode(), str(array.dtype).encode('ascii'), ','.join(str(n) for n in array.shape).encode('ascii'), np.ascontiguousarray(array).tobytes()):
            digest.update(len(part).to_bytes(8,'big'))
            digest.update(part)
    return digest.hexdigest()


def fingerprint_check() -> None:
    """逐欄植錯：漏掉任一幾何、索引、設定或巢狀音色葉子就紅。"""
    c = cases.constants(cases.scheme_case('zoned'))
    original = _fingerprint(c, c._fields)
    for name, value in zip(c._fields, c, strict=True):
        assert isinstance(value, (np.ndarray, int, float, complex, bool))
        changed = np.asarray(value).copy()
        assert changed.dtype.kind != 'O' and changed.size > 0, name
        if changed.dtype.kind == 'b':
            changed.flat[0] = ~changed.flat[0]
        elif changed.dtype.kind == 'U':
            changed.flat[0] = '#'+str(changed.flat[0])[1:]
        else:
            changed.flat[0] += 1
        assert _fingerprint(c._replace(**{name:changed}), c._fields) != original, name


def patch_order_check(name: str) -> None:
    """抓：面片排序抹掉方案區序，或幾何表與邊界面片表不一致。"""
    scheme = cases.scheme_case(name)
    c = cases.constants(scheme)
    assert tuple(c.patch_ids) == surface_ids(scheme_patch_keys(scheme))
    assert tuple(c.patch_ids[2:4]) == (('x0@b','x0@a') if name=='zoned' else ('x0@a','x0@b'))


if __name__ == '__main__':
    if sys.argv[1] == 'design-mask':
        assert 'constant_material' not in cases.constants(cases.scheme_case('whole'))._fields
    elif sys.argv[1] == 'furniture-window':
        scheme = cases.scheme_case('late_furniture')
        c = cases.constants(scheme)
        _, pairs = checked_inputs(scheme, capabilities=load_capabilities(config_path('capabilities.toml')), directivity=DIRECTIVITY)
        window = reflection_setting(load_quality_targets(cases.TARGETS).purpose(scheme.purpose), 'window_upper_ms').value
        assert isinstance(window, (float, int))
        for channel in scheme.channel_group.channels:
            item = pairs[channel.speaker_id, scheme.receiver_set.primary.receiver_id][1]
            room = item.room_m
            furniture = furniture_lane_inputs(item.furniture, (room.Lx, room.Ly, room.Lz), tuple(c.frequencies_hz), float(c.rho_c),
                contact_rel=furniture_contact_rel(default_precision_contracts_path()))
            assert furniture is not None
            paths = single_bounce_furniture_paths(item.source_m.as_tuple(), item.receiver_m.as_tuple(), furniture.furniture,
                c=item.sound_speed_m_s, margin_m=furniture.margin_m)
            assert paths
            direct = np.linalg.norm(np.asarray(item.source_m.as_tuple())-item.receiver_m.as_tuple())/item.sound_speed_m_s
            assert all(p.delay_s-direct > float(window)/1000 for p in paths)
        assert np.count_nonzero(c.reflection_furniture_energy) == 0
    elif sys.argv[1] == 'band-guard':
        real = target_intervals
        def changed(purpose: QualityPurpose) -> dict[float, tuple[float, float]]:
            intervals = real(purpose)
            return dict(tuple(intervals.items())[1:])
        with patch.object(builder, 'target_intervals', changed):
            with pytest.raises(ValueError, match='帶中心'):
                cases.constants(cases.scheme_case('whole'))
    elif sys.argv[1] == 'mismatched-order':
        _, original = checked_inputs(cases.scheme_case('zoned'),
            capabilities=load_capabilities(config_path('capabilities.toml')), directivity=DIRECTIVITY)
        with pytest.raises(ValueError, match='面片順序'):
            builder.build_constants(cases.scheme_case('reordered'), pair_inputs={key:value[1] for key,value in original.items()})
    elif sys.argv[1] == 'fingerprint':
        fingerprint_check()
    elif sys.argv[1] == 'patch-order':
        patch_order_check(sys.argv[2])
    else:
        raise SystemExit(f'未知檢查 {sys.argv[1]}')
