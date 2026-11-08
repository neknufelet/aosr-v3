"""六類常數兩題要在雙精度行程裡建常數的斷言；由 ``test_six_category_jax`` 在冷子行程呼叫。"""
from __future__ import annotations

import hashlib
import sys

import numpy as np

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
    if sys.argv[1] == 'fingerprint':
        fingerprint_check()
    elif sys.argv[1] == 'patch-order':
        patch_order_check(sys.argv[2])
    else:
        raise SystemExit(f'未知檢查 {sys.argv[1]}')
