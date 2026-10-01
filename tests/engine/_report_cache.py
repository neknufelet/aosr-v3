"""同一次 pytest 的報表上游資料；xdist 工人共用暫存根與檔案鎖。"""
from __future__ import annotations

import fcntl
import hashlib
import inspect
import json
from functools import partial
import pickle
from collections.abc import Callable, Iterator, Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from contextlib import contextmanager
from contextvars import ContextVar
from typing import ParamSpec, TypeVar, cast

import pytest
from pydantic import BaseModel, ConfigDict

from aosr.physics import report_io, three_lane_report, three_lane_report_batch
from aosr.physics.reflection_screen import ReflectionScreen, build_reflection_screen
from aosr.physics.reflection_window import ReflectionWindow, build_reflection_window
from aosr.physics.report_output import output_from_report
from aosr.physics.third_octave_decay import ThirdOctaveDecay, build_third_octave_decay
from tests.engine import _source_model_control as stand_ins

_T = TypeVar("_T")
_P = ParamSpec("_P")
_PAIR_CACHE: ContextVar[tuple[pytest.TempPathFactory, str]] = ContextVar("control_pair_cache")
_CONTROL_STAND_INS = (
    (three_lane_report, "_solve_fem_energy", stand_ins.fake_fem_energy),
    (three_lane_report, "_solve_report_late_decay", stand_ins.fast_late_decay),
    (three_lane_report_batch, "solve_geometric_late_energy", stand_ins.fake_late_energy),
)


def shared_report(
    tmp_path_factory: pytest.TempPathFactory, worker_id: str, name: str,
    produce: Callable[[], _T],
) -> _T:
    """同一跑只求解一次；每跑的新暫存根讓產品突變必須重新求解。"""
    if worker_id == "master":
        return produce()
    root = tmp_path_factory.getbasetemp().parent
    path = root / f"{name}.pickle"
    with (root / f"{name}.lock").open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if path.exists():
            return cast(_T, pickle.loads(path.read_bytes()))
        result = produce()
        temporary = path.with_suffix(f".{worker_id}.tmp")
        try:
            temporary.write_bytes(pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL))
            restored = cast(_T, pickle.loads(temporary.read_bytes()))
            assert restored == result
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return restored


def _key_value(value: object) -> object:
    """轉成穩定的完整值；模型不把行程各自排序的 fields_set 寫進摘要。"""
    if isinstance(value, BaseModel):
        return (type(value).__module__, type(value).__qualname__,
                _key_value(value.model_dump(mode="json")))
    if isinstance(value, Enum):
        return (type(value).__module__, type(value).__qualname__, _key_value(value.value))
    if is_dataclass(value) and not isinstance(value, type):
        return (type(value).__module__, type(value).__qualname__,
                tuple((field.name, _key_value(getattr(value, field.name))) for field in fields(value)))
    if isinstance(value, Mapping):
        return tuple(sorted(((_key_value(key), _key_value(item)) for key, item in value.items()), key=repr))
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, tuple(_key_value(item) for item in value))
    if isinstance(value, (set, frozenset)):
        return (type(value).__name__, tuple(sorted((_key_value(item) for item in value), key=repr)))
    if isinstance(value, Path):
        return ("path", str(value))
    if value is None or isinstance(value, (str, bytes, bool, int, float, complex)):
        return value
    raise TypeError(f"報表呼叫的參數不能序列化：{type(value).__name__}")


def shared_report_by_input(
    tmp_path_factory: pytest.TempPathFactory, worker_id: str,
    full_input: object, produce: Callable[[], _T],
) -> _T:
    """鍵只取完整輸入與替身標籤的摘要，不取候選或呼叫者自選的名字。"""
    return shared_report(tmp_path_factory, worker_id, report_key(full_input), produce)


def report_key(full_input: object) -> str:
    """共用檔名只由完整輸入的摘要生成。"""
    key = hashlib.sha256(pickle.dumps(_key_value(full_input), protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()
    return f"report-{key}"


def shared_call(
    tmp_path_factory: pytest.TempPathFactory, worker_id: str,
    function: Callable[_P, _T], stand_in_labels: tuple[str, ...],
    *args: _P.args, **kwargs: _P.kwargs,
) -> _T:
    """收被包函式的實際簽章全部引數，連缺省值與替身標籤一起入鍵。"""
    base = function.func if isinstance(function, partial) else function
    key_args = function.args + args if isinstance(function, partial) else args
    key_kwargs = function.keywords | kwargs if isinstance(function, partial) else kwargs
    arguments = inspect.signature(base).bind(*key_args, **key_kwargs)
    arguments.apply_defaults()
    full_input = (base.__module__, base.__qualname__, arguments.arguments, stand_in_labels)
    return shared_report_by_input(tmp_path_factory, worker_id, full_input,
                                  lambda: function(*args, **kwargs))


@contextmanager
def control_pair_cache_context(
    tmp_path_factory: pytest.TempPathFactory, worker_id: str,
) -> Iterator[None]:
    """只提供本次 pytest 的暫存根，不替換 _solve 的模組屬性。"""
    token = _PAIR_CACHE.set((tmp_path_factory, worker_id))
    try:
        yield
    finally:
        _PAIR_CACHE.reset(token)


@pytest.fixture(autouse=True)
def control_pair_cache(
    tmp_path_factory: pytest.TempPathFactory, worker_id: str,
) -> Iterator[None]:
    with control_pair_cache_context(tmp_path_factory, worker_id):
        yield


class ControlPairPhysics(BaseModel):
    """單對的物理輸出；角色、代號、出身標籤由每題自己組。"""
    model_config = ConfigDict(frozen=True)
    report: report_io.ReportOutput
    screen: ReflectionScreen
    window: ReflectionWindow
    third_octave_decay: ThirdOctaveDecay


def _control_pair(inputs: report_io.ReportInput, window_s: float) -> str:
    for module, name, stand_in in _CONTROL_STAND_INS:
        assert getattr(module, name) is stand_in, name
    solved = report_io.solver_inputs(inputs)
    raw = three_lane_report.solve_three_lane_report(**solved._asdict())
    lane = raw.geometric_lane
    calculated = ControlPairPhysics(
        report=output_from_report(raw, inputs=inputs, with_points=True, path_table_inputs=solved),
        screen=build_reflection_screen(inputs, lane.frequencies_hz),
        window=build_reflection_window(inputs, frequencies_hz=lane.frequencies_hz,
                                       scattering_coefficient=lane.scattering, window_s=window_s),
        third_octave_decay=build_third_octave_decay(raw, inputs),
    )
    text = calculated.model_dump_json()
    assert ControlPairPhysics.model_validate_json(text) == calculated
    return text


def shared_control_pair(inputs: report_io.ReportInput, window_s: float) -> ControlPairPhysics:
    """全向與解析共用同一支生產者；模型內容在實際 ReportInput 的鍵裡。"""
    for module, name, stand_in in _CONTROL_STAND_INS:
        assert getattr(module, name) is stand_in, name
    tmp_path_factory, worker_id = _PAIR_CACHE.get()
    labels = tuple(f"{fake.__module__}.{fake.__qualname__}" for _, _, fake in _CONTROL_STAND_INS)
    text = shared_report_by_input(
        tmp_path_factory, worker_id,
        (_control_pair.__module__, _control_pair.__qualname__, json.dumps(inputs.model_dump(mode="json"), sort_keys=True), window_s, labels),
        lambda: _control_pair(inputs, window_s),
    )
    return ControlPairPhysics.model_validate_json(text)


def shared_control_omnidirectional_report(
    tmp_path_factory: pytest.TempPathFactory, worker_id: str, inputs: report_io.ReportInput,
) -> three_lane_report.ThreeLaneReport:
    """控制組全向報表：同一份實際求解輸入、同三替身；生產者自行掛替身。"""
    assert inputs.source_model.kind.value == "omnidirectional"
    solved = report_io.solver_inputs(inputs)
    with pytest.MonkeyPatch.context() as patch:
        for module, name, fake in _CONTROL_STAND_INS:
            patch.setattr(module, name, fake)
        labels = tuple(f"{fake.__module__}.{fake.__qualname__}" for _, _, fake in _CONTROL_STAND_INS)
        return shared_call(tmp_path_factory, worker_id, three_lane_report.solve_three_lane_report,
                           labels, **solved._asdict())
