"""分段探針的同步共用結果：讀到哪段才算哪段，只有 shared_json 的檔案鎖，沒有背景行程或輪詢。"""
from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

import pytest

from tests.engine._scheme_cache import shared_json

Record = dict[str, object]


class DeferredRecord(dict[str, object]):
    """保留考卷原來的逐鍵讀法；缺鍵直接交給產物判定，不用預設值補洞。"""

    def __init__(self, read: Callable[[str], object]) -> None:
        super().__init__()
        self.read = read

    def __missing__(self, key: str) -> object:
        value = self.read(key)
        self[key] = value
        return value


class Sections:
    """同一場每段只產生一次 JSON；同一工人也記住已讀的結果（含不開 xdist 的跑法）。"""

    def __init__(self, factory: pytest.TempPathFactory, worker: str, name: str, directory: Path,
                 produce: Callable[[str], Record], prepare: Callable[[str], None] | None = None) -> None:
        self.factory, self.worker, self.name = factory, worker, name
        self.directory, self.produce = directory, produce
        self.prepare = prepare
        self.results: dict[str, Record] = {}

    def load(self, group: str) -> Record:
        if group not in self.results:
            if self.prepare is not None:
                self.prepare(group)

            def produce() -> str:
                clock = time.perf_counter()
                started = time.time()
                output = self.produce(group)
                text = json.dumps(output, ensure_ascii=False)
                (self.directory / f"fixture-{group}.json").write_text(text, encoding="utf-8")
                (self.directory / f"seconds-{group}.json").write_text(
                    json.dumps({"group": group, "seconds": time.perf_counter() - clock,
                                "started": started, "finished": time.time(), "worker": self.worker,
                                "pid": os.getpid()}), encoding="utf-8")
                return text

            text = shared_json(self.factory, self.worker, f"{self.name}-split-v1-{group}", produce)
            self.results[group] = cast(Record, json.loads(text))
        return self.results[group]

    def record(self, groups: Mapping[str, str]) -> Record:
        return DeferredRecord(lambda key: self.load(groups[key])[key])
