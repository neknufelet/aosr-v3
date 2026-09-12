"""新家第 5 層：物理。

這個檔刻意不 re-export 任何東西，理由跟套件根的門面一樣（規矩卡
``layers-import-downward-only`` 把資料夾的 ``__init__.py`` 當成站在所有層之上的門面）。
要拿哪一支就寫全名（``from aosr.physics.room_paths import image_source_paths``）。
"""
from __future__ import annotations
