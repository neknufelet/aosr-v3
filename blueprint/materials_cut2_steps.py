"""票 #134 第二刀（``freq_axis``／``experiment_schema``／``material_loader``）的**零件**。

第一刀那三支（``response``／``source``／``registry``）的零件住
:mod:`blueprint.materials_cut1_steps`，**這一支不動它**：第一刀的答案已經在雲端綠過一次，
動它就等於把已經比過的那幾百格重新變成「沒比過」。這一支只做兩件事：

1. 把第一刀的記號與步驟工廠**原封不動借過來**（``from ... import``），這一刀的 case 用的
   是同一套寫法；
2. 補這一刀才需要的那幾種：**巢狀的一層表**（pydantic 的模型是一層一層套的，
   ``ExperimentConfig(frequency_axis={...})``）、**暫存目錄底下的檔**（載入器要真的讀檔），
   還有 **讀模組層級的名字**（``FREQ_AXIS`` 那種常數要進環境才讀得到它的屬性）。

**這一支不准 import donor 或新家的任何東西，也不准 import JAX、pydantic、yaml。** 真正碰
模組與碰檔案系統的是 :mod:`blueprint.materials_cut2_probe`。

**暫存目錄的位置不寫在這裡。** 表上只寫「暫存根底下那個相對名字」（:func:`under`），真的
是哪一個目錄由探針每一筆 case 現開一個；錯誤訊息裡那一段絕對路徑由探針換成
``<TMP>``——**只換那一段**，訊息其他部分一個字都不放寬。
"""
from __future__ import annotations

from blueprint.materials_cut1_steps import (
    CaseEntry,
    ModuleSpec,
    Step,
    Tagged,
    call,
    contains,
    cplx,
    false,
    flt,
    method,
    minus_inf,
    nan,
    none,
    plus_inf,
    read,
    reads,
    ref,
    report_names,
    resolve,
    seq,
    signature,
    size_of,
    strs,
    true,
    write,
)

__all__ = [
    "CaseEntry",
    "ModuleSpec",
    "Step",
    "Tagged",
    "call",
    "contains",
    "cplx",
    "false",
    "flt",
    "get",
    "make_dir",
    "make_file",
    "method",
    "minus_inf",
    "nan",
    "none",
    "obj",
    "plus_inf",
    "read",
    "reads",
    "ref",
    "report_names",
    "resolve",
    "seq",
    "signature",
    "size_of",
    "strs",
    "true",
    "under",
    "under_text",
    "write",
]


# ── 這一刀才需要的記號 ──────────────────────────────────────────────────────
def obj(**entries: Tagged | int | str) -> Tagged:
    """一層表（還原成 ``dict``）。

    pydantic 的模型是一層套一層的：``ExperimentConfig(room={"geometry_level": 0})``。
    這一格讓 case 表寫得出巢狀的輸入，而且鍵是什麼、值是什麼都走同一套記號。
    """
    return {"k": "obj", "v": dict(entries)}


def under(name: str) -> Tagged:
    """暫存根底下那一個相對名字，還原成 ``Path``（載入器收 ``str | Path`` 兩種）。"""
    return {"k": "under", "v": name}


def under_text(name: str) -> Tagged:
    """同 :func:`under`，但還原成**字串**——``str`` 那一支路也要有人走過。"""
    return {"k": "under_str", "v": name}


# ── 這一刀才需要的步驟 ──────────────────────────────────────────────────────
def make_dir(name: str) -> Step:
    """在暫存根底下開一個目錄（父目錄一起開）。"""
    return {"do": "mkdir", "name": name}


def make_file(name: str, text: str) -> Step:
    """在暫存根底下寫一個檔（UTF-8；父目錄一起開）。

    內容寫在 case 表裡，所以「這一筆餵的是哪一份 YAML」是版控裡看得到的東西。
    """
    return {"do": "file", "name": name, "value": {"k": "str", "v": text}}


def get(fn: str, to: str) -> Step:
    """把模組層級的一個名字讀進環境（``freq_axis.FREQ_AXIS`` 那種常數）。

    第一刀不需要這一種：那三支的常數都是直接凍結值。這一刀要問「``FREQ_AXIS`` 這個常數
    的 ``n_freq`` 是多少」——那得先把它放進環境才讀得到屬性。
    """
    return {"do": "get", "fn": fn, "to": to}
