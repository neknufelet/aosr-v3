"""材料編輯頁（#1082 段 2-2a）：列內建與使用者材料、存與刪使用者材料；格式與檢查全交給材料庫（#1086）。

使用者材料庫是網頁資料夾裡的一份 JSON（:func:`user_library_path`，之後 1-3 的材料選單、2-3a 的搜尋也讀這一份）。
每次讀寫都走材料庫自己的 ``load_user_library``／``dump_user_library``：內建唯讀、代號與顯示名不准撞。
檔案讀不進來（壞掉、或後來新增的內建跟它撞名）時照實講原因、拒絕寫入——不當成空的，否則下一次存檔會把檔洗掉。
寫入先寫同資料夾暫存檔再換名；檢查不過就不動原檔，一個位元組都不變。
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Final, cast

from pydantic import BaseModel, TypeAdapter, ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Route

from aosr.gui.problem_text import MATERIAL_MODELS, _plain_material_message, material_form_spec
from aosr.materials.library import (
    PROVENANCE_LABELS,
    FixedEntry,
    LibraryEntry,
    MeasuredPoint,
    MeasuredTable,
    OptimizableEntry,
    ThicknessSpec,
    basis_label,
    builtin_entries,
    dump_user_library,
    load_user_library,
)

USER_LIBRARY_NAME: Final = "materials.json"
PAGE: Final = Path(__file__).resolve().parent / "static" / "materials.html"
LAYER_KINDS: Final = ("porous", "air", "board", "fabric", "micro_perforated", "perforated", "slotted")
FIXED_KINDS: Final = ("assembly", "constant_impedance", "measured_impedance")
ENTRY_KIND_LABELS: Final = {"optimizable": "可最佳化材料", "fixed": "固定材料"}
_ENTRY: Final[TypeAdapter[LibraryEntry]] = TypeAdapter(LibraryEntry)
_LIBRARY_MODELS: Final[tuple[type[BaseModel], ...]] = (OptimizableEntry, FixedEntry, ThicknessSpec, MeasuredTable, MeasuredPoint)
_INDEX_WORDS: Final = {"layers": "第 {n} 層", "points": "第 {n} 列", "choices_m": "第 {n} 個產品厚度",
                       "links": "第 {n} 行", "notes": "第 {n} 行", "preset_members": "第 {n} 個"}
_LIST_NAMES: Final = frozenset({"layers", "points"})
_SHORT_LABELS: Final = {"range_m": "範圍", "base_assembly": "底構造", "porous": "多孔層", "layers": "層",
                        "points": "表格列", "id": "代號", "label": "顯示名", "source": "出處與條件", "material": "材料",
                        "provenance": "數字從哪來", "links": "出處網址", "notes": "註記"}
_TYPE_WORDS: Final = {
    "missing": "必填", "float_type": "要填數字", "float_parsing": "要填數字", "finite_number": "要填有限的數字",
    "bool_type": "要選是或否", "string_type": "要填文字", "string_too_short": "不可空白",
    "string_pattern_mismatch": "格式不對：小寫英文字母開頭，只能用小寫英數與底線",
    "extra_forbidden": "多了不認得的欄位", "union_tag_invalid": "種類不認得", "union_tag_not_found": "缺種類",
    "tuple_type": "要是一串值", "list_type": "要是一串值", "dict_type": "格式不對", "model_type": "格式不對",
}


class MaterialEditError(ValueError):
    """存或刪不成的原因（中文）；``status`` 是要回的狀態碼。"""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def user_library_path(data_dir: Path) -> Path:
    """使用者材料庫在網頁資料夾裡的位置；編輯頁、1-3 材料選單、2-3a 搜尋共用這一個。"""
    return data_dir / USER_LIBRARY_NAME


def _description_label(description: str) -> str:
    inner = description.split("（", 1)[-1]
    for mark in ("，", "；", "）"):
        inner = inner.split(mark, 1)[0]
    return inner


def _field_label(name: str, kind: str | None) -> str:
    spec = material_form_spec().get(kind or "")
    if spec is not None:
        for table in ("fields", "point_fields"):
            fields = cast(dict[str, dict[str, object]], spec.get(table, {}))
            if name in fields:
                return str(fields[name]["label"])
    for model in _LIBRARY_MODELS:
        field = model.model_fields.get(name)
        if field is not None and field.description:
            return _description_label(field.description)
    return {"kind": "種類"}.get(name, name)


def _location(loc: Sequence[int | str]) -> str:
    words: list[str] = []
    labels = {str(kind): str(spec["label"]) for kind, spec in material_form_spec().items()}
    kind: str | None = None
    for index, part in enumerate(loc):
        previous = loc[index - 1] if index else None
        following = loc[index + 1] if index + 1 < len(loc) else None
        if isinstance(part, int):
            if previous == "range_m":
                words.append(("下限", "上限")[part] if part in (0, 1) else f"第 {part + 1} 個")
            else:
                words.append(_INDEX_WORDS.get(str(previous), "第 {n} 項").format(n=part + 1))
        elif part in ENTRY_KIND_LABELS or (part == "material" and following is not None) or (
                part in _LIST_NAMES and isinstance(following, int)):
            pass  # 材料種類那一層、「材料」那一層、清單名本身不寫：畫面上已經在那一格，後面接第幾層／第幾列
        elif part in labels and (previous == "material" or isinstance(previous, int)):
            kind = part
            words.append(labels[part])
        else:
            words.append(_SHORT_LABELS.get(part) or _field_label(part, kind))
            kind = {"porous": "porous", "base_assembly": "assembly"}.get(part, kind)
    return "›".join(words)


def _number(value: object) -> object:
    return int(value) if isinstance(value, float) and value.is_integer() else value


def _message(error: dict[str, object]) -> str:
    kind = str(error["type"])
    context = {key: _number(value) for key, value in cast(dict[str, object], error.get("ctx") or {}).items()}
    if kind == "value_error":
        # 物理模型自己的檢查訊息會帶模型與欄位代號；跟方案輸入頁同一支換成表單上的中文
        return _plain_material_message(str(context.get("error", error["msg"])))
    if "input" in error and error["input"] is None:
        return "必填"  # 表單空白送的是 null：講「必填」，不印檢查器的英文代碼
    bounds = {"greater_than": "要大於 {gt}", "greater_than_equal": "要大於或等於 {ge}", "less_than": "要小於 {lt}",
              "less_than_equal": "要小於或等於 {le}", "too_short": "至少要 {min_length} 筆", "too_long": "最多 {max_length} 筆"}
    if kind in bounds:
        return bounds[kind].format(**context)
    if kind == "literal_error":
        return f"只能是 {context.get('expected', '')}"
    return _TYPE_WORDS.get(kind, f"格式不對（{kind}）")


def _shadowed(error: dict[str, object], errors: Sequence[dict[str, object]]) -> bool:
    """清單裡有一筆壞掉時，檢查器另外把整串當成少於下限再報一次；底下已有更細的錯就不重複講。"""
    if error["type"] != "too_short":
        return False
    loc = tuple(cast(Sequence[int | str], error["loc"]))
    return any(tuple(cast(Sequence[int | str], other["loc"]))[:len(loc)] == loc and other is not error for other in errors)


def plain_error(exc: Exception) -> str:
    """把材料庫的檢查錯誤寫成中文：哪一格、為什麼；多條用分號接。"""
    if isinstance(exc, ValidationError):
        parts = []
        errors = cast(list[dict[str, object]], exc.errors(include_url=False))
        for error in errors:
            if _shadowed(error, errors):
                continue
            where = _location(cast(Sequence[int | str], error["loc"]))
            parts.append(f"{where}：{_message(error)}" if where else _message(error))
        return "；".join(dict.fromkeys(parts))
    if isinstance(exc, json.JSONDecodeError):
        return "使用者材料庫不是有效的 JSON"
    return str(exc)


def _required_fields() -> dict[str, list[str]]:
    return {kind: [name for name, field in MATERIAL_MODELS[kind].model_fields.items() if field.is_required()]
            for kind in (*LAYER_KINDS, *FIXED_KINDS)}


class MaterialStore:
    """一個網頁資料夾的使用者材料庫；同一時間只准一個寫入。"""

    def __init__(self, data_dir: Path) -> None:
        self.path = user_library_path(data_dir)
        self._lock = threading.Lock()

    def _user_entries(self) -> tuple[LibraryEntry, ...]:
        try:
            return load_user_library(self.path)
        except (ValueError, ValidationError, OSError) as exc:
            raise MaterialEditError(f"使用者材料庫（{self.path.name}）讀不進來，先修好檔案再存：{plain_error(exc)}", 409) from exc

    def listing(self) -> dict[str, object]:
        """內建在前、使用者在後；使用者檔讀不進來時照講原因，清單只列內建。"""
        problem = None
        try:
            user = self._user_entries()
        except MaterialEditError as exc:
            user, problem = (), str(exc)
        rows = [{"entry": entry.model_dump(mode="json"), "builtin": True, "basis": basis_label(entry)}
                for entry in builtin_entries()]
        rows += [{"entry": entry.model_dump(mode="json"), "builtin": False, "basis": basis_label(entry)} for entry in user]
        return {"materials": rows, "user_file_problem": problem, "material_form": material_form_spec(),
                "required_fields": _required_fields(), "layer_kinds": list(LAYER_KINDS), "fixed_kinds": list(FIXED_KINDS),
                "provenance_labels": PROVENANCE_LABELS, "entry_kind_labels": ENTRY_KIND_LABELS}

    def _write(self, entries: tuple[LibraryEntry, ...]) -> None:
        text = dump_user_library(entries)
        descriptor, name = tempfile.mkstemp(dir=self.path.parent, prefix=f".{self.path.name}.", suffix=".tmp")
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(text)
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def save(self, material_id: str, document: object, *, create: bool = False) -> LibraryEntry:
        """新增（``create``：代號已在就拒收，不默默蓋掉）或改一筆使用者材料；內建代號、網址代號跟內文對不上、檢查不過一律不寫。"""
        if material_id in {entry.id for entry in builtin_entries()}:
            raise MaterialEditError("內建材料唯讀，要改請先複製成新材料", 403)
        try:
            entry = _ENTRY.validate_python(document)
        except ValidationError as exc:
            raise MaterialEditError(plain_error(exc)) from exc
        if entry.id != material_id:
            raise MaterialEditError(f"網址的代號 {material_id} 跟材料內文的代號 {entry.id} 不一樣")
        with self._lock:
            current = self._user_entries()
            ids = [item.id for item in current]
            if create and material_id in ids:
                raise MaterialEditError(f"已經有代號 {material_id} 的材料；換一個代號，或打開那一筆再改", 409)
            taken = next((item for item in (*builtin_entries(), *current)
                          if item.id != material_id and item.label == entry.label), None)
            if taken is not None:
                raise MaterialEditError(f"顯示名「{entry.label}」已經被代號 {taken.id} 用了，換一個顯示名", 409)
            entries = (tuple(entry if item.id == material_id else item for item in current) if material_id in ids
                       else (*current, entry))
            try:
                self._write(entries)
            except (ValueError, ValidationError) as exc:
                raise MaterialEditError(plain_error(exc)) from exc
        return entry

    def delete(self, material_id: str) -> None:
        """刪一筆使用者材料；內建不准刪。"""
        if material_id in {entry.id for entry in builtin_entries()}:
            raise MaterialEditError("內建材料唯讀，不能刪", 403)
        with self._lock:
            current = self._user_entries()
            if material_id not in {item.id for item in current}:
                raise MaterialEditError(f"沒有這筆使用者材料：{material_id}", 404)
            self._write(tuple(item for item in current if item.id != material_id))


def material_routes(data_dir: Path) -> list[Route]:
    """材料編輯頁與它的三個介面：清單、存（PUT；``If-None-Match: *`` 表示新增）、刪（POST）。"""
    store = MaterialStore(data_dir)

    def failed(exc: MaterialEditError) -> JSONResponse:
        return JSONResponse({"error": str(exc)}, status_code=exc.status)

    async def page(request: Request) -> Response:
        return FileResponse(PAGE, media_type="text/html")

    async def listing(request: Request) -> Response:
        return JSONResponse(await run_in_threadpool(store.listing))

    async def item(request: Request) -> Response:
        material_id = request.path_params["material_id"]
        document = cast(object, await request.json())
        create = request.headers.get("if-none-match") == "*"
        try:
            entry = await run_in_threadpool(lambda: store.save(material_id, document, create=create))
        except MaterialEditError as exc:
            return failed(exc)
        return JSONResponse({"material_id": entry.id, "message": "材料已儲存"})

    async def delete(request: Request) -> Response:
        try:
            await run_in_threadpool(store.delete, request.path_params["material_id"])
        except MaterialEditError as exc:
            return failed(exc)
        return JSONResponse({"material_id": request.path_params["material_id"], "message": "材料已刪除"})

    return [Route("/materials", page), Route("/api/materials", listing),
            Route("/api/materials/{material_id}", item, methods=["PUT"]),
            Route("/api/materials/{material_id}/delete", delete, methods=["POST"])]
