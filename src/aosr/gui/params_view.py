"""進階參數清單的網頁出口（#1343）：``/api/params`` 唯讀吐 JSON（含吸音模型目錄 ``models``）、``/params`` 唯讀「所有參數」頁。

各頁讀預設值的地方也從這裡拿（``advanced_number``／``advanced_text``），不另寫第二份數字。
清單在開機那一刻讀一次：改了清單要重開網頁才生效，跟其他設定檔一樣。
"""
from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Final

from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Route

from aosr.config.paths import CONFIG_DIR
from aosr.gui.advanced_params import AdvancedParams, Value, load_advanced_params, resolved_default
from aosr.gui.model_catalogue import catalogue_listing

REGISTRY_FILE: Final = "advanced_params.toml"
# 清單跟本套件放一起（不進 aosr/config：那會動到計算指紋）；source 指的引擎設定檔照舊在 CONFIG_DIR。
REGISTRY: Final = Path(__file__).resolve().parent / REGISTRY_FILE
CATALOGUE_FILE: Final = "model_catalogue.toml"
CATALOGUE: Final = Path(__file__).resolve().parent / CATALOGUE_FILE
RANGES: Final = CONFIG_DIR / "porous_model_ranges.toml"
PAGE: Final = Path(__file__).resolve().parent / "static" / "params.html"


@cache
def model_catalogue() -> list[dict[str, object]]:
    return catalogue_listing(CATALOGUE, RANGES)


@cache
def advanced_params() -> AdvancedParams:
    return load_advanced_params(REGISTRY)


def advanced_default(key: str) -> Value | None:
    """清單上這一條實際的預設（指到別檔的現抓）；沒有固定值回 ``None``。"""
    return resolved_default(advanced_params().by_key(key), CONFIG_DIR)


def advanced_number(key: str) -> float:
    value = advanced_default(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"進階參數 {key} 不是數字：{value!r}")
    return float(value)


def advanced_text(key: str) -> str:
    value = advanced_default(key)
    if not isinstance(value, str):
        raise TypeError(f"進階參數 {key} 不是文字：{value!r}")
    return value


def params_listing() -> dict[str, object]:
    """整份清單，預設值已解析；``default_from`` 講數字住哪一檔哪一格。"""
    registry = advanced_params()
    rows: list[dict[str, object]] = []
    for param in registry.param:
        row = param.model_dump(mode="json", exclude={"source", "default"})
        row["default"] = resolved_default(param, CONFIG_DIR)
        row["default_from"] = (f"{param.source.file} 的 {param.source.key}" if param.source is not None
                               else REGISTRY_FILE if param.default is not None else None)
        rows.append(row)
    return {"registry": REGISTRY_FILE, "pages": [page.model_dump(mode="json") for page in registry.page], "params": rows,
            "catalogue": CATALOGUE_FILE, "models": model_catalogue()}


async def _listing(request: Request) -> Response:
    return JSONResponse(params_listing())


async def _page(request: Request) -> Response:
    return FileResponse(PAGE, media_type="text/html")


def params_routes() -> list[Route]:
    return [Route("/params", _page), Route("/api/params", _listing)]
