"""正式精度初始化後以 ASGI（同進程網頁請求）讀回實測結果；不開連接埠。"""
from __future__ import annotations

from aosr.runtime import configure_calculation_jax
configure_calculation_jax()

import asyncio
from collections.abc import Callable
from pathlib import Path
import sys
from unittest.mock import patch

from httpx import ASGITransport, AsyncClient
from aosr.gui.app import GuiSettings, create_app


async def inline(function: Callable[..., object], *args: object, **kwargs: object) -> object:
    """沙箱不喚醒跨執行緒 portal（請求入口）；載入、重評與顯示算法仍全部實跑。"""
    return function(*args, **kwargs)


async def main(folder: Path, identifier: str) -> None:
    app = create_app(GuiSettings(engine_commit='a' * 40, data_dir=folder))
    with patch('aosr.gui.app.run_in_threadpool', inline):
        async with AsyncClient(transport=ASGITransport(app=app), base_url='http://localhost') as client:
            response = await client.get(f'/api/results/{identifier}')
            assert response.status_code == 200, response.text
            lines = response.json()['construction_notes']
            assert any('地板' in line and '模型補值' in line for line in lines)
            assert any('底構造第 1 層' in line and '泡棉材料類別' in line for line in lines)
            assert any('底構造第 1 層' in line and '外推照算' in line for line in lines)
            assert all('底構造第 1 層：底構造第' not in line for line in lines)
            assert all('None' not in line and 'measured_table_model_fill' not in line for line in lines)


if __name__ == '__main__':
    asyncio.run(main(Path(sys.argv[1]), sys.argv[2]))
