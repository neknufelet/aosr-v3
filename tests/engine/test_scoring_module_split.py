"""契約底座與未評估原因搬家後，兩邊拿到的必須是同一個東西（票 #351）。

``contract.py`` 與 ``ranking.py`` 都貼著單檔行數上限，新的類別 payload 住自己的模組、要用契約底座，
所以底座搬去 ``contract_base``、未評估原因搬去 ``category_registry``，舊的入口原名再匯出。
這一份守兩件事：舊入口拿到的是同一個物件（不是抄一份——抄的那份一改就分叉，排名層認得的
原因碼跟評估器發的就對不上）；底座不准回頭拿 ``contract``（否則新 payload 一放進聯集就繞成一圈）。
"""
from __future__ import annotations

import ast
import inspect

import pytest

from aosr.scoring import category_registry, contract, contract_base, ranking


@pytest.mark.parametrize(
    "name",
    ["FROZEN", "Flag", "FrequencyRange", "MetricState", "ReasonCode"],
)
def test_contract_reexports_the_base_objects_themselves(name: str) -> None:
    assert getattr(contract, name) is getattr(contract_base, name)


def test_contract_models_are_built_on_the_shared_frozen_base() -> None:
    assert issubclass(contract.CategoryEvaluation, contract_base.FrozenModel)
    # pydantic 每個類別各抄一份設定，所以比內容不比物件
    assert contract.RawQuantity.model_config == contract_base.FROZEN


def test_ranking_reexports_the_registry_not_evaluated_reason() -> None:
    assert ranking.NotEvaluatedReason is category_registry.NotEvaluatedReason


def _absolute(node: ast.ImportFrom) -> str:
    """相對寫法（``from . import x``、``from .contract import y``）換算成完整名字再判。"""
    if node.level == 0:
        return node.module or ""
    package = contract_base.__name__.split(".")[: -node.level]
    return ".".join([*package, *([node.module] if node.module else [])])


def test_contract_base_takes_nothing_from_the_scoring_layer() -> None:
    tree = ast.parse(inspect.getsource(contract_base))
    imported = {
        _absolute(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    scoring = {module for module in imported if module.startswith("aosr.scoring")}
    assert scoring == set(), f"契約底座回頭拿了評分層：{sorted(scoring)}"
