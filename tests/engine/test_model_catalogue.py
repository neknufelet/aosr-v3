"""吸音模型目錄（#1343 追加）：每個類別真的在程式裡、構造格式的層與整面寫法一個不漏、
每條出處那一行真的含作者姓或標記字、Komatsu 範圍等於範圍登記簿、頁面那段預設收起。

清單從型別推（``Layer``、``SurfaceMaterial`` 兩個聯合），不手列；名字找不找得到用程式結構（ast）找，不 import 私有名。
"""
from __future__ import annotations

import ast
import re
import tomllib
import typing
from functools import cache
from pathlib import Path

import pytest
from pydantic import ValidationError
from starlette.testclient import TestClient

from aosr.config.paths import config_path
from aosr.config.porous_model_ranges import KomatsuRanges, load_porous_model_ranges
from aosr.gui.app import GuiSettings, create_app
from aosr.gui.model_catalogue import NO_RANGE, RELATED_NOTE, ModelCatalogue, load_model_catalogue
from aosr.gui.params_view import CATALOGUE, params_listing
from aosr.materials import construction_models
from aosr.materials.construction_models import Layer, PorousLayer, SurfaceMaterial

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "src" / "aosr" / "gui" / "static"


def _catalogue() -> ModelCatalogue:
    return load_model_catalogue(CATALOGUE)


@cache
def _names(path: str) -> frozenset[str]:
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    return frozenset(node.name for node in tree.body if isinstance(node, ast.ClassDef | ast.FunctionDef))


def _line(at: str) -> str:
    path, _, number = at.rpartition(":")
    lines = (ROOT / path).read_text(encoding="utf-8").splitlines()
    assert 1 <= int(number) <= len(lines), f"{at} 超出檔尾"
    return lines[int(number) - 1]


def _union(alias: object) -> set[str]:
    """``Annotated[A | B, Field(...)]`` 的成員類別名。"""
    return {member.__name__ for member in typing.get_args(typing.get_args(alias)[0])}


def test_catalogue_stays_out_of_the_identities() -> None:
    """目錄只有網頁讀：跟進階參數清單一樣住 gui 底下；gui 以外沒有人讀它。"""
    assert CATALOGUE.parent.name == "gui" and CATALOGUE.parent.parent.name == "aosr"
    readers = [path for path in (ROOT / "src" / "aosr").rglob("*.py")
               if "model_catalogue" in path.read_text(encoding="utf-8") and path.parent.name != "gui"]
    assert readers == []


def test_every_code_and_compute_name_exists_in_the_program() -> None:
    catalogue = _catalogue()
    codes = [model.code for model in catalogue.model] + [code for model in catalogue.model for code in model.compute]
    codes += [item.code for item in catalogue.not_model]
    for code in codes:
        path, _, name = code.partition("::")
        assert (ROOT / path).is_file(), f"{code}：檔不在"
        assert name in _names(path), f"{code}：{path} 最外層沒有 {name}"


def test_every_layer_and_surface_kind_is_listed() -> None:
    """構造格式的每一種層、每一種整面寫法都在目錄裡（外殼另列在 not_model 並寫理由），沒有漏的也沒有多的。"""
    catalogue = _catalogue()
    prefix = "src/aosr/materials/construction_models.py::"
    listed = {model.code.removeprefix(prefix) for model in catalogue.model if model.code.startswith(prefix)}
    shells = {item.code.removeprefix(prefix) for item in catalogue.not_model}
    kinds = _union(Layer) | _union(SurfaceMaterial)
    assert kinds <= listed | shells, f"目錄漏了 {sorted(kinds - listed - shells)}"
    assert shells <= kinds and not (shells & listed)
    assert listed - kinds <= {"_BackingFabric"}, "目錄多列了構造格式裡沒有的類別"


def test_every_porous_model_choice_is_listed() -> None:
    choices = set(typing.get_args(PorousLayer.model_fields["model"].annotation))
    variants = {model.variant for model in _catalogue().model if model.code.endswith("::PorousLayer")}
    assert variants == choices


def test_every_param_names_a_real_field() -> None:
    for model in _catalogue().model:
        for param in model.params:
            owner, _, name = param.partition(".")
            assert name in getattr(construction_models, owner).model_fields, f"{model.id}：{param}"


def test_every_source_line_holds_its_author_or_mark() -> None:
    """引用那一行要含第一作者姓；範圍、來源那一行要含 mark。檔在、行在。"""
    for model in _catalogue().model:
        if model.citation is not None:
            assert model.citation.authors[0] in _line(model.citation.at), f"{model.id} 引用 {model.citation.at}"
        if model.range.at is not None:
            assert model.range.mark is not None and model.range.mark in _line(model.range.at), f"{model.id} 範圍 {model.range.at}"
        assert model.origin.mark in _line(model.origin.at), f"{model.id} 來源 {model.origin.at}"


def test_ranges_without_a_source_say_so() -> None:
    for model in _catalogue().model:
        if model.range.registry is None and model.range.at is None:
            assert model.range.note == NO_RANGE, model.id


def test_komatsu_range_and_reference_come_from_the_range_registry() -> None:
    ranges = load_porous_model_ranges(config_path("porous_model_ranges.toml")).komatsu
    listing = params_listing()["models"]
    assert isinstance(listing, list)
    komatsu = next(model for model in listing if model["id"] == "porous_komatsu")
    values = [row["value"] for row in komatsu["range"]["rows"]]
    expected = [list(value) if isinstance(value, tuple) else value
                for key, value in ranges.model_dump().items() if key != "provenance"]
    assert values == expected and len(values) == len(KomatsuRanges.model_fields) - 1
    assert komatsu["citation"]["reference"] == ranges.provenance.reference
    assert komatsu["citation"]["doi"] == ranges.provenance.doi
    with CATALOGUE.open("rb") as handle:
        raw = next(model for model in tomllib.load(handle)["model"] if model["id"] == "porous_komatsu")
    assert set(raw["range"]) == {"registry", "at", "mark"}, "Komatsu 範圍又抄了一份"
    assert not {"title", "venue", "doi"} & set(raw["citation"]), "Komatsu 書目又抄了一份"


def test_catalogue_toml_writes_no_range_numbers() -> None:
    """範圍與參數界線都現讀：目錄檔的範圍欄位不准出現數字＋單位的寫法。"""
    with CATALOGUE.open("rb") as handle:
        raw = tomllib.load(handle)
    for model in raw["model"]:
        note = model["range"].get("note", "")
        assert not re.search(r"\d+(\.\d+)?\s*(Pa|Hz|kg|m²|m\b)", note), model["id"]


def test_api_serves_the_catalogue_and_page_folds_it(tmp_path: Path) -> None:
    with TestClient(create_app(GuiSettings(engine_commit="a" * 40, data_dir=tmp_path)), base_url="http://127.0.0.1") as client:
        body = client.get("/api/params").json()
    assert [model["id"] for model in body["models"]] == [model.id for model in _catalogue().model]
    for model in body["models"]:
        assert {"name", "code", "params", "range", "citation", "blurb", "origin"} <= set(model)
        for param in model["params"]:
            assert param["name"], f"{model['id']} 的 {param['key']} 沒有中文名"
    html = (STATIC / "params.html").read_text(encoding="utf-8")
    tag = re.search(r"<details[^>]*id=\"models\"[^>]*>", html)
    assert tag is not None and "open" not in tag.group(0), "吸音模型那段要預設收起"
    assert "<summary>吸音模型</summary>" in html
    script = (STATIC / "params.js").read_text(encoding="utf-8")
    assert "body.models" in script and not re.search(r"\.open\s*=|setAttribute\(\"open\"", script)


def _bad_sample(tmp_path: Path, model: str) -> Path:
    params = "" if "params =" in model else "params = []\n"
    path = tmp_path / "catalogue.toml"
    path.write_text('[[model]]\nid = "m"\nname = "名"\ncode = "src/aosr/gui/model_catalogue.py::Model"\ncompute = ["src/aosr/gui/model_catalogue.py::Model"]\n'
                    f'blurb = "說明"\n{params}origin = {{ text = "t", at = "x.py:1", mark = "t" }}\n{model}\n', encoding="utf-8")
    return path


def test_bad_sample_base_loads_without_the_bad_part(tmp_path: Path) -> None:
    """壞樣本的控制組：同一份底稿不加壞料要載得進來，壞樣本紅才是那一格的錯。"""
    catalogue = load_model_catalogue(_bad_sample(tmp_path, 'range = { note = "未列有效範圍" }\ncitation = { authors = ["A"], at = "x.py:1" }'))
    assert catalogue.model[0].range.note == NO_RANGE


@pytest.mark.parametrize("model", [
    'range = { note = "大概 100 Hz 以上" }\ncitation = { authors = ["A"], at = "x.py:1" }',
    'range = { note = "未列有效範圍", at = "x.py:1" }',
    'range = { registry = "komatsu", note = "x", at = "x.py:1", mark = "K" }',
    'range = { note = "未列有效範圍" }\ncitation = { authors = ["K"], registry = "komatsu", doi = "d", at = "x.py:1" }',
    'range = { note = "未列有效範圍" }\nparams = ["PorousLayer.made_up"]',
], ids=["range note without source", "at without mark", "registry and note", "registry and copied doi", "made-up param"])
def test_bad_catalogue_refuses_to_load(tmp_path: Path, model: str) -> None:
    with pytest.raises(ValidationError):
        load_model_catalogue(_bad_sample(tmp_path, model))


# 主對話 11-28 用 Crossref 核過的外部書目（工作區 verified-references.md）：代號 → DOI 網址。考卷不連網，只核結構與字串。
VERIFIED_DOIS = {
    "miki_1990": "https://doi.org/10.1250/ast.11.19",
    "maa_1998": "https://doi.org/10.1121/1.423870",
    "ingard_1953": "https://doi.org/10.1121/1.1907235",
    "kristiansen_vigran_1994": "https://doi.org/10.1016/0003-682X(94)90039-6",
    "delany_bazley_1970": "https://doi.org/10.1016/0003-682X(70)90031-9",
    "allard_champoux_1992": "https://doi.org/10.1121/1.402824",
    "johnson_koplik_dashen_1987": "https://doi.org/10.1017/S0022112087000727",
    "champoux_allard_1991": "https://doi.org/10.1063/1.349482",
}


def test_verified_references_carry_doi_and_check_stamp() -> None:
    references = {item.id: item for item in _catalogue().reference}
    assert {key: item.doi for key, item in references.items()} == VERIFIED_DOIS
    assert all(item.verified == "Crossref 2026-11-28" and item.text for item in references.values())


def test_verified_references_hang_where_the_note_says() -> None:
    """書目核過≠aosr 照它實作：Miki 掛引用並註明係數未核；孔板三筆只當相關文獻；Komatsu 掛背景；JCA 兩筆只在「沒有實作」。"""
    models = {model.id: model for model in _catalogue().model}
    miki = models["porous_miki"].citation
    assert miki is not None and miki.reference == "miki_1990" and "未核對" in miki.note
    assert models["micro_perforated"].related == ("maa_1998",)
    assert models["perforated"].related == ("ingard_1953",)
    assert models["slotted"].related == ("kristiansen_vigran_1994",)
    assert set(models["porous_komatsu"].background) == {"delany_bazley_1970", "allard_champoux_1992"}
    jca = {"johnson_koplik_dashen_1987", "champoux_allard_1991"}
    for model in models.values():
        hung = set(model.related) | set(model.background) | ({model.citation.reference} if model.citation else set())
        assert not hung & jca, f"JCA 書目不准掛在現有模型 {model.id}"
    unimplemented = {item.id: item for item in _catalogue().unimplemented}
    assert set(unimplemented["jca"].references) == jca
    assert "JCA" in _line(unimplemented["jca"].at)


def test_api_shows_references_with_their_caveats(tmp_path: Path) -> None:
    with TestClient(create_app(GuiSettings(engine_commit="a" * 40, data_dir=tmp_path)), base_url="http://127.0.0.1") as client:
        body = client.get("/api/params").json()
    models = {model["id"]: model for model in body["models"]}
    assert models["porous_miki"]["citation"]["doi"] == VERIFIED_DOIS["miki_1990"]
    assert "未核對" in models["porous_miki"]["citation"]["note"]
    related = models["micro_perforated"]["related"]
    assert related["note"] == RELATED_NOTE and related["items"][0]["doi"] == VERIFIED_DOIS["maa_1998"]
    assert [item["id"] for item in body["unimplemented"]] == ["jca"]
    script = (STATIC / "params.js").read_text(encoding="utf-8")
    assert "body.unimplemented" in script and "related" in script and "background" in script


def test_bad_reference_id_refuses_to_load(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        load_model_catalogue(_bad_sample(tmp_path, 'range = { note = "未列有效範圍" }\nrelated = ["nobody_2000"]'))
