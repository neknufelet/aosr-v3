"""型別與高度考卷：答案由 #559 施工單第 1–3、8 條手算；浮點推出值明列。"""
import json
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from aosr.config.paths import config_path
from aosr.reporting import scheme_cli
from aosr.reporting.scheme import Scheme
from aosr.reporting.validation import SchemeValidationError, checked_inputs, furniture_problems, validate_scheme
from tests.engine import _speaker_setup_cases as cases
from tests.engine._directivity import DIRECTIVITY
from tests.engine._furniture_cases import CAPABILITIES, fingerprint, pair, reference_document


@pytest.mark.parametrize("kind,mount,accepted", [
    ("bookshelf", "stand", True), ("bookshelf", "desk", True), ("floorstanding", "floor", True),
    ("floorstanding", "stand", False), ("floorstanding", "desk", False), ("bookshelf", "floor", False),
])
def test_kind_and_mount_combinations(kind: str, mount: str, accepted: bool) -> None:
    document = cases.document(mount)
    document["speaker_setup"] = cases.setup(kind, mount)
    if accepted:
        scheme = Scheme.model_validate(document)
        assert scheme.model_dump(mode="json")["speaker_setup"] == document["speaker_setup"]
    else:
        with pytest.raises(ValidationError, match="書架喇叭|落地喇叭"):
            Scheme.model_validate(document)


@pytest.mark.parametrize("field", ["cabinet", "representative"])
def test_cabinet_and_representative_are_required(field: str) -> None:
    document = cases.document()
    cast(dict[str, object], document["speaker_setup"]).pop(field)
    with pytest.raises(ValidationError, match=field):
        Scheme.model_validate(document)


@pytest.mark.parametrize("field", ["width_m", "depth_m", "height_m", "acoustic_center_behind_front_m",
                                  "acoustic_center_above_bottom_m"])
def test_every_cabinet_field_is_required(field: str) -> None:
    document = cases.document()
    cast(dict[str, object], cast(dict[str, object], document["speaker_setup"])["cabinet"]).pop(field)
    with pytest.raises(ValidationError, match=field):
        Scheme.model_validate(document)


@pytest.mark.parametrize("field,value", [
    ("width_m", 0.0), ("depth_m", -0.1), ("height_m", 0.0), ("height_m", float("nan")),
    ("acoustic_center_behind_front_m", -0.01), ("acoustic_center_behind_front_m", 0.28),
    ("acoustic_center_above_bottom_m", -0.01), ("acoustic_center_above_bottom_m", 0.356),
])
def test_invalid_cabinet_is_rejected(field: str, value: float) -> None:
    document = cases.document()
    cast(dict[str, object], cast(dict[str, object], document["speaker_setup"])["cabinet"])[field] = value
    with pytest.raises(ValidationError):
        Scheme.model_validate(document)


@pytest.mark.parametrize("value", ["true", 1, None])
def test_representative_requires_a_boolean(value: object) -> None:
    document = cases.document()
    cast(dict[str, object], document["speaker_setup"])["representative"] = value
    with pytest.raises(ValidationError):
        Scheme.model_validate(document)


def test_omitted_and_null_setup_have_identical_output_and_scene() -> None:
    omitted = Scheme.model_validate(reference_document())
    explicit_null = Scheme.model_validate(reference_document() | {"speaker_setup": None})
    assert explicit_null == omitted
    assert explicit_null.model_dump_json() == omitted.model_dump_json()
    for scheme in (omitted, explicit_null):
        assert "speaker_setup" not in scheme.model_dump()
        assert "speaker_setup" not in json.loads(scheme.model_dump_json())
    with_setup = Scheme.model_validate(reference_document() | {"speaker_setup": cases.setup()})
    assert pair(with_setup) == pair(omitted)
    assert fingerprint(with_setup) == fingerprint(omitted)


@pytest.mark.parametrize("kinds,accepted", [((), False), (("coffee_table",), True), (("desk",), True),
                                          (("coffee_table", "desk"), False)])
def test_desk_mount_requires_exactly_one_table(kinds: tuple[str, ...], accepted: bool) -> None:
    document = cases.document("desk")
    document["furniture"] = [cases.table(kind, kind) for kind in kinds]
    if accepted:
        scheme = Scheme.model_validate(document)
        assert scheme.furniture and {item.kind.value for item in scheme.furniture} == set(kinds)
    else:
        with pytest.raises(ValidationError, match=f"喇叭放桌面.*現在有 {len(kinds)} 件"):
            Scheme.model_validate(document)


@pytest.mark.parametrize("mount,left_z,right_z", [
    ("desk", 0.935, 0.935), ("floor", 0.8, 0.8), ("stand", 0.42, 1.83),
])
def test_valid_heights_preserve_coordinates(mount: str, left_z: float, right_z: float) -> None:
    # 手算 0.70 + 0.03 + 0.205 = 0.935；Python 加總的尾差必須確實存在。
    assert (0.70 + 0.03) + 0.205 == 0.9349999999999999
    assert (0.70 + 0.03) + 0.205 != 0.935
    scheme, _ = checked_inputs(cases.document(mount, left_z=left_z, right_z=right_z),
                               capabilities=CAPABILITIES, directivity=DIRECTIVITY)
    assert scheme.speakers["left"].z == left_z
    assert scheme.speakers["right"].z == right_z
    assert not furniture_problems(scheme)


@pytest.mark.parametrize("mount,z,message", [
    ("desk", 0.935 + 1e-6, "喇叭 left 的高度 0.9350010000000001 m 跟擺法推出值不同："
     "桌面頂 0.73 m＋聲學中心離箱底 0.205 m＝0.9349999999999999 m"),
    ("floor", 0.81, "喇叭 left 的高度 0.81 m 跟擺法推出值不同：落地喇叭聲學中心離地 0.8 m"),
])
def test_height_mismatch_is_rejected_without_overwrite(mount: str, z: float, message: str) -> None:
    document = cases.document(mount, left_z=z)
    scheme = Scheme.model_validate(document)
    if mount == "floor":
        assert scheme.furniture is None
    assert [(problem.path, problem.message) for problem in furniture_problems(scheme)] == [("speakers.left.z", message)]
    assert [(problem.path, problem.message) for problem in validate_scheme(document, capabilities=CAPABILITIES,
                                                                         directivity=DIRECTIVITY)] == [("speakers.left.z", message)]
    with pytest.raises(SchemeValidationError, match="高度"):
        checked_inputs(document, capabilities=CAPABILITIES, directivity=DIRECTIVITY)
    assert scheme.speakers["left"].z == z


@pytest.mark.parametrize("mount", ["desk", "floor"])
def test_cli_run_rejects_height_before_calculation(tmp_path: Path, mount: str) -> None:
    path, output = tmp_path / "scheme.json", tmp_path / "result.json"
    path.write_text(json.dumps(cases.document(mount, left_z=0.81 if mount == "floor" else 0.936)))
    # 主線一般 run 會把驗證例外往外傳，實際 CLI 因此非零退出；不改既有退出契約。
    with pytest.raises(SchemeValidationError, match="speakers.left.z：喇叭 left 的高度"):
        scheme_cli.main(["run", str(path), "--out", str(output), "--engine-commit", "test",
                         "--capabilities", str(config_path("capabilities.toml"))])
    assert not output.exists()
